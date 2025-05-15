import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("torch") 
install("peft")    
install("torchvision")
install("backoff")
install("flash-attn")
install("tqdm")
install("jiwer")
install("librosa")

import argparse
import torch
import numpy as np
import os
import json
import random
import types
from collections import defaultdict
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from jiwer import wer
import librosa
from tqdm import tqdm


# Metadata from the provided file
CMU_ARCTIC_VARIETIES = {
    'aew': 'US English', 'ahw': 'German English', 'aup': 'Indian English',
    'awb': 'Scottish English', 'axb': 'Indian English', 'bdl': 'US English',
    'clb': 'US English', 'eey': 'US English', 'fem': 'German English',
    'gka': 'Indian English', 'jmk': 'Canadian English', 'ksp': 'Indian English',
    'ljm': 'US English', 'lnh': 'US English', 'rms': 'US English',
    'rxr': 'Israeli English', 'slt': 'US English', 'slp': 'Indian English'
}

HISP_ENG_ORIGINS = {
    '0': None, '1': 'Argentina', '2': 'Mexico', '3': 'Argentina', '4': 'Argentina',
    '5': 'Argentina', '6': 'Argentina', '7': None, '8': None, '9': None, '10': None,
    '11': None, '12': None, '13': None, '14': None, '15': None, '16': None, '17': None,
    '18': 'Argentina', '19': 'Chile'
}


def monkey_patch_phi4_model(model):
    """
    Apply a monkey patch to the Phi-4 model to handle the num_logits_to_keep=None issue.
    This directly patches the model's forward method to handle the None case.
    """
    original_forward = model.forward
    
    def patched_forward(self, *args, **kwargs):
        # Store the original num_logits_to_keep
        original_num_logits_to_keep = getattr(self.config, 'num_logits_to_keep', None)
        
        # Set a default value if it's None
        if original_num_logits_to_keep is None:
            self.config.num_logits_to_keep = 1
        
        try:
            # Call the original forward method
            return original_forward(*args, **kwargs)
        finally:
            # Restore the original value
            self.config.num_logits_to_keep = original_num_logits_to_keep
    
    # Replace the forward method
    model.forward = types.MethodType(patched_forward, model)
    
    return model


def load_phi4_model(cache_dir=None):
    """Load Phi-4 multimodal model with patching for num_logits_to_keep issue"""
    print("Loading Phi-4 model...")
    model_path = 'microsoft/Phi-4-multimodal-instruct'
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        attn_implementation='flash_attention_2',
        cache_dir=cache_dir
    ).cuda()
    
    # Apply the monkey patch to fix the num_logits_to_keep issue
    model = monkey_patch_phi4_model(model)
    
    generation_config = GenerationConfig.from_pretrained(model_path)
    
    # Also set num_logits_to_keep in the generation config for good measure
    if not hasattr(generation_config, 'num_logits_to_keep') or generation_config.num_logits_to_keep is None:
        generation_config.num_logits_to_keep = 1
    
    return model, processor, generation_config


def normalize_audio(audio_data):
    """
    Safely normalize audio data to float32 in range [-1.0, 1.0]
    Handles various input types and detects/corrects FLAC normalization issues
    """
    if audio_data is None or len(audio_data) == 0:
        return np.array([], dtype=np.float32)

    # Convert list to numpy array if needed
    if isinstance(audio_data, list):
        audio_data = np.array(audio_data, dtype=np.float32)

    # Handle different dtypes
    if np.issubdtype(audio_data.dtype, np.integer):
        # For integer types, normalize based on the max possible value for that type
        max_value = float(np.iinfo(audio_data.dtype).max)
        audio_float = audio_data.astype(np.float32) / max_value
    else:
        # For float types, just ensure it's float32
        audio_float = audio_data.astype(np.float32)

    # Check for FLAC normalization bug where -1.0 flips to +1.0
    # This can cause extreme positive peaks in otherwise normal audio
    if np.max(audio_float) > 0.99 and np.min(audio_float) > -0.5:
        # This pattern might indicate the FLAC bug where negative values got flipped
        print("Possible FLAC normalization bug detected (missing negative values)")
        # Look for unusual positive peaks that might actually be negative
        threshold = 0.9
        potential_flips = audio_float > threshold
        if np.any(potential_flips):
            # Flip the potentially affected values
            audio_float[potential_flips] = -audio_float[potential_flips]
            print(f"Corrected {np.sum(potential_flips)} potentially flipped samples")

    # Clip extreme values to ensure the range is reasonable
    if np.any(np.abs(audio_float) > 1.1):
        print(f"Clipping extreme audio values (max={np.max(np.abs(audio_float)):.2f})")
        audio_float = np.clip(audio_float, -1.0, 1.0)

    return audio_float


def resample_audio(audio_data, orig_sr, target_sr=16000):
    """Resample audio to target sample rate"""
    # Validate input data
    if audio_data is None or len(audio_data) == 0:
        raise ValueError("Empty audio data provided for resampling")

    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Invalid sample rates: orig={orig_sr}, target={target_sr}")

    if orig_sr == target_sr:
        return audio_data

    try:
        # First normalize the audio to proper float representation
        audio_float = normalize_audio(audio_data)

        # Resample using librosa
        try:
            resampled_audio = librosa.resample(
                y=audio_float,
                orig_sr=orig_sr,
                target_sr=target_sr
            )
        except Exception as e:
            # If that fails, try the legacy resampling approach
            print(f"Standard resampling failed: {e}. Trying legacy resampler...")
            resampled_audio = librosa.resample(
                y=audio_float,
                orig_sr=orig_sr,
                target_sr=target_sr,
                res_type='kaiser_fast'
            )

        # Verify resampled data is valid
        if resampled_audio is None or len(resampled_audio) == 0:
            raise ValueError("Resampling resulted in empty audio data")

        return resampled_audio
    except Exception as e:
        print(f"Failed to resample audio: {str(e)}")
        raise


def normalize_text(text):
    """Normalize text for WER calculation"""
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    for punct in ['.', ',', '?', '!', ';', ':', '"', "'", '(', ')', '[', ']']:
        text = text.replace(punct, ' ')
    # Normalize whitespace
    return ' '.join(text.split())


def transcribe_with_shots(model, processor, generation_config, test_audio, test_sr, examples=None, num_shots=0, 
                          prompt_type="standard", speaker_condition="same"):
    """
    Run ASR with a specified number of in-context examples (shots)
    
    Parameters:
    - speaker_condition: "same" or "different" - whether to refer to the test audio as from the same or different speaker
    """
    
    # Ensure audio is normalized and at 16kHz
    test_audio = normalize_audio(test_audio)
    if test_sr != 16000:
        test_audio = resample_audio(test_audio, test_sr, 16000)
    
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    # For zero-shot, use simpler prompt matching Colab notebook
    if num_shots == 0:
        # Choose between standard and variation prompt
        if prompt_type == "variation":
            # Variation for non-native speakers
            prompt = f"{user_prompt}<|audio_1|>Transcribe the audio clip from a non-native English speaker into text.{prompt_suffix}{assistant_prompt}"
        else:
            # Standard prompt
            prompt = f"{user_prompt}<|audio_1|>Transcribe the audio clip into text.{prompt_suffix}{assistant_prompt}"
        
        # Pass audio data as a list of tuples with (audio_data, sample_rate)
        inputs = processor(text=prompt, audios=[(test_audio, 16000)], return_tensors='pt').to('cuda')
    else:
        # For few-shot, use in-context examples
        all_audios = []
        
        # Process examples (take only up to num_shots)
        limited_examples = examples[:num_shots]
        for example in limited_examples:
            # Ensure example audio is properly normalized and at 16kHz
            audio = normalize_audio(example['audio'])
            if example['sample_rate'] != 16000:
                audio = resample_audio(audio, example['sample_rate'], 16000)
            # Store as tuples of (audio_data, sample_rate)
            all_audios.append((audio, 16000))
            
        # Add test audio as the final audio
        all_audios.append((test_audio, 16000))
        
        # Set speaker context based on condition
        speaker_reference = "the same speaker" if speaker_condition == "same" else "a different speaker"
        
        # Use the Colab notebook prompting approach with the speaker condition
        prompt = f"{user_prompt}I'll provide {'an example' if num_shots == 1 else f'{num_shots} examples'} of speech from a non-native English speaker, followed by the correct transcription. Then I'll give you a new audio from {speaker_reference} to transcribe.{prompt_suffix}{assistant_prompt}I understand. I'll listen to the {'example' if num_shots == 1 else 'examples'} and use {'it' if num_shots == 1 else 'them'} to accurately transcribe the final audio.{prompt_suffix}"
        
        # Add each example with its transcript
        for i, example in enumerate(limited_examples):
            if prompt_type == "variation":
                # Variation with "Transcription:" prefix
                prompt += f"{user_prompt}<|audio_{i+1}|>Transcribe this audio:{prompt_suffix}{assistant_prompt}Transcription: {example['transcript']}{prompt_suffix}"
            else:
                # Standard approach
                prompt += f"{user_prompt}<|audio_{i+1}|>Transcribe this audio:{prompt_suffix}{assistant_prompt}{example['transcript']}{prompt_suffix}"
        
        # Add the test audio for transcription with speaker reference
        if prompt_type == "variation":
            # Variation with "Transcription:" prefix for output
            prompt += f"{user_prompt}<|audio_{len(limited_examples)+1}|>Please transcribe this audio from {speaker_reference}:{prompt_suffix}{assistant_prompt}Transcription:"
        else:
            # Standard approach
            prompt += f"{user_prompt}<|audio_{len(limited_examples)+1}|>Please transcribe this audio from {speaker_reference}:{prompt_suffix}{assistant_prompt}"
        
        # Process with the model
        inputs = processor(text=prompt, audios=all_audios, return_tensors='pt').to('cuda')
    
    # Set num_logits_to_keep at all levels to ensure it's not None
    if not hasattr(generation_config, 'num_logits_to_keep') or generation_config.num_logits_to_keep is None:
        generation_config.num_logits_to_keep = 1
    
    if hasattr(model.config, 'num_logits_to_keep') and model.config.num_logits_to_keep is None:
        model.config.num_logits_to_keep = 1
    
    # Keep the original model generation parameters unchanged
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=1200,
            do_sample=False,  # Use greedy decoding
            num_beams=1,      # Simple beam search
            num_logits_to_keep=1,  # Explicitly pass parameter
        )
    
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return response


def find_transcript_field(sample):
    """Find the transcript field in a sample"""
    possible_fields = ['transcript', 'text']
    for field in possible_fields:
        if field in sample:
            return field
    return None


def get_audio_duration(sample):
    """Get the duration of an audio sample with error handling"""
    try:
        if 'duration' in sample:
            return sample['duration']
        
        if 'audio' in sample and isinstance(sample['audio'], dict):
            if 'duration' in sample['audio']:
                return sample['audio']['duration']
            
            if 'array' in sample['audio'] and 'sampling_rate' in sample['audio']:
                return len(sample['audio']['array']) / sample['audio']['sampling_rate']
        
        return None
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return None


def get_variety_from_dataset(dataset_name, sample):
    """Get variety information from a sample based on the dataset"""
    if dataset_name == "l2-arctic":
        return sample.get('l1', 'unknown')
    elif dataset_name == "cmu-arctic":
        # For CMU-Arctic, we need to extract variety from sample
        # This might be in the audio path or we need to use the mapping
        if 'audio' in sample and 'path' in sample['audio']:
            # Extract speaker ID from path (assuming format like 'arctic_/path/to/speaker_id/audio.wav')
            path = sample['audio']['path']
            for speaker_id in CMU_ARCTIC_VARIETIES:
                if speaker_id in path:
                    return CMU_ARCTIC_VARIETIES[speaker_id]
        return sample.get('variety', 'unknown')
    elif dataset_name == "hisp-eng":
        # For HISP-ENG, use the speaker number to get origin
        speaker = sample.get('speaker', 'unknown')
        return HISP_ENG_ORIGINS.get(str(speaker), 'unknown')
    else:
        return 'unknown'


def get_speaker_from_dataset(dataset_name, sample, idx=None):
    """Get speaker information from a sample based on the dataset"""
    if dataset_name == "cmu-arctic":
        # For CMU-Arctic, speaker is extracted from audio path
        if 'audio' in sample and 'path' in sample['audio']:
            path = sample['audio']['path']
            for speaker_id in CMU_ARCTIC_VARIETIES:
                if speaker_id in path:
                    return speaker_id
        return f"speaker_{idx}" if idx is not None else "unknown"
    else:
        # For L2-Arctic and HISP-ENG, there should be a speaker field
        possible_fields = ['speaker', 'speaker_id', 'speakerId', 'speaker_ID']
        for field in possible_fields:
            if field in sample:
                return str(sample[field])
        return f"speaker_{idx}" if idx is not None else "unknown"


def filter_dataset_by_variety(dataset_name, dataset, max_trials, max_shots, global_seed=42):
    """
    Filter a dataset based on variety requirements:
    1. Speakers must belong to a variety with at least one other speaker
    2. Speakers must have at least (max_shots + 1) audio files for max_shots evaluation
    """
    print(f"Filtering {dataset_name} dataset with {len(dataset)} samples")
    print(f"Parameters: max_trials={max_trials}, max_shots={max_shots}, min_duration=2.5s")
    
    # Group by variety and speaker
    variety_speakers = defaultdict(lambda: defaultdict(list))
    
    # Collect all samples by variety and speaker
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            speaker = get_speaker_from_dataset(dataset_name, sample, i)
            variety = get_variety_from_dataset(dataset_name, sample)
            
            # Skip samples with unknown variety for HISP-ENG
            if dataset_name == "hisp-eng" and variety == 'unknown':
                continue
                
            variety_speakers[variety][speaker].append(i)
            
        except Exception as e:
            print(f"Error accessing sample {i}: {e}")
            continue
    
    print(f"Found {len(variety_speakers)} varieties")
    
    # Filter varieties with at least 2 speakers
    eligible_varieties = {}
    for variety, speakers in variety_speakers.items():
        if len(speakers) >= 2:
            eligible_varieties[variety] = speakers
            print(f"Variety '{variety}': {len(speakers)} speakers")
    
    print(f"Found {len(eligible_varieties)} varieties with ≥2 speakers")
    
    # Check each speaker for sufficient samples (max_shots + 1 minimum)
    final_samples = []
    min_samples_needed = max_shots + 1  # max_shots for examples + 1 for test
    
    for variety, speakers in eligible_varieties.items():
        print(f"\nProcessing variety: {variety}")
        
        for speaker, indices in speakers.items():
            print(f"  Speaker {speaker}: checking {len(indices)} samples...")
            
            # Verify samples and check duration
            valid_samples = []
            for idx in indices:
                try:
                    sample = dataset[idx]
                    
                    # Check duration
                    duration = get_audio_duration(sample)
                    if duration is None or duration < 2.5:
                        continue
                    
                    # Check if audio and transcript are valid
                    if ('audio' in sample and isinstance(sample['audio'], dict) and
                        'array' in sample['audio'] and len(sample['audio']['array']) > 0):
                        transcript_field = find_transcript_field(sample)
                        if transcript_field and sample[transcript_field]:
                            valid_samples.append(sample)
                except Exception as e:
                    continue
            
            if len(valid_samples) >= min_samples_needed:
                # Shuffle and take up to 50 samples
                speaker_seed = global_seed + hash(f"{variety}_{speaker}") % 10000
                random.seed(speaker_seed)
                random.shuffle(valid_samples)
                final_samples.extend(valid_samples[:50])
                print(f"    Added {min(len(valid_samples), 50)} samples for speaker {speaker}")
            else:
                print(f"    Speaker {speaker} has only {len(valid_samples)}/{min_samples_needed} valid samples - skipping")
    
    return final_samples


def pick_examples(pool, k, rng):
    """Return k unique items from pool without replacement."""
    return rng.sample(pool, k) if len(pool) >= k else []


def other_speaker_pool(variety_map, variety_id, exclude_spk):
    """Return list of speakers in this variety except the one to exclude."""
    return [s for s in variety_map[variety_id] if s != exclude_spk]


def evaluate_dataset_multi_shot(dataset_name, model, processor, generation_config, 
                                output_dir, n_shots, max_trials, prompt_type,
                                speaker_condition, max_shots, global_seed=42):
    """Runs one (dataset, n_shots, prompt_type, speaker_condition) slice."""
    shot_tag = f"shot{n_shots}"
    
    print(f"\n=== EVALUATING {dataset_name.upper()} DATASET ({shot_tag}) ===\n")
    print(f"Using prompt type: {prompt_type}")
    print(f"Using speaker condition: {speaker_condition}")
    print(f"Running up to {max_trials} trials per speaker")
    
    # Special handling for CMU-Arctic
    if dataset_name == "cmu-arctic":
        return evaluate_cmu_arctic_multi_shot(
            model, processor, generation_config, 
            output_dir, n_shots, max_trials, 
            prompt_type, speaker_condition, global_seed
        )
    
    # Load dataset
    try:
        if dataset_name == "l2-arctic":
            print("Loading L2-Arctic dataset...")
            dataset = load_dataset("NathanRoll/l2-arctic-dataset-250", split="train")
        elif dataset_name == "hisp-eng":
            print("Loading HISP-ENG dataset...")
            dataset = load_dataset("NathanRoll/hisp-eng", split="train")
        else:
            print(f"Unknown dataset: {dataset_name}")
            return
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        return
    
    # Filter dataset
    filtered_samples = filter_dataset_by_variety(
        dataset_name, dataset, max_trials, max_shots, global_seed
    )
    
    if not filtered_samples:
        print(f"No eligible speakers found in {dataset_name}. Skipping dataset.")
        return
    
    # Group samples by variety and speaker
    variety_speaker_samples = defaultdict(lambda: defaultdict(list))
    
    for sample in filtered_samples:
        speaker_id = get_speaker_from_dataset(dataset_name, sample)
        variety_id = get_variety_from_dataset(dataset_name, sample)
        
        # Skip samples with unknown variety for HISP-ENG
        if dataset_name == "hisp-eng" and variety_id == 'unknown':
            continue
            
        variety_speaker_samples[variety_id][speaker_id].append(sample)
    
    # Create output directory
    dataset_output_dir = os.path.join(output_dir, f"{dataset_name}_{speaker_condition}_speaker_{prompt_type}")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Results dictionary with dynamic keys
    all_results = {
        "overall": {
            "samples": 0,
            "runs_per_condition": {shot_tag: 0},
            f"{shot_tag}_wer": 0,
            f"{shot_tag}_wer_avg": 0
        },
        "speakers": {}
    }
    
    # Process each variety and speaker
    for variety_id, speakers in variety_speaker_samples.items():
        print(f"\nProcessing variety: {variety_id}")
        
        for speaker_id, speaker_samples in speakers.items():
            print(f"  Evaluating speaker: {speaker_id}")
            
            # Get other speakers in same variety for different speaker condition
            other_speakers = [s for s in speakers.keys() if s != speaker_id]
            
            # Pre-collect valid clips (up to 50)
            pool = speaker_samples[:50]
            
            # Shuffle the pool for variety
            pool_seed = global_seed + hash(speaker_id) % 10000
            random.seed(pool_seed)
            random.shuffle(pool)
            
            # Check eligibility
            if len(pool) < n_shots + 1:
                print(f"    Skipping speaker {speaker_id} - not enough samples ({len(pool)} < {n_shots + 1})")
                continue
            
            # Calculate number of trials
            trials = min(len(pool) - n_shots, max_trials)
            
            # Initialize speaker results
            speaker_results = {
                "variety": variety_id,
                "runs_per_condition": {shot_tag: 0},
                f"{shot_tag}_wer_avg": 0,
                f"{shot_tag}_wer_values": []
            }
            
            # Process each trial
            for trial_idx in range(trials):
                # Use the first trial items as test clips
                test_sample = pool[trial_idx]
                
                try:
                    transcript_field = find_transcript_field(test_sample)
                    if not transcript_field:
                        print(f"    No transcript field found in trial {trial_idx}")
                        continue
                    
                    ground_truth = test_sample[transcript_field]
                    test_audio = test_sample['audio']['array']
                    test_sr = test_sample['audio']['sampling_rate']
                    
                    # Build examples list
                    examples = []
                    
                    if speaker_condition == "same":
                        # Use samples from same speaker as context
                        # Build fresh candidate list each trial (exclude the test clip)
                        context_candidates = [c for c in pool if c is not test_sample]
                        
                        # Drop any clip whose transcript is identical to the test transcript
                        test_norm = normalize_text(ground_truth)
                        context_candidates = [
                            c for c in context_candidates 
                            if normalize_text(c[find_transcript_field(c)]) != test_norm
                        ]
                        
                        if len(context_candidates) >= n_shots:
                            rng = random.Random(global_seed + hash(f"{speaker_id}_{trial_idx}") % 10000)
                            selected_context = pick_examples(context_candidates, n_shots, rng)
                            
                            for ctx_sample in selected_context:
                                ctx_field = find_transcript_field(ctx_sample)
                                examples.append({
                                    "audio": ctx_sample["audio"]["array"],
                                    "sample_rate": ctx_sample["audio"]["sampling_rate"],
                                    "transcript": ctx_sample[ctx_field]
                                })
                    
                    else:  # different speaker condition
                        # Use samples from other speakers in same variety
                        if other_speakers:
                            speaker_seed = global_seed + hash(f"{speaker_id}_{trial_idx}") % 10000
                            rng = random.Random(speaker_seed)
                            
                            # Pick a random other speaker from same variety
                            other_speaker = rng.choice(other_speakers)
                            other_context = variety_speaker_samples[variety_id][other_speaker]
                            
                            # Filter out samples with same transcript as test audio
                            available_context = []
                            test_transcript_norm = normalize_text(ground_truth)
                            
                            for ctx_sample in other_context:
                                ctx_transcript_field = find_transcript_field(ctx_sample)
                                if ctx_transcript_field:
                                    ctx_transcript_norm = normalize_text(ctx_sample[ctx_transcript_field])
                                    if ctx_transcript_norm != test_transcript_norm:
                                        available_context.append(ctx_sample)
                            
                            # Select n_shots context examples
                            if len(available_context) >= n_shots:
                                selected_context = pick_examples(available_context, n_shots, rng)
                                
                                for ctx_sample in selected_context:
                                    ctx_transcript_field = find_transcript_field(ctx_sample)
                                    if ctx_transcript_field:
                                        examples.append({
                                            'audio': ctx_sample['audio']['array'],
                                            'sample_rate': ctx_sample['audio']['sampling_rate'],
                                            'transcript': ctx_sample[ctx_transcript_field]
                                        })
                    
                    # Run evaluation
                    if n_shots == 0 or len(examples) == n_shots:
                        result = transcribe_with_shots(
                            model, processor, generation_config,
                            test_audio, test_sr, examples, n_shots,
                            prompt_type, speaker_condition
                        )
                        
                        normalized_truth = normalize_text(ground_truth)
                        normalized_result = normalize_text(result)
                        wer_score = wer(normalized_truth, normalized_result)
                        
                        speaker_results[f"{shot_tag}_wer_values"].append(wer_score)
                        speaker_results["runs_per_condition"][shot_tag] += 1
                        all_results["overall"][f"{shot_tag}_wer"] += wer_score
                        all_results["overall"]["runs_per_condition"][shot_tag] += 1
                        all_results["overall"]["samples"] += 1
                        
                        print(f"    Trial {trial_idx}, {shot_tag} WER: {wer_score:.4f}")
                    else:
                        print(f"    Could not get {n_shots} unique examples for trial {trial_idx}")
                
                except Exception as e:
                    print(f"    Error processing trial {trial_idx}: {e}")
                    continue
            
            # Calculate speaker averages
            if speaker_results["runs_per_condition"][shot_tag] > 0:
                speaker_results[f"{shot_tag}_wer_avg"] = sum(speaker_results[f"{shot_tag}_wer_values"]) / speaker_results["runs_per_condition"][shot_tag]
            
            # Store speaker results
            all_results["speakers"][f"{variety_id}_{speaker_id}"] = speaker_results
            
            # Save individual speaker results
            speaker_file = os.path.join(dataset_output_dir, f"results_{variety_id}_{speaker_id}_{shot_tag}_{prompt_type}.json")
            with open(speaker_file, 'w') as f:
                json.dump(speaker_results, f, indent=2)
            
            print(f"  Speaker summary: {shot_tag}: {speaker_results[f'{shot_tag}_wer_avg']:.4f}")
    
    # Calculate overall averages
    if all_results["overall"]["runs_per_condition"][shot_tag] > 0:
        all_results["overall"][f"{shot_tag}_wer_avg"] = all_results["overall"][f"{shot_tag}_wer"] / all_results["overall"]["runs_per_condition"][shot_tag]
    
    # Print overall summary
    print(f"\n\n========= OVERALL SUMMARY ({dataset_name.upper()}, {shot_tag}) =========")
    print(f"Total speakers: {len(all_results['speakers'])}")
    print(f"{shot_tag}: {all_results['overall']['runs_per_condition'][shot_tag]} runs, Average WER: {all_results['overall'][f'{shot_tag}_wer_avg']:.4f}")
    
    # Save overall results
    overall_file = os.path.join(dataset_output_dir, f"results_overall_{speaker_condition}_{shot_tag}_{prompt_type}.json")
    with open(overall_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved overall results to {overall_file}")
    return all_results


def evaluate_cmu_arctic_multi_shot(model, processor, generation_config, output_dir, 
                                   n_shots, max_trials, prompt_type="standard", 
                                   speaker_condition="same", global_seed=42):
    """Special case for CMU-Arctic dataset which has a different structure"""
    shot_tag = f"shot{n_shots}"
    
    print(f"\n=== EVALUATING CMU-ARCTIC DATASET ({shot_tag}) ===\n")
    print(f"Using prompt type: {prompt_type}")
    print(f"Using speaker condition: {speaker_condition}")
    print(f"Running up to {max_trials} trials per speaker")
    
    try:
        # Load CMU-Arctic (speakers are split names)
        print("Loading CMU-Arctic dataset...")
        dataset = load_dataset("NathanRoll/cmu-arctic")
        
        # Get available speakers
        speakers = list(dataset.keys())
        print(f"Found {len(speakers)} speakers: {', '.join(speakers)}")
        
        # Group speakers by variety for filtering
        variety_speakers = defaultdict(list)
        for speaker in speakers:
            variety = CMU_ARCTIC_VARIETIES.get(speaker, 'unknown')
            variety_speakers[variety].append(speaker)
        
        # Only keep varieties with at least 2 speakers
        eligible_varieties = {v: s for v, s in variety_speakers.items() if len(s) >= 2}
        print(f"Found {len(eligible_varieties)} varieties with ≥2 speakers")
        
        # Create output directory
        dataset_output_dir = os.path.join(output_dir, f"cmu-arctic_{speaker_condition}_speaker_{prompt_type}")
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Results dictionary
        all_results = {
            "overall": {
                "samples": 0,
                "runs_per_condition": {shot_tag: 0},
                f"{shot_tag}_wer": 0,
                f"{shot_tag}_wer_avg": 0
            },
            "speakers": {}
        }
        
        # Process each speaker in eligible varieties
        for variety, speakers_in_variety in eligible_varieties.items():
            print(f"\nProcessing variety: {variety}")
            
            for speaker in speakers_in_variety:
                print(f"  Evaluating speaker: {speaker}")
                
                try:
                    # Get speaker dataset
                    speaker_dataset = dataset[speaker]
                    
                    # We need enough samples for all test scenarios
                    needed_samples = max_trials + n_shots + 1
                    
                    # Find valid samples with duration >= 2.5s
                    valid_samples = []
                    for i in range(len(speaker_dataset)):
                        if len(valid_samples) >= 50:  # Limit to 50 samples
                            break
                            
                        try:
                            sample = speaker_dataset[i]
                            
                            # Check duration
                            duration = get_audio_duration(sample)
                            if duration is None or duration < 2.5:
                                continue
                                
                            # Check if audio is valid
                            if ('audio' in sample and isinstance(sample['audio'], dict) and
                                'array' in sample['audio'] and len(sample['audio']['array']) > 0):
                                # Find transcript field
                                transcript_field = find_transcript_field(sample)
                                if transcript_field:
                                    valid_samples.append(sample)
                        except Exception as e:
                            # Skip problematic samples
                            continue
                    
                    print(f"    Found {len(valid_samples)} valid samples with duration >= 2.5s")
                    
                    # Pre-collect valid clips (up to 50)
                    pool = valid_samples[:50]
                    
                    # Check eligibility
                    if len(pool) < n_shots + 1:
                        print(f"    Skipping speaker {speaker} - not enough samples ({len(pool)} < {n_shots + 1})")
                        continue
                    
                    # Calculate number of trials
                    trials = min(len(pool) - n_shots, max_trials)
                    
                    # Get other speakers in same variety for different speaker condition
                    other_speakers = [s for s in speakers_in_variety if s != speaker]
                    
                    # Initialize speaker results
                    speaker_results = {
                        "variety": variety,
                        "runs_per_condition": {shot_tag: 0},
                        f"{shot_tag}_wer_avg": 0,
                        f"{shot_tag}_wer_values": []
                    }
                    
                    # Use consistent seed for speaker
                    speaker_seed = global_seed + hash(speaker) % 10000
                    random.seed(speaker_seed)
                    random.shuffle(pool)
                    
                    # Process each trial
                    for trial_idx in range(trials):
                        # Use the first trial items as test clips
                        test_sample = pool[trial_idx]
                        
                        try:
                            transcript_field = find_transcript_field(test_sample)
                            if not transcript_field:
                                print(f"      No transcript field found in trial {trial_idx}")
                                continue
                            
                            ground_truth = test_sample[transcript_field]
                            test_audio = test_sample['audio']['array']
                            test_sr = test_sample['audio']['sampling_rate']
                            
                            # Build examples list
                            examples = []
                            
                            if speaker_condition == "same":
                                # Use samples from same speaker as context
                                # Build candidates excluding the current test sample
                                context_candidates = [s for s in pool if s != test_sample]
                                
                                # Filter out samples with same transcript as test audio
                                test_transcript_norm = normalize_text(ground_truth)
                                filtered_candidates = []
                                for ctx_sample in context_candidates:
                                    ctx_transcript_field = find_transcript_field(ctx_sample)
                                    if ctx_transcript_field:
                                        ctx_transcript_norm = normalize_text(ctx_sample[ctx_transcript_field])
                                        if ctx_transcript_norm != test_transcript_norm:
                                            filtered_candidates.append(ctx_sample)
                                
                                # Select n_shots context examples
                                if len(filtered_candidates) >= n_shots:
                                    trial_seed = global_seed + hash(f"{speaker}_{trial_idx}") % 10000
                                    rng = random.Random(trial_seed)
                                    selected_context = pick_examples(filtered_candidates, n_shots, rng)
                                    
                                    for ctx_sample in selected_context:
                                        ctx_transcript_field = find_transcript_field(ctx_sample)
                                        if ctx_transcript_field:
                                            examples.append({
                                                'audio': ctx_sample['audio']['array'],
                                                'sample_rate': ctx_sample['audio']['sampling_rate'],
                                                'transcript': ctx_sample[ctx_transcript_field]
                                            })
                            
                            else:  # different speaker condition
                                # Use samples from other speakers in same variety
                                if other_speakers:
                                    trial_seed = global_seed + hash(f"{speaker}_{trial_idx}") % 10000
                                    rng = random.Random(trial_seed)
                                    
                                    # Pick a random other speaker from same variety
                                    other_speaker = rng.choice(other_speakers)
                                    other_speaker_dataset = dataset[other_speaker]
                                    
                                    # Collect samples from the other speaker
                                    other_context = []
                                    for i in range(min(50, len(other_speaker_dataset))):  # Limit to reduce processing
                                        try:
                                            sample = other_speaker_dataset[i]
                                            duration = get_audio_duration(sample)
                                            if duration and duration >= 2.5:
                                                other_context.append(sample)
                                        except:
                                            continue
                                    
                                    # Filter out samples with same transcript as test audio
                                    available_context = []
                                    test_transcript_norm = normalize_text(ground_truth)
                                    
                                    for ctx_sample in other_context:
                                        ctx_transcript_field = find_transcript_field(ctx_sample)
                                        if ctx_transcript_field:
                                            ctx_transcript_norm = normalize_text(ctx_sample[ctx_transcript_field])
                                            if ctx_transcript_norm != test_transcript_norm:
                                                available_context.append(ctx_sample)
                                    
                                    # Select n_shots context examples
                                    if len(available_context) >= n_shots:
                                        selected_context = pick_examples(available_context, n_shots, rng)
                                        
                                        for ctx_sample in selected_context:
                                            ctx_transcript_field = find_transcript_field(ctx_sample)
                                            if ctx_transcript_field:
                                                examples.append({
                                                    'audio': ctx_sample['audio']['array'],
                                                    'sample_rate': ctx_sample['audio']['sampling_rate'],
                                                    'transcript': ctx_sample[ctx_transcript_field]
                                                })
                            
                            # Run evaluation
                            if n_shots == 0 or len(examples) == n_shots:
                                result = transcribe_with_shots(
                                    model, processor, generation_config,
                                    test_audio, test_sr, examples, n_shots,
                                    prompt_type, speaker_condition
                                )
                                
                                normalized_truth = normalize_text(ground_truth)
                                normalized_result = normalize_text(result)
                                wer_score = wer(normalized_truth, normalized_result)
                                
                                speaker_results[f"{shot_tag}_wer_values"].append(wer_score)
                                speaker_results["runs_per_condition"][shot_tag] += 1
                                all_results["overall"][f"{shot_tag}_wer"] += wer_score
                                all_results["overall"]["runs_per_condition"][shot_tag] += 1
                                all_results["overall"]["samples"] += 1
                                
                                print(f"      Trial {trial_idx}, {shot_tag} WER: {wer_score:.4f}")
                            else:
                                print(f"      Could not get {n_shots} unique examples for trial {trial_idx}")
                        
                        except Exception as e:
                            print(f"      Error processing trial {trial_idx}: {e}")
                            continue
                    
                    # Calculate speaker averages
                    if speaker_results["runs_per_condition"][shot_tag] > 0:
                        speaker_results[f"{shot_tag}_wer_avg"] = sum(speaker_results[f"{shot_tag}_wer_values"]) / speaker_results["runs_per_condition"][shot_tag]
                    
                    # Store speaker results
                    all_results["speakers"][f"{variety}_{speaker}"] = speaker_results
                    
                    # Save individual speaker results
                    speaker_file = os.path.join(dataset_output_dir, f"results_{variety}_{speaker}_{shot_tag}_{prompt_type}.json")
                    with open(speaker_file, 'w') as f:
                        json.dump(speaker_results, f, indent=2)
                    
                    print(f"    Speaker summary: {shot_tag}: {speaker_results[f'{shot_tag}_wer_avg']:.4f}")
                
                except Exception as e:
                    print(f"    Error processing speaker {speaker}: {e}")
                    continue
        
        # Calculate overall averages
        if all_results["overall"]["runs_per_condition"][shot_tag] > 0:
            all_results["overall"][f"{shot_tag}_wer_avg"] = all_results["overall"][f"{shot_tag}_wer"] / all_results["overall"]["runs_per_condition"][shot_tag]
        
        # Print overall summary
        print(f"\n\n========= OVERALL SUMMARY (CMU-ARCTIC, {shot_tag}) =========")
        print(f"Total speakers: {len(all_results['speakers'])}")
        print(f"{shot_tag}: {all_results['overall']['runs_per_condition'][shot_tag]} runs, Average WER: {all_results['overall'][f'{shot_tag}_wer_avg']:.4f}")
        
        # Save overall results
        overall_file = os.path.join(dataset_output_dir, f"results_overall_{speaker_condition}_{shot_tag}_{prompt_type}.json")
        with open(overall_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Saved overall results to {overall_file}")
        return all_results
    
    except Exception as e:
        print(f"Error evaluating CMU-Arctic dataset: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phi-4-MM 0 through max_shots ASR performance")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save results")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for HuggingFace models")
    parser.add_argument("--dataset", type=str, default="l2-arctic",
                        choices=["l2-arctic", "hisp-eng", "cmu-arctic", "all"],
                        help="Dataset to evaluate (default: l2-arctic)")
    parser.add_argument("--max_shots", type=int, default=5,
                        help="Highest n-shot level to evaluate (default 5)")
    parser.add_argument("--max_trials", type=int, default=50,
                        help="Upper limit on trials per speaker per shot level (default: 50)")
    parser.add_argument("--prompt_type", type=str, default="standard",
                        choices=["standard", "variation", "both"],
                        help="Type of prompt to use (default: standard)")
    parser.add_argument("--speaker_condition", type=str, default="same",
                        choices=["same", "different"],
                        help="Whether context is from same or different speaker (default: same)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create speaker condition directory
    speaker_condition_dir = os.path.join(args.output_dir, f"{args.speaker_condition}_speaker_condition")
    os.makedirs(speaker_condition_dir, exist_ok=True)
    
    # Load model
    model, processor, generation_config = load_phi4_model(args.cache_dir)
    
    # Datasets to evaluate
    if args.dataset == "all":
        datasets = ["l2-arctic", "hisp-eng", "cmu-arctic"]
    else:
        datasets = [args.dataset]
    
    # Determine which prompt types to use
    prompt_types = ["standard", "variation"] if args.prompt_type == "both" else [args.prompt_type]
    
    # Run evaluation on each dataset with each prompt type
    all_results = {}
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"EVALUATING {dataset_name.upper()}")
        print(f"{'='*50}\n")
        
        dataset_results = {}
        
        # Loop over prompt types and shot counts
        for prompt_type in prompt_types:
            print(f"\n--- Speaker condition: {args.speaker_condition}, Prompt type: {prompt_type} ---\n")
            
            prompt_results = {}
            
            for n_shots in range(args.max_shots + 1):
                results = evaluate_dataset_multi_shot(
                    dataset_name,
                    model, processor, generation_config,
                    speaker_condition_dir,
                    n_shots,
                    args.max_trials,
                    prompt_type,
                    args.speaker_condition,
                    args.max_shots,
                    args.seed
                )
                
                if results:
                    prompt_results[f"shot{n_shots}"] = results
            
            dataset_results[prompt_type] = prompt_results
        
        all_results[dataset_name] = dataset_results
    
    # Create summary file with all results
    summary_file = os.path.join(speaker_condition_dir, f"all_results_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nEvaluation complete. Summary saved to {summary_file}")


if __name__ == "__main__":
    main()