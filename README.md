# 🎙️ Frontier Speech Recognition via In-Context Speaker Adaptation
<img width="1072" alt="image" src="https://github.com/user-attachments/assets/4efef8f5-b7e9-4be9-bb42-21690144a9bd" />
**TL;DR:** We found that giving a modern speech recognition model just a few transcribed examples (>5 seconds) from a speaker can reduce word error rates by ~20% on average, with especially dramatic improvements for non-native speakers. No retraining required! 

## 🤔 What's This About?

Ever notice how humans quickly adapt to new speakers, accents, or speaking styles? We wanted to see if large multimodal models like Phi-4 could do the same thing. Turns out they can, and it's pretty remarkable.

Here's the key insight: instead of trying to adapt the model through training (expensive, slow), we just give it a few examples of what the speaker sounds like along with their transcripts. Think of it like showing someone a few samples of handwriting before asking them to read more from the same person.

## 🚀 Main Findings

- **Average improvement:** 19.7% relative reduction in word error rate with just 5+ seconds of context
- **Biggest wins for underrepresented speakers:** Korean English saw 50%+ improvement, Hindi English 40%+
- **Diminishing returns:** Most benefits come from the first few examples, plateaus around 6-10 shots
- **Speaker vs. variety effects:** Same-speaker examples help most at 4-6 shots, but with more context, variety-level patterns become more important

### The Data

We tested on four English speech corpora:
- **L2-ARCTIC**: Non-native speakers (Hindi, Korean, Mandarin, Spanish, Arabic, Vietnamese)
- **CMU-ARCTIC**: Native American English speakers (with some regional variety)
- **HEC**: Spanish heritage speakers in the US
- **Speech Accent Archive**: Global variety of accents reading the same paragraph

## 📊 Key Results

The most striking finding? **Low-resource speaker varieties benefit the most.** While US English speakers saw modest 15-20% improvements, speakers with less representation in training data saw huge gains. This suggests in-context learning might be a practical way to make ASR more equitable without requiring massive amounts of additional training data for every variety.

## 🛠️ How It Works

We use a simple but effective prompting scheme with Phi-4 Multimodal:

1. Start with an instruction about transcribing from a speaker
2. Provide N audio-transcript example pairs
3. Present the target audio for transcription

The model learns to adapt its acoustic-phonetic expectations based on the examples. No gradients, no fine-tuning, just clever prompting!

## 🏃‍♂️ Running the Experiments

### Prerequisites

You'll need:
- Access to a machine with GPU (we used A100s)
- The repo environment set up with required packages
- Datasets loaded (scripts handle the downloading)

### Running Same-Speaker Adaptation

```bash
nlprun -q sphinx -g 1 -r 40G -c 16 -p standard \
"export TRANSFORMERS_CACHE=/nlp/scr/$USER/.cache/huggingface && \
export HF_HOME=/nlp/scr/$USER/.cache/huggingface && \
. /nlp/scr/$USER/miniconda3/bin/activate myenv && \
python asr_adaptation.py \
--output_dir /nlp/scr/$USER/20250513_shot_results \
--cache_dir /nlp/scr/$USER/.cache/huggingface \
--max_shots 12 \
--max_trials 50 \
--dataset all \
--speaker_condition same \
--prompt_type both \
--seed 42" \
-n same_speaker
```

### Running Different-Speaker Adaptation

```bash
nlprun -q sphinx -g 1 -r 40G -c 16 -p standard \
"export TRANSFORMERS_CACHE=/nlp/scr/$USER/.cache/huggingface && \
export HF_HOME=/nlp/scr/$USER/.cache/huggingface && \
. /nlp/scr/$USER/miniconda3/bin/activate myenv && \
python asr_adaptation.py \
--output_dir /nlp/scr/$USER/20250513_shot_results \
--cache_dir /nlp/scr/$USER/.cache/huggingface \
--max_shots 12 \
--max_trials 50 \
--dataset all \
--speaker_condition different \
--prompt_type both \
--seed 42" \
-n diff_speaker
```

### Analyzing Results

Check out `ASR_Adaptation_analysis.ipynb` for detailed analysis and visualization of the results.

## 📂 Repository Contents

- `asr_adaptation.py`: Main experiment script that runs the adaptation experiments
- `ASR_Adaptation_analysis.ipynb`: Analysis notebook with plots and detailed breakdowns
- `all_results_summary_same.json`: Results from same-speaker adaptation experiments  
- `all_results_summary_diff.json`: Results from different-speaker adaptation experiments
- `README.md`: You're reading it! 

## 🤝 Why This Matters

**For Practitioners:** You can improve ASR for underrepresented speakers without collecting massive new datasets or retraining models. Just collect a few good examples per speaker/variety and use this prompting approach.

**For Researchers:** This opens up new questions about how large models internalize acoustic adaptation and whether other modalities might benefit from similar approaches.

**For Speech Technology:** This could be a step toward more equitable speech tech that works well for everyone, not just speakers well-represented in training data.

## 🔮 Future Directions

- Extending to other languages and modalities
- Testing on streaming/real-time scenarios  
- Understanding the precise mechanisms behind the adaptation
- Exploring optimal context selection strategies

---
