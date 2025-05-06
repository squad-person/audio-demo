# Proof-of-Concept Results

This document summarizes the findings from the model evaluation phase for the iOS Speech Enhancement App.

## Model Performance Summary

-   **DTLN**: Very mild results for speech enhancement, did almost nothing to make audio better.
-   **VoiceFixer**: Very good results on speech samples, but on real-life videos makes voices robotic and doesn't do anything useful.
-   **RNNoise**: Very basic noise filtering, did almost nothing to the sample audio.
-   **Supervoice Flow (SpeechFlow)**: Shown the most promising results for sample audio, did poorly for real life videos from cameras.

## Conclusion

In overall - no single model was able to satisfy our needs as outlined in `docs/intro.md` and `docs/plan.md`. So to proceed further with an app sample does not make sense at this point.

## Method Comparison: Approach and How It Works

| Method             | Overall Approach and How It Works                                                                                                                                                                                                                            |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **DTLN**           | - **Core Technique:** Neural Network (LSTM-based)<br>- **Processing Stages:** Two-stage (frequency domain analysis, then time-domain refinement)<br>- **Key Operations:** Identifies noise in frequency characteristics, processes sound wave, aims to improve clarity.<br>- **Primary Goal:** Speech denoising. |
| **VoiceFixer**     | - **Core Technique:** Deep Learning (ResUNet + Neural Vocoder)<br>- **Processing Stages:** Two-stage (analysis/enhancement, then speech synthesis)<br>- **Key Operations:** Analyzes for multiple issues (noise, echo, quality), then rebuilds clean speech.<br>- **Primary Goal:** Comprehensive speech restoration. |
| **RNNoise**        | - **Core Technique:** Hybrid (DSP + Recurrent Neural Network)<br>- **Processing Stages:** Integrated (traditional processing with neural network guidance)<br>- **Key Operations:** Identifies speech vs. noise in frequency bands, applies filtering.<br>- **Primary Goal:** Noise suppression. |
| **Supervoice Flow (SpeechFlow)** | - **Core Technique:** Generative Neural Network (Flow Matching)<br>- **Processing Stages:** Transformation (learns to map noisy audio to clean audio)<br>- **Key Operations:** Generates a cleaner version of speech based on learned patterns.<br>- **Primary Goal:** Speech enhancement by generation. | 