# Implementation Plan: iOS Speech Enhancement App

This plan outlines the steps to create an iOS sample app that enhances the speech audio in videos selected from the user's gallery, based on the requirements in `requirements.md` and research in `intro.md`.

**Chosen Technologies**:
- DTLN (Dual-Transform Layer Network) via Core ML conversion.
- VoiceFixer (ResUNet + Vocoder) via Core ML conversion.

*Rationale*: Provide options for speech enhancement.
- `DTLN`: Offers a good balance of quality, performance, and integration effort for general noise reduction.
- `VoiceFixer`: Offers potentially higher quality and broader restoration (noise, reverb, bandwidth expansion, declipping), suitable for more degraded audio, but with higher complexity and potentially slower processing.

Both models are open-source and will be used for offline processing of video files.

## Plan

1.  [x] Set up initial project structure.
2.  [x] Integrate RNNoise using CFFI bindings.
3.  [x] Create basic evaluation script (`run_evaluation_pipeline.py`).
    *   [x] Find input audio files.
    *   [x] Resample audio (16k for RNNoise, 44k for VoiceFixer).
    *   [x] Process audio with RNNoise.
    *   [x] Process audio with DTLN.
    *   [x] Process audio with VoiceFixer.
    *   [x] Generate an HTML summary (`summary.html`) with audio players for comparison.
4.  [ ] (Optional) Add objective audio quality metrics (e.g., PESQ, STOI) if possible.
5.  [ ] (Optional) Add spectrogram visualization to the HTML report.
6.  [ ] Refine documentation and usage instructions.

*Note: DTLN and VoiceFixer integrations were initially commented out but have now been re-enabled.*

## Phase 0: Model Evaluation (Offline)

*Goal*: Verify the effectiveness of the chosen models on sample data before iOS integration.

**Common Setup Tasks**:
- [x] **Set up Python environment**: Create a virtual environment (`venv` or `conda`) and install necessary libraries.
- [x] **Obtain representative sample videos**: Collect a few short videos (15-60 seconds) with varying audio quality.
- [x] **Extract audio**: Use `ffmpeg` command-line tool or a script to extract audio tracks as WAV files. Example: `ffmpeg -i input_video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 1 output_audio_44k.wav`.
- [x] **Modular project structure**:
    - [x] Create `methods/` directory with subdirectories for each evaluated method.
    - [x] Each method directory contains `lib/`, `output/`, `run.py`, `README.md`, `requirements.txt`.
    - [x] Create `preparation.py` and `summary.py` at the root level.
- [x] Implement `preparation.py`.
- [x] Implement `summary.py` (to be updated as methods are evaluated).
- [x] Create root README.md explaining the workflow.

**Method Evaluation**:
*(Status Key: [x] Completed, [>] In Progress, [-] Skipped, [B] Blocked)*

- **DTLN** ([x] Completed)
    - [x] Clone repository.
    - [x] Run inference on samples (Tested `model.h5`, `DTLN_norm_500h.h5`).
    - [x] Implement `methods/dtln/run.py`, `README.md`, `requirements.txt`.
    - [x] Resolve TensorFlow installation issues.
    - [x] Initial analysis of performance and complexity.
    - **Result**: Degraded quality, significant artifacts. Did not meet quality goal.

- **VoiceFixer** ([x] Completed)
    - [x] Install PyPI package.
    - [x] Run inference on samples (Tested `mode=0`, `mode=2`).
    - [x] Implement `methods/voice_fixer/run.py`, `README.md`, `requirements.txt`.
    - [x] Initial analysis of performance and complexity.
    - **Result**: Degraded quality, significant distortion/"robovoice" artifacts. Did not meet quality goal.

- **RNNoise** ([x] Completed)
    - [x] Build C library from source.
    - [x] Create CFFI build script (`build_rnnoise_cffi.py`).
    - [x] Create Python wrapper (`rnnoise_cffi_wrapper.py`).
    - [x] Implement `methods/rnnoise/run.py`, `README.md`, `requirements.txt`.
    - [x] Evaluate on test dataset.
    - [x] Resolve dynamic library loading issues.
    - **Result**: Baseline established; provides some denoising.

- **UniAudio** ([B] Blocked)
    - [>] Clone repository (`yangdongchao/UniAudio`).
    - [ ] Set up environment and download models.
    - [ ] Run inference (SE task).
    - **Reason**: Blocked due to corrupted pre-trained checkpoint download (~10GB), significant dependency conflicts (fairseq/omegaconf), and required code patching in the library.

- **Supervoice Flow (SpeechFlow)** ([x] Completed)
    - [x] Clone repository (`ex3ndr/supervoice-flow`).
    - [x] Set up environment and use `torch.hub`.
    - [x] Implement `methods/supervoice_flow/run.py`, `README.md`, `requirements.txt`.
    - **Result**: (Add evaluation result here when available)

- **AudioSR** ([-] Skipped / [x] Setup Completed)
    - [x] Install package/clone repository (`haoheliu/versatile_audio_super_resolution`).
    - [-] Run inference. **Reason**: Skipped due to persistent internal errors (`Invalid file: tensor(...)`) during `super_resolution` call, despite input format adjustments (tensor, float32).
    - [x] Create directory structure (`methods/audiosr/...`).
    - [x] Implement `run.py` for AudioSR.
    - [-] Update `summary.py` to include AudioSR. **Reason**: Skipped as `run.py` did not complete successfully.

- **AnyEnhance** ([B] Blocked)
    - [x] Locate implementations.
    - **Reason**: No public code found.

- **CleanMel** ([B] Blocked)
    - [x] Locate implementations.
    - [ ] Create directory structure (`methods/cleanmel/...`).
    - **Reason**: `mamba-ssm` dependency incompatible with macOS.

- **UnDiff** ([B] Blocked)
    - [x] Locate implementations (Completed: SamsungLabs/Undiff).
    - **Reason**: Official implementation lacks explicit 'denoising' task (focuses on BWE, declipping, source separation). No further testing planned.

- **SGMSE** ([-] Skipped / [>] In Progress)
    - [x] Locate implementations (Completed: sp-uhh/sgmse).
    - [X] Test implementation setup (Completed, but required commenting out 'pesq' imports due to build/arch issues).
    - [x] Create directory structure (`methods/sgmse/...`).
    - [x] Implement `run.py` for SGMSE.
    - [-] Run `run.py` for SGMSE on sample audio. **Reason**: Skipped due to runtime errors during execution.
    - [-] Update `summary.py` to include SGMSE. **Reason**: Skipped as `run.py` did not complete successfully.

**Overall Evaluation & Selection**:
- [s] ~**(Optional) Objective Metrics**~: Use libraries like `pypesq` or `pystoi` if feasible. (Skipped - Difficulty with real-world videos, PESQ build issues).
- [x] **Initial Subjective Evaluation (DTLN, VoiceFixer)**: Listen to original vs. enhanced audio. Documented findings.
- [ ] **Subjective Evaluation (New Models)**: Listen to samples from UniAudio, Supervoice Flow, AudioSR, SGMSE (if possible), etc.
    - [ ] Assess clarity, noise reduction, artifacts.
    - [ ] Compare effectiveness vs. baseline (RNNoise).
- [ ] **Select Best Candidate(s)**: Choose the most promising model(s) for potential iOS integration based on quality, performance, complexity.
- [x] **Update Summary Script**: Update `summary.py` to include results from newly evaluated models (as they become available).
  *   **Output**: Comparison table/report (e.g., in `results.md` or integrated into `summary.html`) showing metrics (inference time, CPU/memory usage if measurable) and subjective quality assessment for all evaluated methods.

**Project Status Update (as of YYYY-MM-DD - please fill in date):**
Based on the results summarized in `docs/poc_results.md`, no single model met the required quality and performance criteria for real-world video enhancement. Therefore, further development of the iOS app (Phase 1 onwards) is currently on hold pending re-evaluation of available methods or a change in project scope.

## Phase 1: Basic iOS App & Video Handling

*Goal*: Set up the project, allow video selection, model selection, and basic playback.

**Tasks**:
- [ ] **Create Xcode Project**: Start a new iOS App project in Xcode, selecting SwiftUI for the interface and Swift as the language.
- [ ] **Implement Video Selection UI**: Use SwiftUI's `PhotosPicker` view modifier to present the system's photo library interface, filtering for videos. Store the selected `PhotosPickerItem`.
- [ ] **Implement Video Preview/Loading**:
    - [ ] Load an `AVAsset` from the selected `PhotosPickerItem`.
    - [ ] Display a video preview using `AVPlayer` and `VideoPlayer` views.
- [ ] **Implement Model Selection UI**: Present a list of available speech enhancement models (e.g., RNNoise, DTLN, VoiceFixer) for the user to choose from.
- [ ] **Implement Audio Extraction**: Extract the audio track from the selected video using `AVAssetReader` and `AVAssetWriter`.
- [ ] **Implement Audio Processing**: Process the extracted audio using the selected speech enhancement model.
- [ ] **Implement Audio Replacement**: Replace the original audio track in the video with the processed audio using `AVAssetReader`, `AVAssetWriter`, and `AVAssetWriterInput`.
- [ ] **Implement Video Export**: Export the processed video to the user's photo library using `AVAssetExportSession`.
- [ ] **Implement Playback**: Allow the user to play back the processed video using `AVPlayer` and `VideoPlayer` views.
- [ ] **Implement Sharing**: Allow the user to share the processed video using the iOS Share Sheet.
- [ ] **Implement Feedback**: Allow the user to provide feedback on the processed video, which can be collected for future model training and improvement.

## Phase 2: Advanced iOS App & Video Handling

*Goal*: Enhance the app with additional features and optimizations.

**Tasks**:
- [ ] **Implement Background Processing**: Allow the app to process videos in the background using `AVAssetExportSession` with `exportAsynchronously(with:)`.
- [ ] **Implement Progress Indicators**: Display progress indicators during video processing, allowing the user to cancel the operation if desired.
- [ ] **Implement Multiple Model Selection**: Allow the user to select multiple speech enhancement models and apply them sequentially to the video.
- [ ] **Implement Model Comparison**: Allow the user to compare the output of multiple speech enhancement models side-by-side.
- [ ] **Implement Performance Optimizations**: Optimize the app's performance by using efficient data structures, algorithms, and techniques.
- [ ] **Implement User Preferences**: Allow the user to customize the app's behavior and appearance using a settings screen.
- [ ] **Implement Localization**: Allow the app to be localized into multiple languages using Xcode's localization features.
- [ ] **Implement Accessibility**: Ensure the app is accessible to users with disabilities by following Apple's accessibility guidelines.
- [ ] **Implement Testing**: Write unit tests and UI tests for the app using Xcode's testing frameworks.
- [ ] **Implement Continuous Integration**: Set up continuous integration using a service like Jenkins or Travis CI to automatically build and test the app on every commit.
- [ ] **Implement Deployment**: Deploy the app to the App Store using Xcode's archiving and distribution features.

## Phase 3: Maintenance and Updates

*Goal*: Maintain the app and keep it up-to-date with new features and improvements.

**Tasks**:
- [ ] **Monitor Feedback**: Monitor user feedback and bug reports to identify areas for improvement.
- [ ] **Implement New Features**: Implement new features based on user feedback and market trends.
- [ ] **Update Dependencies**: Keep the app's dependencies up-to-date to ensure compatibility and security.
- [ ] **Fix Bugs**: Fix bugs reported by users and identified during testing.
- [ ] **Improve Performance**: Continuously optimize the app's performance to ensure it runs smoothly on a variety of devices.
- [ ] **Improve User Experience**: Continuously improve the app's user experience based on user feedback and usability testing.
- [ ] **Maintain Code Quality**: Maintain the app's code quality by following best practices and writing clean, maintainable code.
- [ ] **Maintain Documentation**: Maintain the app's documentation to ensure it is up-to-date and accurate.
- [ ] **Maintain Compatibility**: Maintain the app's compatibility with new versions of iOS and hardware.
- [ ] **Maintain Security**: Maintain the app's security by following best practices and keeping it up-to-date with the latest security patches.