# Audio Enhancement Method Comparison

This project provides a framework to evaluate and compare different speech enhancement algorithms. It allows users to process their own audio files with various methods and generate an HTML summary for easy comparison.

## Prerequisites

*   **Conda**: For managing Python environments. It is highly recommended to use separate Conda environments for each enhancement method to avoid dependency conflicts.
*   **Python**: Python 3.x is required for the scripts.
*   **Git**: For cloning method repositories if necessary.

## Step-by-Step Guide to Generate a Comparison

Follow these steps to prepare your audio, run enhancement methods, and generate a comparative summary:

### 1. Prepare Input Audio

*   **Place Original Audio**: Put your original `.wav` audio files into the `assets/` directory.
*   **Run Preparation Script**: This script resamples your audio files to the rates required by the different methods (e.g., 16kHz, 44.1kHz) and places them in `assets/prepared/`.
    *   From the project root directory, run:
        ```bash
        python preparation.py
        ```
    *   *Note: The environment used to run `preparation.py` needs `librosa` and `soundfile`. You can create a dedicated 'base' or 'prepare' Conda environment for this: `conda create -n prepare_audio python=3.9 librosa soundfile -c conda-forge` then `conda activate prepare_audio`.*

### 2. Run Enhancement Methods

Each speech enhancement method is self-contained within its subdirectory under `methods/`.

*   **General Process for Each Method**:
    1.  **Navigate to the Method's Directory**:
        ```bash
        cd methods/<method_name> 
        # Example: cd methods/voice_fixer
        ```
    2.  **Consult the Method-Specific README**: Inside each method's directory (e.g., `methods/voice_fixer/`), you will find a `README.md`. This crucial file contains detailed instructions for:
        *   Setting up its specific Conda environment.
        *   Installing its dependencies (usually via `pip install -r requirements.txt`).
        *   Any special build steps or library cloning (e.g., for DTLN, RNNoise).
        *   The exact command(s) to execute its `run.py` script.
    3.  **Execute the `run.py` Script**: Once the environment is set up according to its README, run the script. It will typically read audio from `../../assets/prepared/` and save enhanced audio to its local `output/` folder (e.g., `methods/voice_fixer/output/`).
        ```bash
        # Example for RNNoise, after setting up as per its README and DYLD_LIBRARY_PATH if on macOS
        python run.py 
        ```
    4.  Return to the project root directory:
        ```bash
        cd ../..
        ```

*   **Method-Specific README Locations**:
    *   VoiceFixer: `methods/voice_fixer/README.md`
    *   DTLN: `methods/dtln/README.md`
    *   RNNoise: `methods/rnnoise/README.md`
    *   Supervoice Flow: `methods/supervoice_flow/README.md` 
    *   *(Refer to the respective `README.md` in `methods/<new_method_name>/` for any newly added methods.)*

    **Important**: Always follow the detailed instructions in the method-specific `README.md` as setup and run commands can vary.

### 3. Generate the Summary Report

After processing your audio with all desired methods:

*   **(Optional) Prepare Spectrogram Images**: If you plan to include spectrograms in your `summary.html` (and have modified `summary.py` to support this):
    *   Generate spectrogram images (e.g., as `.png` files) for your original and processed audio.
    *   This is typically done using Python with libraries like `Librosa` (for STFT/mel-spectrogram computation) and `Matplotlib` (for saving plots). Spectrograms can be generated in Python using libraries like Librosa (for STFT) and Matplotlib (for plotting).
    *   Ensure images are stored where `summary.py` can find them (e.g., alongside audio files or in a dedicated `spectrograms/` directory, according to your `summary.py` logic).
*   **Run the Summary Script**: From the project root directory, execute:
    ```bash
    python summary.py
    ```
    *   This script collects the original audio from `assets/prepared/` and the processed audio from each `methods/<method_name>/output/` directory to generate `summary.html`.
    *   *Note: The basic `summary.py` requires standard Python. If you've extended it for spectrograms or other features, ensure its environment has the necessary libraries (e.g., for image handling).*

### 4. View Results

*   Open the `summary.html` file (located in the project root) in your web browser
*   You can listen to the original audio and the versions processed by each enhancement method side-by-side.