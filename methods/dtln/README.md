# DTLN Environment Setup

This document describes how to set up the environment to run the DTLN enhancement script.

## Conda Environment

It's recommended to use a Conda environment to manage dependencies.

1.  **Create Conda environment**:
    ```bash
    # DTLN often uses specific TensorFlow versions. Check the DTLN repo for recommendations.
    # Using TF 2.x as an example:
    conda create -n dtln python=3.8
    ```

2.  **Activate environment**:
    ```bash
    conda activate dtln
    ```

3.  **Clone DTLN Repository**:
    The DTLN processing script likely relies on code from the original DTLN repository.
    ```bash
    cd lib
    git clone https://github.com/breizhn/DTLN.git
    cd .. 
    ```
    *Note: The `run.py` script will assume the DTLN code is available in the `lib/DTLN` subdirectory.*

4.  **Install dependencies**:
    Install TensorFlow (CPU or GPU version depending on your hardware and needs) and other libraries.
    ```bash
    # Install dependencies using the requirements file:
    pip install -r requirements.txt
    # Note: This file includes TensorFlow, soundfile, librosa, numpy, and wavinfo.
    # Check requirements.txt for specific versions or further details.
    ```

## Running the script

Once the environment is set up and activated, you can run the enhancement script:

```