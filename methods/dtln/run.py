import os
import glob
import argparse
import logging
import numpy as np
import soundfile as sf
import tensorflow as tf
import sys

# Add the DTLN library path to sys.path to import its modules if needed
# Adjust the path if your DTLN repo is located elsewhere
DTLN_LIB_PATH = os.path.join(os.path.dirname(__file__), 'lib', 'DTLN')
if DTLN_LIB_PATH not in sys.path:
    sys.path.append(DTLN_LIB_PATH)

# Attempt to import DTLN utilities if available and needed, otherwise use direct TF/Numpy
from DTLN_model import DTLN_model # Import the DTLN model class

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants - Copied from DTLN common practices
SAMPLE_RATE = 16000
SAMPLE_RATE_KEY = "_16k" # To identify correct input files
BLOCK_LEN = 512        # Corresponds to 32ms
BLOCK_SHIFT = 128      # Corresponds to 8ms (75% overlap)

DEFAULT_INPUT_DIR = "../../assets/prepared"
DEFAULT_OUTPUT_DIR = "./output"
# Use the specific model file path from the command line argument or default
# DEFAULT_MODEL_PATH = os.path.join(DTLN_LIB_PATH, "pretrained_model/", "DTLN_norm_500h.h5")


def process_audio(model, audio_data):
    """Processes audio data through the DTLN model block by block."""
    logging.info(f"Starting block processing for audio of length {len(audio_data)} samples.")
    # Pre-allocate buffer for enhanced audio
    out_file = np.zeros((len(audio_data)))
    # Create buffer for processing blocks
    in_buffer = np.zeros((BLOCK_LEN))
    out_buffer = np.zeros((BLOCK_LEN))
    # Calculate number of blocks
    num_blocks = (audio_data.shape[0] - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT

    logging.debug(f"Processing {num_blocks} blocks...")
    # Iterate over blocks
    for idx in range(num_blocks):
        if (idx + 1) % 100 == 0:
            logging.debug(f"Processed block {idx + 1}/{num_blocks}")
        # Shift buffer
        in_buffer[:-BLOCK_SHIFT] = in_buffer[BLOCK_SHIFT:]
        # Read new audio data
        in_buffer[-BLOCK_SHIFT:] = audio_data[idx * BLOCK_SHIFT : idx * BLOCK_SHIFT + BLOCK_SHIFT]
        # --- DTLN Processing --- #
        # Expand dims for model (expects batch size 1)
        in_block = np.expand_dims(in_buffer, axis=0).astype('float32')

        # *** Use model.predict_on_batch() as potentially expected by DTLN_model ***
        # (Check DTLN_model.py or run_evaluation.py if this causes issues)
        # Process block through the model
        # out_block = model.predict(in_block, batch_size=1) # Original
        out_block = model.predict_on_batch(in_block) # Using predict_on_batch

        # Squeeze batch dimension
        out_block = np.squeeze(out_block, axis=0)
        # --- Overlap-Add --- #
        # Shift output buffer
        out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
        out_buffer[-BLOCK_SHIFT:] = np.zeros((BLOCK_SHIFT))
        # Add new processed block
        out_buffer += out_block
        # Write output to file
        out_file[idx * BLOCK_SHIFT : idx * BLOCK_SHIFT + BLOCK_SHIFT] = out_buffer[:BLOCK_SHIFT]

    logging.info(f"Finished block processing ({num_blocks} blocks processed).")
    return out_file


def process_file(model, input_path, output_path):
    """Loads an audio file, processes it, and saves the result."""
    try:
        logging.info(f"Processing {input_path}...")
        # Load audio file
        audio, sr = sf.read(input_path, dtype='float32')

        if sr != SAMPLE_RATE:
            logging.warning(f"Input sample rate {sr} doesn't match expected {SAMPLE_RATE}. Skipping file.")
            # Optionally, add resampling here if needed, but preparation.py should handle it.
            return

        if len(audio.shape) > 1:
            logging.warning("Audio is not mono, converting to mono by averaging channels.")
            audio = np.mean(audio, axis=1)

        # *** Pad audio similar to run_evaluation.py for potential stateful model ***
        # (This assumes the model might be stateful or requires specific padding)
        # get length of file
        len_orig = len(audio)
        # pad audio
        zero_pad = np.zeros(BLOCK_LEN) # Pad with block length
        audio = np.concatenate((zero_pad, audio, zero_pad), axis=0)

        # Process the padded audio data
        enhanced_audio_padded = process_audio(model, audio)

        # *** Unpad the enhanced audio ***
        enhanced_audio = enhanced_audio_padded[BLOCK_LEN : BLOCK_LEN + len_orig]


        # Save the enhanced audio file
        sf.write(output_path, enhanced_audio, SAMPLE_RATE)
        logging.info(f"Saved enhanced audio to {output_path}")

    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")
        # Log traceback for debugging if needed
        # import traceback
        # logging.error(traceback.format_exc())


def main(input_dir, output_dir, model_path):
    """Finds prepared 16k audio files and processes them with DTLN."""
    logging.info(f"Starting DTLN processing...")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Model path: {model_path}")

    if not os.path.exists(model_path):
        logging.error(f"DTLN model weights file not found at {model_path}.")
        # logging.error("You might need to download it or check the path in methods/dtln/lib/DTLN/models/")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Find only the 16k prepared files
    search_pattern = os.path.join(input_dir, f"*{SAMPLE_RATE_KEY}.wav")
    audio_files = glob.glob(search_pattern)

    if not audio_files:
        logging.warning(f"No *{SAMPLE_RATE_KEY}.wav files found in {input_dir}. Did you run preparation.py?")
        return

    logging.info(f"Found {len(audio_files)} audio file(s) to process.")

    # Load the DTLN model using the library's method
    try:
        logging.info("Building DTLN model structure...")
        # Determine if normalization is used based on model filename convention
        if model_path.find('_norm_') != -1:
            norm_stft = True
            logging.info("Detected model uses STFT normalization.")
        else:
            norm_stft = False
            logging.info("Detected model does not use STFT normalization.")

        # Create class instance
        modelClass = DTLN_model()
        # Set constants based on the script's values (might be redundant if DTLN_model uses defaults)
        modelClass.blockLen = BLOCK_LEN
        modelClass.block_shift = BLOCK_SHIFT
        # Build the model structure (might need adaptation if DTLN_model expects different args)
        modelClass.build_DTLN_model(norm_stft=norm_stft) # Use build_DTLN_model (stateless for inference)

        logging.info(f"Loading model weights from {model_path}...")
        modelClass.model.load_weights(model_path)
        logging.info("DTLN model loaded successfully.")
        model_for_processing = modelClass.model # Get the actual Keras model

    except Exception as e:
        logging.error(f"Error loading DTLN model from {model_path}: {e}")
        # Optionally log traceback
        # import traceback
        # logging.error(traceback.format_exc())
        return

    for input_file in audio_files:
        filename = os.path.basename(input_file)
        # Construct output path
        output_filename = filename.replace(SAMPLE_RATE_KEY, "_dtln_enhanced")
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            logging.info(f"Skipping {output_path}, file already exists.")
            continue

        # Pass the loaded Keras model to process_file
        process_file(model_for_processing, input_file, output_path)

    logging.info("DTLN processing finished.")


if __name__ == "__main__":
    # Define default model path using the DTLN library structure
    DEFAULT_MODEL_PATH = os.path.join(DTLN_LIB_PATH, "pretrained_model", "DTLN_norm_500h.h5")

    parser = argparse.ArgumentParser(description="Run DTLN enhancement on prepared audio files.")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing prepared 16k audio files (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save enhanced audio files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to the DTLN model weights file (.h5) (default: {DEFAULT_MODEL_PATH})")

    args = parser.parse_args()

    # Basic input validation
    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    # Model path validation moved inside main()

    main(args.input_dir, args.output_dir, args.model)
