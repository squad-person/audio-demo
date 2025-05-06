import os
import glob
import argparse
import logging
import soundfile as sf
from voicefixer import VoiceFixer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SAMPLE_RATE_KEY = "_44k"
EXPECTED_SAMPLE_RATE = 44100
DEFAULT_INPUT_DIR = "../../assets/prepared"
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_MODE = 1 # Hardcode mode 1
DEFAULT_SUFFIX_TAG = "mode1" # Hardcode suffix for mode 1

def process_file(vf, input_path, output_path, mode):
    """Processes a single audio file using VoiceFixer with a specific mode."""
    try:
        logging.info(f"Processing {input_path} with mode {mode}...")
        # VoiceFixer processes the file directly
        # mode=0: original model
        # mode=1: Add microphone noise suppression
        # mode=2: Add speech restoration
        vf.restore(input=input_path, output=output_path, mode=mode) # Use the specified mode
        logging.info(f"Saved enhanced audio to {output_path}")
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")

def main(input_dir, output_dir, mode, suffix_tag):
    """Finds prepared 44k audio files and processes them with VoiceFixer."""
    logging.info(f"Starting VoiceFixer processing (Mode: {mode}, Suffix Tag: '{suffix_tag}')...")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Find only the 44k prepared files
    search_pattern = os.path.join(input_dir, f"*{SAMPLE_RATE_KEY}.wav")
    audio_files = glob.glob(search_pattern)

    if not audio_files:
        logging.warning(f"No *{SAMPLE_RATE_KEY}.wav files found in {input_dir}. Did you run preparation.py?")
        return

    logging.info(f"Found {len(audio_files)} audio file(s) to process.")

    # Initialize VoiceFixer (loads models)
    # Consider adding try-except block for model loading if needed
    logging.info("Initializing VoiceFixer model...")
    vf = VoiceFixer() # cuda=True can be added if GPU is available and configured
    logging.info("VoiceFixer model initialized.")

    for input_file in audio_files:
        filename = os.path.basename(input_file)
        # Construct output path using the suffix tag
        # If tag is empty (default mode 0), uses original suffix
        tag_part = f"_{suffix_tag}" if suffix_tag else ""
        output_filename = filename.replace(SAMPLE_RATE_KEY, f"_vf{tag_part}_enhanced")
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            logging.info(f"Skipping {output_path}, file already exists.")
            continue

        process_file(vf, input_file, output_path, mode) # Pass mode here

    logging.info(f"VoiceFixer processing finished for mode {mode}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VoiceFixer enhancement (Mode 1) on prepared audio files.")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing prepared audio files (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save enhanced audio files (default: {DEFAULT_OUTPUT_DIR})")
    # Remove mode and suffix_tag arguments
    # parser.add_argument("--mode", type=int, default=DEFAULT_MODE, choices=[0, 1, 2],
    #                     help=f"VoiceFixer processing mode (0: basic, 1: mic noise sup., 2: speech restore) (default: {DEFAULT_MODE})")
    # parser.add_argument("--suffix-tag", type=str, default=DEFAULT_SUFFIX_TAG,
    #                     help=f"Tag to add to output filenames before '_enhanced' (e.g., 'mode1') (default: '{DEFAULT_SUFFIX_TAG}' for mode 0)")

    args = parser.parse_args()

    # Remove validation logic for mode/suffix
    # if args.mode != DEFAULT_MODE and not args.suffix_tag:
    #     # Automatically assign a tag if none provided for non-default modes
    #     args.suffix_tag = f"mode{args.mode}"
    #     logging.warning(f"No suffix tag provided for mode {args.mode}. Automatically using tag: '{args.suffix_tag}'")
    # elif args.mode == DEFAULT_MODE and args.suffix_tag:
    #     logging.warning(f"Suffix tag '{args.suffix_tag}' provided for default mode {args.mode}. Filenames will include the tag.")

    # Call main with hardcoded mode and suffix
    main(args.input_dir, args.output_dir, DEFAULT_MODE, DEFAULT_SUFFIX_TAG)
