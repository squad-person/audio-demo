import os
import glob
import argparse
import logging
import numpy as np
import soundfile as sf
import sys

# Import the wrapper (assuming it's in the same directory)
try:
    from rnnoise_cffi_wrapper import RNNoiseCFFI as RNNoise
except ImportError as e:
    logging.error(f"Failed to import RNNoise wrapper: {e}")
    logging.error("Ensure rnnoise_cffi_wrapper.py is in the same directory and the CFFI module is built (run build_rnnoise_cffi.py).")
    logging.error("Also ensure the RNNoise C library is findable (check DYLD_LIBRARY_PATH/LD_LIBRARY_PATH as per README).")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SAMPLE_RATE = 48000 # RNNoise *internally* processes at 48kHz, though it expects 16-bit PCM frames.
                    # The wrapper handles the sample rate expectation based on its internal frame size.
RNNOISE_FRAME_SIZE = 480 # RNNoise process frames of 480 samples (10 ms at 48kHz).
EXPECTED_INPUT_SR = 16000 # We expect 16k input files from preparation.py
SAMPLE_RATE_KEY = "_16k"
DEFAULT_INPUT_DIR = "../../assets/prepared"
DEFAULT_OUTPUT_DIR = "./output"

def process_audio_rnnoise(denoiser, audio_data_float32):
    """Processes float32 audio data through RNNoise frame by frame."""
    # Ensure input is float32
    if audio_data_float32.dtype != np.float32:
        raise TypeError(f"process_audio_rnnoise expects float32 input, got {audio_data_float32.dtype}")

    # RNNoise operates on frames of a specific size (e.g., 480 samples for 10ms at 48kHz)
    # The CFFI wrapper expects frames of this size.
    # Use the constant defined in this script
    frame_size = RNNOISE_FRAME_SIZE

    num_samples = len(audio_data_float32)
    num_frames = num_samples // frame_size
    output_audio_float32 = np.zeros_like(audio_data_float32)

    logging.debug(f"Processing {num_frames} frames of size {frame_size}...")
    for i in range(num_frames):
        frame_start = i * frame_size
        frame_end = frame_start + frame_size
        frame = audio_data_float32[frame_start:frame_end]

        # Process the float32 frame using the wrapper
        # The wrapper now returns (vad_prob, denoised_frame_float32)
        try:
            vad_prob, denoised_frame = denoiser.process_frame(frame)
            output_audio_float32[frame_start:frame_end] = denoised_frame
        except Exception as e:
            logging.error(f"Error processing frame {i}: {e}")
            # Decide how to handle frame errors: skip frame? fill with silence? stop?
            # For now, just copy original frame data to output
            output_audio_float32[frame_start:frame_end] = frame

    # Handle the last partial frame if any
    remaining_samples = num_samples % frame_size
    if remaining_samples > 0:
        logging.debug(f"Handling last partial frame of {remaining_samples} samples.")
        last_frame_start = num_frames * frame_size
        # Pad the last frame to frame_size with zeros
        last_frame = np.zeros(frame_size, dtype=np.float32)
        last_frame[:remaining_samples] = audio_data_float32[last_frame_start:]
        try:
            vad_prob, denoised_last_frame = denoiser.process_frame(last_frame)
            # Copy back only the original number of samples
            output_audio_float32[last_frame_start:] = denoised_last_frame[:remaining_samples]
        except Exception as e:
             logging.error(f"Error processing final partial frame: {e}")
             # Copy original partial frame data to output
             output_audio_float32[last_frame_start:] = last_frame[:remaining_samples]


    logging.debug("Frame processing complete.")
    # No need to convert back to float32, it already is
    return output_audio_float32

def process_file(denoiser, input_path, output_path):
    """Loads a 16k audio file, processes it with RNNoise, and saves the result."""
    try:
        logging.info(f"Processing {input_path}...")
        # Load audio file, ensure it's float32 for potential conversion later
        audio, sr = sf.read(input_path, dtype='float32')

        if sr != EXPECTED_INPUT_SR:
            logging.warning(f"Input sample rate {sr} doesn't match expected {EXPECTED_INPUT_SR}. Skipping file.")
            # NOTE: RNNoise C lib *expects* 48k internally, but the wrapper might abstract this.
            # However, for consistency with preparation, we check for 16k input.
            # If the wrapper needs 48k, resampling should happen here or in the wrapper.
            # Assuming the wrapper handles the conversion or the C lib is flexible.
            return

        if len(audio.shape) > 1:
            logging.warning("Audio is not mono, converting to mono by averaging channels.")
            audio = np.mean(audio, axis=1)

        # Process the audio data
        enhanced_audio = process_audio_rnnoise(denoiser, audio)

        # Save the enhanced audio file (as float32, common format)
        # RNNoise output is technically 16-bit, but saving as float avoids potential scaling issues
        sf.write(output_path, enhanced_audio, EXPECTED_INPUT_SR) # Save with original input sample rate
        logging.info(f"Saved enhanced audio to {output_path}")

    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")
        # import traceback
        # logging.error(traceback.format_exc())

def main(input_dir, output_dir):
    """Finds prepared 16k audio files and processes them with RNNoise."""
    logging.info(f"Starting RNNoise processing...")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Find only the 16k prepared files
    search_pattern = os.path.join(input_dir, f"*{SAMPLE_RATE_KEY}.wav")
    audio_files = glob.glob(search_pattern)

    if not audio_files:
        logging.warning(f"No *{SAMPLE_RATE_KEY}.wav files found in {input_dir}. Did you run preparation.py?")
        return

    logging.info(f"Found {len(audio_files)} audio file(s) to process.")

    # Define path relative to this script's location
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, "lib", "rnnoise", "weights_blob.bin")
    model_to_use = None

    if os.path.exists(model_path):
        logging.info(f"Found model file: {model_path}")
        model_to_use = model_path
    else:
        logging.warning(f"Model file not found at {model_path}. Using default built-in model.")

    # Initialize RNNoise denoiser from the wrapper
    try:
        logging.info(f"Initializing RNNoise denoiser (Model: {'Loaded from file' if model_to_use else 'Default built-in'})...")
        denoiser = RNNoise(model_path=model_to_use)
        logging.info("RNNoise denoiser initialized.")
    except Exception as e:
        logging.error(f"Error initializing RNNoise: {e}. Have you built the CFFI module and set library paths? (See README)")
        return # Exit if denoiser cannot be created

    for input_file in audio_files:
        filename = os.path.basename(input_file)
        # Construct output path
        model_tag = "_model" if model_to_use else "_default"
        output_filename = filename.replace(SAMPLE_RATE_KEY, f"_rnnoise{model_tag}_enhanced")
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            logging.info(f"Skipping {output_path}, file already exists.")
            continue

        process_file(denoiser, input_file, output_path)

    # Clean up RNNoise instance (if the wrapper has a cleanup method)
    if hasattr(denoiser, 'destroy') and callable(denoiser.destroy):
        denoiser.destroy()
        logging.info("RNNoise denoiser explicitly destroyed.")

    logging.info("RNNoise processing finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RNNoise enhancement on prepared audio files using CFFI wrapper.")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing prepared 16k audio files (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save enhanced audio files (default: {DEFAULT_OUTPUT_DIR})")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
