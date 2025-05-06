import os
import glob
import librosa
import soundfile as sf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = "assets/audio_samples/original_rate"
OUTPUT_DIR = "assets/prepared"
TARGET_RATES = {
    "16k": 16000,
    "44k": 44100,
}

def resample_audio(input_path, output_path, target_sr):
    """Resamples an audio file to the target sample rate."""
    try:
        logging.info(f"Loading {input_path}...")
        # Load audio file using librosa, forcing mono and using original sample rate
        y, sr = librosa.load(input_path, sr=None, mono=True)

        if sr == target_sr:
            logging.info(f"Audio already at target rate {target_sr} Hz. Copying directly.")
            # If already at target rate, just copy to avoid potential quality loss
            # Using soundfile for copy to ensure format consistency if needed
            sf.write(output_path, y, sr)
        else:
            logging.info(f"Resampling from {sr} Hz to {target_sr} Hz...")
            # Resample using librosa
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

            # Save the resampled audio file using soundfile
            logging.info(f"Saving resampled audio to {output_path}...")
            sf.write(output_path, y_resampled, target_sr)

        logging.info(f"Successfully processed {input_path} -> {output_path}")

    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")

def main():
    """Finds audio files and prepares resampled versions."""
    logging.info("Starting audio preparation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    audio_files = glob.glob(os.path.join(INPUT_DIR, "*.wav")) # Adjust pattern if needed (e.g., include .mp3)
    if not audio_files:
        logging.warning(f"No .wav files found in {INPUT_DIR}. Exiting.")
        return

    logging.info(f"Found {len(audio_files)} audio file(s) in {INPUT_DIR}.")

    for input_file in audio_files:
        filename = os.path.basename(input_file)
        name, ext = os.path.splitext(filename)

        for rate_key, target_sr in TARGET_RATES.items():
            output_filename = f"{name}_{rate_key}{ext}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            # Check if the output file already exists to avoid reprocessing
            if os.path.exists(output_path):
                logging.info(f"Skipping {output_path}, file already exists.")
                continue

            resample_audio(input_file, output_path, target_sr)

    logging.info("Audio preparation finished.")

if __name__ == "__main__":
    main()
