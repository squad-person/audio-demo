import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_spectrogram(audio_path, output_path, figsize=(8, 4)):
    """
    Create a spectrogram from an audio file and save it as an image.
    
    Args:
        audio_path (str): Path to the audio file
        output_path (str): Path where to save the spectrogram image
        figsize (tuple): Figure size (width, height) in inches
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logging.debug(f"Loading audio file: {audio_path}")
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    logging.debug(f"Generating spectrogram for: {os.path.basename(audio_path)}")
    # Create spectrogram
    plt.figure(figsize=figsize)
    plt.specgram(y, Fs=sr, cmap='viridis')
    plt.axis('off')  # Remove axes for cleaner look
    
    # Save with tight layout and transparent background
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=100)
    plt.close()
    logging.debug(f"Saved spectrogram to: {output_path}")

def generate_spectrograms_for_directory(audio_dir, output_dir):
    """
    Generate spectrograms for all audio files in a directory.
    
    Args:
        audio_dir (str): Directory containing audio files
        output_dir (str): Directory where to save spectrogram images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Count total number of WAV files
    total_files = sum(1 for root, _, files in os.walk(audio_dir) 
                     for file in files if file.endswith('.wav'))
    
    if total_files == 0:
        logging.info(f"No WAV files found in {audio_dir}")
        return
    
    logging.info(f"Starting spectrogram generation for {total_files} files in {audio_dir}")
    processed_count = 0
    
    # Process all wav files
    for root, _, files in os.walk(audio_dir):
        wav_files = [f for f in files if f.endswith('.wav')]
        for file in wav_files:
            processed_count += 1
            audio_path = os.path.join(root, file)
            # Create relative path structure in output directory
            rel_path = os.path.relpath(audio_path, audio_dir)
            output_path = os.path.join(output_dir, rel_path.replace('.wav', '_spectrogram.png'))
            
            # Create spectrogram
            try:
                logging.info(f"[{processed_count}/{total_files}] Processing: {rel_path}")
                create_spectrogram(audio_path, output_path)
            except Exception as e:
                logging.error(f"Error processing {audio_path}: {e}")
    
    logging.info(f"Completed spectrogram generation for {audio_dir}")
    logging.info(f"Successfully processed: {processed_count}/{total_files} files")
    logging.info(f"Output directory: {output_dir}")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate spectrograms from audio files")
    parser.add_argument("--audio-dir", type=str, help="Directory containing audio files")
    parser.add_argument("--output-dir", type=str, default="spectrograms", help="Output directory for spectrograms")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.audio_dir:
        generate_spectrograms_for_directory(args.audio_dir, args.output_dir)
    else:
        logging.error("Please provide --audio-dir argument") 