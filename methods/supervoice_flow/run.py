import argparse
import os
import torch
import torchaudio
import soundfile as sf # Using soundfile for saving, torchaudio for loading/resampling
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable for the model to avoid reloading it for every file
model = None
device = None

def load_model():
    """Loads the Supervoice Enhance model using torch.hub."""
    global model, device
    if model is None:
        logging.info("Loading Supervoice Enhance model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Force reload might be needed if cache is stale or testing changes
            # model = torch.hub.load(repo_or_dir='ex3ndr/supervoice-enhance', model='enhance', vocoder=True, force_reload=True)
            model = torch.hub.load(repo_or_dir='ex3ndr/supervoice-enhance', model='enhance', vocoder=True)
            model.to(device)
            model.eval()
            logging.info(f"Model loaded successfully on {device}.")
            logging.info(f"Model expected sample rate: {model.sample_rate}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    return model, device

def enhance_audio(input_path, output_path, enhancement_steps=8):
    """
    Loads audio, runs Supervoice Enhance enhancement, and saves the output.
    """
    global model, device
    if model is None:
        load_model() # Ensure model is loaded

    if model is None: # Check again if loading failed
        logging.error("Model not loaded. Skipping enhancement.")
        return

    logging.info(f"Processing file: {input_path}")

    try:
        # Load audio using torchaudio
        audio, sr = torchaudio.load(input_path)
        audio = audio.to(device)
        logging.info(f"Loaded audio with sample rate: {sr}")

        # Resample if necessary
        if sr != model.sample_rate:
            logging.info(f"Resampling audio from {sr} Hz to {model.sample_rate} Hz")
            resampler = torchaudio.transforms.Resample(sr, model.sample_rate).to(device)
            audio = resampler(audio)
            sr = model.sample_rate

        # Convert to mono if necessary
        if audio.shape[0] > 1:
            logging.info("Converting audio to mono")
            audio = audio.mean(dim=0, keepdim=True)

        # Remove batch dimension if added by torchaudio (model expects single waveform tensor)
        if audio.dim() > 1 and audio.shape[0] == 1:
            audio = audio.squeeze(0)

        # Perform enhancement
        logging.info(f"Starting enhancement with {enhancement_steps} steps...")
        with torch.no_grad(): # Inference doesn't need gradients
            enhanced_audio = model.enhance(waveform=audio, steps=enhancement_steps)
        logging.info("Enhancement complete.")

        # Move back to CPU for saving
        enhanced_audio_cpu = enhanced_audio.cpu()

        # Save the enhanced audio using soundfile
        # Ensure output is 1D or 2D [frames, channels]
        if enhanced_audio_cpu.dim() == 1:
            enhanced_audio_cpu = enhanced_audio_cpu.unsqueeze(-1) # Add channel dim if mono

        # Soundfile expects [frames, channels]
        sf.write(output_path, enhanced_audio_cpu.numpy(), model.sample_rate)
        logging.info(f"Saved enhanced audio to: {output_path} with sample rate {model.sample_rate}")

    except Exception as e:
        logging.error(f"Failed during enhancement for {input_path}: {e}")
        # Optionally re-raise or handle specific errors
        # raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Supervoice Flow/Enhance Speech Enhancement")
    parser.add_argument("--input", required=True, help="Path to the input audio file or directory.")
    parser.add_argument("--output_dir", default="output", help="Directory to save enhanced audio files.")
    parser.add_argument("--steps", type=int, default=8, help="Number of enhancement steps (default: 8, try 32 for potentially higher quality).")

    args = parser.parse_args()

    # Load the model once before processing files
    try:
        load_model()
    except Exception as e:
        logging.error(f"Exiting due to model loading failure: {e}")
        exit(1)

    # Construct absolute paths based on the current working directory
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve input path relative to the script directory
    input_path_abs = os.path.abspath(os.path.join(script_dir, args.input))
    # Resolve output directory relative to the script directory
    output_dir_abs = os.path.abspath(os.path.join(script_dir, args.output_dir))


    # Create output directory if it doesn't exist
    logging.info(f"Ensuring output directory exists: {output_dir_abs}")
    os.makedirs(output_dir_abs, exist_ok=True)

    if os.path.isdir(input_path_abs):
        logging.info(f"Processing all .wav files in directory: {input_path_abs}")
        wav_files = [f for f in os.listdir(input_path_abs) if f.lower().endswith(".wav")]
        if not wav_files:
            logging.warning(f"No .wav files found in {input_path_abs}")
        else:
            logging.info(f"Found {len(wav_files)} .wav files to process.")
            for filename in wav_files:
                input_file_path = os.path.join(input_path_abs, filename)
                # Use the resolved absolute output directory
                output_filename = f"{os.path.splitext(filename)[0]}_supervoiceenhance.wav" # Changed suffix
                output_file_path = os.path.join(output_dir_abs, output_filename)
                enhance_audio(input_file_path, output_file_path, args.steps)

    elif os.path.isfile(input_path_abs) and input_path_abs.lower().endswith(".wav"):
        logging.info(f"Processing single file: {input_path_abs}")
        filename = os.path.basename(input_path_abs)
        # Use the resolved absolute output directory
        output_filename = f"{os.path.splitext(filename)[0]}_supervoiceenhance.wav" # Changed suffix
        output_file_path = os.path.join(output_dir_abs, output_filename)
        enhance_audio(input_path_abs, output_file_path, args.steps)
    else:
        # Use the originally provided input argument in the error message for clarity
        logging.error(f"Invalid input: {args.input}. Must be an existing .wav file or a directory containing .wav files.")

    logging.info("Processing complete.") 