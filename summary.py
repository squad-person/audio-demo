import os
import glob
import logging
import argparse
from collections import defaultdict
import re # Import regex
from utils.spectrogram import generate_spectrograms_for_directory
import shutil
from jinja2 import Template

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PREPARED_DIR = "assets/prepared"
METHODS_DIR = "methods"
OUTPUT_HTML = "summary.html"
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"
SPECTROGRAMS_DIR = "spectrograms"

# Define suffixes and a way to extract base names AND config hints if possible
METHOD_CONFIG = {
    "voice_fixer_mode_0": {
        "expected_suffix": "_vf_enhanced.wav",
        "config_extraction": lambda fname: "Mode: 0 (Original)"
    },
    "voice_fixer_mode_1": {
        "expected_suffix": "_vf_mode1_enhanced.wav",
        "config_extraction": lambda fname: "Mode: 1 (With Preprocessing)"
    },
    "dtln": {
        "expected_suffix": "_dtln_enhanced.wav",
        # Try to extract model from a potential future naming convention, else default
        "config_extraction": lambda fname: (
            (m := re.match(r'.*_dtln_(.*?)_enhanced\.wav', fname)) and f"Model: {m[1]}"
        ) or "Model: DTLN_norm_500h.h5 (Default)"
    },
    "rnnoise": {
        "expected_suffix": "_rnnoise_model_enhanced.wav", # Current expected suffix
        "config_extraction": lambda fname: "Model: Default"
    },
    "supervoice_flow": { # Added for Supervoice Enhance
        "expected_suffix": "_supervoiceenhance.wav",
        "config_extraction": lambda fname: "Config: Default (torch.hub, steps=8)"
    }     
}

# Define which original sample rate corresponds to which method's input
METHOD_INPUT_RATE_KEY = {
    "voice_fixer_0": "_44k",
    "voice_fixer_1": "_44k",
    "dtln": "_16k",
    "rnnoise": "_16k", # Assuming rnnoise uses 16k input based on run.py
    "supervoice_flow": "_44k", # Uses 24k internally, but prefers higher rate input for resampling
    "rnnoise_default": "_16k", # If we split rnnoise methods
}

# Updated Row Template without heading rows
ROW_TEMPLATE = """
<tr class="data-row">
    <td class="method-cell filename-cell" data-method="original">
        <div class="font-medium">{base_name}</div>
    </td>
    <td class="method-cell" data-method="original">
        <div class="audio-container">
            <audio controls class="audio-player" src="{original_path}"></audio>
            <div class="spectrogram-container">
                <div class="playback-cursor"></div>
                <img 
                    src="{original_spectrogram}" 
                    alt="Spectrogram" 
                    class="spectrogram-img mt-2 w-full rounded-lg shadow-sm"
                    loading="lazy"
                />
            </div>
        </div>
    </td>
    {method_cells}
</tr>
"""

# Updated Cell Template with modern styling
CELL_TEMPLATE = """
<td class="method-cell" data-method="{method_name}">
    {audio_player}
</td>
"""

# Updated Audio Player Template with playback indicator
AUDIO_PLAYER_TEMPLATE = (
    '<div class="audio-container">'
    '<audio controls class="audio-player" src="{path}"></audio>'
    '<div class="spectrogram-container">'
    '<div class="playback-cursor"></div>'
    '<img '
    'src="{spectrogram_path}" '
    'alt="Spectrogram" '
    'class="spectrogram-img mt-2 w-full rounded-lg shadow-sm" '
    'loading="lazy" '
    '/>'
    '</div>'
    '</div>'
)

# Updated placeholder for missing files
NO_FILE_PLACEHOLDER = '<div class="text-gray-400 dark:text-gray-600 italic">File not found</div>'

def ensure_directory_exists(directory):
    """Ensure that a directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def copy_static_files():
    """Copy static files to their locations."""
    ensure_directory_exists(STATIC_DIR)
    
    # Create spectrograms directory if it doesn't exist
    ensure_directory_exists('spectrograms')
    ensure_directory_exists('spectrograms/prepared')

def find_files():
    """Finds original prepared files, corresponding enhanced files, and determines method configs."""
    results = defaultdict(lambda: {'original_16k': None, 'original_44k': None, 'methods': {}})
    methods_present = [d for d in os.listdir(METHODS_DIR) if os.path.isdir(os.path.join(METHODS_DIR, d))]
    logging.info(f"Detected methods (directories): {methods_present}")

    active_method_configs = {m: METHOD_CONFIG[m] for m in methods_present if m in METHOD_CONFIG}
    logging.info(f"Processing configuration for methods: {list(active_method_configs.keys())}")

    method_configs_summary = {m: "Config N/A" for m in active_method_configs} # Initialize with N/A

    # Find original prepared files
    prepared_files = glob.glob(os.path.join(PREPARED_DIR, "*.wav"))
    for prep_file in prepared_files:
        basename = os.path.basename(prep_file)
        if "_16k.wav" in basename:
            base_name = basename.replace("_16k.wav", "")
            results[base_name]['original_16k'] = prep_file
        elif "_44k.wav" in basename:
            base_name = basename.replace("_44k.wav", "")
            results[base_name]['original_44k'] = prep_file
        else:
            logging.warning(f"Skipping unexpected file in prepared dir: {prep_file}")

    # Find enhanced files and capture first config per method
    for method, config_details in active_method_configs.items():
        method_output_dir = os.path.join(METHODS_DIR, method, "output")
        enhanced_files = glob.glob(os.path.join(method_output_dir, "*.wav"))
        suffix = config_details["expected_suffix"]
        config_extractor = config_details["config_extraction"]
        config_found_for_method = False # Flag to capture first config

        logging.debug(f"Searching for suffix '{suffix}' in {method_output_dir}")

        for enh_file in enhanced_files:
            enh_basename_full = os.path.basename(enh_file)

            is_match = False
            config_text = ""
            potential_base = ""
            base_name_match = ""

            # Check primary suffix
            if enh_basename_full.endswith(suffix):
                is_match = True
                temp_base = enh_basename_full.replace(suffix, "")
                if temp_base.endswith("_16k"):
                    base_name_match = temp_base[:-4]
                elif temp_base.endswith("_44k"):
                    base_name_match = temp_base[:-4]
                else:
                    base_name_match = temp_base
                    logging.debug(f"No rate suffix found for {enh_basename_full}, using {base_name_match} as base.")

                config_text = config_extractor(enh_basename_full)
            # Check alternative rnnoise suffix
            elif method == "rnnoise" and enh_basename_full.endswith("_rnnoise_default_enhanced.wav"):
                is_match = True
                temp_base = enh_basename_full.replace("_rnnoise_default_enhanced.wav", "")
                if temp_base.endswith("_16k"):
                    base_name_match = temp_base[:-4]
                elif temp_base.endswith("_44k"):
                    base_name_match = temp_base[:-4]
                else:
                    base_name_match = temp_base
                    logging.debug(f"No rate suffix found for {enh_basename_full}, using {base_name_match} as base.")

                config_text = "Model: Default"

            if is_match:
                # Use the correctly extracted base_name_match for lookup
                if base_name_match in results:
                    relative_path = os.path.relpath(enh_file)
                    # Store only the path now in the main results
                    results[base_name_match]['methods'][method] = {
                        'path': relative_path
                    }
                    logging.debug(f"Matched {enh_basename_full} to base {base_name_match} for method {method}")

                    # Capture the config for the method header (only once)
                    if not config_found_for_method and config_text:
                        method_configs_summary[method] = config_text
                        config_found_for_method = True
                        logging.info(f"Captured config for method '{method}': {config_text}")

                else:
                    logging.warning(f"Found enhanced file '{enh_basename_full}' for method '{method}', but no matching original base name '{base_name_match}'.")
            else:
                logging.debug(f"Skipping file with non-matching suffix in {method} output: {enh_basename_full} (expected suffix: {suffix} or alt)")

    # Filter out entries with no original files found
    valid_results = {k: v for k, v in results.items() if v['original_16k'] or v['original_44k']}
    logging.info(f"Found results for {len(valid_results)} base audio files.")
    processed_methods = list(active_method_configs.keys())
    processed_methods.sort() # Sort here for consistency
    return valid_results, processed_methods, method_configs_summary

def generate_spectrograms(methods):
    """Generate spectrograms for all audio files."""
    logging.info("Generating spectrograms for all audio files...")
    
    # Generate spectrograms for prepared files
    generate_spectrograms_for_directory(PREPARED_DIR, os.path.join(SPECTROGRAMS_DIR, "prepared"))
    
    # Generate spectrograms for each method's output
    for method in methods:
        method_output_dir = os.path.join(METHODS_DIR, method, "output")
        if os.path.exists(method_output_dir):
            generate_spectrograms_for_directory(
                method_output_dir,
                os.path.join(SPECTROGRAMS_DIR, method)
            )

def generate_html(results, methods, method_configs, regenerate_spectrograms=False):
    """Generates the HTML summary page with config info in headers."""
    # Generate spectrograms if requested
    if regenerate_spectrograms:
        generate_spectrograms(methods)
    else:
        logging.info("Skipping spectrogram generation. Use --regenerate-spectrograms to regenerate.")

    # Read the template file first
    try:
        with open(os.path.join(TEMPLATES_DIR, 'summary.html'), 'r') as f:
            template_content = f.read()
    except IOError as e:
        logging.error(f"Failed to read template file: {e}")
        return

    # Generate method filters
    method_filters = ""
    for method in methods:
        method_title = method.replace('_', ' ').title()
        if method == "supervoice_flow":
            method_title = "SuperVoice"
        
        method_filters += f'''
            <button
                id="btn-{method}"
                class="btn btn-outline"
                onclick="toggleMethod('{method}')"
                data-tooltip="{method_configs.get(method, 'Config N/A')}"
            >
                {method_title}
            </button>
        '''

    # Generate method headers with config info and data-method attribute
    method_headers = ""
    for method in methods:
        config_str = method_configs.get(method, 'Config N/A')
        method_title = method.replace('_', ' ').title()
        if method == "supervoice_flow":
            method_title = "SuperVoice"
        
        method_headers += f'''
            <th class="method-header" data-method="{method}" data-tooltip="{config_str}">
                {method_title}
                <div class="text-xs text-gray-600 mt-1">{config_str}</div>
            </th>
        '''

    table_rows = []
    sorted_base_names = sorted(results.keys())

    for base_name in sorted_base_names:
        data = results[base_name]
        method_cells = ""

        # Determine which original to show (same logic as before)
        original_to_display = data['original_16k'] # Default
        original_rate_key = "16k"
        if not original_to_display and data['original_44k']:
             original_to_display = data['original_44k']
             original_rate_key = "44k"
        elif data['original_44k']:
            # Prefer 44k if any active method expects it
            active_methods_for_base = data['methods'].keys()
            if any(METHOD_INPUT_RATE_KEY.get(m) == '_44k' for m in active_methods_for_base):
                original_to_display = data['original_44k']
                original_rate_key = "44k"

        if not original_to_display:
            logging.warning(f"No prepared original found for base name '{base_name}'. Skipping row.")
            continue

        original_path_relative = os.path.relpath(original_to_display)
        original_spectrogram_path = original_path_relative.replace(
            PREPARED_DIR,
            "spectrograms/prepared"
        ).replace(".wav", "_spectrogram.png")

        # Build cells for each method
        for method in methods:
            method_data = data['methods'].get(method)
            if method_data and method_data.get('path'):
                audio_path = method_data['path']
                spectrogram_path = os.path.join(
                    "spectrograms",
                    method,
                    os.path.basename(audio_path).replace(".wav", "_spectrogram.png")
                )
                audio_player = AUDIO_PLAYER_TEMPLATE.format(
                    path=audio_path,
                    spectrogram_path=spectrogram_path
                )
                method_label = method.replace('_', ' ').title()
                if method == "supervoice_flow":
                    method_label = "SuperVoice"
                cell_content = CELL_TEMPLATE.format(
                    audio_player=audio_player,
                    method_name=method
                )
            else:
                method_label = method.replace('_', ' ').title()
                if method == "supervoice_flow":
                    method_label = "SuperVoice"
                cell_content = CELL_TEMPLATE.format(
                    audio_player=NO_FILE_PLACEHOLDER,
                    method_name=method
                )
            method_cells += cell_content

        # Update the original audio cell to include its spectrogram
        table_rows.append(ROW_TEMPLATE.format(
            base_name=base_name,
            original_path=original_path_relative,
            original_spectrogram=original_spectrogram_path,
            method_cells=method_cells
        ))

    # Generate final HTML using Jinja2 template
    try:
        template = Template(template_content)
        final_html = template.render(
            method_filters=method_filters,
            method_headers=method_headers,
            table_rows="\n".join(table_rows)
        )
        
        with open(OUTPUT_HTML, 'w') as f:
            f.write(final_html)
        logging.info(f"Successfully generated summary HTML: {OUTPUT_HTML}")
    except Exception as e:
        logging.error(f"Failed to generate HTML: {e}")

def main():
    """Main function to find files and generate summary."""
    parser = argparse.ArgumentParser(description="Generate HTML summary of audio enhancement results.")
    parser.add_argument('--regenerate-spectrograms', action='store_true', 
                      help='Regenerate spectrograms for all audio files')
    args = parser.parse_args()

    # Ensure directories exist and copy static files
    copy_static_files()
    
    # Create spectrograms directory if it doesn't exist
    ensure_directory_exists(SPECTROGRAMS_DIR)

    # Find files, active methods, and their determined configs
    results, methods_found, method_configs = find_files()

    if not results:
        logging.warning("No processed files found to generate summary.")
        return

    # Generate HTML using the found results, methods, and configs
    generate_html(results, methods_found, method_configs, args.regenerate_spectrograms)

if __name__ == "__main__":
    main()
