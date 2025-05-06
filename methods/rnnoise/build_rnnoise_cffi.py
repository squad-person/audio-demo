import os
from cffi import FFI

# Define the C header declarations for CFFI
# Updated to match rnnoise.h for model loading capabilities
cdef_source = r"""
    // Opaque types
    typedef struct DenoiseState DenoiseState;
    typedef struct RNNModel RNNModel;
    // For FILE* from rnnoise_model_from_file, use void* as CFFI doesn't handle FILE*
    typedef void FILE;

    // Core functions
    int rnnoise_get_size(void);
    int rnnoise_get_frame_size(void);
    DenoiseState *rnnoise_create(RNNModel *model);
    int rnnoise_init(DenoiseState *st, RNNModel *model); // Initialize pre-allocated state
    void rnnoise_destroy(DenoiseState *st);
    float rnnoise_process_frame(DenoiseState *st, float *out, const float *in);

    // Model loading functions
    RNNModel *rnnoise_model_from_buffer(const void *ptr, int len);
    RNNModel *rnnoise_model_from_file(FILE *f);
    RNNModel *rnnoise_model_from_filename(const char *filename);
    void rnnoise_model_free(RNNModel *model);
"""

# Create an FFI builder instance
ffibuilder = FFI()

# Set the C definitions
ffibuilder.cdef(cdef_source)

# Specify the source code for CFFI to compile
# This tells CFFI where to find the header file and which library to link against
# Adjust library name and path if necessary (e.g., librnnoise.so on Linux)

# Calculate path relative to the script's directory
script_dir = os.path.dirname(__file__) # This is methods/rnnoise

# Correctly define the path to the cloned rnnoise library source
rnnoise_lib_dir = os.path.abspath(os.path.join(script_dir, 'lib', 'rnnoise'))
# The compiled library (.dylib/.so) is typically in the .libs subdirectory
rnnoise_compiled_libs_dir = os.path.join(rnnoise_lib_dir, '.libs')

# The output directory for the CFFI module (.so/.dylib) should be this script's directory
# so that rnnoise_cffi_wrapper.py can find it easily.
output_dir = script_dir
cffi_module_name = "_rnnoise_cffi" # This is the Python import name


# Add linker argument to embed RPATH relative to the final module location ($ORIGIN)
# This helps the CFFI module find the librnnoise.dylib/.so at runtime
# when they are in different directories (if output_dir != rnnoise_compiled_libs_dir)
# If output_dir IS the same as script_dir, and librnnoise is in lib/rnnoise/.libs
# the relative path needs to account for that.
# Relative path from output_dir (script_dir) to rnnoise_compiled_libs_dir
relative_lib_path = os.path.relpath(rnnoise_compiled_libs_dir, output_dir)
ffibuilder.set_source(cffi_module_name,
    r"""
        #include <rnnoise.h> // Include the actual header
    """,
    # rnnoise.h is usually in the root of the source dir after configure
    # UPDATE: Found it in the include/ subdirectory
    include_dirs=[os.path.join(rnnoise_lib_dir, 'include')],
    # Path to the compiled library directory
    library_dirs=[rnnoise_compiled_libs_dir],
    # Name CFFI looks for (e.g., librnnoise.dylib or librnnoise.so)
    libraries=['rnnoise'],
    # On macOS use @loader_path, on Linux use $ORIGIN
    # Using $ORIGIN works on macOS too in many cases via linker translation
    extra_link_args=[f'-Wl,-rpath,$ORIGIN/{relative_lib_path}'] # $ORIGIN refers to the dir of the loading module (_rnnoise_cffi.so)
)

if __name__ == "__main__":
    # Compile the C extension module, placing output in this directory (script_dir)
    print(f"Compiling RNNoise CFFI interface into {output_dir}...")
    # Use target instead of tmpdir to control the final output file name and location
    # target = f"{os.path.join(output_dir, cffi_module_name)}.*" # Causes issues on some platforms
    # Let CFFI handle naming, just ensure it lands in output_dir
    ffibuilder.compile(verbose=True) # Compile in the current directory (which should be script_dir)
    print("Compilation successful!") 