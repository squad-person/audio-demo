import numpy as np
from _rnnoise_cffi import ffi, lib
import os
from typing import Union, Tuple

# Constants based on RNNoise
FRAME_SIZE_MS = 10
# RNNoise C lib operates at 48kHz internally
RNNOISE_SAMPLE_RATE = 48000
SAMPLES_PER_FRAME = (RNNOISE_SAMPLE_RATE // 1000) * FRAME_SIZE_MS # Should be 480

class RNNoiseCFFIError(Exception):
    pass

class RNNoiseCFFI:
    def __init__(self, model_path: Union[str, None] = None):
        self._model = None
        self._state = None
        self._destroyed = False
        self._state_created_internally = False # Flag to track how state was created
        
        if model_path:
            if not os.path.exists(model_path):
                 raise RNNoiseCFFIError(f"Model file not found: {model_path}")
            
            # Convert path to bytes for CFFI
            model_path_bytes = model_path.encode('utf-8')
            c_model_path = ffi.new("char[]", model_path_bytes)
            
            self._model = lib.rnnoise_model_from_filename(c_model_path)
            if not self._model:
                raise RNNoiseCFFIError(f"Failed to load RNNoise model from {model_path}")
            
            # Use rnnoise_init with the loaded model
            # Get the size needed for the state
            state_size = lib.rnnoise_get_size()
            # Allocate memory for the state
            # Keep the buffer around to prevent GC
            self._state_buffer = ffi.new(f"char[{state_size}]") 
            self._state = ffi.cast("DenoiseState *", self._state_buffer)
            
            ret = lib.rnnoise_init(self._state, self._model)
            if ret != 0:
                 if self._model: lib.rnnoise_model_free(self._model) # Clean up model if init fails
                 raise RNNoiseCFFIError(f"Failed to initialize RNNoise state with model {model_path}")
            self._state_created_internally = False # State initialized in buffer, not created by lib
        else:
            # Create RNNoise state using the default built-in model
            self._state = lib.rnnoise_create(ffi.NULL) 
            if not self._state:
                raise RNNoiseCFFIError("Failed to create RNNoise state with default model.")
            self._state_created_internally = True # State created (and allocated) by lib

    def _check_destroyed(self):
        if self._destroyed:
            raise RNNoiseCFFIError("RNNoise state has already been destroyed.")

    def process_frame(self, frame_float32):
        """Processes a single frame of audio.

        Args:
            frame_float32: A numpy array of exactly SAMPLES_PER_FRAME float32 samples.

        Returns:
            Tuple[float, np.ndarray]: A tuple containing:
                - vad_probability (float): The voice activity probability.
                - denoised_frame (np.ndarray): A numpy array containing the denoised
                                               SAMPLES_PER_FRAME float32 samples.
        """
        self._check_destroyed()
        
        if not isinstance(frame_float32, np.ndarray) or frame_float32.dtype != np.float32:
             raise TypeError(f"Input frame must be a numpy array of float32, got {type(frame_float32)} with dtype {frame_float32.dtype}")
             
        if frame_float32.shape != (SAMPLES_PER_FRAME,):
            raise ValueError(f"Input frame must have exactly {SAMPLES_PER_FRAME} samples, got {frame_float32.shape}")

        # CFFI requires pointers to the data
        # Create an output buffer (copy of input, will be modified in-place by C func)
        out_frame_float32 = np.copy(frame_float32)
        
        in_ptr = ffi.cast("const float *", ffi.from_buffer(frame_float32))
        out_ptr = ffi.cast("float *", ffi.from_buffer(out_frame_float32))

        # Call the C function
        vad_prob = lib.rnnoise_process_frame(self._state, out_ptr, in_ptr)

        return vad_prob, out_frame_float32
        
    def destroy(self):
        if self._destroyed:
             return 
        
        # Only call destroy if the state was created by rnnoise_create
        if self._state_created_internally and hasattr(self, '_state') and self._state:
            lib.rnnoise_destroy(self._state)
        # Else: If state was created via init (buffer managed by CFFI), 
        # we don't call destroy on it. CFFI/Python GC will handle the buffer.
        
        self._state = None # Clear state reference regardless
            
        # Free the model if it was loaded externally
        if hasattr(self, '_model') and self._model:
            lib.rnnoise_model_free(self._model)
            self._model = None 
            
        self._destroyed = True

    def __del__(self):
        # Use getattr to safely check _destroyed, default to True if not set (e.g., init failed)
        if not getattr(self, '_destroyed', True):
             self.destroy() 