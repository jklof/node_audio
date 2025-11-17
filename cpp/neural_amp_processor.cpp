#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <memory>
#include <filesystem>
#include <stdexcept> // For std::exception
// Include NAM core headers
#include "NAM/dsp.h" 
#include "NAM/get_dsp.h" 

#if defined(_WIN32)
    #define PLUGIN_API extern "C" __declspec(dllexport)
#else
    #define PLUGIN_API extern "C"
#endif

// Forward declare NAM_SAMPLE to use it in the class, aligning with the CMake fix
#ifdef NAM_SAMPLE_FLOAT
    #define NAM_SAMPLE float
#else
    #define NAM_SAMPLE double
#endif

class NamProcessor {
public:
    NamProcessor()
    {
        _nam_file = "";
        _dsp = nullptr;
        _sample_rate = 0.0;
        _is_loaded = false;
    }

    // Set model file and trigger loading
    void set_parameters(const char *nam_file, float sample_rate, int max_buffer_size) {
        // Only reload if file path or sample rate has changed significantly
        if (nam_file && (std::string(nam_file) == _nam_file) && 
            (std::abs(sample_rate - _sample_rate) < 1e-6) && _is_loaded) {
            return; 
        }

        // --- Core Logic: Load the model and Reset ---
        try {
            std::string new_nam_file = std::string(nam_file);
            
            // 1. Load the DSP object only if the file path has changed
            if (new_nam_file != _nam_file || !_dsp) {
                _dsp = nam::get_dsp(std::filesystem::path(new_nam_file));
                _nam_file = new_nam_file;
            }
            
            if (_dsp) {
                // 2. Reset the DSP with the new parameters
                _sample_rate = (double)sample_rate;
                // Note: The NAM Core uses 'double' for sampleRate in Reset
                _dsp->Reset(_sample_rate, max_buffer_size); 
                
                _is_loaded = true;
            } else {
                _is_loaded = false;
            }
        } catch (const std::exception& e) {
            // Error handling for loading failure
            _is_loaded = false;
            _dsp.reset();
        }
    }

    void process(float* in_channel, float* out_channel, int num_samples) {
        // If not loaded or pointers are invalid, write silence to the output buffer
        if (!_is_loaded || !_dsp || !in_channel || !out_channel || num_samples <= 0) {
             if (out_channel) {
                std::fill(out_channel, out_channel + num_samples, 0.0f);
             }
            return;
        }

        // --- Core Logic: Process audio ---
        // The NAM Core DSP::process expects NAM_SAMPLE*, which is 'float*'
        // This now maps directly to our arguments.
        NAM_SAMPLE* input_signal = (NAM_SAMPLE*)in_channel;
        NAM_SAMPLE* output_signal = (NAM_SAMPLE*)out_channel;

        try {
            _dsp->process(input_signal, output_signal, num_samples);
        } catch (const std::exception& e) {
            // Runtime error in DSP core - put it into an error state
            _is_loaded = false;
            _dsp.reset();
        }
    }

private:
    std::string _nam_file;
    std::unique_ptr<nam::DSP> _dsp;
    double _sample_rate;
    bool _is_loaded;
};

// ... C-style API for Python's ctypes ...

PLUGIN_API void* create_handle() {
    return new (std::nothrow) NamProcessor();
}

PLUGIN_API void delete_handle(void* handle) {
    if (handle) {
        delete static_cast<NamProcessor*>(handle);
    }
}

PLUGIN_API void set_parameters(void* handle, const char *nam_file, float sample_rate, int max_buffer_size) {
    if (handle) {
        static_cast<NamProcessor*>(handle)->set_parameters(nam_file, sample_rate, max_buffer_size);
    }
}

PLUGIN_API void process_block(void* handle, float* in_channel, float* out_channel, int num_samples) {
    if (handle) {
        static_cast<NamProcessor*>(handle)->process(in_channel, out_channel, num_samples);
    }
}