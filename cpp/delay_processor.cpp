#include <vector>
#include <cmath>
#include <cstdint>
#if defined(_WIN32)
    #define PLUGIN_API extern "C" __declspec(dllexport)
#else
    #define PLUGIN_API extern "C"
#endif

class DelayProcessor {
public:
    DelayProcessor()
        : _buffer_size(0),
          _max_channels(0),
          _write_head(0),
          _delay_samples(100.0f),
          _feedback(0.5f),
          _mix(0.5f) {
        _delay_buffer.resize(_buffer_size * _max_channels, 0.0f);
    }

    // Set all parameters in one call to reduce FFI overhead
    void set_parameters(int buffer_size, int max_channels,
                        float delay_samples, float feedback, float mix) {
        // Validate inputs
        if (buffer_size <= 0 || max_channels <= 0) {
            return;
        }

        // Clamp parameters to valid ranges
        delay_samples = std::max(0.0f, std::min(delay_samples, (float)(buffer_size - 1)));
        feedback = std::max(0.0f, std::min(feedback, 1.0f));
        mix = std::max(0.0f, std::min(mix, 1.0f));

        // Resize buffer if needed
        if (buffer_size != _buffer_size || max_channels != _max_channels) {
            _buffer_size = buffer_size;
            _max_channels = max_channels;
            _delay_buffer.resize(_buffer_size * _max_channels, 0.0f);
            _write_head = 0;  // Reset write head on buffer resize
        }

        _delay_samples = delay_samples;
        _feedback = feedback;
        _mix = mix;
    }

    void process(float** in_channels, float** out_channels, int num_channels, int num_samples) {
        // Validate inputs
        if (!in_channels || !out_channels || num_samples <= 0 || _buffer_size <= 0 || num_channels <= 0) {
            return;
        }

        // Check that num_channels does not exceed max_channels
        if (num_channels > _max_channels) {
            num_channels = _max_channels;
        }

        // Check that all channel pointers are valid
        for (int ch = 0; ch < num_channels; ++ch) {
            if (!in_channels[ch] || !out_channels[ch]) {
                return;
            }
        }

        for (int i = 0; i < num_samples; ++i) {
            // --- 1. Calculate read position with linear interpolation ---
            float read_head_float = static_cast<float>(_write_head) - _delay_samples;
            int index_floor = static_cast<int>(floorf(read_head_float));
            int index_ceil = index_floor + 1;
            float fraction = read_head_float - index_floor;

            // Wrap indices around the buffer
            if (index_floor < 0) {
                index_floor += _buffer_size;
            } else if (index_floor >= _buffer_size) {
                index_floor -= _buffer_size;
            }

            if (index_ceil < 0) {
                index_ceil += _buffer_size;
            } else if (index_ceil >= _buffer_size) {
                index_ceil -= _buffer_size;
            }

            for (int ch = 0; ch < num_channels; ++ch) {
                // --- 2. Read from delay buffer with interpolation ---
                float sample1 = _delay_buffer[ch * _buffer_size + index_floor];
                float sample2 = _delay_buffer[ch * _buffer_size + index_ceil];
                float delayed_sample = sample1 * (1.0f - fraction) + sample2 * fraction;

                float dry_sample = in_channels[ch][i];

                // --- 3. Mix dry and wet signals to create output ---
                out_channels[ch][i] = (dry_sample * (1.0f - _mix)) + (delayed_sample * _mix);

                // --- 4. Create feedback signal and write to buffer ---
                float feedback_sample = dry_sample + (delayed_sample * _feedback);
                _delay_buffer[ch * _buffer_size + _write_head] = feedback_sample;
            }

            // --- 5. Advance write head ---
            _write_head = (_write_head + 1) % _buffer_size;
        }
    }

private:
    std::vector<float> _delay_buffer;
    int _buffer_size;
    int _max_channels;
    int _write_head;

    // Parameters
    float _delay_samples;
    float _feedback;
    float _mix;
};

// ===================================================
// C-style API for Python's ctypes
// ===================================================

PLUGIN_API void* create_handle() {
    return new (std::nothrow) DelayProcessor();
}

PLUGIN_API void delete_handle(void* handle) {
    if (handle) {
        delete static_cast<DelayProcessor*>(handle);
    }
}

PLUGIN_API void set_parameters(void* handle, int buffer_size, int max_channels,
                               float delay_samples, float feedback, float mix) {
    if (handle) {
        static_cast<DelayProcessor*>(handle)->set_parameters(buffer_size, max_channels,
                                                             delay_samples, feedback, mix);
    }
}

PLUGIN_API void process_block(void* handle, float** in_channels, float** out_channels,
                              int num_channels, int num_samples) {
    if (handle) {
        static_cast<DelayProcessor*>(handle)->process(in_channels, out_channels,
                                                       num_channels, num_samples);
    }
}
