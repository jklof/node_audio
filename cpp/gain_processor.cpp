#include <cstdint>
#include <cmath>

#if defined(_WIN32)
    #define PLUGIN_API extern "C" __declspec(dllexport)
#else
    #define PLUGIN_API extern "C"
#endif

// - The actual DSP logic class
class GainProcessor {
public:
    GainProcessor() : gain_factor(1.0f) {}

    void process(float* buffer, int num_channels, int num_samples) {
        for (int i = 0; i < num_channels * num_samples; ++i) {
            buffer[i] *= gain_factor;
        }
    }

    void set_gain_db(float db) {
        // Basic dB to amplitude conversion
        gain_factor = powf(10.0f, db / 20.0f);
    }

private:
    float gain_factor;
};

// The C-style API that Python will talk to

// create the instance (required)
PLUGIN_API void* create_handle() {
    return new GainProcessor();
}

// cleans up the instance (required)
PLUGIN_API void remove_handle(void* handle) {
    if (handle) {
        delete static_cast<GainProcessor*>(handle);
    }
}

// Sets a parameter on an existing instance
PLUGIN_API void set_gain_db(void* handle, float db) {
    if (handle) {
        static_cast<GainProcessor*>(handle)->set_gain_db(db);
    }
}

// Processes a block of audio
PLUGIN_API void process_block(void* handle, float* buffer, int num_channels, int num_samples) {
    if (handle) {
        static_cast<GainProcessor*>(handle)->process(buffer, num_channels, num_samples);
    }
}
