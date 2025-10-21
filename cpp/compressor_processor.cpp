#include <vector>
#include <cmath>
#include <algorithm> // For std::max, std::min
#include <cstdint>   // For size_t

#if defined(_WIN32)
    #define PLUGIN_API extern "C" __declspec(dllexport)
#else
    #define PLUGIN_API extern "C"
#endif

// --- Constants ---
constexpr int SIDECHAIN_DOWNSAMPLE_FACTOR = 16;
constexpr float EPSILON = 1e-9f;

// ==============================================================================
// 1. CompressorProcessor Class Definition
// ==============================================================================
class CompressorProcessor {
public:
    CompressorProcessor();

    void set_parameters(float samplerate, float threshold_db, float ratio,
                        float attack_ms, float release_ms, float knee_db);

    void process(float** in_channels, float** sidechain_channels,
                 float** out_channels, int num_channels, int num_samples);

private:
    // Parameters
    float _samplerate;
    float _threshold_db;
    float _ratio;
    float _knee_db;
    float _attack_coeff;
    float _release_coeff;

    // DSP State
    float _envelope;
    std::vector<float> _delay_buffer;
    int _delay_samples;
    int _write_head;
    int _max_channels;
};

// ==============================================================================
// 2. CompressorProcessor Class Implementation
// ==============================================================================
CompressorProcessor::CompressorProcessor()
    : _samplerate(44100.0f),
      _threshold_db(-20.0f),
      _ratio(4.0f),
      _knee_db(6.0f),
      _attack_coeff(0.0f),
      _release_coeff(0.0f),
      _envelope(0.0f),
      _delay_samples(SIDECHAIN_DOWNSAMPLE_FACTOR / 2),
      _write_head(0),
      _max_channels(2) {
    // Initialize delay buffer for default channel count. The size is channels * delay_length.
    _delay_buffer.resize(static_cast<size_t>(_max_channels) * _delay_samples, 0.0f);
}

void CompressorProcessor::set_parameters(float samplerate, float threshold_db, float ratio,
                                       float attack_ms, float release_ms, float knee_db) {
    _samplerate = samplerate > 0 ? samplerate : 44100.0f;
    _threshold_db = threshold_db;
    _ratio = std::max(1.0f, ratio);
    _knee_db = std::max(0.0f, knee_db);

    float downsampled_samplerate = _samplerate / static_cast<float>(SIDECHAIN_DOWNSAMPLE_FACTOR);
    float attack_s = attack_ms / 1000.0f;
    float release_s = release_ms / 1000.0f;

    // Coefficients are calculated based on the downsampled rate
    _attack_coeff = (attack_s > EPSILON) ? expf(-1.0f / (downsampled_samplerate * attack_s)) : 0.0f;
    _release_coeff = (release_s > EPSILON) ? expf(-1.0f / (downsampled_samplerate * release_s)) : 0.0f;
}

void CompressorProcessor::process(float** in_channels, float** sidechain_channels,
                                  float** out_channels, int num_channels, int num_samples) {
    if (!in_channels || !out_channels || num_channels <= 0 || num_samples <= 0) {
        return;
    }

    // Resize delay buffer if channel count has changed since last time
    if (num_channels != _max_channels) {
        _max_channels = num_channels;
        _delay_buffer.assign(static_cast<size_t>(_max_channels) * _delay_samples, 0.0f);
        _write_head = 0; // Reset write head on resize
    }

    // Determine which signal to use for level detection
    float** detection_channels = (sidechain_channels != nullptr) ? sidechain_channels : in_channels;
    
    // A single buffer for the downsampled power values
    std::vector<float> power_sidechain(num_samples);
    // A buffer to hold the final linear gain values for each sample
    std::vector<float> gain_reduction_linear(num_samples);

    // --- Step 1: Level Detection (Full Sample Rate) ---
    // Create a mono power signal by finding the max absolute value across channels, then squaring it.
    for (int i = 0; i < num_samples; ++i) {
        float max_val = 0.0f;
        for (int ch = 0; ch < num_channels; ++ch) {
            if (detection_channels[ch]) {
                max_val = std::max(max_val, std::abs(detection_channels[ch][i]));
            }
        }
        power_sidechain[i] = max_val * max_val;
    }

    // --- Step 2: Envelope Following & Gain Calculation (Downsampled Rate) ---
    float current_linear_gain = 1.0f;
    for (int i = 0; i < num_samples; ++i) {
        // Only perform the expensive calculations at the downsampled rate
        if (i % SIDECHAIN_DOWNSAMPLE_FACTOR == 0) {
            // Average the power over the downsampling window
            float avg_power = 0.0f;
            int end_idx = std::min(i + SIDECHAIN_DOWNSAMPLE_FACTOR, num_samples);
            for (int j = i; j < end_idx; ++j) {
                avg_power += power_sidechain[j];
            }
            avg_power /= static_cast<float>(end_idx - i);

            // Update envelope (ballistics)
            float target = avg_power;
            float coeff = (target > _envelope) ? _attack_coeff : _release_coeff;
            _envelope = target + coeff * (_envelope - target);

            // Convert envelope to dB and calculate gain reduction
            float envelope_db = 10.0f * log10f(_envelope + EPSILON);
            float gain_reduction_db = 0.0f;
            float slope = 1.0f / _ratio - 1.0f;
            float knee_start = _threshold_db - _knee_db / 2.0f;
            float knee_end = _threshold_db + _knee_db / 2.0f;

            if (envelope_db > knee_end) {
                gain_reduction_db = (envelope_db - _threshold_db) * slope;
            } else if (envelope_db > knee_start) {
                float x = envelope_db - knee_start;
                // Use a safe division for the knee calculation
                gain_reduction_db = (slope / (2.0f * std::max(EPSILON, _knee_db))) * x * x;
            }
            
            // Convert dB gain reduction to a linear multiplier
            current_linear_gain = powf(10.0f, gain_reduction_db / 20.0f);
        }
        // Apply the same gain multiplier to all samples within the downsample window
        gain_reduction_linear[i] = current_linear_gain;
    }

    // --- Step 3: Apply Gain with Delay Compensation (Full Sample Rate) ---
    for (int i = 0; i < num_samples; ++i) {
        // Calculate the read head for the circular buffer to get the delayed sample
        int read_head = (_write_head - (_delay_samples - 1) + _delay_samples) % _delay_samples;
        
        for (int ch = 0; ch < num_channels; ++ch) {
            // Planar buffer index calculation: the buffer is a flat array
            size_t write_idx = static_cast<size_t>(ch) * _delay_samples + _write_head;
            size_t read_idx = static_cast<size_t>(ch) * _delay_samples + read_head;

            // 1. Read the delayed sample from the buffer
            float delayed_sample = _delay_buffer[read_idx];
            
            // 2. Apply gain and write to the output
            out_channels[ch][i] = delayed_sample * gain_reduction_linear[i];

            // 3. Write the current unprocessed input sample into the buffer for the future
            _delay_buffer[write_idx] = in_channels[ch][i];
        }
        // Advance the write head for the next sample, wrapping around if necessary
        _write_head = (_write_head + 1) % _delay_samples;
    }
}

// ==============================================================================
// 3. C-style API for Python's ctypes
// ==============================================================================

PLUGIN_API void* create_handle() {
    return new (std::nothrow) CompressorProcessor();
}

PLUGIN_API void delete_handle(void* handle) {
    if (handle) {
        delete static_cast<CompressorProcessor*>(handle);
    }
}

PLUGIN_API void set_parameters(void* handle, float samplerate, float threshold_db, float ratio,
                               float attack_ms, float release_ms, float knee_db) {
    if (handle) {
        static_cast<CompressorProcessor*>(handle)->set_parameters(samplerate, threshold_db, ratio,
                                                                 attack_ms, release_ms, knee_db);
    }
}

PLUGIN_API void process_block(void* handle, float** in_channels, float** sidechain_channels,
                              float** out_channels, int num_channels, int num_samples) {
    if (handle) {
        static_cast<CompressorProcessor*>(handle)->process(in_channels, sidechain_channels,
                                                         out_channels, num_channels, num_samples);
    }
}