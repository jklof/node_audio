#include <vector>
#include <cmath>
#include <algorithm> // For std::min/max
#include <cstdint>

#if defined(_WIN32)
    #define PLUGIN_API extern "C" __declspec(dllexport)
#else
    #define PLUGIN_API extern "C"
#endif

// Define a portable, compile-time constant for Pi
constexpr float PI = 3.14159265358979323846f;

// A small constant for floating point comparisons and preventing division by zero
constexpr float EPSILON = 1e-9f;

// Enum to be shared with Python (values must match)
enum class ModulationEffectType {
    CHORUS = 0,
    FLANGER = 1,
    PHASER = 2
};

// --- Define the number of phaser stages ---
constexpr int PHASER_STAGES = 6;


class ModulationProcessor {
public:
    // --- Lifecycle and Configuration ---

    ModulationProcessor()
        : _sampleRate(44100), _maxChannels(0), _maxBlockSize(0), _bufferSizeSamples(0),
          _mode(ModulationEffectType::CHORUS), _rateHz(0.5f), _depth(0.5f),
          _feedback(0.0f), _mix(0.5f), _lfoPhase(0.0f), _writeHead(0) {
    }

    void prepare(int sampleRate, int maxBlockSize, int maxChannels) {
        _sampleRate = sampleRate;
        _maxChannels = maxChannels;
        _maxBlockSize = maxBlockSize;
        
        constexpr float MAX_DELAY_S = 0.05f;
        _bufferSizeSamples = static_cast<int>(MAX_DELAY_S * _sampleRate);

        _delayBuffer.resize(_maxChannels, std::vector<float>(_bufferSizeSamples, 0.0f));
        
        // --- Resize phaser state buffer for IIR filters ---
        // Each of the 6 stages needs 2 state variables per channel:
        // z[0] = x[n-1] (previous input)
        // z[1] = y[n-1] (previous output)
        _phaserZ.resize(PHASER_STAGES, std::vector<std::vector<float>>(2, std::vector<float>(_maxChannels, 0.0f)));
        
        _lfoBuffer.resize(_maxBlockSize, 0.0f);

        reset();
    }

    void reset() {
        for (auto& channel_buffer : _delayBuffer) {
            std::fill(channel_buffer.begin(), channel_buffer.end(), 0.0f);
        }
        // --- reset the 3D phaser state buffer ---
        for (auto& stage_buffer : _phaserZ) {
            for (auto& state_vec : stage_buffer) {
                std::fill(state_vec.begin(), state_vec.end(), 0.0f);
            }
        }
        _writeHead = 0;
        _lfoPhase = 0.0f;
    }

    // --- Parameter Setting ---

    void setParameters(int mode, float rateHz, float depth, float feedback, float mix) {
        ModulationEffectType newMode = static_cast<ModulationEffectType>(mode);
        if (_mode != newMode) {
            reset();
        }
        _mode = newMode;
        _rateHz = std::max(0.01f, rateHz);
        _depth = std::max(0.0f, std::min(1.0f, depth));
        _feedback = std::max(0.0f, std::min(0.98f, feedback));
        _mix = std::max(0.0f, std::min(1.0f, mix));
    }

    // --- Real-time Processing ---

    void process(float** inChannels, float** outChannels, int numChannels, int numSamples) {
        if (!inChannels || !outChannels || numChannels <= 0 || numSamples <= 0 || numChannels > _maxChannels) {
            return;
        }

        const float phaseIncrement = _rateHz / static_cast<float>(_sampleRate);
        for (int i = 0; i < numSamples; ++i) {
            // Generate a bipolar LFO signal (-1 to 1)
            _lfoBuffer[i] = sinf(2.0f * PI * _lfoPhase);
            _lfoPhase += phaseIncrement;
            if (_lfoPhase >= 1.0f) _lfoPhase -= 1.0f;
        }

        switch (_mode) {
            case ModulationEffectType::CHORUS:
                processDelayEffect(inChannels, outChannels, numChannels, numSamples, 25.0f, 20.0f);
                break;
            case ModulationEffectType::FLANGER:
                processDelayEffect(inChannels, outChannels, numChannels, numSamples, 5.0f, 4.9f);
                break;
            case ModulationEffectType::PHASER:
                processPhaser(inChannels, outChannels, numChannels, numSamples);
                break;
        }
    }

private:
    // --- DSP Helper Methods ---

    void processDelayEffect(float** inChannels, float** outChannels, int numChannels, int numSamples, float centerDelayMs, float depthMs) {
        const float center_delay_samples = centerDelayMs / 1000.0f * _sampleRate;
        const float depth_samples = (depthMs * _depth) / 1000.0f * _sampleRate;

        for (int i = 0; i < numSamples; ++i) {
            float current_delay_samps = center_delay_samples + _lfoBuffer[i] * depth_samples;
            
            float read_head_float = static_cast<float>(_writeHead) - current_delay_samps;
            int index_floor = static_cast<int>(floorf(read_head_float));
            float fraction = read_head_float - index_floor;

            for (int ch = 0; ch < numChannels; ++ch) {
                int wrapped_floor = (index_floor % _bufferSizeSamples + _bufferSizeSamples) % _bufferSizeSamples;
                int wrapped_ceil = ((index_floor + 1) % _bufferSizeSamples + _bufferSizeSamples) % _bufferSizeSamples;

                float sample1 = _delayBuffer[ch][wrapped_floor];
                float sample2 = _delayBuffer[ch][wrapped_ceil];
                float wet_sample = sample1 * (1.0f - fraction) + sample2 * fraction;
                float dry_sample = inChannels[ch][i];
                
                outChannels[ch][i] = (dry_sample * (1.0f - _mix)) + (wet_sample * _mix);
                _delayBuffer[ch][_writeHead] = dry_sample + (wet_sample * _feedback);
            }
            _writeHead = (_writeHead + 1) % _bufferSizeSamples;
        }
    }

    // --- Phaser Implementation ---
    void processPhaser(float** inChannels, float** outChannels, int numChannels, int numSamples) {
        const float min_freq = 100.0f;
        const float max_freq = 4000.0f;
        const float sweep_width = (max_freq - min_freq) * _depth;

        for (int i = 0; i < numSamples; ++i) {
            // Map bipolar LFO (-1 to 1) to unipolar (0 to 1) for frequency control
            float unipolar_lfo = (_lfoBuffer[i] + 1.0f) * 0.5f;
            float center_freq = min_freq + sweep_width * unipolar_lfo;

            // Calculate the all-pass filter coefficient 'a' based on the modulated frequency.
            // This is the core of the sweeping effect.
            float tan_val = tanf(PI * center_freq / static_cast<float>(_sampleRate));
            float a = (1.0f - tan_val) / (1.0f + tan_val + EPSILON);

            for (int ch = 0; ch < numChannels; ++ch) {
                // Get the feedback from the output of the final stage from the PREVIOUS sample tick.
                float feedback_input = _phaserZ[PHASER_STAGES - 1][1][ch] * _feedback;

                // The input to the first stage is the dry signal plus the feedback.
                float current_stage_input = inChannels[ch][i] - feedback_input;

                // Cascade through all all-pass filter stages.
                for (int stage = 0; stage < PHASER_STAGES; ++stage) {
                    // Retrieve state variables from the previous sample tick for this stage.
                    float prev_input = _phaserZ[stage][0][ch];
                    float prev_output = _phaserZ[stage][1][ch];

                    // Apply the first-order IIR all-pass filter difference equation:
                    // y[n] = a*x[n] + x[n-1] - a*y[n-1]
                    float current_stage_output = a * current_stage_input + prev_input - a * prev_output;

                    // Update the state variables for the NEXT sample tick.
                    _phaserZ[stage][0][ch] = current_stage_input;  // Store current input as x[n-1] for next time.
                    _phaserZ[stage][1][ch] = current_stage_output; // Store current output as y[n-1] for next time.

                    // The output of this stage becomes the input for the next one.
                    current_stage_input = current_stage_output;
                }

                // The final output of the filter chain is our wet signal.
                float wet_sample = current_stage_input;
                float dry_sample = inChannels[ch][i];

                // Mix the dry and wet signals.
                outChannels[ch][i] = (dry_sample * (1.0f - _mix)) + (wet_sample * _mix);
            }
        }
    }


    // --- Member Variables ---
    int _sampleRate;
    int _maxChannels;
    int _maxBlockSize;
    int _bufferSizeSamples;

    ModulationEffectType _mode;
    float _rateHz;
    float _depth;
    float _feedback;
    float _mix;

    float _lfoPhase;
    std::vector<float> _lfoBuffer;
    
    // For Chorus/Flanger
    std::vector<std::vector<float>> _delayBuffer;
    int _writeHead;
    
    // For Phaser: A 3D vector to hold state for each stage, state variable, and channel.
    // Dims: [stage_idx][state_var_idx][channel_idx]
    // state_var_idx: 0 for previous_input (x[n-1]), 1 for previous_output (y[n-1])
    std::vector<std::vector<std::vector<float>>> _phaserZ;
};

// ===================================================
// C-style API for Python's ctypes (unchanged)
// ===================================================

PLUGIN_API void* create_handle() {
    return new ModulationProcessor();
}

PLUGIN_API void delete_handle(void* handle) {
    if (handle) {
        delete static_cast<ModulationProcessor*>(handle);
    }
}

PLUGIN_API void prepare(void* handle, int sampleRate, int maxBlockSize, int maxChannels) {
    if (handle) {
        static_cast<ModulationProcessor*>(handle)->prepare(sampleRate, maxBlockSize, maxChannels);
    }
}

PLUGIN_API void reset(void* handle) {
    if (handle) {
        static_cast<ModulationProcessor*>(handle)->reset();
    }
}

PLUGIN_API void set_parameters(void* handle, int mode, float rateHz, float depth, float feedback, float mix) {
    if (handle) {
        static_cast<ModulationProcessor*>(handle)->setParameters(mode, rateHz, depth, feedback, mix);
    }
}

PLUGIN_API void process_block(void* handle, float** in_channels, float** out_channels, int num_channels, int num_samples) {
    if (handle) {
        static_cast<ModulationProcessor*>(handle)->process(in_channels, out_channels, num_channels, num_samples);
    }
}