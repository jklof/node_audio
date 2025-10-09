#include <vector>
#include <cmath>
#include <algorithm> // For std::min/max
#include <cstdint>

#if defined(_WIN32)
    #define PLUGIN_API extern "C" __declspec(dllexport)
#else
    #define PLUGIN_API extern "C"
#endif

// Define a portable, compile-time constant for PI
constexpr float PI = 3.14159265358979323846f;

// A small constant for floating point comparisons and preventing division by zero
constexpr float EPSILON = 1e-9f;

// Enum to be shared with Python (values must match)
enum class ModulationEffectType {
    CHORUS = 0,
    FLANGER = 1,
    PHASER = 2
};

class ModulationProcessor {
public:
    // --- Lifecycle and Configuration ---

    ModulationProcessor()
        : _sampleRate(44100), _maxChannels(0), _maxBlockSize(0), _bufferSizeSamples(0),
          _mode(ModulationEffectType::CHORUS), _rateHz(0.5f), _depth(0.5f),
          _feedback(0.0f), _mix(0.5f), _lfoPhase(0.0f), _writeHead(0) {
        // Constructor is minimal; setup happens in prepare()
    }

    void prepare(int sampleRate, int maxBlockSize, int maxChannels) {
        _sampleRate = sampleRate;
        _maxChannels = maxChannels;
        _maxBlockSize = maxBlockSize;
        
        constexpr float MAX_DELAY_S = 0.05f;
        _bufferSizeSamples = static_cast<int>(MAX_DELAY_S * _sampleRate);

        _delayBuffer.resize(_maxChannels, std::vector<float>(_bufferSizeSamples, 0.0f));
        _phaserZ.resize(6, std::vector<float>(_maxChannels, 0.0f)); // 6 phaser stages
        _lfoBuffer.resize(_maxBlockSize, 0.0f);

        reset();
    }

    void reset() {
        for (auto& channel_buffer : _delayBuffer) {
            std::fill(channel_buffer.begin(), channel_buffer.end(), 0.0f);
        }
        for (auto& stage_buffer : _phaserZ) {
            std::fill(stage_buffer.begin(), stage_buffer.end(), 0.0f);
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

        // 1. Pre-calculate LFO for the entire block
        const float phaseIncrement = _rateHz / static_cast<float>(_sampleRate);
        for (int i = 0; i < numSamples; ++i) {
            _lfoBuffer[i] = sinf(2.0f * PI * _lfoPhase); // <-- FIX: Use PI
            _lfoPhase += phaseIncrement;
            if (_lfoPhase >= 1.0f) _lfoPhase -= 1.0f;
        }

        // 2. Process based on the current mode
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

    void processPhaser(float** inChannels, float** outChannels, int numChannels, int numSamples) {
        constexpr int PHASER_STAGES = 6;
        for (int i = 0; i < numSamples; ++i) {
            float sweep_width = (4000.0f - 100.0f) * _depth;
            float center_freq = 100.0f + sweep_width / 2.0f + ((4000.0f - 100.0f - sweep_width) / 2.0f) * (_lfoBuffer[i] + 1.0f);
            float d = (1.0f - (PI * center_freq / _sampleRate)) / (1.0f + (PI * center_freq / _sampleRate) + EPSILON);

            for (int ch = 0; ch < numChannels; ++ch) {
                float feedback_input = _phaserZ[PHASER_STAGES - 1][ch] * _feedback;
                float stage_input = inChannels[ch][i] + feedback_input;

                for (int stage = 0; stage < PHASER_STAGES; ++stage) {
                    float output_sample = d * stage_input + (1.0f - d) * _phaserZ[stage][ch];
                    _phaserZ[stage][ch] = stage_input;
                    stage_input = output_sample;
                }
                
                float wet_sample = stage_input;
                float dry_sample = inChannels[ch][i];
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
    std::vector<std::vector<float>> _delayBuffer;
    int _writeHead;
    std::vector<std::vector<float>> _phaserZ;
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