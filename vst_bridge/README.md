Rust VST3 Bridge for node_audio

Overview

This workspace builds two VST3 plugins that bridge audio and optional MIDI between a DAW and the Python node_audio app over a simple TCP protocol.

- node_audio_vst_fx: Audio effect (in-place or sends audio to app and returns processed audio).
- node_audio_vst_instrument: Instrument (generates audio from the app; receives MIDI in future).

Transport Protocol

- TCP to localhost:61000 by default. The Python app runs a bridge server when the "Network Bridge Output" node is the active clock. If present, a "Network Bridge Input" node provides the effect input audio to the graph.
- Handshake (client→server):
  NABR + u32 sample_rate + u32 blocksize + u16 in_channels + u16 out_channels
- Per block (client→server):
  opcode=1 (u8), seq=u32, interleaved float32 input samples [frames x in_channels]
- Per block (server→client):
  opcode=2 (u8), seq=u32, interleaved float32 output samples [frames x out_channels]

Build

Prereqs: Rust stable, `cargo`, and VST3 SDK runtime dependencies handled by the `nih-plug` crate (cross-platform VST3 wrapper).

Commands:

```bash
cargo build -p node_audio_vst_fx --release
cargo build -p node_audio_vst_instrument --release
```

On macOS the built VST3 bundles are in `target/release/`. For development, most DAWs can load from a custom path.

Usage

1) Start the Python app and add nodes:
   - Add "Network Bridge Output" (set as clock).
   - Optional for FX: add "Network Bridge Input" and route it into your graph.
   - Connect your graph to the bridge output.
   - Start processing (F5).

2) In your DAW:
   - Load the FX plugin on an insert to process track audio via the app.
   - Or load the Instrument plugin on a MIDI/instrument track.

Troubleshooting

- If the plugin can't connect, ensure the app is running and the bridge node is the active clock.
- Block size mismatches increase latency; prefer 512 frames in the DAW.
- Firewall prompts: allow local connections on localhost.


