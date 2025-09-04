use nih_plug::prelude::*;
use parking_lot::Mutex;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::num::NonZeroU32;
use std::time::{Duration, Instant};
use nih_plug_egui::{egui, create_egui_editor, EguiState};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 61000;

fn write_all(stream: &mut TcpStream, buf: &[u8]) -> anyhow::Result<()> {
    stream.write_all(buf)?;
    Ok(())
}

fn read_exact(stream: &mut TcpStream, buf: &mut [u8]) -> anyhow::Result<()> {
    stream.read_exact(buf)?;
    Ok(())
}

struct BridgeClient {
    stream: Option<TcpStream>,
    seq: u32,
    last_config: (u32, u32, u16, u16),
    next_retry_at: Option<Instant>,
    retry_interval: Duration,
}

impl BridgeClient {
    fn new() -> Self {
        Self {
            stream: None,
            seq: 0,
            last_config: (0, 0, 0, 0),
            next_retry_at: None,
            retry_interval: Duration::from_millis(1000),
        }
    }

    fn is_connected(&self) -> bool { self.stream.is_some() }

    fn force_reconnect(&mut self) {
        self.stream = None;
        self.seq = 0;
        self.next_retry_at = Some(Instant::now());
    }

    fn ensure_connected(
        &mut self,
        sample_rate: f32,
        block_size: usize,
        in_ch: usize,
        out_ch: usize,
    ) -> anyhow::Result<()> {
        let cfg = (sample_rate as u32, block_size as u32, in_ch as u16, out_ch as u16);
        let reconnect = match self.stream { None => true, Some(_) => self.last_config != cfg };
        if reconnect {
            // Respect retry cooldown when not connected
            if self.stream.is_none() {
                if let Some(when) = self.next_retry_at {
                    if Instant::now() < when { return Ok(()); }
                }
            }

            match TcpStream::connect((DEFAULT_HOST, DEFAULT_PORT)) {
                Ok(mut stream) => {
                    let mut hdr = Vec::with_capacity(4 + 4 + 4 + 2 + 2);
                    hdr.extend_from_slice(b"NABR");
                    hdr.extend_from_slice(&(cfg.0).to_le_bytes());
                    hdr.extend_from_slice(&(cfg.1).to_le_bytes());
                    hdr.extend_from_slice(&(cfg.2).to_le_bytes());
                    hdr.extend_from_slice(&(cfg.3).to_le_bytes());
                    write_all(&mut stream, &hdr)?;
                    stream.set_nodelay(true).ok();
                    self.stream = Some(stream);
                    self.last_config = cfg;
                    self.seq = 0;
                    self.next_retry_at = None;
                }
                Err(_e) => {
                    // Schedule next retry and remain disconnected
                    self.stream = None;
                    self.next_retry_at = Some(Instant::now() + self.retry_interval);
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) -> anyhow::Result<()> {
        let (_sr, bs, in_ch, out_ch) = self.last_config;
        let mut stream = match self.stream.as_ref() {
            Some(s) => s.try_clone()?,
            None => anyhow::bail!("not connected"),
        };

        // send request: opcode=1, seq, payload
        let mut header = [0u8; 1 + 4];
        header[0] = 1;
        header[1..5].copy_from_slice(&self.seq.to_le_bytes());
        write_all(&mut stream, &header)?;
        if in_ch > 0 {
            write_all(&mut stream, bytemuck::cast_slice(input))?;
        }

        // recv response: opcode=2, seq, payload
        let mut resp_header = [0u8; 1 + 4];
        read_exact(&mut stream, &mut resp_header)?;
        if resp_header[0] != 2 {
            anyhow::bail!("invalid opcode from server");
        }
        let rsp_seq = u32::from_le_bytes(resp_header[1..5].try_into().unwrap());
        if rsp_seq != self.seq {
            // Out-of-order; continue anyway
        }
        let expected = (bs as usize) * (out_ch as usize);
        let mut bytes = vec![0u8; expected * 4];
        read_exact(&mut stream, &mut bytes)?;
        let samples: &[f32] = bytemuck::cast_slice(&bytes);
        output.copy_from_slice(samples);
        self.seq = self.seq.wrapping_add(1);
        Ok(())
    }
}

struct NodeAudioFx {
    client: Arc<Mutex<BridgeClient>>,
    params: Arc<EmptyParams>,
    editor_state: Arc<EguiState>,
}

impl Default for NodeAudioFx {
    fn default() -> Self {
        Self {
            client: Arc::new(Mutex::new(BridgeClient::new())),
            params: Arc::new(EmptyParams {}),
            editor_state: EguiState::from_size(280, 120),
        }
    }
}

#[derive(Params)]
struct EmptyParams {}

impl Plugin for NodeAudioFx {
    const NAME: &'static str = "NodeAudio Bridge FX";
    const VENDOR: &'static str = "NodeAudio";
    const URL: &'static str = "https://localhost";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = "0.1.0";
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: Some(NonZeroU32::new(2).unwrap()),
        main_output_channels: Some(NonZeroU32::new(2).unwrap()),
        ..AudioIOLayout::const_default()
    }];

    type SysExMessage = (); // Not used for now

    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> { self.params.clone() }

    fn initialize(&mut self, _audio_io_layout: &AudioIOLayout, buffer_config: &BufferConfig, _context: &mut impl InitContext<Self>) -> bool {
        let mut client = self.client.lock();
        client.ensure_connected(buffer_config.sample_rate, buffer_config.max_buffer_size as usize, 2, 2).ok();
        true
    }

    fn reset(&mut self) {
        // Reconnect on reset
        let _ = self.client.lock().ensure_connected(44100.0, 512, 2, 2);
    }

    fn process(&mut self, buffer: &mut Buffer, _aux: &mut AuxiliaryBuffers, context: &mut impl ProcessContext<Self>) -> ProcessStatus {
        let frames = buffer.samples();
        let chans = buffer.as_slice();
        let num_chans = chans.len();
        let in_ch = num_chans.min(2);
        let out_ch = num_chans.min(2);

        // Interleave input as f32 frames x in_ch
        let mut input_interleaved = vec![0.0f32; frames * in_ch.max(1)];
        if in_ch > 0 {
            for f in 0..frames {
                for c in 0..in_ch {
                    input_interleaved[f * in_ch + c] = chans[c][f];
                }
            }
        }

        let mut output_interleaved = vec![0.0f32; frames * out_ch.max(1)];
        let result = {
            let mut client = self.client.lock();
            client.ensure_connected(context.transport().sample_rate, frames, in_ch, out_ch).ok();
            client.process_block(&input_interleaved, &mut output_interleaved)
        };
        if result.is_err() || output_interleaved.iter().all(|s| s.abs() < f32::EPSILON) {
            // If disconnected or failed, pass input through to output for overlapping channels
            let copy_ch = in_ch.min(out_ch);
            for f in 0..frames {
                for c in 0..copy_ch {
                    output_interleaved[f * out_ch + c] = input_interleaved[f * in_ch + c];
                }
            }
        }

        // De-interleave into outputs
        for f in 0..frames {
            for c in 0..out_ch {
                chans[c][f] = output_interleaved[f * out_ch + c];
            }
        }

        ProcessStatus::Normal
    }

    fn editor(&mut self, _executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let client = self.client.clone();
        let editor = create_egui_editor(
            self.editor_state.clone(),
            self.params.clone(),
            move |_ctx, _| {
                // No-op build step
            },
            move |ctx, _setter, _| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let connected = { client.lock().is_connected() };
                    let status_text = if connected { "Connected" } else { "Disconnected" };
                    let color = if connected { egui::Color32::from_rgb(0, 200, 0) } else { egui::Color32::from_rgb(200, 0, 0) };
                    ui.horizontal(|ui| {
                        ui.colored_label(color, status_text);
                        if !connected {
                            if ui.button("Retry now").clicked() {
                                client.lock().force_reconnect();
                            }
                        }
                    });
                });
            },
        );
        editor
    }
}

impl ClapPlugin for NodeAudioFx {
    const CLAP_ID: &'static str = "com.nodeaudio.bridgefx";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Bridge to node_audio over TCP");
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo, ClapFeature::Utility];
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
}

impl Vst3Plugin for NodeAudioFx {
    const VST3_CLASS_ID: [u8; 16] = *b"NodeBrdgFxRust!!";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Fx, Vst3SubCategory::Tools];
}

nih_export_clap!(NodeAudioFx);
nih_export_vst3!(NodeAudioFx);


