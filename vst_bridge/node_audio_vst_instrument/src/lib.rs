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
    connection_attempts: u32,
    last_error: Option<String>,
}

impl BridgeClient {
    fn new() -> Self {
        Self { 
            stream: None, 
            seq: 0, 
            last_config: (0, 0, 0, 0), 
            next_retry_at: None, 
            retry_interval: Duration::from_millis(1000),
            connection_attempts: 0,
            last_error: None,
        }
    }

    fn is_connected(&self) -> bool { self.stream.is_some() }

    fn force_reconnect(&mut self) {
        self.stream = None;
        self.seq = 0;
        self.next_retry_at = Some(Instant::now());
        self.last_error = None;
    }

    fn get_status_info(&self) -> (bool, u32, Option<String>) {
        (self.is_connected(), self.connection_attempts, self.last_error.clone())
    }

    fn ensure_connected(&mut self, sample_rate: f32, block_size: usize, out_ch: usize) -> anyhow::Result<()> {
        // Instrument: in_ch=0, out_ch=desired
        let cfg = (sample_rate as u32, block_size as u32, 0u16, out_ch as u16);
        let reconnect = match self.stream { None => true, Some(_) => self.last_config != cfg };
        if reconnect {
            if self.stream.is_none() {
                if let Some(when) = self.next_retry_at { if Instant::now() < when { return Ok(()); } }
            }
            match TcpStream::connect((DEFAULT_HOST, DEFAULT_PORT)) {
                Ok(mut stream) => {
                    let mut hdr = Vec::with_capacity(4 + 4 + 4 + 2 + 2);
                    hdr.extend_from_slice(b"NABR");
                    hdr.extend_from_slice(&(cfg.0).to_le_bytes());
                    hdr.extend_from_slice(&(cfg.1).to_le_bytes());
                    hdr.extend_from_slice(&(cfg.2).to_le_bytes());
                    hdr.extend_from_slice(&(cfg.3).to_le_bytes());
                    match write_all(&mut stream, &hdr) {
                        Ok(_) => {
                            stream.set_nodelay(true).ok();
                            self.stream = Some(stream);
                            self.last_config = cfg;
                            self.seq = 0;
                            self.next_retry_at = None;
                            self.connection_attempts = 0;
                            self.last_error = None;
                        }
                        Err(e) => {
                            self.connection_attempts += 1;
                            self.last_error = Some(format!("Handshake failed: {}", e));
                            self.stream = None;
                            self.next_retry_at = Some(Instant::now() + self.retry_interval);
                            return Ok(());
                        }
                    }
                }
                Err(e) => {
                    self.connection_attempts += 1;
                    self.last_error = Some(format!("Connection failed: {}", e));
                    self.stream = None;
                    self.next_retry_at = Some(Instant::now() + self.retry_interval);
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn process_block(&mut self, output: &mut [f32]) -> anyhow::Result<()> {
        let (_sr, bs, _in_ch, out_ch) = self.last_config;
        let mut stream = match self.stream.as_ref() { Some(s) => s.try_clone()?, None => anyhow::bail!("not connected") };
        
        // Set a timeout to prevent hanging if server becomes unresponsive
        stream.set_read_timeout(Some(Duration::from_millis(500)))?;
        stream.set_write_timeout(Some(Duration::from_millis(500)))?;
        
        // send request with opcode=1 and zero input payload
        let mut header = [0u8; 1 + 4];
        header[0] = 1;
        header[1..5].copy_from_slice(&self.seq.to_le_bytes());
        write_all(&mut stream, &header).map_err(|e| {
            // Force disconnect on write error
            self.stream = None;
            self.connection_attempts += 1;
            self.last_error = Some(format!("Write error: {}", e));
            self.next_retry_at = Some(Instant::now() + self.retry_interval);
            e
        })?;

        // recv response
        let mut resp_header = [0u8; 1 + 4];
        read_exact(&mut stream, &mut resp_header).map_err(|e| {
            // Force disconnect on read error
            self.stream = None;
            self.connection_attempts += 1;
            self.last_error = Some(format!("Read header error: {}", e));
            self.next_retry_at = Some(Instant::now() + self.retry_interval);
            e
        })?;
        if resp_header[0] != 2 { 
            self.stream = None;
            self.connection_attempts += 1;
            self.last_error = Some("Invalid opcode from server".to_string());
            self.next_retry_at = Some(Instant::now() + self.retry_interval);
            anyhow::bail!("invalid opcode from server"); 
        }
        let _rsp_seq = u32::from_le_bytes(resp_header[1..5].try_into().unwrap());
        let expected = (bs as usize) * (out_ch as usize);
        let mut bytes = vec![0u8; expected * 4];
        read_exact(&mut stream, &mut bytes).map_err(|e| {
            self.stream = None;
            self.connection_attempts += 1;
            self.last_error = Some(format!("Read payload error: {}", e));
            self.next_retry_at = Some(Instant::now() + self.retry_interval);
            e
        })?;
        let samples: &[f32] = bytemuck::cast_slice(&bytes);
        output.copy_from_slice(samples);
        self.seq = self.seq.wrapping_add(1);
        Ok(())
    }
}

struct NodeAudioInstrument {
    client: Arc<Mutex<BridgeClient>>,
    params: Arc<EmptyParams>,
    editor_state: Arc<EguiState>,
}

impl Default for NodeAudioInstrument {
    fn default() -> Self {
        Self { client: Arc::new(Mutex::new(BridgeClient::new())), params: Arc::new(EmptyParams {}), editor_state: EguiState::from_size(300, 150) }
    }
}

#[derive(Params)]
struct EmptyParams {}

impl Plugin for NodeAudioInstrument {
    const NAME: &'static str = "NodeAudio Bridge Instrument";
    const VENDOR: &'static str = "NodeAudio";
    const URL: &'static str = "https://localhost";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = "0.1.0";
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: None,
        main_output_channels: Some(NonZeroU32::new(2).unwrap()),
        ..AudioIOLayout::const_default()
    }];

    type SysExMessage = (); // TODO: map MIDI to future messages
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> { self.params.clone() }

    fn initialize(&mut self, _audio_io_layout: &AudioIOLayout, buffer_config: &BufferConfig, _context: &mut impl InitContext<Self>) -> bool {
        let mut client = self.client.lock();
        client.ensure_connected(buffer_config.sample_rate, buffer_config.max_buffer_size as usize, 2).ok();
        true
    }

    fn process(&mut self, buffer: &mut Buffer, _aux: &mut AuxiliaryBuffers, context: &mut impl ProcessContext<Self>) -> ProcessStatus {
        let frames = buffer.samples();
        let chans = buffer.as_slice();
        let out_ch = chans.len().min(2);

        let mut output_interleaved = vec![0.0f32; frames * out_ch.max(1)];
        let result = {
            let mut client = self.client.lock();
            client.ensure_connected(context.transport().sample_rate, frames, out_ch).ok();
            client.process_block(&mut output_interleaved)
        };
        if result.is_err() {
            // stay silent if not connected
        }

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
                    let (connected, attempts, last_error) = { 
                        let client_lock = client.lock();
                        client_lock.get_status_info()
                    };
                    
                    // Status display
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            let status_text = if connected { "Connected" } else { "Disconnected" };
                            let color = if connected { 
                                egui::Color32::from_rgb(0, 200, 0) 
                            } else { 
                                egui::Color32::from_rgb(200, 0, 0) 
                            };
                            ui.colored_label(color, status_text);
                            
                            // Always show reconnect button
                            if ui.button("Reconnect").clicked() {
                                client.lock().force_reconnect();
                            }
                        });
                        
                        // Show connection attempts if there have been any
                        if attempts > 0 {
                            ui.label(format!("Connection attempts: {}", attempts));
                        }
                        
                        // Show last error if any
                        if let Some(error) = last_error {
                            ui.colored_label(egui::Color32::from_rgb(255, 200, 0), 
                                           format!("Last error: {}", error));
                        }
                        
                        ui.label(format!("Server: {}:{}", DEFAULT_HOST, DEFAULT_PORT));
                    });
                });
            },
        );
        editor
    }
}

impl ClapPlugin for NodeAudioInstrument {
    const CLAP_ID: &'static str = "com.nodeaudio.bridgeinst";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Instrument bridged from node_audio over TCP");
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::Instrument, ClapFeature::Stereo, ClapFeature::Utility];
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
}

impl Vst3Plugin for NodeAudioInstrument {
    const VST3_CLASS_ID: [u8; 16] = *b"NodeBrdgInstRust";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Instrument, Vst3SubCategory::Tools];
}

nih_export_clap!(NodeAudioInstrument);
nih_export_vst3!(NodeAudioInstrument);


