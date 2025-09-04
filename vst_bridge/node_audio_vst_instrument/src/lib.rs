use anyhow::Context;
use nih_plug::prelude::*;
use parking_lot::Mutex;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::num::NonZeroU32;

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
}

impl BridgeClient {
    fn new() -> Self {
        Self { stream: None, seq: 0, last_config: (0, 0, 0, 0) }
    }

    fn ensure_connected(&mut self, sample_rate: f32, block_size: usize, out_ch: usize) -> anyhow::Result<()> {
        // Instrument: in_ch=0, out_ch=desired
        let cfg = (sample_rate as u32, block_size as u32, 0u16, out_ch as u16);
        let reconnect = match self.stream { None => true, Some(_) => self.last_config != cfg };
        if reconnect {
            let mut stream = TcpStream::connect((DEFAULT_HOST, DEFAULT_PORT))
                .with_context(|| "connect to node_audio bridge server")?;
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
        }
        Ok(())
    }

    fn process_block(&mut self, output: &mut [f32]) -> anyhow::Result<()> {
        let (_sr, bs, _in_ch, out_ch) = self.last_config;
        let mut stream = match self.stream.as_ref() { Some(s) => s.try_clone()?, None => anyhow::bail!("not connected") };
        // send request with opcode=1 and zero input payload
        let mut header = [0u8; 1 + 4];
        header[0] = 1;
        header[1..5].copy_from_slice(&self.seq.to_le_bytes());
        write_all(&mut stream, &header)?;

        // recv response
        let mut resp_header = [0u8; 1 + 4];
        read_exact(&mut stream, &mut resp_header)?;
        if resp_header[0] != 2 { anyhow::bail!("invalid opcode from server"); }
        let _rsp_seq = u32::from_le_bytes(resp_header[1..5].try_into().unwrap());
        let expected = (bs as usize) * (out_ch as usize);
        let mut bytes = vec![0u8; expected * 4];
        read_exact(&mut stream, &mut bytes)?;
        let samples: &[f32] = bytemuck::cast_slice(&bytes);
        output.copy_from_slice(samples);
        self.seq = self.seq.wrapping_add(1);
        Ok(())
    }
}

struct NodeAudioInstrument {
    client: Arc<Mutex<BridgeClient>>,
}

impl Default for NodeAudioInstrument {
    fn default() -> Self {
        Self { client: Arc::new(Mutex::new(BridgeClient::new())) }
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

    fn params(&self) -> Arc<dyn Params> { Arc::new(EmptyParams {}) }

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
        {
            let mut client = self.client.lock();
            client.ensure_connected(context.transport().sample_rate, frames, out_ch).ok();
            client.process_block(&mut output_interleaved).ok();
        }

        for f in 0..frames {
            for c in 0..out_ch {
                chans[c][f] = output_interleaved[f * out_ch + c];
            }
        }

        ProcessStatus::Normal
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


