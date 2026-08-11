// Blade port of the inkwell-webgpu-water engine.
// Open-ocean, optimized mode, surface view — a faithful port of
// WebGpuWaterEngine from inkwell-webgpu-water/src/lib/webgpu-water-engine.ts.

use blade_graphics as gpu;
use blade_macros::ShaderData;
use bytemuck::{Pod, Zeroable};
use std::{mem, ptr, time::Instant};

const TERRAIN_EXTENT: f32 = 390.0;
const TERRAIN_FIELD_RESOLUTION: u32 = 512;
const TERRAIN_FIELD_TEXELS: u32 = TERRAIN_FIELD_RESOLUTION + 1;
const WORLD_UNIFORM_BYTES: u64 = 256;
const SIMULATION_PARAM_BYTES: u64 = 32;
const SPECTRAL_RESOLUTION: u32 = 128;
const SPECTRAL_LOG_SIZE: u32 = 7;
const SPECTRAL_FFT_PASSES: usize = (SPECTRAL_LOG_SIZE * 2) as usize;
const BREAKER_EVENT_RESOLUTION: u32 = 256;
const BREAKER_PATCH_ALONG_RESOLUTION: u32 = 256;
const BREAKER_PATCH_ACROSS_RESOLUTION: u32 = 48;
const WATER_CLIPMAP_RESOLUTION: u32 = 64;
const WATER_CLIPMAP_LEVELS: u32 = 4;
const TETHYS_WATER_LEVEL: f32 = 1.4;
const TETHYS_WATER_FIELD_SIZE: f32 = 192.0;
const MESH_RESOLUTION: u32 = 240;
const SIMULATION_RESOLUTION: u32 = 256;

const CASCADES: [CascadeConfig; 3] = [
    CascadeConfig {
        length_scale: 240.0,
        cutoff_low: 0.024,
        cutoff_high: 0.36,
        amplitude_scale: 0.45,
        secondary_scale: 0.22,
        seed: 0x51f15e,
    },
    CascadeConfig {
        length_scale: 64.0,
        cutoff_low: 0.30,
        cutoff_high: 1.42,
        amplitude_scale: 0.45,
        secondary_scale: 0.08,
        seed: 0x72a93b,
    },
    CascadeConfig {
        length_scale: 12.0,
        cutoff_low: 1.22,
        cutoff_high: 24.0,
        amplitude_scale: 0.82,
        secondary_scale: 0.0,
        seed: 0x19ce47,
    },
];

#[derive(Clone, Copy)]
struct CascadeConfig {
    length_scale: f64,
    cutoff_low: f64,
    cutoff_high: f64,
    amplitude_scale: f64,
    secondary_scale: f64,
    seed: u32,
}

// ---------------------------------------------------------------------------
// Host-side mirror of the WGSL WorldUniforms (256-byte uniform buffer).
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WorldUniforms {
    view_proj: [[f32; 4]; 4],
    camera_time: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    camera_forward: [f32; 4],
    sun_water: [f32; 4],
    terrain: [f32; 4],
    simulation: [f32; 4],
    player: [f32; 4],
    interaction: [f32; 4],
    environment: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimulationParams {
    impulse: [f32; 4],
    step_foam_shift: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FftParams {
    axis: u32,
    stage: u32,
    size: u32,
    finalize: u32,
}

// ---------------------------------------------------------------------------
// ShaderData layouts. Bindings are assigned from field order; the WGSL has no
// @group/@binding annotations. Groups are split at blade's 8-resource limit.
// ---------------------------------------------------------------------------
#[derive(ShaderData)]
struct TerrainData {
    uniforms: gpu::BufferPiece,
    field_out: gpu::TextureView,
}

#[derive(ShaderData)]
struct SimulationData0 {
    uniforms: gpu::BufferPiece,
    sim_params: gpu::BufferPiece,
    previous_state: gpu::TextureView,
    next_state: gpu::TextureView,
    terrain_field: gpu::TextureView,
    long_field0: gpu::TextureView,
    long_field1: gpu::TextureView,
    medium_field0: gpu::TextureView,
}
#[derive(ShaderData)]
struct SimulationData1 {
    medium_field1: gpu::TextureView,
    spectrum_sampler: gpu::Sampler,
}

#[derive(ShaderData)]
struct BreakerEventData0 {
    uniforms: gpu::BufferPiece,
    previous_events: gpu::TextureView,
    next_events: gpu::TextureView,
    terrain_field: gpu::TextureView,
    water_state: gpu::TextureView,
    field_sampler: gpu::Sampler,
    long_field0: gpu::TextureView,
    long_field1: gpu::TextureView,
}
#[derive(ShaderData)]
struct BreakerEventData1 {
    medium_field0: gpu::TextureView,
    medium_field1: gpu::TextureView,
    spectrum_sampler: gpu::Sampler,
}

#[derive(ShaderData)]
struct SpectrumEvolutionData {
    uniforms: gpu::BufferPiece,
    initial_spectrum: gpu::TextureView,
    wave_data: gpu::TextureView,
    field0: gpu::TextureView,
    field1: gpu::TextureView,
}

#[derive(ShaderData)]
struct SpectralFftData {
    fft_params: gpu::BufferPiece,
    twiddle_table: gpu::TextureView,
    input0: gpu::TextureView,
    input1: gpu::TextureView,
    output0: gpu::TextureView,
    output1: gpu::TextureView,
}

#[derive(ShaderData)]
struct SkyData {
    uniforms: gpu::BufferPiece,
}

#[derive(ShaderData)]
struct TerrainRenderData0 {
    uniforms: gpu::BufferPiece,
    terrain_field: gpu::TextureView,
    field_sampler: gpu::Sampler,
    medium_field0: gpu::TextureView,
    medium_field1: gpu::TextureView,
    short_field0: gpu::TextureView,
    short_field1: gpu::TextureView,
    spectrum_sampler: gpu::Sampler,
}
#[derive(ShaderData)]
struct TerrainRenderData1 {
    water_state: gpu::TextureView,
}

#[derive(ShaderData)]
struct WaterData0 {
    uniforms: gpu::BufferPiece,
    terrain_field: gpu::TextureView,
    water_state: gpu::TextureView,
    field_sampler: gpu::Sampler,
    long_field0: gpu::TextureView,
    long_field1: gpu::TextureView,
    medium_field0: gpu::TextureView,
    medium_field1: gpu::TextureView,
}
#[derive(ShaderData)]
struct WaterData1 {
    short_field0: gpu::TextureView,
    short_field1: gpu::TextureView,
    spectrum_sampler: gpu::Sampler,
    breaker_events: gpu::TextureView,
}

// ---------------------------------------------------------------------------
// Deterministic spectrum generation (ported from buildSpectralOceanData).
// ---------------------------------------------------------------------------
struct DeterministicRandom {
    state: u32,
}

impl DeterministicRandom {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }
    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6d2b79f5);
        let mut value = self.state;
        value = (value ^ (value >> 15)).wrapping_mul(value | 1);
        value ^= value.wrapping_add((value ^ (value >> 7)).wrapping_mul(61));
        ((value ^ (value >> 14)) >> 0) as f64 / 4294967296.0
    }
}

fn wrap_angle(mut angle: f64) -> f64 {
    while angle > std::f64::consts::PI {
        angle -= std::f64::consts::PI * 2.0;
    }
    while angle < -std::f64::consts::PI {
        angle += std::f64::consts::PI * 2.0;
    }
    angle
}

fn spectrum_normalisation_factor(spread: f64) -> f64 {
    let s2 = spread * spread;
    let s3 = s2 * spread;
    let s4 = s3 * spread;
    if spread < 5.0 {
        -0.000564 * s4 + 0.00776 * s3 - 0.044 * s2 + 0.192 * spread + 0.163
    } else {
        -4.8e-8 * s4 + 1.07e-5 * s3 - 9.53e-4 * s2 + 5.9e-2 * spread + 0.393
    }
}

struct SpectralData {
    initial: Vec<f32>,
    wave: Vec<f32>,
    twiddle: Vec<f32>,
}

fn build_spectral_ocean_data(size: usize, config: &CascadeConfig) -> SpectralData {
    let gravity: f64 = 9.81;
    let depth: f64 = 54.0;
    let wind_speed: f64 = 11.5;
    let fetch: f64 = 120_000.0;
    let wind_angle: f64 = -0.48;
    let peak_enhancement: f64 = 3.3;
    let swell: f64 = 0.38;
    let delta_k = std::f64::consts::PI * 2.0 / config.length_scale;
    let alpha = 0.076 * (gravity * fetch / (wind_speed * wind_speed)).powf(-0.22);
    let peak_omega = 22.0 * (wind_speed * fetch / (gravity * gravity)).powf(-0.33);

    let mut initial_k = vec![0.0f32; size * size * 2];
    let mut wave = vec![0.0f32; size * size * 4];
    let mut random = DeterministicRandom::new(config.seed);
    let gaussian = |random: &mut DeterministicRandom| -> (f64, f64) {
        let u = random.next().max(1e-7);
        let v = random.next();
        let radius = (-2.0 * u.ln()).sqrt();
        let angle = std::f64::consts::PI * 2.0 * v;
        (radius * angle.cos(), radius * angle.sin())
    };

    let half = size as f64 / 2.0;
    for y in 0..size {
        for x in 0..size {
            let nx = x as f64 - half;
            let nz = y as f64 - half;
            let kx = nx * delta_k;
            let kz = nz * delta_k;
            let k_length = (kx * kx + kz * kz).sqrt();
            let pixel = y * size + x;
            let wave_offset = pixel * 4;
            if k_length < config.cutoff_low || k_length > config.cutoff_high {
                wave[wave_offset..wave_offset + 4].copy_from_slice(&[0.0, 1.0, 0.0, 0.0]);
                continue;
            }
            let kh = (k_length * depth).min(20.0);
            let tanh_kh = kh.tanh();
            let omega = (gravity * k_length * tanh_kh).sqrt();
            let sech_squared = 1.0 - tanh_kh * tanh_kh;
            let frequency_derivative = gravity
                * (depth * k_length * sech_squared + tanh_kh)
                / (omega * 2.0).max(1e-5);
            let omega_h = omega * (depth / gravity).sqrt();
            let tma = if omega_h <= 1.0 {
                0.5 * omega_h * omega_h
            } else if omega_h < 2.0 {
                1.0 - 0.5 * (2.0 - omega_h) * (2.0 - omega_h)
            } else {
                1.0
            };
            let sigma = if omega <= peak_omega { 0.07 } else { 0.09 };
            let peak_distance = (omega - peak_omega) / (sigma * peak_omega).max(1e-5);
            let peak_shape = (-0.5 * peak_distance * peak_distance).exp();
            let peak_ratio = peak_omega / omega;
            let jonswap = tma * alpha * gravity * gravity / omega.powf(5.0)
                * (-1.25 * peak_ratio.powf(4.0)).exp()
                * peak_enhancement.powf(peak_shape);
            let theta = wrap_angle((kz).atan2(kx) - wind_angle);
            let omega_ratio = omega / peak_omega;
            let spread_power = ((if omega > peak_omega {
                9.77 * omega_ratio.powf(-2.5)
            } else {
                6.97 * omega_ratio.powf(5.0)
            }) + 16.0 * omega_ratio.min(20.0).tanh() * swell * swell)
                * 0.58;
            let focused_direction = spectrum_normalisation_factor(spread_power)
                * (theta * 0.5).cos().abs().powf(2.0 * spread_power);
            let broad_direction = 2.0 / std::f64::consts::PI * theta.cos().max(0.0).powi(2);
            let direction = focused_direction * 0.68 + broad_direction * 0.32;
            let short_wave_fade = (-0.00016 * k_length * k_length).exp();
            let mut spectral_density = jonswap * direction * short_wave_fade;
            if config.secondary_scale > 0.0 {
                let swell_wind_speed = 8.4;
                let swell_fetch = 310_000.0;
                let swell_peak_omega =
                    22.0 * (swell_wind_speed * swell_fetch / (gravity * gravity)).powf(-0.33);
                let swell_alpha = 0.076
                    * (gravity * swell_fetch / (swell_wind_speed * swell_wind_speed)).powf(-0.22);
                let swell_sigma = if omega <= swell_peak_omega { 0.07 } else { 0.09 };
                let swell_peak_distance =
                    (omega - swell_peak_omega) / (swell_sigma * swell_peak_omega).max(1e-5);
                let swell_peak_shape = (-0.5 * swell_peak_distance * swell_peak_distance).exp();
                let swell_peak_ratio = swell_peak_omega / omega;
                let swell_spectrum = tma * swell_alpha * gravity * gravity / omega.powf(5.0)
                    * (-1.25 * swell_peak_ratio.powf(4.0)).exp()
                    * 2.6f64.powf(swell_peak_shape);
                let swell_theta = wrap_angle((kz).atan2(kx) - (wind_angle + 0.82));
                let swell_ratio = omega / swell_peak_omega;
                let swell_spread = ((if omega > swell_peak_omega {
                    9.77 * swell_ratio.powf(-2.5)
                } else {
                    6.97 * swell_ratio.powf(5.0)
                }) + 9.0)
                    * 0.72;
                let swell_direction = spectrum_normalisation_factor(swell_spread)
                    * (swell_theta * 0.5).cos().abs().powf(2.0 * swell_spread);
                spectral_density += swell_spectrum * swell_direction * short_wave_fade
                    * config.secondary_scale;
            }
            let amplitude = (2.0 * spectral_density.max(0.0) * frequency_derivative.abs()
                / k_length
                * delta_k
                * delta_k)
                .sqrt()
                * config.amplitude_scale;
            let (noise_x, noise_y) = gaussian(&mut random);
            initial_k[pixel * 2] = (noise_x * amplitude) as f32;
            initial_k[pixel * 2 + 1] = (noise_y * amplitude) as f32;
            wave[wave_offset..wave_offset + 4]
                .copy_from_slice(&[kx as f32, (1.0 / k_length) as f32, kz as f32, omega as f32]);
        }
    }

    let mut initial = vec![0.0f32; size * size * 4];
    for y in 0..size {
        for x in 0..size {
            let pixel = y * size + x;
            let mirror = ((size - y) % size) * size + ((size - x) % size);
            initial[pixel * 4] = initial_k[pixel * 2];
            initial[pixel * 4 + 1] = initial_k[pixel * 2 + 1];
            initial[pixel * 4 + 2] = initial_k[mirror * 2];
            initial[pixel * 4 + 3] = -initial_k[mirror * 2 + 1];
        }
    }

    let log_size = SPECTRAL_LOG_SIZE as usize;
    let mut twiddle = vec![0.0f32; log_size * size * 4];
    for stage in 0..log_size {
        let block = size >> (stage + 1);
        for output in 0..size / 2 {
            let first = (2 * block * (output / block) + output % block) % size;
            let angle = -2.0 * std::f64::consts::PI / size as f64 * (output / block) as f64
                * block as f64;
            let cosine = angle.cos();
            let sine = angle.sin();
            let base = (stage * size + output) * 4;
            let opposite = (stage * size + output + size / 2) * 4;
            twiddle[base..base + 4]
                .copy_from_slice(&[cosine as f32, sine as f32, first as f32, (first + block) as f32]);
            twiddle[opposite..opposite + 4].copy_from_slice(&[
                -cosine as f32,
                -sine as f32,
                first as f32,
                (first + block) as f32,
            ]);
        }
    }

    SpectralData {
        initial,
        wave,
        twiddle,
    }
}

// ---------------------------------------------------------------------------
// Camera math (ported verbatim from the TS engine, same row-major layout).
// ---------------------------------------------------------------------------
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    [v[0] / length, v[1] / length, v[2] / length]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn look_at(eye: [f32; 3], target: [f32; 3]) -> [f32; 16] {
    let z = normalize3([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
    let x = normalize3(cross3([0.0, 1.0, 0.0], z));
    let y = cross3(z, x);
    [
        x[0], y[0], z[0], 0.0, x[1], y[1], z[1], 0.0, x[2], y[2], z[2], 0.0,
        -dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1.0,
    ]
}

fn perspective(fov_radians: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_radians / 2.0).tan();
    [
        f / aspect, 0.0, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 0.0, far / (near - far), -1.0,
        0.0, 0.0, (near * far) / (near - far), 0.0,
    ]
}

fn multiply(left: &[f32; 16], right: &[f32; 16]) -> [f32; 16] {
    let mut output = [0.0f32; 16];
    for column in 0..4 {
        for row in 0..4 {
            output[column * 4 + row] = left[row] * right[column * 4]
                + left[4 + row] * right[column * 4 + 1]
                + left[8 + row] * right[column * 4 + 2]
                + left[12 + row] * right[column * 4 + 3];
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
struct Cascade {
    initial: gpu::Texture,
    initial_view: gpu::TextureView,
    wave_data: gpu::Texture,
    wave_data_view: gpu::TextureView,
    fields: [[gpu::Texture; 2]; 2],
    field_views: [[gpu::TextureView; 2]; 2],
}

pub struct WaterEngine {
    world_uniforms: gpu::Buffer,
    sim_params: gpu::Buffer,
    fft_param_buffers: Vec<gpu::Buffer>,
    field_sampler: gpu::Sampler,
    spectrum_sampler: gpu::Sampler,
    terrain_texture: gpu::Texture,
    terrain_view: gpu::TextureView,
    water_textures: [gpu::Texture; 2],
    water_views: [gpu::TextureView; 2],
    breaker_event_textures: [gpu::Texture; 2],
    breaker_event_views: [gpu::TextureView; 2],
    cascades: Vec<Cascade>,
    twiddle_texture: gpu::Texture,
    twiddle_view: gpu::TextureView,
    depth_texture: gpu::Texture,
    depth_view: gpu::TextureView,
    terrain_compute: gpu::ComputePipeline,
    simulation_pipeline: gpu::ComputePipeline,
    breaker_event_pipeline: gpu::ComputePipeline,
    spectrum_evolution: gpu::ComputePipeline,
    spectral_fft: gpu::ComputePipeline,
    sky_pipeline: gpu::RenderPipeline,
    terrain_pipeline: gpu::RenderPipeline,
    water_pipeline: gpu::RenderPipeline,
    breaker_patch_pipeline: gpu::RenderPipeline,
    terrain_prepared: bool,
    active_simulation: usize,
    active_breaker_events: usize,
    yaw: f32,
    pitch: f32,
    radius: f32,
    start: Instant,
    last_wake_at: f32,
    extent: gpu::Extent,
    pub gpu_frame_ms: f32,
}

impl WaterEngine {
    fn create_texture(
        context: &gpu::Context,
        name: &str,
        format: gpu::TextureFormat,
        size: [u32; 2],
        usage: gpu::TextureUsage,
    ) -> gpu::Texture {
        context.create_texture(gpu::TextureDesc {
            name,
            format,
            size: gpu::Extent {
                width: size[0],
                height: size[1],
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage,
            external: None,
        })
    }

    fn create_view(
        context: &gpu::Context,
        name: &str,
        texture: gpu::Texture,
        format: gpu::TextureFormat,
    ) -> gpu::TextureView {
        context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name,
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        )
    }

    pub fn new(
        context: &gpu::Context,
        extent: gpu::Extent,
        surface_format: gpu::TextureFormat,
    ) -> Self {
        let shader_source = include_str!("water.wgsl");
        let shader = context.create_shader(gpu::ShaderDesc {
            source: shader_source,
            naga_module: None,
        });

        let terrain_layout = <TerrainData as gpu::ShaderData>::layout();
        let simulation0_layout = <SimulationData0 as gpu::ShaderData>::layout();
        let simulation1_layout = <SimulationData1 as gpu::ShaderData>::layout();
        let breaker0_layout = <BreakerEventData0 as gpu::ShaderData>::layout();
        let breaker1_layout = <BreakerEventData1 as gpu::ShaderData>::layout();
        let spectrum_layout = <SpectrumEvolutionData as gpu::ShaderData>::layout();
        let fft_layout = <SpectralFftData as gpu::ShaderData>::layout();
        let sky_layout = <SkyData as gpu::ShaderData>::layout();
        let terrain_render0_layout = <TerrainRenderData0 as gpu::ShaderData>::layout();
        let terrain_render1_layout = <TerrainRenderData1 as gpu::ShaderData>::layout();
        let water0_layout = <WaterData0 as gpu::ShaderData>::layout();
        let water1_layout = <WaterData1 as gpu::ShaderData>::layout();

        let terrain_compute = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "terrain field",
            data_layouts: &[&terrain_layout],
            compute: shader.at("buildTerrain"),
        });
        let simulation_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "water simulation",
            data_layouts: &[&simulation0_layout, &simulation1_layout],
            compute: shader.at("simulate"),
        });
        let breaker_event_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "breaker events",
            data_layouts: &[&breaker0_layout, &breaker1_layout],
            compute: shader.at("updateBreakerEvents"),
        });
        let spectrum_evolution = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "spectrum evolution",
            data_layouts: &[&spectrum_layout],
            compute: shader.at("evolveSpectrum"),
        });
        let spectral_fft = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "spectral inverse FFT",
            data_layouts: &[&fft_layout],
            compute: shader.at("inverseFftStage"),
        });

        let depth_stencil_off = |depth_write_enabled: bool, depth_compare: gpu::CompareFunction| {
            Some(gpu::DepthStencilState {
                format: gpu::TextureFormat::Depth32Float,
                depth_write_enabled,
                depth_compare,
                stencil: gpu::StencilState::default(),
                bias: gpu::DepthBiasState::default(),
            })
        };

        let sky_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "sky",
            data_layouts: &[&sky_layout],
            vertex: shader.at("skyVertex"),
            vertex_fetches: &[],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            fragment: Some(shader.at("skyFragment")),
            color_targets: &[surface_format.into()],
            multisample_state: gpu::MultisampleState::default(),
        });
        let terrain_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "terrain",
            data_layouts: &[&terrain_render0_layout, &terrain_render1_layout],
            vertex: shader.at("terrainVertex"),
            vertex_fetches: &[],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            fragment: Some(shader.at("terrainFragment")),
            color_targets: &[surface_format.into()],
            multisample_state: gpu::MultisampleState::default(),
        });
        let water_target = gpu::ColorTargetState {
            format: surface_format,
            blend: Some(gpu::BlendState::ALPHA_BLENDING),
            write_mask: gpu::ColorWrites::default(),
        };
        fn water_pipeline_desc<'a>(
            data_layouts: &'a [&'a gpu::ShaderDataLayout],
            vertex: gpu::ShaderFunction<'a>,
            fragment: gpu::ShaderFunction<'a>,
            color_targets: &'a [gpu::ColorTargetState],
        ) -> gpu::RenderPipelineDesc<'a> {
            gpu::RenderPipelineDesc {
                name: "water",
                data_layouts,
                vertex,
                vertex_fetches: &[],
                primitive: gpu::PrimitiveState {
                    topology: gpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(gpu::DepthStencilState {
                    format: gpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: gpu::CompareFunction::Less,
                    stencil: gpu::StencilState::default(),
                    bias: gpu::DepthBiasState::default(),
                }),
                fragment: Some(fragment),
                color_targets,
                multisample_state: gpu::MultisampleState::default(),
            }
        }
        let water_layouts = [&water0_layout, &water1_layout];
        let water_targets = [water_target.clone()];
        let water_pipeline = context.create_render_pipeline(water_pipeline_desc(
            &water_layouts,
            shader.at("waterVertex"),
            shader.at("waterFragment"),
            &water_targets,
        ));
        let breaker_patch_pipeline = context.create_render_pipeline(water_pipeline_desc(
            &water_layouts,
            shader.at("breakerPatchVertex"),
            shader.at("waterFragment"),
            &[water_target.clone()],
        ));

        let field_sampler = context.create_sampler(gpu::SamplerDesc {
            name: "field",
            address_modes: [gpu::AddressMode::ClampToEdge; 3],
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            ..Default::default()
        });
        let spectrum_sampler = context.create_sampler(gpu::SamplerDesc {
            name: "spectrum",
            address_modes: [gpu::AddressMode::Repeat; 3],
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            ..Default::default()
        });

        // Buffers
        let world_uniforms = context.create_buffer(gpu::BufferDesc {
            name: "world uniforms",
            size: WORLD_UNIFORM_BYTES,
            memory: gpu::Memory::Shared,
        });
        let sim_params = context.create_buffer(gpu::BufferDesc {
            name: "simulation params",
            size: SIMULATION_PARAM_BYTES,
            memory: gpu::Memory::Shared,
        });
        let fft_param_buffers = (0..SPECTRAL_FFT_PASSES)
            .map(|pass| {
                let buffer = context.create_buffer(gpu::BufferDesc {
                    name: "inverse FFT pass params",
                    size: mem::size_of::<FftParams>() as u64,
                    memory: gpu::Memory::Shared,
                });
                let params = FftParams {
                    axis: if pass < SPECTRAL_LOG_SIZE as usize { 0 } else { 1 },
                    stage: (pass % SPECTRAL_LOG_SIZE as usize) as u32,
                    size: SPECTRAL_RESOLUTION,
                    finalize: if pass == SPECTRAL_FFT_PASSES - 1 { 1 } else { 0 },
                };
                unsafe {
                    ptr::copy_nonoverlapping(
                        &params as *const FftParams as *const u8,
                        buffer.data(),
                        mem::size_of::<FftParams>(),
                    );
                }
                context.sync_buffer(buffer, 0, mem::size_of::<FftParams>() as u64);
                buffer
            })
            .collect::<Vec<_>>();

        // Textures
        let terrain_texture = Self::create_texture(
            context,
            "terrain field",
            gpu::TextureFormat::Rgba16Float,
            [TERRAIN_FIELD_TEXELS, TERRAIN_FIELD_TEXELS],
            gpu::TextureUsage::STORAGE | gpu::TextureUsage::RESOURCE,
        );
        let terrain_view = Self::create_view(
            context,
            "terrain field view",
            terrain_texture,
            gpu::TextureFormat::Rgba16Float,
        );
        let water_textures = [
            Self::create_texture(
                context,
                "water state",
                gpu::TextureFormat::Rgba16Float,
                [SIMULATION_RESOLUTION, SIMULATION_RESOLUTION],
                gpu::TextureUsage::STORAGE | gpu::TextureUsage::RESOURCE,
            ),
            Self::create_texture(
                context,
                "water state",
                gpu::TextureFormat::Rgba16Float,
                [SIMULATION_RESOLUTION, SIMULATION_RESOLUTION],
                gpu::TextureUsage::STORAGE | gpu::TextureUsage::RESOURCE,
            ),
        ];
        let water_views = water_textures.map(|texture| {
            Self::create_view(context, "water state view", texture, gpu::TextureFormat::Rgba16Float)
        });
        let breaker_event_textures = [
            Self::create_texture(
                context,
                "breaker event history",
                gpu::TextureFormat::Rgba16Float,
                [BREAKER_EVENT_RESOLUTION, 1],
                gpu::TextureUsage::STORAGE | gpu::TextureUsage::RESOURCE,
            ),
            Self::create_texture(
                context,
                "breaker event history",
                gpu::TextureFormat::Rgba16Float,
                [BREAKER_EVENT_RESOLUTION, 1],
                gpu::TextureUsage::STORAGE | gpu::TextureUsage::RESOURCE,
            ),
        ];
        let breaker_event_views = breaker_event_textures.map(|texture| {
            Self::create_view(
                context,
                "breaker event history view",
                texture,
                gpu::TextureFormat::Rgba16Float,
            )
        });

        let twiddle_texture = Self::create_texture(
            context,
            "Stockham FFT twiddle table",
            gpu::TextureFormat::Rgba32Float,
            [SPECTRAL_RESOLUTION, SPECTRAL_LOG_SIZE],
            gpu::TextureUsage::RESOURCE | gpu::TextureUsage::COPY,
        );
        let twiddle_view = Self::create_view(
            context,
            "Stockham FFT twiddle table view",
            twiddle_texture,
            gpu::TextureFormat::Rgba32Float,
        );

        // Spectral data + upload
        let mut cascades = Vec::with_capacity(CASCADES.len());
        let spectrum_bytes = (SPECTRAL_RESOLUTION * SPECTRAL_RESOLUTION * 16) as usize;
        let twiddle_bytes = (SPECTRAL_RESOLUTION * SPECTRAL_LOG_SIZE * 16) as usize;
        let upload_size = spectrum_bytes * 6 + twiddle_bytes;
        let upload_buffer = context.create_buffer(gpu::BufferDesc {
            name: "spectral upload",
            size: upload_size as u64,
            memory: gpu::Memory::Upload,
        });
        let mut offset = 0usize;
        let mut twiddle_written = false;
        for config in &CASCADES {
            let data = build_spectral_ocean_data(SPECTRAL_RESOLUTION as usize, config);
            unsafe {
                ptr::copy_nonoverlapping(
                    bytemuck::cast_slice(&data.initial).as_ptr(),
                    upload_buffer.data().offset(offset as isize),
                    spectrum_bytes,
                );
            }
            offset += spectrum_bytes;
            unsafe {
                ptr::copy_nonoverlapping(
                    bytemuck::cast_slice(&data.wave).as_ptr(),
                    upload_buffer.data().offset(offset as isize),
                    spectrum_bytes,
                );
            }
            offset += spectrum_bytes;
            if !twiddle_written {
                unsafe {
                    ptr::copy_nonoverlapping(
                        bytemuck::cast_slice(&data.twiddle).as_ptr(),
                        upload_buffer.data().offset(offset as isize),
                        twiddle_bytes,
                    );
                }
                twiddle_written = true;
            }
        }
        context.sync_buffer(upload_buffer, 0, upload_size as u64);

        for _ in 0..CASCADES.len() {
            let initial = Self::create_texture(
                context,
                "initial spectrum",
                gpu::TextureFormat::Rgba32Float,
                [SPECTRAL_RESOLUTION, SPECTRAL_RESOLUTION],
                gpu::TextureUsage::RESOURCE | gpu::TextureUsage::COPY,
            );
            let wave_data = Self::create_texture(
                context,
                "wave vectors",
                gpu::TextureFormat::Rgba32Float,
                [SPECTRAL_RESOLUTION, SPECTRAL_RESOLUTION],
                gpu::TextureUsage::RESOURCE | gpu::TextureUsage::COPY,
            );
            let fields = [0, 1].map(|_| {
                [0, 1].map(|_| {
                    Self::create_texture(
                        context,
                        "spectral field",
                        gpu::TextureFormat::Rgba16Float,
                        [SPECTRAL_RESOLUTION, SPECTRAL_RESOLUTION],
                        gpu::TextureUsage::STORAGE | gpu::TextureUsage::RESOURCE,
                    )
                })
            });
            cascades.push(Cascade {
                initial,
                initial_view: Self::create_view(
                    context,
                    "initial spectrum view",
                    initial,
                    gpu::TextureFormat::Rgba32Float,
                ),
                wave_data,
                wave_data_view: Self::create_view(
                    context,
                    "wave vectors view",
                    wave_data,
                    gpu::TextureFormat::Rgba32Float,
                ),
                field_views: fields.map(|ping| {
                    ping.map(|texture| {
                        Self::create_view(
                            context,
                            "spectral field view",
                            texture,
                            gpu::TextureFormat::Rgba16Float,
                        )
                    })
                }),
                fields,
            });
        }

        // Depth target
        let depth_texture = Self::create_texture(
            context,
            "depth",
            gpu::TextureFormat::Depth32Float,
            [extent.width, extent.height],
            gpu::TextureUsage::TARGET,
        );
        let depth_view = Self::create_view(
            context,
            "depth view",
            depth_texture,
            gpu::TextureFormat::Depth32Float,
        );

        // One-shot init encoder: initialize textures and upload spectral data.
        let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "water init",
            buffer_count: 1,
            manual_barriers: false,
        });
        encoder.start();
        encoder.init_texture(depth_texture);
        encoder.init_texture(twiddle_texture);
        for cascade in &cascades {
            encoder.init_texture(cascade.initial);
            encoder.init_texture(cascade.wave_data);
        }
        let mut transfer_offset = 0usize;
        if let mut transfer = encoder.transfer("spectral uploads") {
            for cascade in &cascades {
                transfer.copy_buffer_to_texture(
                    upload_buffer.at(transfer_offset as u64),
                    SPECTRAL_RESOLUTION * 16,
                    cascade.initial.into(),
                    gpu::Extent {
                        width: SPECTRAL_RESOLUTION,
                        height: SPECTRAL_RESOLUTION,
                        depth: 1,
                    },
                );
                transfer_offset += spectrum_bytes;
                transfer.copy_buffer_to_texture(
                    upload_buffer.at(transfer_offset as u64),
                    SPECTRAL_RESOLUTION * 16,
                    cascade.wave_data.into(),
                    gpu::Extent {
                        width: SPECTRAL_RESOLUTION,
                        height: SPECTRAL_RESOLUTION,
                        depth: 1,
                    },
                );
                transfer_offset += spectrum_bytes;
            }
            transfer.copy_buffer_to_texture(
                upload_buffer.at(transfer_offset as u64),
                SPECTRAL_RESOLUTION * 16,
                twiddle_texture.into(),
                gpu::Extent {
                    width: SPECTRAL_RESOLUTION,
                    height: SPECTRAL_LOG_SIZE,
                    depth: 1,
                },
            );
        }
        let sync_point = context.submit(&mut encoder);
        let _ = context.wait_for(&sync_point, !0);
        context.destroy_command_encoder(&mut encoder);
        context.destroy_buffer(upload_buffer);

        Self {
            world_uniforms,
            sim_params,
            fft_param_buffers,
            field_sampler,
            spectrum_sampler,
            terrain_texture,
            terrain_view,
            water_textures,
            water_views,
            breaker_event_textures,
            breaker_event_views,
            cascades,
            twiddle_texture,
            twiddle_view,
            depth_texture,
            depth_view,
            terrain_compute,
            simulation_pipeline,
            breaker_event_pipeline,
            spectrum_evolution,
            spectral_fft,
            sky_pipeline,
            terrain_pipeline,
            water_pipeline,
            breaker_patch_pipeline,
            terrain_prepared: false,
            active_simulation: 0,
            active_breaker_events: 0,
            yaw: 0.56,
            pitch: 0.07,
            radius: 58.0,
            start: Instant::now(),
            last_wake_at: -10.0,
            extent,
            gpu_frame_ms: 0.0,
        }
    }

    pub fn resize(&mut self, context: &gpu::Context, extent: gpu::Extent) {
        if extent.width == 0 || extent.height == 0 {
            return;
        }
        self.extent = extent;
        context.destroy_texture_view(self.depth_view);
        context.destroy_texture(self.depth_texture);
        self.depth_texture = Self::create_texture(
            context,
            "depth",
            gpu::TextureFormat::Depth32Float,
            [extent.width, extent.height],
            gpu::TextureUsage::TARGET,
        );
        self.depth_view = Self::create_view(
            context,
            "depth view",
            self.depth_texture,
            gpu::TextureFormat::Depth32Float,
        );
        let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "depth init",
            buffer_count: 1,
            manual_barriers: false,
        });
        encoder.start();
        encoder.init_texture(self.depth_texture);
        let sync_point = context.submit(&mut encoder);
        let _ = context.wait_for(&sync_point, !0);
        context.destroy_command_encoder(&mut encoder);
    }

    pub fn set_camera(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch.clamp(-0.24, 1.08);
    }

    pub fn camera(&self) -> (f32, f32) {
        (self.yaw, self.pitch)
    }

    pub fn zoom(&mut self, factor: f32) {
        self.radius = (self.radius * factor).clamp(18.0, 145.0);
    }

    fn player(&self, elapsed: f32) -> ([f32; 2], [f32; 2]) {
        let angle = elapsed * 0.22;
        let position = [angle.sin() * 7.5, -18.0 + angle.cos() * 5.2];
        let velocity = [angle.cos() * 1.65, -angle.sin() * 1.14];
        (position, velocity)
    }

    fn write_uniforms(&mut self, context: &gpu::Context, elapsed: f32) {
        let underwater = false;
        let target = [0.0, TETHYS_WATER_LEVEL, -22.0];
        let horizontal = self.pitch.cos() * self.radius;
        let vertical_orbit = self.pitch.sin() * self.radius;
        let eye = [
            target[0] + self.yaw.sin() * horizontal,
            5.2 + vertical_orbit,
            target[2] + self.yaw.cos() * horizontal,
        ];
        let forward = normalize3([
            target[0] - eye[0],
            target[1] - eye[1],
            target[2] - eye[2],
        ]);
        let right = normalize3(cross3(forward, [0.0, 1.0, 0.0]));
        let up = normalize3(cross3(right, forward));
        let aspect = self.extent.width as f32 / self.extent.height.max(1) as f32;
        let projection = perspective(52.0 * std::f32::consts::PI / 180.0, aspect, 0.12, 560.0);
        let view = look_at(eye, target);
        let view_projection = multiply(&projection, &view);
        let (player_position, player_velocity) = self.player(elapsed);
        let tan_half_fov = (52.0 * std::f32::consts::PI / 360.0).tan();
        let sun = normalize3([-0.52, 0.30, -0.80]);
        let world = WorldUniforms {            view_proj: [
                [
                    view_projection[0],
                    view_projection[1],
                    view_projection[2],
                    view_projection[3],
                ],
                [
                    view_projection[4],
                    view_projection[5],
                    view_projection[6],
                    view_projection[7],
                ],
                [
                    view_projection[8],
                    view_projection[9],
                    view_projection[10],
                    view_projection[11],
                ],
                [
                    view_projection[12],
                    view_projection[13],
                    view_projection[14],
                    view_projection[15],
                ],
            ],
            camera_time: [eye[0], eye[1], eye[2], elapsed],
            camera_right: [right[0], right[1], right[2], tan_half_fov * aspect],
            camera_up: [up[0], up[1], up[2], tan_half_fov],
            camera_forward: [forward[0], forward[1], forward[2], 0.0],
            sun_water: [sun[0], sun[1], sun[2], TETHYS_WATER_LEVEL],
            terrain: [
                TERRAIN_EXTENT,
                MESH_RESOLUTION as f32,
                SIMULATION_RESOLUTION as f32,
                if underwater { 1.0 } else { 0.0 },
            ],
            simulation: [
                0.0,
                -12.0,
                TETHYS_WATER_FIELD_SIZE,
                1.0 / SIMULATION_RESOLUTION as f32,
            ],
            player: [
                player_position[0],
                player_position[1],
                player_velocity[0],
                player_velocity[1],
            ],
            interaction: [
                (player_velocity[0] * player_velocity[0]
                    + player_velocity[1] * player_velocity[1])
                    .sqrt(),
                1.0,
                self.extent.width as f32,
                self.extent.height as f32,
            ],
            environment: [0.0, MESH_RESOLUTION as f32, MESH_RESOLUTION as f32, 0.0],
        };
        // Zero the remaining 32 bytes of the 256-byte uniform buffer.
        let mut bytes = [0u8; WORLD_UNIFORM_BYTES as usize];
        unsafe {
            ptr::copy_nonoverlapping(
                &world as *const WorldUniforms as *const u8,
                bytes.as_mut_ptr(),
                mem::size_of::<WorldUniforms>(),
            );
        }
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), self.world_uniforms.data(), bytes.len());
        }
        context.sync_buffer(self.world_uniforms, 0, WORLD_UNIFORM_BYTES);
    }

    fn write_sim_params(&mut self, context: &gpu::Context, elapsed: f32) {
        let (player_position, _) = self.player(elapsed);
        let wake_due = elapsed - self.last_wake_at >= 0.10;
        let impulse_strength = if wake_due { -0.012 } else { 0.0 };
        if impulse_strength != 0.0 {
            self.last_wake_at = elapsed;
        }
        let impulse_uv_x = (player_position[0] - 0.0) / TETHYS_WATER_FIELD_SIZE + 0.5;
        let impulse_uv_y = (player_position[1] + 12.0) / TETHYS_WATER_FIELD_SIZE + 0.5;
        let params = SimulationParams {
            impulse: [
                impulse_uv_x,
                impulse_uv_y,
                impulse_strength,
                0.54 / TETHYS_WATER_FIELD_SIZE,
            ],
            step_foam_shift: [1.0 / 60.0, 0.72, 0.0, 0.0],
        };
        unsafe {
            ptr::copy_nonoverlapping(
                &params as *const SimulationParams as *const u8,
                self.sim_params.data(),
                mem::size_of::<SimulationParams>(),
            );
        }
        context.sync_buffer(self.sim_params, 0, SIMULATION_PARAM_BYTES);
    }

    pub fn frame(
        &mut self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        surface_view: gpu::TextureView,
    ) {
        let elapsed = self.start.elapsed().as_secs_f32();
        self.write_uniforms(context, elapsed);
        self.write_sim_params(context, elapsed);

        // [BISECT] compute passes disabled
        if let mut pass = encoder.render(
            "scene",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: surface_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            let mut rc = pass.with(&self.sky_pipeline);
            rc.bind(
                0,
                &SkyData {
                    uniforms: self.world_uniforms.into(),
                },
            );
            rc.draw(0, 3, 0, 1);
        }

        self.gpu_frame_ms = encoder
            .timings()
            .iter()
            .map(|(_, duration)| duration.as_secs_f32() * 1000.0)
            .sum();
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_buffer(self.world_uniforms);
        context.destroy_buffer(self.sim_params);
        for buffer in &self.fft_param_buffers {
            context.destroy_buffer(*buffer);
        }
        context.destroy_sampler(self.field_sampler);
        context.destroy_sampler(self.spectrum_sampler);
        context.destroy_texture_view(self.terrain_view);
        context.destroy_texture(self.terrain_texture);
        for view in &self.water_views {
            context.destroy_texture_view(*view);
        }
        for texture in &self.water_textures {
            context.destroy_texture(*texture);
        }
        for view in &self.breaker_event_views {
            context.destroy_texture_view(*view);
        }
        for texture in &self.breaker_event_textures {
            context.destroy_texture(*texture);
        }
        for cascade in &self.cascades {
            context.destroy_texture_view(cascade.initial_view);
            context.destroy_texture(cascade.initial);
            context.destroy_texture_view(cascade.wave_data_view);
            context.destroy_texture(cascade.wave_data);
            for ping in &cascade.field_views {
                for view in ping {
                    context.destroy_texture_view(*view);
                }
            }
            for ping in &cascade.fields {
                for texture in ping {
                    context.destroy_texture(*texture);
                }
            }
        }
        context.destroy_texture_view(self.twiddle_view);
        context.destroy_texture(self.twiddle_texture);
        context.destroy_texture_view(self.depth_view);
        context.destroy_texture(self.depth_texture);
        context.destroy_compute_pipeline(&mut self.terrain_compute);
        context.destroy_compute_pipeline(&mut self.simulation_pipeline);
        context.destroy_compute_pipeline(&mut self.breaker_event_pipeline);
        context.destroy_compute_pipeline(&mut self.spectrum_evolution);
        context.destroy_compute_pipeline(&mut self.spectral_fft);
        context.destroy_render_pipeline(&mut self.sky_pipeline);
        context.destroy_render_pipeline(&mut self.terrain_pipeline);
        context.destroy_render_pipeline(&mut self.water_pipeline);
        context.destroy_render_pipeline(&mut self.breaker_patch_pipeline);
    }
}
