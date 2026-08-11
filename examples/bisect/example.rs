// Bisect harness: bunnymark shell + my sky pipeline from water.wgsl.

use blade_graphics as gpu;
use bytemuck::{Pod, Zeroable};

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

#[derive(blade_macros::ShaderData)]
struct SkyData {
    uniforms: WorldUniforms,
}

pub struct Example {
    pipeline: gpu::RenderPipeline,
    uniforms: gpu::Buffer,
    screen_size: gpu::Extent,
}

impl Example {
    pub fn new(
        context: &gpu::Context,
        screen_size: gpu::Extent,
        surface_format: gpu::TextureFormat,
    ) -> Self {
        let shader_source = include_str!("water.wgsl");
        let shader = context.create_shader(gpu::ShaderDesc {
            source: shader_source,
            naga_module: None,
        });
        let sky_layout = <SkyData as gpu::ShaderData>::layout();

        let pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
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

        let uniforms = context.create_buffer(gpu::BufferDesc {
            name: "world uniforms",
            size: 256,
            memory: gpu::Memory::Shared,
        });

        Self {
            pipeline,
            uniforms,
            screen_size,
        }
    }

    pub fn screen_size(&self) -> gpu::Extent {
        self.screen_size
    }

    pub fn set_screen_size(&mut self, size: gpu::Extent) {
        self.screen_size = size;
    }

    pub fn render(
        &mut self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        target: gpu::TextureView,
    ) {
        let mut uniforms = WorldUniforms {
            view_proj: [[0.0; 4]; 4],
            camera_time: [0.0, 0.0, 0.0, 1.0],
            camera_right: [1.0, 0.0, 0.0, 0.5],
            camera_up: [0.0, 1.0, 0.0, 0.5],
            camera_forward: [0.0, 0.0, 1.0, 0.0],
            sun_water: [0.0, 1.0, 0.0, 0.0],
            terrain: [390.0, 240.0, 256.0, 0.0],
            simulation: [0.0, -12.0, 192.0, 1.0 / 256.0],
            player: [0.0, 0.0, 0.0, 0.0],
            interaction: [0.0, 1.0, 0.0, 0.0],
            environment: [0.0, 240.0, 240.0, 0.0],
        };
        uniforms.view_proj[0][0] = 1.0;
        uniforms.view_proj[1][1] = 1.0;
        uniforms.view_proj[2][2] = 1.0;
        uniforms.view_proj[3][3] = 1.0;
        let _ = context;

        if let mut pass = encoder.render(
            "main",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            let mut rc = pass.with(&self.pipeline);
            rc.bind(0, &SkyData { uniforms });
            rc.draw(0, 3, 0, 1);
        }
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        let _ = context;
        context.destroy_render_pipeline(&mut self.pipeline);
    }
}


impl Example {
    pub fn increase(&mut self) {}
    pub fn step(&mut self, _delta: f32) {}
}
