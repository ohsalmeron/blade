// blade-water — native port of inkwell-webgpu-water (open-ocean, optimized,
// surface view). Drag to orbit, wheel to zoom, Esc to quit.
//
// Controls and telemetry mirror the original web lab: FPS and per-frame GPU
// time are printed every 100 frames.

#![allow(irrefutable_let_patterns)]

mod engine;

use blade_graphics as gpu;
use engine::WaterEngine;
use std::time::Instant;

fn make_surface_config(size: winit::dpi::PhysicalSize<u32>) -> gpu::SurfaceConfig {
    gpu::SurfaceConfig {
        size: gpu::Extent {
            width: size.width,
            height: size.height,
            depth: 1,
        },
        usage: gpu::TextureUsage::TARGET,
        display_sync: gpu::DisplaySync::Recent,
        ..Default::default()
    }
}

struct App {
    engine: Option<WaterEngine>,
    command_encoder: Option<gpu::CommandEncoder>,
    prev_sync_point: Option<gpu::SyncPoint>,
    surface: Option<gpu::Surface>,
    context: Option<gpu::Context>,
    window: Option<winit::window::Window>,
    dragging: bool,
    last_cursor: winit::dpi::PhysicalPosition<f64>,
    last_snapshot: Instant,
    frame_count: u32,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes =
            winit::window::Window::default_attributes().with_title("blade-water (Tethys)");
        let window = event_loop.create_window(window_attributes).unwrap();

        let context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                timing: false,
                ..Default::default()
            })
            .unwrap()
        };
        println!("{:?}", context.device_information());

        let window_size = window.inner_size();
        let surface = context
            .create_surface_configured(&window, make_surface_config(window_size))
            .unwrap();
        let surface_info = surface.info();

        let extent = gpu::Extent {
            width: window_size.width,
            height: window_size.height,
            depth: 1,
        };
        let engine = WaterEngine::new(&context, extent, surface_info.format);

        let command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
            manual_barriers: false,
        });

        self.engine = Some(engine);
        self.command_encoder = Some(command_encoder);
        self.surface = Some(surface);
        self.context = Some(context);
        self.window = Some(window);
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::Resized(size) => {
                let context = self.context.as_ref().unwrap();
                let config = make_surface_config(size);
                context.reconfigure_surface(self.surface.as_mut().unwrap(), config);
                if let Some(engine) = self.engine.as_mut() {
                    engine.resize(
                        context,
                        gpu::Extent {
                            width: size.width,
                            height: size.height,
                            depth: 1,
                        },
                    );
                }
            }
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                if self.dragging {
                    let dx = position.x - self.last_cursor.x;
                    let dy = position.y - self.last_cursor.y;
                    if let Some(engine) = self.engine.as_mut() {
                        let (yaw, pitch) = engine.camera();
                        engine.set_camera(yaw - dx as f32 * 0.005, pitch + dy as f32 * 0.004);
                    }
                }
                self.last_cursor = position;
            }
            winit::event::WindowEvent::MouseInput {
                state, button, ..
            } => {
                if button == winit::event::MouseButton::Left {
                    self.dragging = state == winit::event::ElementState::Pressed;
                }
            }
            winit::event::WindowEvent::MouseWheel { delta, .. } => {
                let pixels = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y * 100.0,
                    winit::event::MouseScrollDelta::PixelDelta(position) => position.y as f32,
                };
                if let Some(engine) = self.engine.as_mut() {
                    engine.zoom((pixels * 0.001).exp());
                }
            }
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => match key_code {
                winit::keyboard::KeyCode::Escape => {
                    event_loop.exit();
                }
                _ => {}
            },
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                let context = self.context.as_ref().unwrap();
                let surface = self.surface.as_mut().unwrap();
                let command_encoder = self.command_encoder.as_mut().unwrap();
                let engine = self.engine.as_mut().unwrap();

                // Wait for the previous frame before rewriting the shared
                // uniform buffers (no frames in flight).
                if let Some(sp) = self.prev_sync_point.take() {
                    let _ = context.wait_for(&sp, !0);
                }

                let frame = surface.acquire_frame();

                command_encoder.start();
                command_encoder.init_texture(frame.texture());
                engine.frame(context, command_encoder, frame.texture_view());
                command_encoder.present(frame);
                let sync_point = context.submit(command_encoder);
                self.prev_sync_point = Some(sync_point);

                self.frame_count += 1;
                if self.frame_count == 100 {
                    let frame_ms = self.last_snapshot.elapsed().as_secs_f32() * 10.0;
                    println!(
                        "Avg frame {:.2} ms ({:.0} fps), GPU {:.2} ms",
                        frame_ms,
                        1000.0 / frame_ms,
                        engine.gpu_frame_ms
                    );
                    self.last_snapshot = Instant::now();
                    self.frame_count = 0;
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = App {
        engine: None,
        command_encoder: None,
        prev_sync_point: None,
        surface: None,
        context: None,
        window: None,
        dragging: false,
        last_cursor: winit::dpi::PhysicalPosition::new(0.0, 0.0),
        last_snapshot: Instant::now(),
        frame_count: 0,
    };
    event_loop.run_app(&mut app).unwrap();

    let context = app.context.as_ref().unwrap();
    if let Some(sp) = app.prev_sync_point.take() {
        let _ = context.wait_for(&sp, !0);
    }
    if let Some(mut engine) = app.engine.take() {
        engine.deinit(context);
    }
    if let Some(mut command_encoder) = app.command_encoder.take() {
        context.destroy_command_encoder(&mut command_encoder);
    }
    if let Some(mut surface) = app.surface.take() {
        context.destroy_surface(&mut surface);
    }
}
