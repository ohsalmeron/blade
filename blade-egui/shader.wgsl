struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

struct Uniforms {
    screen_size: vec2<f32>,
    padding: vec2<f32>,
};
var<uniform> r_uniforms: Uniforms;

// Matches epaint::Vertex (pos, uv, color) so we can fetch from a vertex buffer
// instead of a storage buffer — WebGL2 has no SSBOs.
struct Vertex {
    pos: vec2<f32>,
    uv: vec2<f32>,
    color: u32,
}
}

fn linear_from_gamma(srgb: vec3<f32>) -> vec3<f32> {
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let is_higher = step(vec3<f32>(0.04045), srgb);
    return mix(lower, higher, is_higher);
}

@vertex
fn vs_main(input: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coord = input.uv;
    out.color = unpack4x8unorm(input.color);
    out.position = vec4<f32>(
        2.0 * input.pos.x / r_uniforms.screen_size.x - 1.0,
        1.0 - 2.0 * input.pos.y / r_uniforms.screen_size.y,
        0.0,
        1.0,
    );
    return out;
}

var r_texture: texture_2d<f32>;
var r_sampler: sampler;

// Egui blends in gamma space, see https://github.com/emilk/egui/pull/2071
fn blended_color(in: VertexOutput) -> vec4<f32> {
    return in.color * textureSample(r_texture, r_sampler, in.tex_coord);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // ColorSpace::Linear swapchain (Vulkan/Metal/EGL sRGB storage).
    let blended = blended_color(in);
    return vec4f(linear_from_gamma(blended.xyz), blended.a);
}

@fragment
fn fs_main_srgb(in: VertexOutput) -> @location(0) vec4<f32> {
    // Plain UNORM canvas (WebGL blit to HTML canvas): output gamma-space directly.
    let blended = blended_color(in);
    return vec4f(blended.xyz, blended.a);
}
