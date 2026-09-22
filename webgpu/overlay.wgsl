// overlay.wgsl: the debug strip, drawn over the scene.
//
// The transliteration of DEBUG_HLSL in ../viewer/main.cpp (the five-line shader pair the native viewers compile at
// start-up): a clip-space quad with a texture coordinate, and one texture fetch. The strip's RGBA pixels are blitted on
// the CPU from the tree's own 8x8 font (overlay.js), so the glyphs are the native viewers' glyphs and a browser
// screenshot reads as the same viewer. Alpha blending and the depth test are pipeline state, not shader state.

@group(0) @binding(0) var strip : texture_2d<f32>;
@group(0) @binding(1) var strip_samp : sampler;

struct VSO {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn VSMain(@location(0) pos : vec2<f32>, @location(1) uv : vec2<f32>) -> VSO {
    var o : VSO;
    o.pos = vec4<f32>(pos, 0.0, 1.0);
    o.uv = uv;
    return o;
}

@fragment
fn PSMain(i : VSO) -> @location(0) vec4<f32> {
    return textureSample(strip, strip_samp, i.uv);
}
