// nntc_view.wgsl: the latent textures on a WebGPU sampler, the transliteration of VSMain, DecodeNNTC and PSMain in
// ../viewer/bin/nntc_view.hlsl (which the Direct3D 11 and Direct3D 12 viewers share, and which ../viewer_vk/bin/view.frag
// is the GLSL transliteration of).
//
// It is deliberately a transliteration and not a rewrite. The feature order and the dequantisation rule are the format's
// contract (../docs/FORMAT.md sections 4 and 6), so the arithmetic below is the other shader's line for line: `float4`
// reads `vec4<f32>`, `ddx` / `ddy` read `dpdx` / `dpdy`, `[unroll]` is dropped, `k++` becomes a statement of its own,
// `if (sel.x & 1)` becomes an explicit comparison against zero, and `Texture2D` with its `SamplerState` becomes a
// `texture_2d<f32>` and a `sampler`. The `>> 2` and `& 3` spellings stay for line parity.
//
// The one thing a reader of the native shader must not skip: LEVEL 1 IS SAMPLED WITH ITS GRADIENTS SCALED BY const0.z
// (2^lod_bias_level1 = 4, or 1.0 with the level-1 LOD shift turned off). WebGPU's sampler has NO LOD BIAS FIELD at all,
// so the gradient form the tree already ships is not merely preferable in a browser, it is the only way there. Without
// it the GPU reads level 1 - a quarter of the resolution per axis - two mips finer than the encoder fitted, and the
// picture goes splotchy from mip 1 down while still looking plausible at 1:1.
//
// Nothing here reconstructs texels: the representation is valid under the sampling operator.

// The one uniform block: the Direct3D viewer's SceneConstants followed by its DecoderConstants, 2032 bytes, which is
// what the Vulkan viewer already merges them into under std140. WGSL's uniform layout gives mat4x4<f32> and every vec4
// a 16-byte alignment and array<vec4<f32>, 108> a 16-byte stride, so the bytes are the same and the host packs them
// once for every backend (renderer.js asserts the offsets before the first draw).
struct Constants {
    mvp : mat4x4<f32>,      // built row-major on the host and transposed once, so these are the columns
    texSize : vec4<f32>,    // xy = level 0's base size, zw = level 1's
    lodInfo : vec4<f32>,    // x = level 0's mip count - 1, y = level 1's
    const0 : vec4<f32>,     // x = show level 0 raw, y = show level 1 raw, z = level 1's UV-gradient scale, w = renormalise the shown triple
    const1 : vec4<f32>,     // x = which channels to draw, in every view (0 RGB, 1 R, 2 G, 3 B, 4 A); yzw spare
    lo0 : vec4<f32>,
    hi0 : vec4<f32>,        // level 0's dequantisation per channel: value = lo + sample * (hi - lo)
    lo1 : vec4<f32>,
    hi1 : vec4<f32>,        // level 1's
    dims : vec4<i32>,       // x = C0, y = C1, z = nin, w = nout
    sel : vec4<i32>,        // x = the terms mask, y = the output texture shown, z = level 0's texture count, w = 0
    W : array<vec4<f32>, 108>,   // W row-major: row r, column c at W[r * 6 + c / 4][c % 4]
    bias : array<vec4<f32>, 5>,  // nout <= 18
};

@group(0) @binding(0) var<uniform> c : Constants;
@group(0) @binding(1) var lat0 : texture_2d<f32>;    // level 0: C0 channels (R8 / R8G8 / R8G8B8A8, or BC4 / BC5, or the same planes unpacked on the CPU)
@group(0) @binding(2) var lat1 : texture_2d<f32>;    // level 1: C1 channels, a quarter of the resolution per axis
@group(0) @binding(3) var lat0b : texture_2d<f32>;   // level 0's channels past the first two, when they are a file of their own
@group(0) @binding(4) var samp : sampler;            // level 0's sampler
@group(0) @binding(5) var samp1 : sampler;           // level 1's: the SAME unbiased sampler (no sampler carries a LOD bias; see const0.z)

struct VSOutput {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn VSMain(@location(0) in_pos : vec3<f32>, @location(1) in_uv : vec2<f32>) -> VSOutput {
    var o : VSOutput;
    o.pos = c.mvp * vec4<f32>(in_pos, 1.0);
    o.uv = in_uv;
    return o;
}

// The decode of one sample pair: the affine decoder over the two dequantised rows. Every output of the decoder is
// computed (3 per texture of the material, nout = 3 * textures, up to 18): what a game shader does once for the whole
// material from one pair of samples. The fragment shader then displays one texture's triple.
fn DecodeNNTC(z0 : vec4<f32>, z1 : vec4<f32>) -> array<f32, 18> {
    let C0 = c.dims.x;
    let C1 = c.dims.y;
    let nin = c.dims.z;
    let nout = c.dims.w;
    // The two rows as `var`s rather than the parameters themselves: a runtime index wants a reference, and the
    // parameter is a value. Nothing else about them changes.
    var s = z0;
    var cc = z1;
    var phi : array<f32, 24>;
    for (var i = 0; i < 24; i++) { phi[i] = 0.0; }
    var k = 0;
    if ((c.sel.x & 1) != 0) {                                       // [a]  c_j
        for (var j = 0; j < 4; j++) { if (j < C1) { phi[k] = cc[j]; k++; } }
    }
    if ((c.sel.x & 2) != 0) {                                       // [b]  s_i
        for (var i = 0; i < 4; i++) { if (i < C0) { phi[k] = s[i]; k++; } }
    }
    if ((c.sel.x & 4) != 0) {                                       // [sc] s_i c_j
        for (var i = 0; i < 4; i++) {
            for (var j = 0; j < 4; j++) { if (i < C0 && j < C1) { phi[k] = s[i] * cc[j]; k++; } }
        }
    }
    var outv : array<f32, 18>;
    for (var r = 0; r < 18; r++) {
        var acc = c.bias[r >> 2][r & 3];   // >> 2 and & 3 are / 4 and % 4 on these non-negative indices
        if (r < nout) {
            for (var col = 0; col < nin; col++) { acc = acc + c.W[r * 6 + (col >> 2)][col & 3] * phi[col]; }
        }
        outv[r] = acc;   // the identity output; the caller clamps to [0, 1] as the encoder's 8-bit rounding does
    }
    return outv;
}


// WHICH CHANNELS ARE DRAWN, in all three views. They used to show xyz and nothing else, which left
// a four-channel latent's fourth channel with no way to be looked at at all, and gave no way to tell which of
// three visible ones carried what. A single channel is drawn as GREY - the same value in all three - because a
// lone channel shown in its own colour reads as a tint rather than as a magnitude, and the thing being judged
// here is a magnitude. This branches on a uniform, so every pixel of a frame takes the same path.
fn shownChannels(s : vec4<f32>, which : f32) -> vec4<f32> {
    if (which < 0.5) { return vec4<f32>(s.xyz, 1.0); }
    var v = s.x;
    if (which > 3.5) { v = s.w; } else if (which > 2.5) { v = s.z; } else if (which > 1.5) { v = s.y; }
    return vec4<f32>(v, v, v, 1.0);
}
@fragment
fn PSMain(input : VSOutput) -> @location(0) vec4<f32> {
    // Two standard mipmapped texture samples (three when level 0 is in two files and lat0b is bound): what a game does.
    // Both latents are read through the same explicit-gradient path, so every implementation filters them alike.
    let dx = dpdx(input.uv);
    let dy = dpdy(input.uv);
    var s0 = textureSampleGrad(lat0, samp, input.uv, dx, dy);   // the uncompressed level 0, or a BC4 / BC5, or its CPU unpack (the same sampler either way)
    if (c.sel.z == 2) {
        let s0b = textureSampleGrad(lat0b, samp, input.uv, dx, dy);   // channels 2-3 from the second texture; at C0 3 only .r (channel 2) is read below
        s0 = vec4<f32>(s0.x, s0.y, s0b.x, s0b.y);
    }
    // Level 1 is a quarter of level 0's size, so left alone the GPU reads it two mips finer than level 0. The encoder
    // pairs mip m of level 1 with mip m of level 0, so its UV gradients are scaled by const0.z = 2^lod_bias_level1
    // (4; 1.0 when the level-1 LOD shift is turned off). GRADIENTS RATHER THAN A SAMPLER LOD BIAS: the native viewers
    // chose it because a bias blurs a magnified texture badly on Intel integrated graphics, and WebGPU has no sampler
    // LOD bias field at all, so here it is the only form the rule can take.
    let s1 = textureSampleGrad(lat1, samp1, input.uv, dx * c.const0.z, dy * c.const0.z);
    if (c.const0.x > 0.5) { return shownChannels(s0, c.const1.x); }   // key 1: level 0's texture as stored
    if (c.const0.y > 0.5) { return shownChannels(s1, c.const1.x); }   // key 2: level 1's
    let z0 = c.lo0 + s0 * (c.hi0 - c.lo0);   // the dequantisation AFTER sampling (affine, so the blend of samples is the blend of values)
    let z1 = c.lo1 + s1 * (c.hi1 - c.lo1);
    var outv = DecodeNNTC(z0, z1);           // the whole material decoded (a `var`, so the runtime index below is a reference)
    let t = 3 * c.sel.y;                     // the texture shown
    var rgb = saturate(vec3<f32>(outv[t], outv[t + 1], outv[t + 2]));
    if (c.const0.w > 0.5) {
        // Tangent-space normal renormalisation: unpack the triple to [-1, 1], renormalise to unit length, repack. Only
        // a pixel that can be a tangent-space normal is touched: a grey-ish pixel unpacks to a near-zero vector with no
        // direction to normalise to, and a black-ish pixel (or anything with z < 0) points into the surface, which no
        // tangent-space normal does. Those are left as decoded.
        let n = rgb * 2.0 - 1.0;
        let len = length(n);
        if (len > 0.5 && n.z > 0.0) { rgb = saturate((n / len) * 0.5 + 0.5); }
    }
    // THE DECODED PICTURE GOES THROUGH THE SAME CONTROL, after the renormalisation rather than before it: what
    // is being looked at one channel at a time is what is on screen, not an intermediate. Alpha is 1.0 by
    // construction here - this decoder writes triples - so choosing A in this view draws white, which is the
    // honest answer rather than a hidden case.
    return shownChannels(vec4<f32>(rgb, 1.0), c.const1.x);
}
