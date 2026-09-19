// view.frag: the latent textures on a Vulkan sampler, the transliteration of DecodeNNTC and PSMain in
// viewer/bin/nntc_view.hlsl.
//
// It is deliberately a transliteration and not a rewrite. The feature order and the dequantisation rule are the
// format's contract (../../docs/FORMAT.md section 4 and section 6) and not an implementation detail either viewer is
// free to restate, so the arithmetic below is the other shader's line for line: `float4` reads `vec4`, `saturate` reads
// `clamp(x, 0.0, 1.0)`, `[unroll]` is dropped, and `Texture2D` with its `SamplerState` becomes a `texture2D` and a
// `sampler` combined at the point of use - which is what lets one key rewrite one sampler descriptor and nothing else
// (key M's maxLod; key L's LOD shift is a uniform now, const0.z, and rewrites no descriptor at all).
//
// Two standard mipmapped samples, nothing special (three when level 0 is in two files): what a game does. The hardware
// picks each texture's mip from the same UV derivatives, and level 1's sample SCALES those derivatives by
// 2^lod_bias_level1 (const0.z) so that output mip m reads mip m of BOTH latents - the 1:1 rule the encoder fitted.
// Nothing here reconstructs texels: the representation is valid under the sampling operator.
#version 450

// Repeated verbatim from view.vert, and it has to be: shaderc compiles each stage on its own, and both stages declare
// binding 0. The order is the Direct3D viewer's SceneConstants followed by its DecoderConstants, which under std140 is
// the same 2032 bytes in the same order.
layout(std140, set = 0, binding = 0) uniform Constants {
    mat4  mvp;
    vec4  tex_size;
    vec4  lod_info;
    vec4  const0;    // x / y = show level 0 / level 1 raw, z = level 1's UV-gradient scale, w = renormalise
    vec4  const1;    // spare
    vec4  lo0, hi0;
    vec4  lo1, hi1;
    ivec4 dims;
    ivec4 sel;
    vec4  W[108];
    vec4  bias_[5];
} cb;

layout(set = 0, binding = 1) uniform texture2D lat0;    // level 0: C0 channels (R8 / R8G8 / R8G8B8A8 / BC4 / BC5)
layout(set = 0, binding = 2) uniform texture2D lat1;    // level 1: C1 channels, a quarter of the resolution per axis
layout(set = 0, binding = 3) uniform texture2D lat0b;   // level 0's channels past the first two, when they are a file of their own
layout(set = 0, binding = 4) uniform sampler   samp;    // level 0's sampler
layout(set = 0, binding = 5) uniform sampler   samp1;   // level 1's: the SAME unbiased sampler (no sampler carries a LOD bias; see const0.z)

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_colour;

// The decode of one sample pair: the affine decoder over the two dequantised rows. Every output is computed (3 per
// texture of the material, nout = 3 * textures, up to 18), which is what a game shader does once for the whole
// material from one pair of samples; main() then displays one texture's triple.
void decode_nntc(vec4 z0, vec4 z1, out float outv[18]) {
    const int C0 = cb.dims.x, C1 = cb.dims.y, nin = cb.dims.z, nout = cb.dims.w;
    float phi[24];
    for (int i = 0; i < 24; i++) phi[i] = 0.0;
    int k = 0;
    if ((cb.sel.x & 1) != 0) { for (int j = 0; j < 4; j++) if (j < C1) phi[k++] = z1[j]; }                        // [a]  c_j
    if ((cb.sel.x & 2) != 0) { for (int i = 0; i < 4; i++) if (i < C0) phi[k++] = z0[i]; }                        // [b]  s_i
    if ((cb.sel.x & 4) != 0) { for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) if (i < C0 && j < C1) phi[k++] = z0[i] * z1[j]; }   // [sc] s_i c_j
    for (int r = 0; r < 18; r++) {
        float acc = cb.bias_[r >> 2][r & 3];   // >> 2 and & 3 are / 4 and % 4 on these non-negative indices, as in the HLSL
        if (r < nout) for (int c = 0; c < nin; c++) acc += cb.W[r * 6 + (c >> 2)][c & 3] * phi[c];
        outv[r] = acc;   // the identity output; the caller clamps to [0, 1] as the encoder's 8-bit rounding does
    }
}

void main() {
    // Both latents are read through the same explicit-gradient path, so every implementation filters them alike
    // (an implicit texture() beside a textureGrad() may not be: llvmpipe gives only the implicit one anisotropy).
    const vec2 dx = dFdx(v_uv), dy = dFdy(v_uv);
    vec4 s0 = textureGrad(sampler2D(lat0, samp), v_uv, dx, dy);    // the uncompressed level 0, or a BC4 / BC5 (its channels in .rg)
    if (cb.sel.z == 2) s0.zw = textureGrad(sampler2D(lat0b, samp), v_uv, dx, dy).rg;   // channels 2-3 from the second file; at C0 3 only .r is read below
    // Level 1 is a quarter of level 0's size, so left alone the GPU reads it two mips finer than level 0. The encoder
    // pairs mip m of level 1 with mip m of level 0, so its UV gradients are scaled by const0.z = 2^lod_bias_level1
    // (4; 1.0 when key L turns the rule off) to keep the two in step.
    // Gradients rather than a sampler LOD bias of +2: the bias is fine on NVIDIA and AMD but blurs a magnified
    // texture badly on Intel integrated graphics; scaled gradients work on all three. README.md, "The two ways to
    // apply the level-1 mip shift", compares the two methods.
    vec4 s1 = textureGrad(sampler2D(lat1, samp1), v_uv, dx * cb.const0.z, dy * cb.const0.z);
    if (cb.const0.x > 0.5) { out_colour = vec4(s0.rgb, 1.0); return; }   // level 0's texture as stored
    if (cb.const0.y > 0.5) { out_colour = vec4(s1.rgb, 1.0); return; }   // level 1's
    vec4 z0 = cb.lo0 + s0 * (cb.hi0 - cb.lo0);   // the dequantisation AFTER sampling (affine, so the blend of samples is the blend of values)
    vec4 z1 = cb.lo1 + s1 * (cb.hi1 - cb.lo1);
    float outv[18];
    decode_nntc(z0, z1, outv);                   // the whole material decoded
    const int t = 3 * cb.sel.y;                  // the texture shown
    vec3 rgb = clamp(vec3(outv[t], outv[t + 1], outv[t + 2]), 0.0, 1.0);
    if (cb.const0.w > 0.5) {
        // Tangent-space normal renormalisation: unpack the 8-bit triple to [-1, 1], renormalise to unit length, repack
        // to [0, 1]. The decoded normal is an affine reconstruction, so its length drifts from 1 where the fit is
        // imperfect. Only a pixel that can be a tangent-space normal is touched: a grey-ish pixel unpacks to a
        // near-zero vector with no direction to normalise to, and one with z < 0 points into the surface.
        vec3 n = rgb * 2.0 - 1.0;
        const float len = length(n);
        if (len > 0.5 && n.z > 0.0) rgb = clamp((n / len) * 0.5 + 0.5, 0.0, 1.0);
    }
    out_colour = vec4(rgb, 1.0);
}
