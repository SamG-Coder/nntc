// nntc_view.hlsl: the latent textures on the GPU (SM 5.0). Lifted from the author's shader_deblocking_d3d11 sample (Apache 2.0);
// the sampling and the decode are new here.
//
// The asset is two latent textures with full mip chains (PREFIX_lat0.dds, PREFIX_lat1.dds; the quantisation index per texel with its
// bits replicated over the byte) and PREFIX_nntc.json (the dequantisation and the bilinear decoder). The GPU samples BOTH textures with the
// hardware sampler (point / bilinear / trilinear, two standard mipmapped samples, three when level 0 is in two files and t2 is
// bound), the shader dequantises the samples (affine per
// channel, so the blend of samples is the blend of the encoder's values), builds the feature vector phi(z) in the JSON's order and
// applies the affine decoder out = W phi + b. Nothing here reconstructs texels: the representation is valid under the sampling operator.

cbuffer SceneConstants : register(b0)
{
    float4x4 mvp;       // CPU builds row-major and transposes once at cbuffer-write time
    float4   texSize;   // xy = level 0's base size, zw = level 1's base size
    float4   lodInfo;   // x = level 0's mip count - 1, y = level 1's
    float4   const0;    // x = show level 0's texture raw, y = show level 1's raw (keys 1 / 2), z = the factor level 1's UV gradients are scaled by before its sample (2^lod_bias_level1 with the 1:1 mip rule on, 1 with key L off), w = renormalise the shown triple as a tangent-space normal (key V)
    float4   const1;    // spare (keys 5-8)
};

// The decoder (PREFIX_nntc.json): the dequantisation of both textures and the affine map over phi(z).
cbuffer DecoderConstants : register(b1)
{
    float4 lo0, hi0;    // level 0's dequantisation per channel: value = lo + sample * (hi - lo)
    float4 lo1, hi1;    // level 1's
    int4   dims;        // x = C0 (level 0 channels), y = C1 (level 1 channels), z = nin (|phi|), w = nout (3 per output texture)
    int4   sel;         // x = the terms mask (1 = a: c_j, 2 = b: s_i, 4 = sc: s_i c_j), y = the output texture shown (materials), z = level 0 packed: 0 (no), 1 (t0 = BC4 / BC5), 2 (t0 and t2), w = 0
    float4 W[108];      // W row-major, nout <= 18 rows of nin <= 24 columns: row r, column c at W[r * 6 + c / 4][c % 4]
    float4 bias[5];     // nout <= 18
};

Texture2D    lat0 : register(t0);   // level 0: C0 channels (R8 / R8G8 / R8G8B8A8)
Texture2D    lat1 : register(t1);   // level 1: C1 channels, a quarter of the resolution per axis
Texture2D    lat0b : register(t2);  // level 0's channels past the first two, in a texture of their own (sel.z = the texture count: 0 = the single uncompressed t0)
SamplerState samp : register(s0);
SamplerState samp1 : register(s1);  // level 1's sampler: the SAME unbiased sampler as s0 (no sampler in this viewer carries a LOD bias any more). The 1:1 mip rule lives in the gradients handed to SampleGrad below, through const0.z

struct VSInput  { float3 pos : POSITION; float2 uv : TEXCOORD0; };
struct VSOutput { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

VSOutput VSMain(VSInput input)
{
    VSOutput o;
    o.pos = mul(mvp, float4(input.pos, 1.0));
    o.uv  = input.uv;
    return o;
}

// The decode of one sample pair: the affine decoder over the two dequantised rows. Every output of the decoder is computed
// (3 per texture of the material, nout = 3 * textures, up to 18): what a game shader does once for the whole material from one pair of
// samples. The pixel shader then displays one texture's triple.
void DecodeNNTC(float4 z0, float4 z1, out float outv[18])
{
    const int C0 = dims.x, C1 = dims.y, nin = dims.z, nout = dims.w;
    float phi[24];
    [unroll] for (int i = 0; i < 24; i++) phi[i] = 0.0;
    int k = 0;
    if (sel.x & 1) { [unroll] for (int j = 0; j < 4; j++) if (j < C1) phi[k++] = z1[j]; }                       // [a]  c_j
    if (sel.x & 2) { [unroll] for (int i = 0; i < 4; i++) if (i < C0) phi[k++] = z0[i]; }                       // [b]  s_i
    if (sel.x & 4) { [unroll] for (int i = 0; i < 4; i++) [unroll] for (int j = 0; j < 4; j++) if (i < C0 && j < C1) phi[k++] = z0[i] * z1[j]; }   // [sc] s_i c_j
    [unroll] for (int r = 0; r < 18; r++) {
        float acc = bias[r >> 2][r & 3];   // >> 2 and & 3 are / 4 and % 4 on these non-negative indices, without fxc's integer-division warning
        if (r < nout) for (int c = 0; c < nin; c++) acc += W[r * 6 + (c >> 2)][c & 3] * phi[c];
        outv[r] = acc;   // the identity output; the caller clamps to [0, 1] as the encoder's 8-bit rounding does
    }
}

float4 PSMain(VSOutput input) : SV_Target
{
    // Two standard mipmapped texture samples (three when level 0 is in two files and t2 is bound): what a game does.
    // Both latents are read through the same explicit-gradient path, so every implementation filters them alike
    // (an implicit Sample beside a SampleGrad may not be: llvmpipe gives only the implicit one anisotropy).
    const float2 dx = ddx(input.uv), dy = ddy(input.uv);
    float4 s0 = lat0.SampleGrad(samp, input.uv, dx, dy);   // the uncompressed level 0, or a BC4 / BC5 (its channels in .rg, the same sampler)
    if (sel.z == 2) s0.zw = lat0b.SampleGrad(samp, input.uv, dx, dy).rg;   // channels 2-3 from the second texture; at C0 3 it is a BC4 and only .r (channel 2) is read below
    // Level 1 is a quarter of level 0's size, so left alone the GPU reads it two mips finer than level 0. The encoder
    // pairs mip m of level 1 with mip m of level 0, so its UV gradients are scaled by const0.z = 2^lod_bias_level1
    // (4; 1.0 when key L turns the rule off) to keep the two in step.
    // Gradients rather than a sampler LOD bias of +2: the bias is fine on NVIDIA and AMD but blurs a magnified
    // texture badly on Intel integrated graphics; scaled gradients work on all three. README.md, "The two ways to
    // apply the level-1 mip shift", compares the two methods.
    float4 s1 = lat1.SampleGrad(samp1, input.uv, dx * const0.z, dy * const0.z);
    if (const0.x > 0.5) return float4(s0.rgb, 1.0);   // key 1: level 0's texture as stored (its channels as RGB)
    if (const0.y > 0.5) return float4(s1.rgb, 1.0);   // key 2: level 1's
    float4 z0 = lo0 + s0 * (hi0 - lo0);   // the dequantisation AFTER sampling (affine, so the blend of samples is the blend of values)
    float4 z1 = lo1 + s1 * (hi1 - lo1);
    float outv[18]; DecodeNNTC(z0, z1, outv);   // the whole material decoded
    const int t = 3 * sel.y;                   // the texture shown (key N)
    float3 rgb = saturate(float3(outv[t], outv[t + 1], outv[t + 2]));
    if (const0.w > 0.5) {
        // Tangent-space normal renormalisation (key V, off by default): unpack the 8-bit triple to [-1, 1], renormalise to unit length,
        // repack to [0, 1]. The decoded normal is an affine reconstruction, so its length drifts from 1 where the fit is imperfect; this
        // shows what a shader that normalises after the fetch would see. Only a pixel that can be a tangent-space normal is touched:
        // a grey-ish pixel unpacks to a near-zero vector with no direction to normalise to, and a black-ish pixel (or anything with
        // z < 0, i.e. blue below 0.5) points into the surface, which no tangent-space normal does. Those are left as decoded.
        float3 n = rgb * 2.0 - 1.0;
        const float len = length(n);
        if (len > 0.5 && n.z > 0.0) rgb = saturate((n / len) * 0.5 + 0.5);
    }
    return float4(rgb, 1.0);
}
