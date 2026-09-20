// nntc_view_linalg.hlsl: the Shader Model 6.10 linear-algebra decode, and ../../viewer/bin/nntc_view.hlsl with one part changed.
//
// Everything here is that file's: the same two or three SampleGrad calls through the same samplers, the same dequantisation
// after sampling, the same feature order (docs/FORMAT.md section 4), the same clamp, the same texture selection, the same
// raw modes and the same renormalisation. What differs is the multiply in the middle. Where nntc_view.hlsl writes the 18 by
// 24 product as two nested loops over the constant buffer's W, this file hands it to the SM 6.10 linear-algebra API as ONE
// MultiplyAdd, with the matrix read from a ByteAddressBuffer and the bias read beside it.
//
// The rest of the file is a transliteration for a reason worth restating: the feature order and the dequantisation rule are
// the format's contract, so the two decode paths must not be two readings of it. A difference between the frames this file
// draws and the frames nntc_view.hlsl draws is then the fp16 weights and nothing else, which is what makes that difference
// measurable (docs/D3D12_LINEAR_ALGEBRA_PLAN.md B.6).
//
// THE SHAPE IS ALWAYS 18 BY 24. M and K are template arguments and therefore compile-time constants, and this format's nin
// and nout vary per asset, so the matrix is zero-padded to the largest shape the format allows and phi is zero past nin. A
// zero row contributes an output that is already ignored and a zero column multiplies a zero feature, so the padded product
// is the unpadded one in exact arithmetic.
//
// This file is compiled ONLY by a build made with the CMake option NNTC_D3D12_LINALG and only on a device whose query
// passed. Everywhere else the viewer compiles nntc_view.hlsl alone and never mentions the feature.
//
// THE VERTEX SHADER IS HERE TOO, and it is compiled by DXC rather than by fxc. A pipeline state whose vertex stage is DXBC
// and whose pixel stage is DXIL is refused: CreateGraphicsPipelineState returns E_INVALIDARG (0x80070057) for that pair,
// measured on this tree's machine, which answers the plan's section B.4. So this path's vertex stage is a second compiler's
// output, and the arm that separates "a different compiler" from "a different instruction" is the DXC plain arm of B.6.
#include <dx/linalg.h>
using namespace dx::linalg;

// WHICH TYPES THE MULTIPLY USES is the device's answer and not this file's, so the one place it varies is behind these
// macros, which the viewer defines from the row its query found. There is exactly one shader file either way, because two
// files would be two readings of the same decode and would drift.
//
//   NNTC_LA_FP32 1 - the bias and the RESULT are fp32. The input vector is fp16 either way, so phi is rounded to fp16 on
//                    the way in just as the matrix is: the difference from nntc_view.hlsl is the 11-bit mantissa of the
//                    weights AND of the inputs, not of the weights alone. This is what the viewer asks for second, and
//                    takes when the device grants it (WARP does).
//   NNTC_LA_FP32 0 - the bias and the result are fp16 as well. It is the row Tier 1 makes mandatory, and it is what an
//                    NVIDIA driver enumerates on the machine this was written on: no fp32 result at all. The accumulation
//                    is then the implementation's own, which is why the two paths' frames are compared and the number
//                    published rather than assumed.
#if NNTC_LA_FP32
#define LA_SCALAR float
#else
#define LA_SCALAR half
#endif

// NNTC_LA_MUL_OPTIMAL 1 - the matrix was run through ConvertLinearAlgebraMatrix into the device's own multiply-optimal
//                         layout, whose addressing is the device's own business: the stride passed here is 0, which is
//                         what the specification asks for and what the load ignores. 0 - it is row-major fp16 as the host
//                         wrote it, which the specification permits for a thread-scope load and which needs no conversion
//                         API at all. Both are drawn and compared (the plan's B.5).
#if NNTC_LA_MUL_OPTIMAL
#define LA_LAYOUT MatrixLayoutEnum::MulOptimal
#else
#define LA_LAYOUT MatrixLayoutEnum::RowMajor
#endif

cbuffer SceneConstants : register(b0)
{
    float4x4 mvp;       // CPU builds row-major and transposes once at cbuffer-write time
    float4   texSize;   // xy = level 0's base size, zw = level 1's base size
    float4   lodInfo;   // x = level 0's mip count - 1, y = level 1's
    float4   const0;    // x = show level 0's texture raw, y = show level 1's raw (keys 1 / 2), z = the factor level 1's UV gradients are scaled by before its sample, w = renormalise the shown triple as a tangent-space normal (key V)
    float4   const1;    // spare (keys 5-8)
};

// The decoder constants, the same struct the plain path reads, with two fields at the end this path needs. W and bias stay
// in it and are unread here: the buffer is one struct written once per frame by set_uniforms, and declaring it the same way
// in both shaders is what keeps the offsets a fact rather than a coincidence.
cbuffer DecoderConstants : register(b1)
{
    float4 lo0, hi0;    // level 0's dequantisation per channel: value = lo + sample * (hi - lo)
    float4 lo1, hi1;    // level 1's
    int4   dims;        // x = C0, y = C1, z = nin, w = nout
    int4   sel;         // x = the terms mask, y = the output texture shown, z = the textures level 0 is read from, w = 0
    float4 W[108];      // the plain path's copy of the matrix; this path reads the buffer at t3 instead
    float4 bias[5];     // likewise
    uint4  linalg;      // x = the bias's byte offset in that buffer, y = the matrix's stride in bytes, zw = 0
};

Texture2D    lat0 : register(t0);   // level 0: C0 channels (R8 / R8G8 / R8G8B8A8 / BC4 / BC5)
Texture2D    lat1 : register(t1);   // level 1: C1 channels, a quarter of the resolution per axis
Texture2D    lat0b : register(t2);  // level 0's channels past the first two, in a texture of their own
// The matrix and the bias, in ONE raw buffer. The matrix starts at byte 0, which is the base of the buffer and therefore
// far past the 128-byte alignment a thread-scope load requires, and the bias at linalg.x, which the host rounds up to 128.
ByteAddressBuffer Weights : register(t3);
SamplerState samp : register(s0);
SamplerState samp1 : register(s1);  // level 1's sampler: the SAME unbiased sampler as s0; the 1:1 mip rule lives in the gradients, through const0.z

static const uint LA_M = 18, LA_K = 24;   // the padded shape
typedef Matrix<ComponentType::F16, LA_M, LA_K, MatrixUse::A, MatrixScope::Thread> WMatrix;

struct VSInput  { float3 pos : POSITION; float2 uv : TEXCOORD0; };
struct VSOutput { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

VSOutput VSMain(VSInput input)
{
    VSOutput o;
    o.pos = mul(mvp, float4(input.pos, 1.0));
    o.uv  = input.uv;
    return o;
}

void DecodeNNTC(float4 z0, float4 z1, out float outv[18])
{
    const int C0 = dims.x, C1 = dims.y;
    float phi[24];
    [unroll] for (int i = 0; i < 24; i++) phi[i] = 0.0;
    int k = 0;
    if (sel.x & 1) { [unroll] for (int j = 0; j < 4; j++) if (j < C1) phi[k++] = z1[j]; }                       // [a]  c_j
    if (sel.x & 2) { [unroll] for (int i2 = 0; i2 < 4; i2++) if (i2 < C0) phi[k++] = z0[i2]; }                  // [b]  s_i
    if (sel.x & 4) { [unroll] for (int i3 = 0; i3 < 4; i3++) [unroll] for (int j3 = 0; j3 < 4; j3++) if (i3 < C0 && j3 < C1) phi[k++] = z0[i3] * z1[j3]; }   // [sc] s_i c_j

    // The one instruction the whole path exists for. The input is the phi this shader just built, rounded to fp16 on the
    // way in; the matrix is read as fp16 in the layout the host wrote it in; the bias and the result are whichever of fp32
    // and fp16 the device's row granted. Nothing is transposed: the format publishes W row-major and row-major here means
    // the same order, result[r] = sum over c of input[c] * matrix[stride * r + c].
    vector<half, LA_K> v;
    [unroll] for (uint c = 0; c < LA_K; c++) v[c] = (half)phi[c];
    WMatrix Wm = WMatrix::Load<LA_LAYOUT>(Weights, 0, linalg.y);
    vector<LA_SCALAR, LA_M> b = Weights.Load<vector<LA_SCALAR, LA_M> >(linalg.x);
    vector<LA_SCALAR, LA_M> r = MultiplyAdd<LA_SCALAR>(Wm, v, b);
    [unroll] for (uint o = 0; o < LA_M; o++) outv[o] = (float)r[o];   // the identity output; the caller clamps, as nntc_view.hlsl does
}

float4 PSMain(VSOutput input) : SV_Target
{
    // Two standard mipmapped texture samples (three when level 0 is in two files and t2 is bound): what a game does.
    const float2 dx = ddx(input.uv), dy = ddy(input.uv);
    float4 s0 = lat0.SampleGrad(samp, input.uv, dx, dy);
    if (sel.z == 2) s0.zw = lat0b.SampleGrad(samp, input.uv, dx, dy).rg;
    // Level 1 is a quarter of level 0's size, so left alone the GPU reads it two mips finer than level 0. Its UV gradients
    // are scaled by const0.z = 2^lod_bias_level1 (1.0 when key L turns the rule off) to keep the two in step.
    float4 s1 = lat1.SampleGrad(samp1, input.uv, dx * const0.z, dy * const0.z);
    if (const0.x > 0.5) return float4(s0.rgb, 1.0);   // key 1: level 0's texture as stored
    if (const0.y > 0.5) return float4(s1.rgb, 1.0);   // key 2: level 1's
    float4 z0 = lo0 + s0 * (hi0 - lo0);   // the dequantisation AFTER sampling
    float4 z1 = lo1 + s1 * (hi1 - lo1);
    float outv[18]; DecodeNNTC(z0, z1, outv);
    const int t = 3 * sel.y;
    float3 rgb = saturate(float3(outv[t], outv[t + 1], outv[t + 2]));
    if (const0.w > 0.5) {
        // Tangent-space normal renormalisation (key V), nntc_view.hlsl's line for line.
        float3 n = rgb * 2.0 - 1.0;
        const float len = length(n);
        if (len > 0.5 && n.z > 0.0) rgb = saturate((n / len) * 0.5 + 0.5);
    }
    return float4(rgb, 1.0);
}
