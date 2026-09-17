// view_coopvec.frag: STAGE 4, and view.frag with one line changed.
//
// Everything here is that file's: the same two or three samples through the same samplers, the same dequantisation
// after sampling, the same feature order, the same clamp, the same texture selection, the same raw modes and the same
// renormalisation. What differs is the multiply in the middle. Where view.frag writes the 18 by 24 product as two
// nested loops over the uniform block's W, this file hands it to VK_NV_cooperative_vector as ONE instruction, with the
// matrix read from a storage buffer in the device's own inferencing-optimal layout and the bias read beside it.
//
// The rest of the file is a transliteration for a reason worth restating: the feature order and the dequantisation
// rule are the format's contract (../../docs/FORMAT.md section 4 and section 6), so the two decode paths must not be
// two readings of it. A difference between the frames this file draws and the frames view.frag draws is then the fp16
// weights and nothing else, which is exactly what makes that difference measurable.
//
// THE SHAPE IS ALWAYS 18 BY 24. The shading language requires M, K and the three interpretation arguments to be
// constant expressions, and this format's nin and nout vary per asset, so the matrix is zero-padded to the largest
// shape the format allows and phi is zero past nin. A zero row contributes an output that is already ignored and a
// zero column multiplies a zero feature, so the padded product is the unpadded one in exact arithmetic.
//
// This file is compiled ONLY on a device whose cooperative-vector query passed (the extension, the feature, the
// fragment stage in cooperativeVectorSupportedStages, and a matching type tuple). On every other device the viewer
// builds view.frag alone and never mentions the extension.
#version 450
#extension GL_NV_cooperative_vector : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require

// WHICH TYPES THE MULTIPLY USES is the device's answer and not this file's, so the one place it varies is behind this
// macro, which the viewer defines from the type tuple its query found. There is exactly one shader file either way,
// because two files would be two readings of the same decode and would drift.
//
//   NNTC_CV_FP32 1 - the bias and the RESULT are fp32 and the accumulation is exact. The input vector is fp32 too,
//                    but the instruction reads it through the fp16 INPUT INTERPRETATION (the third argument below,
//                    which matches the tuple the viewer asked the device for), so phi is rounded to fp16 on the way
//                    in just as the matrix is: the difference from view.frag is the 11-bit mantissa of the weights
//                    AND of the inputs, not of the weights alone. This is what the viewer asks the device for first.
//   NNTC_CV_FP32 0 - everything is fp16. It is the fallback, and it is what NVIDIA's driver actually enumerates on the
//                    machine this was written on: no fp32 result type is offered at all. The accumulation is then the
//                    implementation's own, which is why the two paths' frames are compared and the number published
//                    rather than assumed.
#if NNTC_CV_FP32
#define CV_SCALAR float32_t
#define CV_BIAS_INTERPRETATION gl_ComponentTypeFloat32NV
#else
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define CV_SCALAR float16_t
#define CV_BIAS_INTERPRETATION gl_ComponentTypeFloat16NV
#endif

// Repeated verbatim from view.vert and view.frag, and it has to be: shaderc compiles each stage on its own, and every
// stage that declares binding 0 must declare the same block.
layout(std140, set = 0, binding = 0) uniform Constants {
    mat4  mvp;
    vec4  tex_size;
    vec4  lod_info;
    vec4  const0;
    vec4  const1;
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
layout(set = 0, binding = 5) uniform sampler   samp1;   // level 1's: the same filter plus the LOD bias

// The converted matrix and the bias, in ONE storage buffer. The element type is deliberately `uint` and not
// `float16_t`: the extension ignores the array's own scalar type and reads raw bytes according to the interpretation
// arguments below, so declaring the array as 32-bit words keeps this shader off the 16-bit storage feature it would
// otherwise have to require for nothing. The matrix begins at byte 0, which is 64-byte aligned because it is the base
// of the buffer, and the bias at BIAS_OFFSET, which the host rounds up to a multiple of 64 - both offsets are the
// extension's rules (64 for the matrix, 16 for the bias) and not preferences.
layout(set = 0, binding = 6, std430) restrict readonly buffer Weights { uint data[]; } wbuf;

// The host knows where it put the bias and the shader does not, so it arrives as a specialization constant: the size
// the driver's inferencing-optimal layout takes for an 18 by 24 fp16 matrix is the driver's business, and baking a
// guess at it into this file would be a bug waiting for a different GPU.
layout(constant_id = 0) const uint BIAS_OFFSET = 0u;

const uint M = 18u, K = 24u;      // the padded shape; literal constants, as the extension requires
const uint MATRIX_OFFSET = 0u;

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_colour;

void decode_nntc(vec4 z0, vec4 z1, out float outv[18]) {
    const int C0 = cb.dims.x, C1 = cb.dims.y;
    float phi[24];
    for (int i = 0; i < 24; i++) phi[i] = 0.0;
    int k = 0;
    if ((cb.sel.x & 1) != 0) { for (int j = 0; j < 4; j++) if (j < C1) phi[k++] = z1[j]; }                        // [a]  c_j
    if ((cb.sel.x & 2) != 0) { for (int i = 0; i < 4; i++) if (i < C0) phi[k++] = z0[i]; }                        // [b]  s_i
    if ((cb.sel.x & 4) != 0) { for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) if (i < C0 && j < C1) phi[k++] = z0[i] * z1[j]; }   // [sc] s_i c_j

    // The one instruction the whole stage exists for. The input is the phi this shader just built, converted to fp16
    // on the way in; the matrix is read as fp16; the bias and the result are whichever of fp32 and fp16 the device
    // enumerated (see the macro at the top). Nothing is
    // transposed: the format publishes W row-major and row-major here means the same order, result[j] = sum over k of
    // input[k] * matrix[stride * j + k]. The stride argument is ignored for an optimal layout.
    //
    // Writing the input a component at a time is the access pattern the specification itself warns "may be
    // suboptimal", and it is what the format's feature vector forces: phi is built by a loop with a running index.
    coopvecNV<CV_SCALAR, K> pv;
    for (uint i = 0u; i < K; ++i) pv[i] = CV_SCALAR(phi[i]);
    coopvecNV<CV_SCALAR, M> res;
    coopVecMatMulAddNV(res, pv, gl_ComponentTypeFloat16NV,
                       wbuf.data, MATRIX_OFFSET, gl_ComponentTypeFloat16NV,
                       wbuf.data, BIAS_OFFSET,   CV_BIAS_INTERPRETATION,
                       M, K, gl_CooperativeVectorMatrixLayoutInferencingOptimalNV,
                       false, 0u);
    for (uint r = 0u; r < M; ++r) outv[r] = float(res[r]);   // the identity output; the caller clamps, as view.frag does
}

void main() {
    vec4 s0 = texture(sampler2D(lat0, samp), v_uv);    // the uncompressed level 0, or a BC4 / BC5 (its channels in .rg)
    if (cb.sel.z == 2) s0.zw = texture(sampler2D(lat0b, samp), v_uv).rg;   // channels 2-3 from the second file
    vec4 s1 = texture(sampler2D(lat1, samp1), v_uv);   // level 1 through its own sampler (the LOD bias)
    if (cb.const0.x > 0.5) { out_colour = vec4(s0.rgb, 1.0); return; }   // level 0's texture as stored
    if (cb.const0.y > 0.5) { out_colour = vec4(s1.rgb, 1.0); return; }   // level 1's
    vec4 z0 = cb.lo0 + s0 * (cb.hi0 - cb.lo0);   // the dequantisation AFTER sampling
    vec4 z1 = cb.lo1 + s1 * (cb.hi1 - cb.lo1);
    float outv[18];
    decode_nntc(z0, z1, outv);
    const int t = 3 * cb.sel.y;
    vec3 rgb = clamp(vec3(outv[t], outv[t + 1], outv[t + 2]), 0.0, 1.0);
    if (cb.const0.w > 0.5) {
        // Tangent-space normal renormalisation, view.frag's line for line.
        vec3 n = rgb * 2.0 - 1.0;
        const float len = length(n);
        if (len > 0.5 && n.z > 0.0) rgb = clamp((n / len) * 0.5 + 0.5, 0.0, 1.0);
    }
    out_colour = vec4(rgb, 1.0);
}
