// view.vert: the scene's vertex shader, the transliteration of VSMain in viewer/bin/nntc_view.hlsl.
//
// The position is transformed by one matrix and the texture coordinate is passed through; everything that makes the
// picture happens in view.frag. The matrix arrives already transposed, exactly as the Direct3D viewer writes it: that
// viewer builds a row-major matrix and transposes it once at constant-buffer-write time, so the sixteen floats in the
// buffer are the matrix's COLUMNS one after another - which is what std140 means by a mat4. The two viewers therefore
// send the same bytes and multiply them the same way round, and their cameras can be compared parameter for parameter.
//
// Vulkan's clip space has y pointing down where Direct3D's points up, and the difference is paid for on the host with a
// negative viewport height rather than here with a sign, so that this file and the .hlsl stay line-for-line comparable.
#version 450

// The one uniform block, and the reason it is repeated verbatim in view.frag: shaderc compiles each stage on its own
// with no include mechanism wired up, and both stages must declare binding 0 identically. Its layout is the Direct3D
// viewer's two constant buffers laid end to end (SceneConstants then DecoderConstants), which under std140 is the same
// 2032 bytes in the same order, so the host's packing code is a copy of that viewer's.
layout(std140, set = 0, binding = 0) uniform Constants {
    mat4  mvp;          // built row-major on the host and transposed once, so these are the columns
    vec4  tex_size;     // xy = level 0's base size, zw = level 1's
    vec4  lod_info;     // x = level 0's mip count - 1, y = level 1's
    vec4  const0;       // x = show level 0 raw, y = show level 1 raw, z spare, w = renormalise the shown triple
    vec4  const1;       // spare
    vec4  lo0, hi0;     // level 0's dequantisation per channel: value = lo + sample * (hi - lo)
    vec4  lo1, hi1;     // level 1's
    ivec4 dims;         // x = C0, y = C1, z = nin, w = nout
    ivec4 sel;          // x = the terms mask, y = the output texture shown, z = level 0's texture count, w = 0
    vec4  W[108];       // W row-major: row r, column c at W[r * 6 + c / 4][c % 4]
    vec4  bias_[5];     // nout <= 18
} cb;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec2 v_uv;

void main() {
    gl_Position = cb.mvp * vec4(in_pos, 1.0);
    v_uv = in_uv;
}
