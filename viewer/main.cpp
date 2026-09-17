// nntc_view: the asset on a real Direct3D 11 sampler.
//
// Derived from the author's shader_deblocking_d3d11 sample (Apache 2.0, LICENSE at the root of this tree). It loads the
// asset nntc_encode writes (docs/FORMAT.md): the two latent .dds textures with their full mip chains (PREFIX_lat0.dds - or PREFIX_lat0a.dds
// and PREFIX_lat0b.dds - and PREFIX_lat1.dds) and PREFIX_nntc.json (the dequantisation of both and the affine decoder). The window, device,
// swap chain, camera, quad and cube, samplers, debug overlay and frame loop are the sample's; the .dds loader, the JSON reader, the
// decoder constant buffer and the shader (bin/nntc_view.hlsl) are this program's.
//
// Usage: nntc_view <PREFIX_nntc.json> [--bc]   (an uncompressed level 0 is also packed to BC4 / BC5 at load; key 4 switches to it, --bc starts there)
//        --shot FILE.bmp renders ONE frame and exits, and takes no keyboard input at all, so its bytes depend on the command line and the
//        asset and on nothing else (see the Input section).
// Controls: arrows move, W/S zoom, A/D yaw, Q/E pitch, Shift slow, C cube/quad, P/B/T point/bilinear/trilinear, X anisotropic filtering
//           on / off (it applies in trilinear mode only), N next output texture
//           (materials), M mips on / off (the sampler's MaxLOD 0: mip 0 only, nothing reloaded), L level 1's LOD bias on / off (see
//           make_level1_samplers), V renormalise the shown triple as a tangent-space normal (unpack to [-1, 1], unit length, repack;
//           grey-ish pixels (no direction) and black-ish or z < 0 pixels (into the surface) are left alone; off by default), 1 show level 0's texture as stored, 2 show level 1's, 4 level 0 from its BC4 / BC5 pack (made at load;
//           the default is what the file holds), R reload the shader, Space reset, Esc quit.
// Anisotropy: D3D11_FILTER_ANISOTROPIC with MaxAnisotropy = ANISO_MAX, on by default in trilinear mode. It is several trilinear samples
// along the footprint's long axis, and the decoder is affine in the samples, so a blend of latent samples decodes to the blend of the
// decodes and nothing about the fit has to change - the same argument that makes trilinear filtering free (docs/DESIGN.md).
// Sampling: two standard mipmapped Sample() calls, one per latent texture. Level 1's sampler carries MipLODBias = log2(block) so the
// hardware reads mip m of BOTH latents for output mip m, the 1:1 rule the encoder fitted (without it the GPU reads level 1, a quarter
// of the resolution, two mips finer than fitted: right at 1:1, splotchy colour from mip 1 down). The shader dequantises the samples
// and applies the decoder. Nothing else special.
//
// Dependencies: the C runtime, Win32, D3D11, D3DCompiler.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <cstdio>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <filesystem>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

#ifdef _DEBUG
#define ENABLE_D3D_DEBUG 1
#else
#define ENABLE_D3D_DEBUG 0
#endif

template <typename T> static void safe_release(T*& p) { if (p) { p->Release(); p = nullptr; } }

// ---------------------------------------------------------------------------
// 8x8 debug font (g_debug_font8x8_basic from encoder/basisu_enc.cpp, ASCII 32-127).
// Bit order: pixel (x,y) set if (glyph[y] >> x) & 1  (LSB = leftmost, y=0 = top).
// ---------------------------------------------------------------------------
static const uint8_t g_font8x8[96][8] = {
 { 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, { 0x18,0x3C,0x3C,0x18,0x18,0x00,0x18,0x00}, { 0x36,0x36,0x00,0x00,0x00,0x00,0x00,0x00}, { 0x36,0x36,0x7F,0x36,0x7F,0x36,0x36,0x00},
 { 0x0C,0x3E,0x03,0x1E,0x30,0x1F,0x0C,0x00}, { 0x00,0x63,0x33,0x18,0x0C,0x66,0x63,0x00}, { 0x1C,0x36,0x1C,0x6E,0x3B,0x33,0x6E,0x00}, { 0x06,0x06,0x03,0x00,0x00,0x00,0x00,0x00},
 { 0x18,0x0C,0x06,0x06,0x06,0x0C,0x18,0x00}, { 0x06,0x0C,0x18,0x18,0x18,0x0C,0x06,0x00}, { 0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00}, { 0x00,0x0C,0x0C,0x3F,0x0C,0x0C,0x00,0x00},
 { 0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x06}, { 0x00,0x00,0x00,0x3F,0x00,0x00,0x00,0x00}, { 0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x00}, { 0x60,0x30,0x18,0x0C,0x06,0x03,0x01,0x00},
 { 0x3E,0x63,0x73,0x7B,0x6F,0x67,0x3E,0x00}, { 0x0C,0x0E,0x0C,0x0C,0x0C,0x0C,0x3F,0x00}, { 0x1E,0x33,0x30,0x1C,0x06,0x33,0x3F,0x00}, { 0x1E,0x33,0x30,0x1C,0x30,0x33,0x1E,0x00},
 { 0x38,0x3C,0x36,0x33,0x7F,0x30,0x78,0x00}, { 0x3F,0x03,0x1F,0x30,0x30,0x33,0x1E,0x00}, { 0x1C,0x06,0x03,0x1F,0x33,0x33,0x1E,0x00}, { 0x3F,0x33,0x30,0x18,0x0C,0x0C,0x0C,0x00},
 { 0x1E,0x33,0x33,0x1E,0x33,0x33,0x1E,0x00}, { 0x1E,0x33,0x33,0x3E,0x30,0x18,0x0E,0x00}, { 0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x00}, { 0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x06},
 { 0x18,0x0C,0x06,0x03,0x06,0x0C,0x18,0x00}, { 0x00,0x00,0x3F,0x00,0x00,0x3F,0x00,0x00}, { 0x06,0x0C,0x18,0x30,0x18,0x0C,0x06,0x00}, { 0x1E,0x33,0x30,0x18,0x0C,0x00,0x0C,0x00},
 { 0x3E,0x63,0x7B,0x7B,0x7B,0x03,0x1E,0x00}, { 0x0C,0x1E,0x33,0x33,0x3F,0x33,0x33,0x00}, { 0x3F,0x66,0x66,0x3E,0x66,0x66,0x3F,0x00}, { 0x3C,0x66,0x03,0x03,0x03,0x66,0x3C,0x00},
 { 0x1F,0x36,0x66,0x66,0x66,0x36,0x1F,0x00}, { 0x7F,0x46,0x16,0x1E,0x16,0x46,0x7F,0x00}, { 0x7F,0x46,0x16,0x1E,0x16,0x06,0x0F,0x00}, { 0x3C,0x66,0x03,0x03,0x73,0x66,0x7C,0x00},
 { 0x33,0x33,0x33,0x3F,0x33,0x33,0x33,0x00}, { 0x1E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, { 0x78,0x30,0x30,0x30,0x33,0x33,0x1E,0x00}, { 0x67,0x66,0x36,0x1E,0x36,0x66,0x67,0x00},
 { 0x0F,0x06,0x06,0x06,0x46,0x66,0x7F,0x00}, { 0x63,0x77,0x7F,0x7F,0x6B,0x63,0x63,0x00}, { 0x63,0x67,0x6F,0x7B,0x73,0x63,0x63,0x00}, { 0x1C,0x36,0x63,0x63,0x63,0x36,0x1C,0x00},
 { 0x3F,0x66,0x66,0x3E,0x06,0x06,0x0F,0x00}, { 0x1E,0x33,0x33,0x33,0x3B,0x1E,0x38,0x00}, { 0x3F,0x66,0x66,0x3E,0x36,0x66,0x67,0x00}, { 0x1E,0x33,0x07,0x0E,0x38,0x33,0x1E,0x00},
 { 0x3F,0x2D,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, { 0x33,0x33,0x33,0x33,0x33,0x33,0x3F,0x00}, { 0x33,0x33,0x33,0x33,0x33,0x1E,0x0C,0x00}, { 0x63,0x63,0x63,0x6B,0x7F,0x77,0x63,0x00},
 { 0x63,0x63,0x36,0x1C,0x1C,0x36,0x63,0x00}, { 0x33,0x33,0x33,0x1E,0x0C,0x0C,0x1E,0x00}, { 0x7F,0x63,0x31,0x18,0x4C,0x66,0x7F,0x00}, { 0x1E,0x06,0x06,0x06,0x06,0x06,0x1E,0x00},
 { 0x03,0x06,0x0C,0x18,0x30,0x60,0x40,0x00}, { 0x1E,0x18,0x18,0x18,0x18,0x18,0x1E,0x00}, { 0x08,0x1C,0x36,0x63,0x00,0x00,0x00,0x00}, { 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF},
 { 0x0C,0x0C,0x18,0x00,0x00,0x00,0x00,0x00}, { 0x00,0x00,0x1E,0x30,0x3E,0x33,0x6E,0x00}, { 0x07,0x06,0x06,0x3E,0x66,0x66,0x3B,0x00}, { 0x00,0x00,0x1E,0x33,0x03,0x33,0x1E,0x00},
 { 0x38,0x30,0x30,0x3E,0x33,0x33,0x6E,0x00}, { 0x00,0x00,0x1E,0x33,0x3F,0x03,0x1E,0x00}, { 0x1C,0x36,0x06,0x0F,0x06,0x06,0x0F,0x00}, { 0x00,0x00,0x6E,0x33,0x33,0x3E,0x30,0x1F},
 { 0x07,0x06,0x36,0x6E,0x66,0x66,0x67,0x00}, { 0x0C,0x00,0x0E,0x0C,0x0C,0x0C,0x1E,0x00}, { 0x30,0x00,0x30,0x30,0x30,0x33,0x33,0x1E}, { 0x07,0x06,0x66,0x36,0x1E,0x36,0x67,0x00},
 { 0x0E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, { 0x00,0x00,0x33,0x7F,0x7F,0x6B,0x63,0x00}, { 0x00,0x00,0x1F,0x33,0x33,0x33,0x33,0x00}, { 0x00,0x00,0x1E,0x33,0x33,0x33,0x1E,0x00},
 { 0x00,0x00,0x3B,0x66,0x66,0x3E,0x06,0x0F}, { 0x00,0x00,0x6E,0x33,0x33,0x3E,0x30,0x78}, { 0x00,0x00,0x3B,0x6E,0x66,0x06,0x0F,0x00}, { 0x00,0x00,0x3E,0x03,0x1E,0x30,0x1F,0x00},
 { 0x08,0x0C,0x3E,0x0C,0x0C,0x2C,0x18,0x00}, { 0x00,0x00,0x33,0x33,0x33,0x33,0x6E,0x00}, { 0x00,0x00,0x33,0x33,0x33,0x1E,0x0C,0x00}, { 0x00,0x00,0x63,0x6B,0x7F,0x7F,0x36,0x00},
 { 0x00,0x00,0x63,0x36,0x1C,0x36,0x63,0x00}, { 0x00,0x00,0x33,0x33,0x33,0x3E,0x30,0x1F}, { 0x00,0x00,0x3F,0x19,0x0C,0x26,0x3F,0x00}, { 0x38,0x0C,0x0C,0x07,0x0C,0x0C,0x38,0x00},
 { 0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x00}, { 0x07,0x0C,0x0C,0x38,0x0C,0x0C,0x07,0x00}, { 0x6E,0x3B,0x00,0x00,0x00,0x00,0x00,0x00}, { 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
};

// ---------------------------------------------------------------------------
// Config / state
// ---------------------------------------------------------------------------
static int   WINDOW_WIDTH  = 2560;   // wide enough that the overlay's first line fits at 16 px per character
static int   WINDOW_HEIGHT = 1440;
static const float FOV_DEGREES = 90.0f;
static const float Z_MIN = 0.40f, Z_MAX = -50.0f;
static const float Z_SPEED = 1.0f, XY_SPEED = 0.75f, ROT_SPEED = 90.0f;

static ID3D11Device*            g_dev = nullptr;
static ID3D11DeviceContext*     g_ctx = nullptr;
static IDXGISwapChain*          g_swapchain = nullptr;
static ID3D11RenderTargetView*  g_rtv = nullptr;
static ID3D11Texture2D*         g_depth_tex = nullptr;
static ID3D11DepthStencilView*  g_dsv = nullptr;
// The sampler sets. [mips on / off][0=point, 1=bilinear, 2=trilinear, 3=anisotropic]: the MaxLOD = 0 row is key M (the
// hardware held at mip 0 of both textures, nothing reloaded), and slot 3 is the trilinear filter with anisotropy, which
// key X turns on and off. Both sets carry all four, so the LOD bias and the mips toggle keep working under anisotropy.
static const int FILTER_STATES = 4;
static const UINT ANISO_MAX = 8;   // the anisotropy asked for when it is on; one line to change
static ID3D11SamplerState*      g_samplers1[2][FILTER_STATES] = {};  // level 1's samplers: the same with MipLODBias = log2(block) (key L: the 1:1 mip rule the encoder fitted)
static ID3D11SamplerState*      g_samplers[2][FILTER_STATES] = {};   // level 0's
static ID3D11RasterizerState*   g_raster = nullptr;
static ID3D11DepthStencilState* g_depth_on = nullptr, * g_depth_off = nullptr;
static ID3D11BlendState*        g_blend_alpha = nullptr;
static ID3D11Buffer*            g_cbuffer = nullptr;
static ID3D11Buffer*            g_dec_cbuffer = nullptr;   // the decoder constants (b1)
static bool                     g_quit = false;
static bool                     g_bc = false;   // the level-0 textures bound: false = the uncompressed .dds (the default), true = the BC4 / BC5 pack made at load (key 4; --bc starts packed)

// Level 1's sampler set: the same three filters with MipLODBias = log2(block), so the hardware picks level 1's mip as the encoder's 1:1
// rule does (output mip m reads mip m of both latents). Level 1 has a quarter of the texels per axis (block 4), so its natural LOD is 2 below
// level 0's; the bias of +2 lines them up (at 1:1 on screen its LOD becomes 0, still mip 0; at level 0's mip 3 it reads its own mip 3).
// The bias is part of the format's contract (docs/FORMAT.md section 5), so a sampler that could not be created is a
// failure of the load and not something to render around: the caller stops.
static bool make_level1_samplers(int block, float bias) {
    for (int mips = 0; mips < 2; mips++) for (int f = 0; f < FILTER_STATES; f++) {
        safe_release(g_samplers1[mips][f]);
        if (!g_samplers[mips][f]) continue;
        D3D11_SAMPLER_DESC sd{}; g_samplers[mips][f]->GetDesc(&sd); sd.MipLODBias = bias;
        if (FAILED(g_dev->CreateSamplerState(&sd, &g_samplers1[mips][f]))) { fprintf(stderr, "ERROR: level 1's sampler state could not be created\n"); return false; }
    }
    printf("  level 1 sampler: MipLODBias +%.0f (the JSON's lod_bias_level1, log2 of block %d): the 1:1 mip rule; key L toggles it\n", bias, block);
    return true;
}

// The asset: one latent texture per level.
struct LatentTex {
    ID3D11Texture2D*          tex = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11Texture2D*          bc_tex[2] = { nullptr, nullptr };   // level 0 packed: channels 0-1 (BC5, or BC4 for one channel) and what is left (a fourth channel makes a BC5, a third alone a BC4), made at load
    ID3D11ShaderResourceView* bc_srv[2] = { nullptr, nullptr };
    int bc_n = 0; int bc_nc[2] = { 0, 0 }; double bc_psnr = 0, bc_maxerr = 0; bool bc_lossless = false;   // the packed texture count, each one's channel count, the packing quality over the chain, the round trip's largest error
    std::vector<std::vector<uint8_t>> raw; std::vector<std::pair<int, int>> dims;   // the .dds levels as read (for the pack)
    ID3D11Texture2D*          tex_b = nullptr;   // the SECOND file of a two-file level 0 (channels 2-3 as a BC5, or channel 2 alone as a BC4), bound to t2
    ID3D11ShaderResourceView* srv_b = nullptr;
    int file_files = 1;    // the .dds files this level is stored in: one, or two for a BC level 0 of three or four channels
    bool file_bc = false;  // the level came block-compressed in the file, so the viewer makes no pack of its own
    int W = 0, H = 0, C = 0, stored_C = 0, mips = 0, dxgi = 0;
    // The SECOND file's own format and channel count, which are not the first's: a three-channel level 0 is a BC5 of
    // channels 0-1 and a BC4 of channel 2 alone, so the two files differ in both. Zero when there is only one file.
    int stored_C_b = 0, dxgi_b = 0;
    float lo[4] = {0,0,0,0}, hi[4] = {1,1,1,1};
    int bits[4] = {8,8,8,8};
    std::string file;
};
// The decoder constant buffer: must match cbuffer DecoderConstants in nntc_view.hlsl (16-byte rules).
struct DecCB {
    float lo0[4], hi0[4], lo1[4], hi1[4];
    int   dims[4];        // C0, C1, nin, nout
    int   sel[4];         // terms mask, output texture shown, 0, 0
    float W[108][4];      // row r, column c at W[r * 6 + c / 4][c % 4]: six float4 per row, MAX_TEXTURES * 3 rows
    float bias[5][4];     // nout <= 18, so five float4 cover it
};

struct State {
    float x = 0, y = 0, z = -3.0f, yaw = 0, pitch = 0;
    bool  cube = false;
    int   filter_mode = 2;
    bool  aniso = true;      // key X: anisotropic filtering, which applies in TRILINEAR mode only (the point and bilinear states are what they say they are)
    bool  lod_bias = true;   // key L: level 1 sampled with MipLODBias = log2(block) (the encoder's 1:1 rule); off = the plain sampler (the GPU reads level 1 two mips finer than fitted)
    bool  mips_on = true;   // key M: false = the sampler's MaxLOD is 0, so every fetch reads mip 0 (the textures unchanged)
    ID3D11VertexShader*  vs = nullptr;
    ID3D11PixelShader*   ps = nullptr;
    ID3D11InputLayout*   layout = nullptr;
    LatentTex lat[2];
    DecCB  dec{};
    int    textures_out = 1, tex_shown = 0;    // a material decodes 3 * textures_out channels; the shader shows one triple
    int    src_w = 0, src_h = 0, block = 4;
    std::string terms;
    float  const0[4] = {0,0,0,0};
    float  const1[4] = {0,0,0,0};
    ID3D11VertexShader*       debug_vs = nullptr;
    ID3D11PixelShader*        debug_ps = nullptr;
    ID3D11InputLayout*        debug_layout = nullptr;
    ID3D11Buffer*             debug_vb = nullptr, * debug_ib = nullptr;
    ID3D11Texture2D*          debug_tex = nullptr;
    ID3D11ShaderResourceView* debug_srv = nullptr;
    bool   debug_dirty = true;
    std::string shader_path;
};
static State g;

static const char* filter_name(int m) { return m == 0 ? "POINT" : (m == 2 ? "TRILINEAR" : "BILINEAR"); }

// The sampler slot a filter state picks: anisotropy is the trilinear filter with several samples along the footprint's
// long axis, so it is a fourth state of the trilinear mode and not a mode of its own. In point and bilinear the flag is
// remembered and does nothing, which is what the overlay says.
static int filter_slot() { return (g.filter_mode == 2 && g.aniso) ? 3 : g.filter_mode; }

// ---------------------------------------------------------------------------
// 4x4 matrices (row-major with column-vector convention; one transpose at cbuffer-write time).
// ---------------------------------------------------------------------------
struct Mat4 { float m[16]; };
static Mat4 mat_identity() { Mat4 r{}; for (int i=0;i<4;i++) r.m[i*4+i]=1.0f; return r; }
static Mat4 mat_mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) { float s=0; for(int k=0;k<4;k++) s+=a.m[i*4+k]*b.m[k*4+j]; r.m[i*4+j]=s; }
    return r;
}
static Mat4 mat_perspective(float fov_deg, float aspect, float znear, float zfar) {
    Mat4 m{};
    float f = 1.0f / std::tan((fov_deg * 3.14159265358979f / 180.0f) / 2.0f);
    m.m[0*4+0] = f / aspect;
    m.m[1*4+1] = f;
    m.m[2*4+2] = zfar / (znear - zfar);
    m.m[2*4+3] = (zfar * znear) / (znear - zfar);
    m.m[3*4+2] = -1.0f;
    return m;
}
static Mat4 mat_translate(float x, float y, float z) { Mat4 m = mat_identity(); m.m[0*4+3]=x; m.m[1*4+3]=y; m.m[2*4+3]=z; return m; }
static Mat4 mat_rot_y(float deg) {
    Mat4 m = mat_identity(); float r=deg*3.14159265358979f/180.0f, c=std::cos(r), s=std::sin(r);
    m.m[0*4+0]=c; m.m[0*4+2]=s; m.m[2*4+0]=-s; m.m[2*4+2]=c; return m;
}
static Mat4 mat_rot_x(float deg) {
    Mat4 m = mat_identity(); float r=deg*3.14159265358979f/180.0f, c=std::cos(r), s=std::sin(r);
    m.m[1*4+1]=c; m.m[1*4+2]=-s; m.m[2*4+1]=s; m.m[2*4+2]=c; return m;
}
static Mat4 mat_transpose(const Mat4& a) { Mat4 r{}; for (int i=0;i<4;i++) for (int j=0;j<4;j++) r.m[i*4+j]=a.m[j*4+i]; return r; }

// ---------------------------------------------------------------------------
// Shader loading (bin/nntc_view.hlsl, compiled at runtime: edit and press R).
// ---------------------------------------------------------------------------
static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}
// A path from argv is in the process's ANSI code page, so widening it one char at a time turns every byte above 127
// into the wrong character (and a two-byte character into two of them). D3DCompileFromFile wants wide, so it is
// converted properly.
static std::wstring widen(const std::string& s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_ACP, 0, s.c_str(), (int)s.size(), nullptr, 0);
    if (n <= 0) return std::wstring(s.begin(), s.end());
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}
static ID3DBlob* compile_hlsl_file(const std::string& path, const char* entry, const char* target) {
    const std::wstring wpath = widen(path);
    ID3DBlob* code = nullptr, * errors = nullptr;
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
    HRESULT hr = D3DCompileFromFile(wpath.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, target, flags, 0, &code, &errors);
    if (errors) { fprintf(stderr, "%s (%s):\n%s\n", FAILED(hr) ? "SHADER ERROR" : "SHADER WARNING", entry, (const char*)errors->GetBufferPointer()); errors->Release(); }
    if (FAILED(hr)) { safe_release(code); return nullptr; }
    return code;
}
static bool load_shader(const std::string& path) {
    ID3DBlob* vsb = compile_hlsl_file(path, "VSMain", "vs_5_0");
    if (!vsb) return false;
    ID3DBlob* psb = compile_hlsl_file(path, "PSMain", "ps_5_0");
    if (!psb) { vsb->Release(); return false; }
    ID3D11VertexShader* vs = nullptr; ID3D11PixelShader* ps = nullptr; ID3D11InputLayout* layout = nullptr;
    bool ok = SUCCEEDED(g_dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs)) &&
              SUCCEEDED(g_dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps));
    if (ok) {
        const D3D11_INPUT_ELEMENT_DESC il[] = {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };
        ok = SUCCEEDED(g_dev->CreateInputLayout(il, 2, vsb->GetBufferPointer(), vsb->GetBufferSize(), &layout));
    }
    vsb->Release(); psb->Release();
    if (!ok) { safe_release(vs); safe_release(ps); safe_release(layout); return false; }
    safe_release(g.vs); safe_release(g.ps); safe_release(g.layout);
    g.vs = vs; g.ps = ps; g.layout = layout;
    printf("Shader compiled successfully.\n");
    return true;
}
static void reload_shader() { if (!load_shader(g.shader_path)) fprintf(stderr, "Shader reload failed, keeping previous shader.\n"); }

#include "nntc_json.h"   // the JSON reader, in shared/ and shared with bc_check.cpp and the encoder
#include "bc_pack.h"   // the BC4 / BC5 encoder, decoder and level packer (validated by bc_check.cpp against bcdec)
#include "dds.h"   // the DX10 .dds parsing, in shared/ because the Vulkan viewer reads the same files through it

// Level 0's packed textures from its .dds levels, in bc_pack.h's own file rule: channels 0-1 into a BC5 (a BC4 when that
// is all there is), and what is left into a second texture - a BC5 for a fourth channel, a BC4 for a third one alone. It
// is the same split the encoder writes, so key 4 shows the layout the file format would have given the same plane.
static bool bc_pack_level0(LatentTex& L) {
    const int C = L.C; L.bc_n = bc_level0_files(C);
    if ((L.W % 4) || (L.H % 4)) { fprintf(stderr, "WARNING: level 0 is %dx%d, not a multiple of 4: no BC pack\n", L.W, L.H); L.bc_n = 0; return false; }
    double se_all = 0; size_t n_all = 0; bool lossless = true;
    printf("  level 0 packed at load (key 4 switches to it): ");
    for (int p = 0; p < L.bc_n; p++) {
        const int c0 = 2 * p, nc = bc_level0_file_channels(C, p);
        L.bc_nc[p] = nc;
        if (c0 + nc > L.stored_C) { fprintf(stderr, "\nERROR: level 0 stores %d channels, %d needed for the pack\n", L.stored_C, c0 + nc); L.bc_n = 0; return false; }
        printf("%s%s (channel%s %d%s, %d bit%s)", p ? " + " : "", nc == 1 ? "BC4" : "BC5", nc == 1 ? "" : "s", c0, nc == 1 ? "" : (" and " + std::to_string(c0 + 1)).c_str(), L.bits[c0], L.bits[c0] == 1 ? "" : "s");
        for (int c = 0; c < nc; c++) if (L.bits[c0 + c] > 3) lossless = false;   // every packed channel is one the layout uses
        std::vector<std::vector<uint8_t>> packed(L.mips); std::vector<D3D11_SUBRESOURCE_DATA> subs(L.mips);
        for (int i = 0; i < L.mips; i++) {
            double se = 0; size_t n = 0;
            packed[i] = bc_pack_level(L.raw[i].data(), L.dims[i].first, L.dims[i].second, L.stored_C, c0, nc, L.bits, C, se, n);
            {   // The round trip: every packed block decoded on the CPU (the hardware palette) against the exact index value of the source.
                // One decode per BLOCK and then its sixteen texels, rather than a decode per texel, which decoded every block sixteen times.
                const int W = L.dims[i].first, H = L.dims[i].second, bx = (W + 3) / 4, by = (H + 3) / 4;
                for (int byi = 0; byi < by; byi++) for (int bxi = 0; bxi < bx; bxi++) for (int c = 0; c < nc; c++) {
                    float dec[16]; bc4_decode_block(&packed[i][((size_t)byi * bx + bxi) * (nc == 1 ? 8 : 16) + c * 8], dec);
                    const int bits_c = L.bits[c0 + c];
                    for (int t = 0; t < 16; t++) {
                        const int x = bxi * 4 + t % 4, y = byi * 4 + t / 4; if (x >= W || y >= H) continue;
                        const int k = L.raw[i][((size_t)y * W + x) * L.stored_C + c0 + c] >> (8 - bits_c);
                        L.bc_maxerr = std::max(L.bc_maxerr, std::fabs(dec[t] - k * 255.0 / ((1 << bits_c) - 1)));
                    }
                }
            }
            subs[i].pSysMem = packed[i].data(); subs[i].SysMemPitch = (UINT)(((L.dims[i].first + 3) / 4) * (nc == 1 ? 8 : 16)); subs[i].SysMemSlicePitch = 0;
            se_all += se; n_all += n;
        }
        D3D11_TEXTURE2D_DESC td{};
        td.Width = (UINT)L.W; td.Height = (UINT)L.H; td.MipLevels = (UINT)L.mips; td.ArraySize = 1;
        td.Format = nc == 1 ? DXGI_FORMAT_BC4_UNORM : DXGI_FORMAT_BC5_UNORM; td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_IMMUTABLE; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(g_dev->CreateTexture2D(&td, subs.data(), &L.bc_tex[p])) || FAILED(g_dev->CreateShaderResourceView(L.bc_tex[p], nullptr, &L.bc_srv[p]))) {
            fprintf(stderr, "\nERROR: the BC texture creation failed\n");
            for (int q = 0; q < 2; q++) { safe_release(L.bc_srv[q]); safe_release(L.bc_tex[q]); }   // the first texture too, when it is the second that failed
            L.bc_n = 0; return false;
        }
    }
    L.bc_lossless = lossless; L.bc_psnr = n_all && se_all > 0 ? 10.0 * std::log10(255.0 * 255.0 / (se_all / n_all)) : 0.0;
    if (lossless) printf(": lossless (0 and 255 as the endpoints, the index in the selectors; the round trip's largest error %.3g / 255 in float)\n", L.bc_maxerr); else printf(": lossy, packing PSNR %.2f dB over the chain against the exact index values (largest error %.1f / 255)\n", L.bc_psnr, L.bc_maxerr);
    return true;
}

// ---------------------------------------------------------------------------
// The DX10 .dds loader (docs/FORMAT.md section 2): the 148-byte header, the levels base-first and tightly packed, into an immutable
// mipmapped texture and its SRV. Uncompressed R8 / R8G8 / R8G8B8A8, and block-compressed BC4_UNORM (80) / BC5_UNORM (83), which is
// how the encoder writes level 0 by default: a level is then ceil(w/4) x ceil(h/4) blocks of 8 or 16 bytes, so the subresource pitch is
// the block ROW's byte count and not a texel row's. `second` loads the SECOND file of a two-file level 0 into the same LatentTex, which
// is bound to the slot the load-time pack already uses for channels 2-3.
//
// The parsing itself is shared/dds.h, which is this function's own header walk lifted out so that the Vulkan viewer reads the same
// bytes through the same checks; what is left here is the Direct3D tail, which is all that was ever api-specific.
// ---------------------------------------------------------------------------
static bool load_dds(const std::string& path, LatentTex& L, bool second = false) {
    DdsImage img;
    if (!dds_read(path, img)) return false;
    const int W = img.W, H = img.H, nmip = img.mips, dxgi = img.dxgi, C = img.channels;
    const bool bc = img.bc;
    // The two files of one level share the extent and the level count, and NOT the format: a three-channel level 0 is a
    // BC5 of channels 0-1 and a BC4 of channel 2, so only that both are block-compressed (or neither is) is required.
    if (second && (W != L.W || H != L.H || nmip != L.mips)) { fprintf(stderr, "ERROR: '%s' does not match the first file of this level (%dx%d, %d level%s)\n", path.c_str(), L.W, L.H, L.mips, L.mips == 1 ? "" : "s"); return false; }
    if (second && bc != L.file_bc) { fprintf(stderr, "ERROR: '%s' is %s where the first file of this level is not\n", path.c_str(), bc ? "block-compressed" : "uncompressed"); return false; }
    std::vector<D3D11_SUBRESOURCE_DATA> subs(nmip);
    std::vector<std::pair<int, int>> level_dim(nmip);
    for (int i = 0; i < nmip; i++) {
        subs[i].pSysMem = img.bytes.data() + img.levels[i].offset; subs[i].SysMemPitch = (UINT)img.levels[i].pitch; subs[i].SysMemSlicePitch = 0;
        level_dim[i] = { img.levels[i].w, img.levels[i].h };
    }
    DXGI_FORMAT fmt = (DXGI_FORMAT)dxgi;
    D3D11_TEXTURE2D_DESC td{};
    td.Width = (UINT)W; td.Height = (UINT)H; td.MipLevels = (UINT)nmip; td.ArraySize = 1;
    td.Format = fmt; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_IMMUTABLE; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    ID3D11Texture2D* tex = nullptr; ID3D11ShaderResourceView* srv = nullptr;
    HRESULT hr = g_dev->CreateTexture2D(&td, subs.data(), &tex);
    if (FAILED(hr)) { fprintf(stderr, "ERROR: CreateTexture2D failed for '%s' (0x%08X)\n", path.c_str(), (unsigned)hr); return false; }
    hr = g_dev->CreateShaderResourceView(tex, nullptr, &srv);
    if (FAILED(hr)) { tex->Release(); fprintf(stderr, "ERROR: CreateShaderResourceView failed (0x%08X)\n", (unsigned)hr); return false; }
    // A second file replacing a first one that is already there releases it: the two are loaded in sequence, and a
    // failure between them used to leave the first leaked.
    if (second) { safe_release(L.srv_b); safe_release(L.tex_b); L.tex_b = tex; L.srv_b = srv; L.stored_C_b = C; L.dxgi_b = dxgi; }
    else {
        safe_release(L.srv); safe_release(L.tex);
        L.tex = tex; L.srv = srv; L.W = W; L.H = H; L.stored_C = C; L.mips = nmip; L.dxgi = (int)fmt; L.file = path; L.file_bc = bc;
        L.raw.clear(); L.dims = level_dim;
        if (!bc) { L.raw.resize(nmip); for (int i = 0; i < nmip; i++) L.raw[i].assign(img.bytes.begin() + img.levels[i].offset, img.bytes.begin() + img.levels[i].offset + img.levels[i].size); }   // kept for the pack
    }
    printf("  %s: %dx%d, %d mip level%s, DXGI format %d (%s, %d channel%s stored)\n", path.c_str(), W, H, nmip, nmip == 1 ? "" : "s", dxgi, bc ? (dxgi == 80 ? "BC4_UNORM" : "BC5_UNORM") : "uncompressed", C, C == 1 ? "" : "s");
    return true;
}

// ---------------------------------------------------------------------------
// The asset: PREFIX_nntc.json, then the two .dds files it names (next to it), then the decoder constants. The path is
// taken as given - the viewer derives no name of its own - so an asset written under any descriptor name loads.
// ---------------------------------------------------------------------------
static bool load_asset(const std::string& json_path) {
    const std::string text = read_file(json_path);
    if (text.empty()) { fprintf(stderr, "ERROR: cannot read '%s'\n", json_path.c_str()); return false; }
    JParser P(text); JVal root = P.parse();
    if (!P.ok || root.kind != JVal::OBJ) { fprintf(stderr, "ERROR: '%s' is not valid JSON\n", json_path.c_str()); return false; }
    // Assets written before the project was renamed carry the older format string; they are the same format byte for
    // byte, so the viewer opens them and says so once rather than refusing what it can read. It goes to stdout with the
    // rest of the load's running commentary - it is not a diagnostic the caller has to act on - and ahead of the
    // "Loading" line, which is only printed once the descriptor is known to be one this viewer reads.
    if (root.string("format") == "ntc-dds-1") printf("note: format 'ntc-dds-1' is the older name of nntc-dds-1; reading it as such\n");
    else if (root.string("format") != "nntc-dds-1") { fprintf(stderr, "ERROR: unknown format '%s' (expected nntc-dds-1)\n", root.string("format").c_str()); return false; }
    printf("Loading %s\n", json_path.c_str());
    const size_t slash = json_path.find_last_of("/\\"); const std::string dir = slash == std::string::npos ? "" : json_path.substr(0, slash + 1);
    // A file the descriptor names is taken as written when it is an absolute path, and from the descriptor's own
    // directory otherwise: the encoder writes base names, and a hand-authored descriptor may point anywhere.
    const auto beside = [&dir](const std::string& name) {
        const std::filesystem::path p(name);
        return p.is_absolute() ? name : dir + name;
    };
    const JVal* src = root.get("source"); const JVal* decs = root.get("decode");
    g.src_w = src ? (int)src->number("width") : 0; g.src_h = src ? (int)src->number("height") : 0;
    g.textures_out = decs ? (int)decs->number("textures_out", 1) : 1; g.block = (int)root.number("block", 4);
    const JVal* texs = root.get("textures");
    if (!texs || texs->kind != JVal::ARR || texs->arr.size() != 2) { fprintf(stderr, "ERROR: the JSON must describe two textures\n"); return false; }
    for (int l = 0; l < 2; l++) {
        const JVal& t = texs->arr[l]; LatentTex& L = g.lat[l];
        if ((int)t.number("level", -1) != l) { fprintf(stderr, "ERROR: textures[%d] is not level %d\n", l, l); return false; }
        L.C = (int)t.number("channels_used");
        // bits_per_channel decides a shift (byte >> (8 - bits)) and a divisor (2^bits - 1), so 0 or a value above 8
        // would be a negative shift - undefined - or a division by zero. The entries are checked for being numbers at
        // all, too: a JSON that carried a string there would otherwise read as .num = 0.
        const JVal* bits = t.get("bits_per_channel");
        for (int c = 0; c < 4; c++) L.bits[c] = bits && bits->kind == JVal::ARR && c < (int)bits->arr.size() && bits->arr[c].kind == JVal::NUM ? (int)bits->arr[c].num : 8;
        for (int c = 0; c < 4; c++) {
            const int lo_bits = l == 0 ? 1 : 4;   // level 0 is written at 8 bits under --l0 bc8 (the default) and at 1-4 under --l0 palette; level 1 at 4-8
            if (L.bits[c] < lo_bits || L.bits[c] > 8) { fprintf(stderr, "ERROR: level %d: bits_per_channel[%d] is %d, which is outside %d..8\n", l, c, L.bits[c], lo_bits); return false; }
        }
        // "file" names one .dds and "files" names two, which is how a block-compressed level 0 of three or four channels is stored:
        // BC5 carries two channels, so channels 0-1 are the first file and what is left is the second, and the second is bound to t2 -
        // the slot the viewer's own load-time pack already uses for those channels, so the shader is the same either way. Each entry of
        // "files" is an OBJECT carrying that file's own dxgi_format_id and channels_stored, because the two need not agree: three
        // channels are a BC5 and then a BC4 of channel 2 alone. The values are checked against the .dds headers rather than trusted.
        //
        // An entry that is a BARE STRING is the older spelling of the same key - every asset written before the two files were allowed
        // to differ in format carries one - and is read as a name with nothing to cross-check, which is what viewer/bc_check.cpp and
        // tools/dds_decode.py have always done. Refusing it here made every earlier three- or four-channel asset unopenable.
        const JVal* files = t.get("files");
        if (files && files->kind == JVal::ARR) {
            if (files->arr.size() != 2) { fprintf(stderr, "ERROR: level %d: \"files\" must name two .dds files\n", l); return false; }
            std::string names[2];
            int want_dxgi[2] = { 0, 0 }, want_stored[2] = { 0, 0 };
            for (int i = 0; i < 2; i++) {
                const JVal& e = files->arr[i];
                if (e.kind != JVal::OBJ && e.kind != JVal::STR) { fprintf(stderr, "ERROR: level %d: \"files\"[%d] must be a file name or an object carrying one\n", l, i); return false; }
                names[i] = e.kind == JVal::OBJ ? e.string("file") : e.str;
                if (names[i].empty()) { fprintf(stderr, "ERROR: level %d: \"files\"[%d] names no file\n", l, i); return false; }
                want_dxgi[i] = e.kind == JVal::OBJ ? (int)e.number("dxgi_format_id", 0) : 0;      // 0 = the older spelling: nothing published to cross-check
                want_stored[i] = e.kind == JVal::OBJ ? (int)e.number("channels_stored", 0) : 0;
            }
            if (!load_dds(beside(names[0]), L) || !load_dds(beside(names[1]), L, true)) return false;
            L.file_files = 2;
            const int got_dxgi[2] = { L.dxgi, L.dxgi_b }, got_stored[2] = { L.stored_C, L.stored_C_b };
            for (int i = 0; i < 2; i++)
                if ((want_dxgi[i] && want_dxgi[i] != got_dxgi[i]) || (want_stored[i] && want_stored[i] != got_stored[i])) {
                    fprintf(stderr, "ERROR: level %d: the JSON says %s is DXGI %d storing %d channel(s) and the file is DXGI %d storing %d\n",
                            l, names[i].c_str(), want_dxgi[i], want_stored[i], got_dxgi[i], got_stored[i]);
                    return false;
                }
        } else if (!load_dds(beside(t.string("file")), L)) return false;
        // The channels the level's files hold between them: one file stores stored_C, two store stored_C + stored_C_b (2 + 1 for a
        // three-channel BC level 0, which is exactly the three the decoder reads).
        const int stored_all = L.stored_C + (L.file_files > 1 ? L.stored_C_b : 0);
        if (L.C < 1 || L.C > 4 || L.C > stored_all) { fprintf(stderr, "ERROR: level %d: %d channels used of %d stored\n", l, L.C, stored_all); return false; }
        // The load-time pack, for an uncompressed level 0 only; key 4 binds it. It is not fatal when it fails - the
        // file's own plane is still there and is what the viewer shows - but it says so rather than leaving key 4 dead
        // with no explanation, and the raw levels it needed are dropped once it has run.
        if (l == 0 && !L.file_bc) {
            if (!bc_pack_level0(L)) fprintf(stderr, "WARNING: level 0 was not packed at load; key 4 has nothing to bind\n");
            L.raw.clear(); L.raw.shrink_to_fit();
        }
        // Level 1's LOD-biased samplers (key L). The bias is the JSON's own lod_bias_level1, which is what the format
        // says the sampler must carry; log2(block) is only the value the writer puts there, and a file that ever said
        // something else would mean it.
        if (l == 1 && !make_level1_samplers(g.block, (float)root.number("lod_bias_level1", (double)g.block > 1 ? 2.0 : 0.0))) return false;
        const JVal* dq = t.get("dequantise"); const JVal* lo = dq ? dq->get("lo") : nullptr; const JVal* hi = dq ? dq->get("hi") : nullptr;
        if (!lo || !hi || lo->kind != JVal::ARR || hi->kind != JVal::ARR || (int)lo->arr.size() < L.C || (int)hi->arr.size() < L.C) { fprintf(stderr, "ERROR: level %d: no lo / hi per channel\n", l); return false; }
        for (int c = 0; c < L.C; c++) { L.lo[c] = (float)lo->arr[c].num; L.hi[c] = (float)hi->arr[c].num; }
        printf("  level %d: %d channel%s used, bits", l, L.C, L.C == 1 ? "" : "s"); for (int c = 0; c < L.C; c++) printf(" %d", L.bits[c]);
        printf(", dequantise value = lo + sample * (hi - lo) with lo / hi"); for (int c = 0; c < L.C; c++) printf(" [%g, %g]", L.lo[c], L.hi[c]); printf("\n");
    }
    const JVal* dec = root.get("decoder");
    if (!dec) { fprintf(stderr, "ERROR: no decoder in the JSON\n"); return false; }
    if (dec->string("type") != "bilinear") { fprintf(stderr, "ERROR: decoder type '%s': this viewer decodes the bilinear decoder only\n", dec->string("type").c_str()); return false; }
    const JVal* hidden = dec->get("hidden"); if (hidden && hidden->kind == JVal::ARR && !hidden->arr.empty()) { fprintf(stderr, "ERROR: hidden layers are not supported\n"); return false; }
    if (dec->string("output") != "identity") { fprintf(stderr, "ERROR: output rule '%s' is not the identity\n", dec->string("output").c_str()); return false; }
    const int C0 = (int)dec->number("C0"), C1 = (int)dec->number("C1"), nin = (int)dec->number("nin"), nout = (int)dec->number("nout");
    if (C0 != g.lat[0].C || C1 != g.lat[1].C || nin < 1 || nin > 24 || nout < 3 || nout > 18 || nout != 3 * g.textures_out) { fprintf(stderr, "ERROR: decoder shape C0 %d C1 %d nin %d nout %d does not fit the textures (%d / %d channels, %d outputs) or the shader's limits (nin <= 24, nout <= 18)\n", C0, C1, nin, nout, g.lat[0].C, g.lat[1].C, 3 * g.textures_out); return false; }
    g.terms = dec->string("terms"); int mask = 0;
    // The terms are a SEQUENCE, not a set: the shader builds phi in the order a, b, sc, so "b a sc" would name the same
    // three groups and describe a different feature vector. Anything out of that order is refused rather than reordered.
    { std::istringstream ss(g.terms); std::string w; int last = -1; while (ss >> w) { const int which = w == "a" ? 0 : (w == "b" ? 1 : (w == "sc" ? 2 : -1));
        if (which < 0) { fprintf(stderr, "ERROR: the term '%s' is not one the shader builds (a, b, sc)\n", w.c_str()); return false; }
        if (which <= last) { fprintf(stderr, "ERROR: the terms '%s' are not in the order the shader builds phi (a, then b, then sc)\n", g.terms.c_str()); return false; }
        last = which; mask |= 1 << which; } }
    int expect = 0; if (mask & 1) expect += C1; if (mask & 2) expect += C0; if (mask & 4) expect += C0 * C1;
    if (expect != nin) { fprintf(stderr, "ERROR: the terms '%s' give %d features, the decoder has nin %d\n", g.terms.c_str(), expect, nin); return false; }
    const JVal* layers = dec->get("layers");
    if (!layers || layers->kind != JVal::ARR || layers->arr.size() != 1) { fprintf(stderr, "ERROR: expected one layer\n"); return false; }
    const JVal& lay = layers->arr[0]; const JVal* Wv = lay.get("weights"); const JVal* bv = lay.get("bias");
    if ((int)lay.number("rows") != nout || (int)lay.number("cols") != nin || !Wv || !bv || (int)Wv->arr.size() != nout * nin || (int)bv->arr.size() != nout) { fprintf(stderr, "ERROR: the layer is not %d x %d\n", nout, nin); return false; }
    DecCB& d = g.dec; memset(&d, 0, sizeof(d));
    for (int c = 0; c < 4; c++) { d.lo0[c] = g.lat[0].lo[c]; d.hi0[c] = g.lat[0].hi[c]; d.lo1[c] = g.lat[1].lo[c]; d.hi1[c] = g.lat[1].hi[c]; }
    d.dims[0] = C0; d.dims[1] = C1; d.dims[2] = nin; d.dims[3] = nout; d.sel[0] = mask; d.sel[1] = 0;
    for (int r = 0; r < nout; r++) for (int c = 0; c < nin; c++) d.W[r * 6 + c / 4][c % 4] = (float)Wv->arr[(size_t)r * nin + c].num;
    for (int r = 0; r < nout; r++) d.bias[r / 4][r % 4] = (float)bv->arr[r].num;
    printf("  decoder: bilinear, terms '%s', phi %d -> %d outputs (%d texture%s), %d weights\n", g.terms.c_str(), nin, nout, g.textures_out, g.textures_out == 1 ? "" : "s", nout * nin + nout);
    return true;
}

// ---------------------------------------------------------------------------
// Geometry (position float3 + uv float2, interleaved).
// ---------------------------------------------------------------------------
static ID3D11Buffer* g_quad_vb = nullptr, * g_quad_ib = nullptr;
static ID3D11Buffer* g_cube_vb = nullptr, * g_cube_ib = nullptr;
static int g_cube_index_count = 0;

static ID3D11Buffer* make_buffer(const void* data, size_t bytes, UINT bind) {
    D3D11_BUFFER_DESC bd{}; bd.ByteWidth = (UINT)bytes; bd.Usage = D3D11_USAGE_IMMUTABLE; bd.BindFlags = bind;
    D3D11_SUBRESOURCE_DATA sd{}; sd.pSysMem = data;
    ID3D11Buffer* b = nullptr;
    if (FAILED(g_dev->CreateBuffer(&bd, &sd, &b))) return nullptr;
    return b;
}
static void create_quad(float aspect) {
    float hw, hh;
    if (aspect >= 1.0f) { hw=1.0f; hh=1.0f/aspect; } else { hw=aspect; hh=1.0f; }
    // UV origin: (0,0) = top-left = the first uploaded row; the .dds levels are top-row-first, so v runs 0..1 top to bottom with no flip.
    float v[] = {
        -hw,-hh,0.0f, 0.0f,1.0f,  hw,-hh,0.0f, 1.0f,1.0f,
         hw, hh,0.0f, 1.0f,0.0f, -hw, hh,0.0f, 0.0f,0.0f,
    };
    uint32_t idx[] = {0,1,2, 0,2,3};
    g_quad_vb = make_buffer(v, sizeof(v), D3D11_BIND_VERTEX_BUFFER);
    g_quad_ib = make_buffer(idx, sizeof(idx), D3D11_BIND_INDEX_BUFFER);
}
static void create_cube() {
    const float h=0.5f;
    float v[] = {
        -h,-h, h, 0,1,  h,-h, h, 1,1,  h, h, h, 1,0, -h, h, h, 0,0,
         h,-h,-h, 0,1, -h,-h,-h, 1,1, -h, h,-h, 1,0,  h, h,-h, 0,0,
         h,-h, h, 0,1,  h,-h,-h, 1,1,  h, h,-h, 1,0,  h, h, h, 0,0,
        -h,-h,-h, 0,1, -h,-h, h, 1,1, -h, h, h, 1,0, -h, h,-h, 0,0,
        -h, h, h, 0,1,  h, h, h, 1,1,  h, h,-h, 1,0, -h, h,-h, 0,0,
        -h,-h,-h, 0,1,  h,-h,-h, 1,1,  h,-h, h, 1,0, -h,-h, h, 0,0,
    };
    std::vector<uint32_t> idx;
    for (int i=0;i<6;i++){ int b=i*4; idx.insert(idx.end(),{(uint32_t)b,(uint32_t)b+1,(uint32_t)b+2,(uint32_t)b,(uint32_t)b+2,(uint32_t)b+3}); }
    g_cube_vb = make_buffer(v, sizeof(v), D3D11_BIND_VERTEX_BUFFER);
    g_cube_ib = make_buffer(idx.data(), idx.size()*sizeof(uint32_t), D3D11_BIND_INDEX_BUFFER);
    g_cube_index_count = (int)idx.size();
}

// ---------------------------------------------------------------------------
// Debug text overlay (8x8 font rasterized to an RGBA texture, drawn as a quad).
// ---------------------------------------------------------------------------
static const int OVL_W = 2560, OVL_H = 84, FONT_SCALE = 2, LINE_ADV = 20;   // the overlay strip spans the wider window
static const char* DEBUG_HLSL =
    "struct VSO { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "VSO VSMain(float2 pos : POSITION, float2 uv : TEXCOORD0) {\n"
    "    VSO o; o.pos = float4(pos, 0.0, 1.0); o.uv = uv; return o; }\n"
    "Texture2D tex : register(t0); SamplerState samp : register(s0);\n"
    "float4 PSMain(VSO i) : SV_Target { return tex.Sample(samp, i.uv); }\n";

static void blit_char(std::vector<uint8_t>& buf, int px, int py, char ch) {
    int c = (unsigned char)ch; if (c<32 || c>127) c='.';
    const uint8_t* glyph = g_font8x8[c-32];
    for (int y=0;y<8;y++) for (int x=0;x<8;x++) {
        if (!((glyph[y]>>x)&1)) continue;
        for (int sy=0;sy<FONT_SCALE;sy++) for (int sx=0;sx<FONT_SCALE;sx++) {
            int X=px+x*FONT_SCALE+sx, Y=py+y*FONT_SCALE+sy;
            if (X<0||X>=OVL_W||Y<0||Y>=OVL_H) continue;
            uint8_t* d=&buf[(Y*OVL_W+X)*4]; d[0]=255;d[1]=255;d[2]=255;d[3]=255;
        }
    }
}
static void update_debug_text() {
    if (!g.debug_dirty) return;
    std::vector<uint8_t> buf(OVL_W*OVL_H*4);
    for (size_t i=0;i<buf.size();i+=4){ buf[i]=0;buf[i+1]=0;buf[i+2]=0;buf[i+3]=180; }
    char l0[256], l1[256], l2[256];
    // The load-time pack's own name: one BC4 or BC5, and for three or four channels the second texture's format too
    // (three channels put channel 2 alone in a BC4, four put channels 2-3 in a second BC5).
    const char* packname = g.lat[0].C == 1 ? "BC4" : (g.lat[0].C == 2 ? "BC5" : (g.lat[0].C == 3 ? "BC5+BC4" : "BC5+BC5"));
    char bcs[64]; if (g.lat[0].bc_n == 0) snprintf(bcs, sizeof(bcs), "U8"); else if (g.lat[0].bc_lossless) snprintf(bcs, sizeof(bcs), "%s lossless", packname); else snprintf(bcs, sizeof(bcs), "%s %.1fdB", packname, g.lat[0].bc_psnr);
    auto fmt_name = [](int dxgi) { return dxgi == 61 ? "R8" : (dxgi == 49 ? "R8G8" : (dxgi == 28 ? "RGBA8" : (dxgi == 80 ? "BC4" : (dxgi == 83 ? "BC5" : "?")))); };   // the texture's stored format
    // The format line when level 0 arrived block-compressed in the file: nothing was packed here, so it says "(file)" and no quality.
    char bcf[64]; snprintf(bcf, sizeof(bcf), "%s%s%s (file)", fmt_name(g.lat[0].dxgi), g.lat[0].file_files > 1 ? "+" : "", g.lat[0].file_files > 1 ? fmt_name(g.lat[0].dxgi_b) : "");
    // GPU memory of what is bound, in bytes, summed over the base and over the whole chain, as bits per source pixel of the decode size
    // per material texture. An uncompressed texture is one byte per stored channel per texel; a BLOCK-COMPRESSED one is whole 4x4
    // blocks - ceil(w/4) * ceil(h/4) of them at 8 bytes (BC4) or 16 (BC5) - which is what the driver allocates and what the .dds holds.
    // Counting it as w * h * 0.5 or w * h * 1 understates every level whose size is not a multiple of 4: a 10x10 plane is nine blocks
    // per axis-pair, not 6.25, and the deep end of a chain is nearly all such levels.
    double mem_base = 0, mem_all = 0, mem_base_l0 = 0;
    for (int l = 0; l < 2; l++) {
        const LatentTex& L = g.lat[l]; const bool packed = l == 0 && g_bc && L.bc_n > 0;
        // Each bound texture counted at ITS OWN rate, because the two textures of a three-channel level 0 are a BC5 and a
        // BC4 and not two of either: 16 + 8 bytes a block, which is the 12 bpp the file costs.
        double bytes_per_block = 0.0;   // zero when nothing here is block-compressed
        if (packed) { for (int p = 0; p < L.bc_n; p++) bytes_per_block += L.bc_nc[p] == 1 ? 8.0 : 16.0; }
        else if (L.file_bc) bytes_per_block = (L.dxgi == 80 ? 8.0 : 16.0) + (L.file_files > 1 ? (L.dxgi_b == 80 ? 8.0 : 16.0) : 0.0);
        for (int i = 0; i < (int)L.dims.size(); i++) {
            const int w = L.dims[i].first, h = L.dims[i].second;
            const double b = bytes_per_block > 0.0 ? (double)((w + 3) / 4) * ((h + 3) / 4) * bytes_per_block
                                                   : (double)L.stored_C * w * h;
            if (i == 0) { mem_base += b; if (l == 0) mem_base_l0 = b; }
            mem_all += b;
        }
    }
    const double px = (double)std::max(1, g.lat[0].W) * std::max(1, g.lat[0].H) * std::max(1, g.textures_out);   // the decode size x the textures
    // Level 0's own share is called out beside the total: it is the number the file layout decides (4 bpp for one BC4,
    // 8 for one BC5, 12 for a BC5 + a BC4 at three channels, 16 for two BC5s), divided by the material's textures.
    const double bpp_base = 8.0 * mem_base / px, bpp_all = 8.0 * mem_all / px, bpp_l0 = 8.0 * mem_base_l0 / px;
    // per level: size x channels used, the stored texture format (8 bits per channel, the index replicated), the index bits per channel, the mip count; then the GPU memory
    snprintf(l0,sizeof(l0),"Src:%dx%d  L0:%dx%dx%d %s idx%db %dmips  L1:%dx%dx%d %s idx%db %dmips  blk:%d  GPU:%.2fbpp/tex base (L0 %.2f), %.2f with mips",
             g.src_w,g.src_h, g.lat[0].W,g.lat[0].H,g.lat[0].C, g.lat[0].file_bc ? bcf : (g_bc && g.lat[0].bc_n ? bcs : fmt_name(g.lat[0].dxgi)), g.lat[0].bits[0], g.lat[0].mips, g.lat[1].W,g.lat[1].H,g.lat[1].C, fmt_name(g.lat[1].dxgi), g.lat[1].bits[0], g.lat[1].mips, g.block, bpp_base, bpp_l0, bpp_all);
    // Anisotropy applies in trilinear mode only, so in the other two the overlay says the flag is remembered and idle.
    char aniso_s[32];
    if (!g.aniso) snprintf(aniso_s, sizeof(aniso_s), "Aniso:OFF");
    else if (g.filter_mode == 2) snprintf(aniso_s, sizeof(aniso_s), "Aniso:%u", (unsigned)ANISO_MAX);
    else snprintf(aniso_s, sizeof(aniso_s), "Aniso:%u (trilinear only)", (unsigned)ANISO_MAX);
    snprintf(l1,sizeof(l1),"Mode:%-4s Filter:%-9s %-24s Mips:%s L1bias:%s Tex:%d/%d  Show:%s  Renorm:%s",
             g.cube?"CUBE":"QUAD", filter_name(g.filter_mode), aniso_s, g.mips_on?"ON ":"OFF", g.lod_bias?"ON ":"OFF", g.tex_shown, g.textures_out,
             g.const0[0]>0.5f?"LATENT0":(g.const0[1]>0.5f?"LATENT1":"DECODE"), g.const0[3]>0.5f?"ON ":"OFF");
    snprintf(l2,sizeof(l2),"X:%+5.1f Y:%+5.1f Z:%5.1f Yaw:%+6.1f Pitch:%+6.1f", g.x,g.y,g.z,g.yaw,g.pitch);
    const char* l3 = "Move:Arrows/WS Rot:ADQE C:cube B/T/P:filter X:aniso M:mips L:L1bias N:tex V:renorm 1/2:latents 4:BC-pack R:reload Spc:reset Esc";
    const char* lines[4] = {l0,l1,l2,l3};
    for (int li=0; li<4; li++) { int y = 2 + li*LINE_ADV, x = 4; for (const char* p=lines[li]; *p; ++p) { blit_char(buf, x, y, *p); x += 8*FONT_SCALE; } }
    D3D11_MAPPED_SUBRESOURCE map{};
    if (SUCCEEDED(g_ctx->Map(g.debug_tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &map))) {
        for (int y = 0; y < OVL_H; y++) memcpy((uint8_t*)map.pData + y * map.RowPitch, &buf[y * OVL_W * 4], OVL_W * 4);
        g_ctx->Unmap(g.debug_tex, 0);
    }
    g.debug_dirty = false;
}
static bool init_debug() {
    ID3DBlob* vsb = nullptr, * psb = nullptr, * errs = nullptr;
    if (FAILED(D3DCompile(DEBUG_HLSL, strlen(DEBUG_HLSL), "debug_overlay", nullptr, nullptr, "VSMain", "vs_5_0", 0, 0, &vsb, &errs))) {
        if (errs) { fprintf(stderr, "OVERLAY VS ERROR:\n%s\n", (const char*)errs->GetBufferPointer()); errs->Release(); }
        return false;
    }
    safe_release(errs);
    if (FAILED(D3DCompile(DEBUG_HLSL, strlen(DEBUG_HLSL), "debug_overlay", nullptr, nullptr, "PSMain", "ps_5_0", 0, 0, &psb, &errs))) {
        if (errs) { fprintf(stderr, "OVERLAY PS ERROR:\n%s\n", (const char*)errs->GetBufferPointer()); errs->Release(); }
        vsb->Release(); return false;
    }
    safe_release(errs);
    bool ok = SUCCEEDED(g_dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &g.debug_vs)) &&
              SUCCEEDED(g_dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &g.debug_ps));
    if (ok) {
        const D3D11_INPUT_ELEMENT_DESC il[] = {
            { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };
        ok = SUCCEEDED(g_dev->CreateInputLayout(il, 2, vsb->GetBufferPointer(), vsb->GetBufferSize(), &g.debug_layout));
    }
    vsb->Release(); psb->Release();
    if (!ok) return false;
    D3D11_BUFFER_DESC bd{}; bd.ByteWidth = 16*sizeof(float); bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(g_dev->CreateBuffer(&bd, nullptr, &g.debug_vb))) return false;
    uint32_t idx[] = {0,1,2, 0,2,3};
    g.debug_ib = make_buffer(idx, sizeof(idx), D3D11_BIND_INDEX_BUFFER);
    if (!g.debug_ib) return false;
    D3D11_TEXTURE2D_DESC td{};
    td.Width = OVL_W; td.Height = OVL_H; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DYNAMIC; td.BindFlags = D3D11_BIND_SHADER_RESOURCE; td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(g_dev->CreateTexture2D(&td, nullptr, &g.debug_tex))) return false;
    if (FAILED(g_dev->CreateShaderResourceView(g.debug_tex, nullptr, &g.debug_srv))) return false;
    return true;
}
static bool g_overlay = true;   // --nooverlay: draw the scene alone, so two --shot frames can be compared byte for byte (the overlay names the file and its format, so it differs between two assets that must render the same)
static void draw_debug_text() {
    if (!g.debug_vs || !g_overlay) return;
    update_debug_text();
    float w = (float)OVL_W / WINDOW_WIDTH * 2.0f, h = (float)OVL_H / WINDOW_HEIGHT * 2.0f;
    float verts[] = { -1.0f, 1.0f, 0.0f, 0.0f,  -1.0f + w, 1.0f, 1.0f, 0.0f,  -1.0f + w, 1.0f - h, 1.0f, 1.0f,  -1.0f, 1.0f - h, 0.0f, 1.0f };
    D3D11_MAPPED_SUBRESOURCE map{};
    if (SUCCEEDED(g_ctx->Map(g.debug_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &map))) { memcpy(map.pData, verts, sizeof(verts)); g_ctx->Unmap(g.debug_vb, 0); }
    const float blend_factor[4] = {0,0,0,0};
    g_ctx->OMSetBlendState(g_blend_alpha, blend_factor, 0xFFFFFFFF);
    g_ctx->OMSetDepthStencilState(g_depth_off, 0);
    g_ctx->IASetInputLayout(g.debug_layout);
    UINT stride = 4*sizeof(float), offset = 0;
    g_ctx->IASetVertexBuffers(0, 1, &g.debug_vb, &stride, &offset);
    g_ctx->IASetIndexBuffer(g.debug_ib, DXGI_FORMAT_R32_UINT, 0);
    g_ctx->VSSetShader(g.debug_vs, nullptr, 0);
    g_ctx->PSSetShader(g.debug_ps, nullptr, 0);
    g_ctx->PSSetShaderResources(0, 1, &g.debug_srv);
    g_ctx->PSSetSamplers(0, 1, &g_samplers[0][0]);
    g_ctx->DrawIndexed(6, 0, 0);
    g_ctx->OMSetBlendState(nullptr, blend_factor, 0xFFFFFFFF);
    g_ctx->OMSetDepthStencilState(g_depth_on, 0);
}

// ---------------------------------------------------------------------------
// Swap chain / render target (re)creation.
// ---------------------------------------------------------------------------
static bool create_backbuffer_views() {
    ID3D11Texture2D* bb = nullptr;
    if (FAILED(g_swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&bb))) return false;
    HRESULT hr = g_dev->CreateRenderTargetView(bb, nullptr, &g_rtv);
    bb->Release();
    if (FAILED(hr)) return false;
    D3D11_TEXTURE2D_DESC dd{};
    dd.Width = (UINT)WINDOW_WIDTH; dd.Height = (UINT)WINDOW_HEIGHT; dd.MipLevels = 1; dd.ArraySize = 1;
    dd.Format = DXGI_FORMAT_D24_UNORM_S8_UINT; dd.SampleDesc.Count = 1; dd.Usage = D3D11_USAGE_DEFAULT; dd.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    if (FAILED(g_dev->CreateTexture2D(&dd, nullptr, &g_depth_tex))) return false;
    if (FAILED(g_dev->CreateDepthStencilView(g_depth_tex, nullptr, &g_dsv))) return false;
    return true;
}
static void resize_backbuffer(int w, int h) {
    if (!g_swapchain || w <= 0 || h <= 0) return;
    if (w == WINDOW_WIDTH && h == WINDOW_HEIGHT && g_rtv && g_dsv) return;   // the size it already has: nothing to do, and no frame to lose
    WINDOW_WIDTH = w; WINDOW_HEIGHT = h;
    g_ctx->OMSetRenderTargets(0, nullptr, nullptr);
    safe_release(g_rtv); safe_release(g_dsv); safe_release(g_depth_tex);
    g_swapchain->ResizeBuffers(0, (UINT)w, (UINT)h, DXGI_FORMAT_UNKNOWN, 0);
    if (!create_backbuffer_views()) fprintf(stderr, "ERROR: backbuffer view recreation failed after resize\n");
    g.debug_dirty = true;
}

// ---------------------------------------------------------------------------
// Input
//
// --shot TAKES NO INPUT AT ALL, and that is what makes it reproducible. The one frame it draws is a pure function of the
// command line and the asset only if nothing else can reach the state it draws from, and two things could: the keyboard
// state process_held_keys reads with GetAsyncKeyState (which is the live state of the physical keyboard, not this
// process's), and the WM_KEYDOWN messages the pump dispatches while the freshly created, freshly focused window is the
// foreground one. A key held or struck in the few milliseconds a --shot launch lives - the shell the run was started
// from, a keystroke meant for an editor, the release gate running while somebody works - moved the camera or changed the
// filter, the mip or the shown texture for that one frame. It showed up as one shot in ten differing from the others on
// identical binaries and identical assets, over the quad and nowhere else; the frame's own arithmetic was never at
// fault. So in shot mode the held-key scan is not run and key messages are ignored, and the frame depends on nothing
// that is not on the command line.
// ---------------------------------------------------------------------------
static bool g_shot = false;   // --shot: one frame, no input
static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_CLOSE: case WM_DESTROY: g_quit = true; PostQuitMessage(0); return 0;
        case WM_SIZE: if (wp != SIZE_MINIMIZED) resize_backbuffer((int)LOWORD(lp), (int)HIWORD(lp)); return 0;
        case WM_KEYDOWN: {
            if (g_shot) return 0;   // a keystroke that lands in the shot window changes nothing it draws
            if (lp & (1 << 30)) return 0;
            switch (wp) {
                case VK_ESCAPE: g_quit = true; break;
                case 'R': reload_shader(); break;
                case 'B': g.filter_mode=1; g.debug_dirty=true; printf("Filter: BILINEAR\n"); break;
                case 'T': g.filter_mode=2; g.debug_dirty=true; printf("Filter: TRILINEAR\n"); break;
                case 'P': g.filter_mode=0; g.debug_dirty=true; printf("Filter: POINT\n"); break;
                case 'X': g.aniso=!g.aniso; g.debug_dirty=true; printf("Anisotropic filtering: %s%s\n", g.aniso ? "ON (MaxAnisotropy " : "OFF", g.aniso ? (std::to_string((unsigned)ANISO_MAX) + ")").c_str() : ""); if (g.filter_mode != 2) printf("  (it applies in TRILINEAR mode only; press T)\n"); break;
                case 'L': g.lod_bias=!g.lod_bias; g.debug_dirty=true; printf("Level 1 LOD bias: %s\n", g.lod_bias ? "ON (+log2(block): the 1:1 mip rule)" : "OFF (the GPU's own LOD: level 1 two mips finer than fitted)"); break;
                case 'M': g.mips_on=!g.mips_on; g.debug_dirty=true; printf("Mips: %s\n", g.mips_on ? "ON (MaxLOD unlimited)" : "OFF (sampler MaxLOD = 0: mip 0 only)"); break;
                case 'C': g.cube=!g.cube; g.debug_dirty=true; break;
                case 'N': g.tex_shown = (g.tex_shown + 1) % g.textures_out; g.debug_dirty=true; printf("Output texture %d of %d\n", g.tex_shown, g.textures_out); break;
                case '1': g.const0[0]=1.0f-g.const0[0]; g.const0[1]=0; g.debug_dirty=true; break;
                case '2': g.const0[1]=1.0f-g.const0[1]; g.const0[0]=0; g.debug_dirty=true; break;
                case '3': g.const0[2]=1.0f-g.const0[2]; g.debug_dirty=true; break;
                case 'V': g.const0[3]=1.0f-g.const0[3]; g.debug_dirty=true; printf("Normal renormalisation: %s\n", g.const0[3]>0.5f ? "ON (unpack, unit length, repack)" : "OFF"); break;
                case '4': if (g.lat[0].bc_n) { g_bc = !g_bc; g.debug_dirty=true; printf("Level 0: %s\n", g_bc ? "BC4/BC5 pack" : "uncompressed"); } break;
                case '5': g.const1[0]=1.0f-g.const1[0]; g.debug_dirty=true; break;
                case '6': g.const1[1]=1.0f-g.const1[1]; g.debug_dirty=true; break;
                case '7': g.const1[2]=1.0f-g.const1[2]; g.debug_dirty=true; break;
                case '8': g.const1[3]=1.0f-g.const1[3]; g.debug_dirty=true; break;
                case VK_SPACE:
                    g.x=0;g.y=0;g.z=-3.0f;g.yaw=0;g.pitch=0;
                    for(int i=0;i<4;i++){g.const0[i]=0;g.const1[i]=0;}
                    g.tex_shown = 0; g.debug_dirty=true; printf("Reset to initial state\n"); break;
            }
            return 0;
        }
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}
static bool key_down(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }
static void process_held_keys(HWND hwnd, float dt) {
    if (g_shot) return;   // see above: the shot frame is a function of the command line and the asset, and of nothing else
    if (GetForegroundWindow() != hwnd) return;
    if (key_down(VK_SHIFT)) dt *= 1.0f/3.0f;
    bool moved=false;
    if (key_down('W')) {g.z+=Z_SPEED*dt;moved=true;}
    if (key_down('S')) {g.z-=Z_SPEED*dt;moved=true;}
    if (key_down(VK_LEFT))  {g.x+=XY_SPEED*dt;moved=true;}
    if (key_down(VK_RIGHT)) {g.x-=XY_SPEED*dt;moved=true;}
    if (key_down(VK_UP))    {g.y+=XY_SPEED*dt;moved=true;}
    if (key_down(VK_DOWN))  {g.y-=XY_SPEED*dt;moved=true;}
    if (key_down('A')) {g.yaw+=ROT_SPEED*dt;moved=true;}
    if (key_down('D')) {g.yaw-=ROT_SPEED*dt;moved=true;}
    if (key_down('Q')) {g.pitch+=ROT_SPEED*dt;moved=true;}
    if (key_down('E')) {g.pitch-=ROT_SPEED*dt;moved=true;}
    if (g.z < Z_MAX) g.z=Z_MAX; if (g.z > Z_MIN) g.z=Z_MIN;
    if (moved) g.debug_dirty=true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
struct CBData {          // must match cbuffer SceneConstants in nntc_view.hlsl (16-byte rules)
    float mvp[16];
    float texSize[4];
    float lodInfo[4];
    float const0[4];
    float const1[4];
};
static void set_uniforms() {
    Mat4 proj = mat_perspective(FOV_DEGREES, (float)WINDOW_WIDTH/WINDOW_HEIGHT, 0.001f, 100.0f);
    Mat4 model = mat_mul(mat_mul(mat_translate(g.x,g.y,g.z), mat_rot_y(g.yaw)), mat_rot_x(g.pitch));
    Mat4 mvp = mat_mul(proj, model);
    CBData cb{};
    Mat4 mvp_t = mat_transpose(mvp);
    memcpy(cb.mvp, mvp_t.m, sizeof(cb.mvp));
    cb.texSize[0]=(float)g.lat[0].W; cb.texSize[1]=(float)g.lat[0].H; cb.texSize[2]=(float)g.lat[1].W; cb.texSize[3]=(float)g.lat[1].H;
    cb.lodInfo[0]=(float)(g.lat[0].mips-1); cb.lodInfo[1]=(float)(g.lat[1].mips-1);
    memcpy(cb.const0, g.const0, sizeof(cb.const0));
    memcpy(cb.const1, g.const1, sizeof(cb.const1));
    D3D11_MAPPED_SUBRESOURCE map{};
    if (SUCCEEDED(g_ctx->Map(g_cbuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &map))) { memcpy(map.pData, &cb, sizeof(cb)); g_ctx->Unmap(g_cbuffer, 0); }
    g.dec.sel[1] = g.tex_shown;
    if (SUCCEEDED(g_ctx->Map(g_dec_cbuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &map))) { memcpy(map.pData, &g.dec, sizeof(g.dec)); g_ctx->Unmap(g_dec_cbuffer, 0); }
}

// EVERY flag, not two of them. The usage named --bc and --nooverlay alone while the loop below accepted fourteen, so
// the twelve it did not name could be found only by reading this file - and --help itself fell through to "ignoring
// unknown option" and then to a usage printed with an exit code of 1, which is a refusal rather than an answer.
static void viewer_usage(const char* exe) {
    printf("Usage: %s <PREFIX_nntc.json> [flags]\n", exe);
    printf("  The asset's descriptor and the .dds files it names, as written by nntc_encode. The path is taken as\n");
    printf("  given: the viewer derives no name of its own.\n\n");
    printf("  --shot FILE.bmp  render ONE frame to a 24-bit .bmp and exit, taking no keyboard input at all, so the\n");
    printf("                   bytes are a function of the command line and the asset alone (a check without a desktop)\n");
    printf("  --tex N          show output texture N of a material (the key N cycles them)\n");
    printf("  --cube           the cube rather than the quad (the key C)\n");
    printf("  --z F            the camera distance\n");
    printf("  --yaw F          the camera yaw, in degrees\n");
    printf("  --pitch F        the camera pitch, in degrees\n");
    printf("  --nomips         the sampler's MaxLOD is 0, so every fetch reads mip 0 (the key M off)\n");
    printf("  --nobias         level 1 without its MipLODBias of log2(block) (the key L off). The bias is what the\n");
    printf("                   encoder's 1:1 rule becomes on the GPU, so this is the control and not a preference\n");
    printf("  --noaniso        anisotropy off, which applies in the trilinear mode only (the key X off)\n");
    printf("  --renorm         renormalise the decoded triple as a tangent-space normal (the key V)\n");
    printf("  --raw0           show level 0's own channels instead of the decode (the key 1)\n");
    printf("  --raw1           show level 1's own channels instead of the decode (the key 2)\n");
    printf("  --bc             start with level 0 on its BC4 / BC5 pack, made at load when the file itself is\n");
    printf("                   uncompressed (the key 4 switches; lossless for 1-3 bits, an endpoint search for 4+)\n");
    printf("  --nooverlay      draw the scene without the debug strip, so two --shot frames can be compared byte for\n");
    printf("                   byte (the strip names the file and its format, which two identical renders differ in)\n\n");
    printf("  The shader is loaded from nntc_view.hlsl in the working directory (or next to the executable).\n");
}

// A numeric flag's value must be the whole argument and finite: `--tex nope` used to be texture 0 through atoi, which
// is the kind of silent acceptance the encoder refuses, and the viewer refuses it the same way.
static bool parse_number(const char* text, double& out) {
    char* end = nullptr; errno = 0; const double v = strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(v)) return false;
    out = v; return true;
}
static bool parse_integer(const char* text, int& out) {
    char* end = nullptr; errno = 0; const long v = strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || v < -1000000 || v > 1000000) return false;
    out = (int)v; return true;
}

int main(int argc, char** argv) {
    std::string json_path, shot_path;   // --shot FILE.bmp: render one frame (with --cube, --z F, --yaw F, --tex N) to a .bmp and exit (a check without a desktop)
    for (int i = 1; i < argc; i++) { std::string a = argv[i];
        if (a == "--help" || a == "-h") { viewer_usage(argv[0]); return 0; }   // asking for the usage is not a refusal
        else if (a == "--shot") { if (i + 1 >= argc) { fprintf(stderr, "ERROR: --shot needs a file name\n"); return 1; } shot_path = argv[++i]; g_shot = true; }
        else if (a == "--cube") g.cube = true;
        else if (a == "--nomips") g.mips_on = false;   // the key M state for --shot
        else if (a == "--nobias") g.lod_bias = false;   // the key L state for --shot
        else if (a == "--noaniso") g.aniso = false;   // the key X state for --shot
        else if (a == "--renorm") g.const0[3] = 1.0f;   // the key V state for --shot
        else if (a == "--z" || a == "--yaw" || a == "--pitch") {
            double v; if (i + 1 >= argc || !parse_number(argv[i + 1], v)) { fprintf(stderr, "ERROR: %s needs a number, not '%s'\n", a.c_str(), i + 1 < argc ? argv[i + 1] : ""); return 1; }
            i++; if (a == "--z") g.z = (float)v; else if (a == "--yaw") g.yaw = (float)v; else g.pitch = (float)v; }
        else if (a == "--tex") {
            int v; if (i + 1 >= argc || !parse_integer(argv[i + 1], v)) { fprintf(stderr, "ERROR: --tex needs a texture index, not '%s'\n", i + 1 < argc ? argv[i + 1] : ""); return 1; }
            i++; g.tex_shown = v; }
        else if (a == "--bc") g_bc = true;
        else if (a == "--nooverlay") g_overlay = false;
        else if (a == "--raw0") g.const0[0] = 1.0f;
        else if (a == "--raw1") g.const0[1] = 1.0f;
        else if (!a.empty() && a[0] == '-') { fprintf(stderr, "ERROR: unknown option '%s'\n", a.c_str()); viewer_usage(argv[0]); return 1; }
        else if (json_path.empty()) json_path = a;
        else { fprintf(stderr, "ERROR: one descriptor only; '%s' is a second\n", a.c_str()); return 1; } }
    // Nothing is printed about the run's state before it is known that there IS a run: named no asset, the program owes
    // the caller the usage and nothing else.
    if (json_path.empty()) {
        viewer_usage(argv[0]);
        return 1;   // no asset was named, which IS a refusal, unlike --help above
    }
    // The descriptor is opened here, before a window, a device or a shader exists, so that a path this program cannot
    // read costs one line and an exit rather than a window that appears and vanishes. load_asset reads it again below;
    // one extra open of a file of a few kilobytes is worth the tidy refusal.
    //
    // "cannot read" used to cover three different mistakes, because an empty read_file is all three: a path that is not
    // there, a directory named where a descriptor was meant, and a file that exists but holds nothing. The caller is
    // told which, since each has its own fix, and the status is 1 for all of them.
    {
        std::error_code ec;
        const std::filesystem::path p(json_path);
        if (!std::filesystem::exists(p, ec)) {
            fprintf(stderr, "ERROR: cannot read '%s'\n", json_path.c_str());
            return 1;
        }
        if (!std::filesystem::is_regular_file(p, ec)) {
            fprintf(stderr, "ERROR: '%s' is not a file\n", json_path.c_str());
            return 1;
        }
        if (read_file(json_path).empty()) {
            fprintf(stderr, "ERROR: '%s' is empty\n", json_path.c_str());
            return 1;
        }
    }
    // `level 0` here is what the VIEWER will bind, which for a file that is already block-compressed is the file's own
    // blocks and not a pack of this program's making; load_asset says which, once it has read the file.
    printf("start: mips %s, level 1 LOD bias %s, anisotropy %s, level 0 %s\n", g.mips_on ? "on" : "off", g.lod_bias ? "on" : "off", g.aniso ? "on (trilinear only)" : "off", g_bc ? "the BC pack, if the file is uncompressed" : "as the file holds it");

    WNDCLASSEXA wc{}; wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW; wc.lpfnWndProc = wnd_proc; wc.hInstance = GetModuleHandleA(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW); wc.lpszClassName = "nntc_view_wc";
    RegisterClassExA(&wc);
    RECT r{0,0,WINDOW_WIDTH,WINDOW_HEIGHT};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowA(wc.lpszClassName, "NNTC latent viewer (D3D11)", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
                              r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) { fprintf(stderr, "ERROR: window creation failed\n"); return 1; }

    // The size the window ACTUALLY got. AdjustWindowRect asks for a 2560x1440 client area, but Windows clamps a window
    // to the work area, so on a smaller desktop the client rect is not what was asked for. Two things follow from
    // taking it here rather than trusting the constants:
    //
    //   the swap chain is created at the client size, so nothing is stretched on Present; and
    //
    //   the first WM_SIZE - which is SENT during CreateWindow, while g_swapchain is still null, and again later when
    //   the window is shown - no longer changes anything, because resize_backbuffer now sees the size it already has.
    //   That is what made a fresh --shot unsteady: whether the posted WM_SIZE reached the message pump before or after
    //   the one frame --shot draws was a race, and it decided whether the frame was 2560x1440 or the clamped size. The
    //   first launch of a newly built executable is exactly when the timing shifts, which is why it showed up there.
    { RECT cr{}; if (GetClientRect(hwnd, &cr) && cr.right > cr.left && cr.bottom > cr.top) { WINDOW_WIDTH = (int)(cr.right - cr.left); WINDOW_HEIGHT = (int)(cr.bottom - cr.top); } }
    printf("window: %dx%d client area\n", WINDOW_WIDTH, WINDOW_HEIGHT);

    // Non-sRGB backbuffer and UNORM textures throughout: the latents are not colours, and the decoder's output is written as it is.
    DXGI_SWAP_CHAIN_DESC scd{};
    scd.BufferCount = 2; scd.BufferDesc.Width = (UINT)WINDOW_WIDTH; scd.BufferDesc.Height = (UINT)WINDOW_HEIGHT;
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow = hwnd; scd.SampleDesc.Count = 1; scd.Windowed = TRUE; scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    const D3D_FEATURE_LEVEL want_fl = D3D_FEATURE_LEVEL_11_0;
    D3D_FEATURE_LEVEL got_fl{};
    UINT dev_flags = 0;
#if ENABLE_D3D_DEBUG
    dev_flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, dev_flags, &want_fl, 1, D3D11_SDK_VERSION, &scd, &g_swapchain, &g_dev, &got_fl, &g_ctx);
#if ENABLE_D3D_DEBUG
    if (FAILED(hr)) {
        fprintf(stderr, "WARNING: D3D11 debug layer unavailable; retrying without it\n");
        dev_flags &= ~D3D11_CREATE_DEVICE_DEBUG;
        hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, dev_flags, &want_fl, 1, D3D11_SDK_VERSION, &scd, &g_swapchain, &g_dev, &got_fl, &g_ctx);
    }
#endif
    if (FAILED(hr)) { fprintf(stderr, "ERROR: D3D11CreateDeviceAndSwapChain failed (0x%08X)\n", (unsigned)hr); return 1; }
    printf("Direct3D 11, feature level 0x%04X%s\n", (unsigned)got_fl, (dev_flags & D3D11_CREATE_DEVICE_DEBUG) ? "  (debug layer active)" : "");
    if (!create_backbuffer_views()) { fprintf(stderr, "ERROR: backbuffer view creation failed\n"); return 1; }

    {
        D3D11_RASTERIZER_DESC rd{}; rd.FillMode = D3D11_FILL_SOLID; rd.CullMode = D3D11_CULL_NONE; rd.DepthClipEnable = TRUE;
        if (FAILED(g_dev->CreateRasterizerState(&rd, &g_raster))) { fprintf(stderr, "ERROR: the rasterizer state could not be created\n"); return 1; }
        D3D11_DEPTH_STENCIL_DESC dd{};
        dd.DepthEnable = TRUE; dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL; dd.DepthFunc = D3D11_COMPARISON_LESS;
        if (FAILED(g_dev->CreateDepthStencilState(&dd, &g_depth_on))) { fprintf(stderr, "ERROR: the depth-on state could not be created\n"); return 1; }
        dd.DepthEnable = FALSE; dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        if (FAILED(g_dev->CreateDepthStencilState(&dd, &g_depth_off))) { fprintf(stderr, "ERROR: the depth-off state could not be created\n"); return 1; }
        D3D11_BLEND_DESC bd{};
        bd.RenderTarget[0].BlendEnable = TRUE; bd.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA; bd.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        bd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD; bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE; bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
        bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD; bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        if (FAILED(g_dev->CreateBlendState(&bd, &g_blend_alpha))) { fprintf(stderr, "ERROR: the blend state could not be created\n"); return 1; }
        // Four sampler states (P/B/T and T with anisotropy), all CLAMP: the encoder fits the taps clamped at the edges.
        // Every field is set: a zeroed D3D11_SAMPLER_DESC has ComparisonFunc 0, which is not a legal value, and
        // MaxAnisotropy 0, which is not one either - the runtime accepts both because it ignores them for these
        // filters, but make_level1_samplers reads the desc back with GetDesc and creates a sampler FROM it, so an
        // illegal field would come back and be used.
        D3D11_SAMPLER_DESC sd{};
        sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
        sd.MinLOD = 0.0f;
        for (int mips = 0; mips < 2; mips++) {   // a second set with MaxLOD = 0: the hardware clamps the LOD to mip 0 (key M), the textures and shader untouched
            sd.MaxLOD = mips ? D3D11_FLOAT32_MAX : 0.0f; sd.MipLODBias = 0.0f;
            static const D3D11_FILTER filters[FILTER_STATES] = { D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_FILTER_ANISOTROPIC };
            for (int f = 0; f < FILTER_STATES; f++) {
                sd.Filter = filters[f];
                sd.MaxAnisotropy = f == 3 ? ANISO_MAX : 1;   // anisotropy is several trilinear samples along the footprint's long axis
                if (FAILED(g_dev->CreateSamplerState(&sd, &g_samplers[mips][f]))) { fprintf(stderr, "ERROR: sampler state %d could not be created\n", f); return 1; }
            }
        }
        D3D11_BUFFER_DESC cbd{}; cbd.ByteWidth = sizeof(CBData); cbd.Usage = D3D11_USAGE_DYNAMIC; cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        // Both constant buffers are checked: a decoder constant buffer that failed to create would leave the shader
        // decoding with whatever the slot holds and render a wrong picture rather than fail.
        if (FAILED(g_dev->CreateBuffer(&cbd, nullptr, &g_cbuffer))) { fprintf(stderr, "ERROR: the scene constant buffer could not be created\n"); return 1; }
        cbd.ByteWidth = sizeof(DecCB);
        if (FAILED(g_dev->CreateBuffer(&cbd, nullptr, &g_dec_cbuffer))) { fprintf(stderr, "ERROR: the decoder constant buffer could not be created\n"); return 1; }
    }

    g.shader_path = "nntc_view.hlsl";
    if (read_file(g.shader_path).empty()) { std::string a = argv[0]; size_t s = a.find_last_of("/\\"); if (s != std::string::npos) g.shader_path = a.substr(0,s+1) + "nntc_view.hlsl"; }
    if (!load_shader(g.shader_path)) return 1;
    if (!load_asset(json_path)) return 1;
    // --tex is given before the asset is read, so the material's texture count is only known now; an out-of-range
    // value would index the shader's output triples past the end.
    if (g.tex_shown < 0 || g.tex_shown >= g.textures_out) { fprintf(stderr, "WARNING: --tex %d is outside 0..%d; showing texture 0\n", g.tex_shown, g.textures_out - 1); g.tex_shown = 0; }

    create_quad((float)g.lat[0].W / (float)g.lat[0].H);
    create_cube();
    if (!init_debug()) fprintf(stderr, "WARNING: debug overlay init failed (continuing without it)\n");

    const float clear_color[4] = {0.2f, 0.2f, 0.2f, 1.0f};
    LARGE_INTEGER qpf, last, now; QueryPerformanceFrequency(&qpf); QueryPerformanceCounter(&last);
    while (!g_quit) {
        MSG msg;
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageA(&msg); }
        if (g_quit) break;
        QueryPerformanceCounter(&now);
        float dt = (float)((double)(now.QuadPart - last.QuadPart) / (double)qpf.QuadPart);
        last = now;
        if (dt > 0.1f) dt = 0.1f;
        process_held_keys(hwnd, dt);
        if (!g_rtv || !g_dsv) { g_swapchain->Present(1, 0); continue; }
        D3D11_VIEWPORT vp{}; vp.Width = (float)WINDOW_WIDTH; vp.Height = (float)WINDOW_HEIGHT; vp.MaxDepth = 1.0f;
        g_ctx->RSSetViewports(1, &vp);
        g_ctx->RSSetState(g_raster);
        g_ctx->OMSetRenderTargets(1, &g_rtv, g_dsv);
        g_ctx->ClearRenderTargetView(g_rtv, clear_color);
        g_ctx->ClearDepthStencilView(g_dsv, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
        g_ctx->OMSetDepthStencilState(g_depth_on, 0);
        // The frame's bindings are DECIDED FIRST, before anything is uploaded, because the shader is told in the decoder constant
        // buffer (sel.z) how many textures level 0 is read from. Deciding after the upload would leave one frame in which the new
        // textures are read with the previous frame's count, which would flash on a key-4 toggle of a three or four channel level 0.
        // The samplers are picked here too, so keys 4, L and M all take effect on the same frame as the state they change.
        const bool packed = g_bc && g.lat[0].bc_n > 0;
        // t0 level 0, t1 level 1, t2 level 0's channels past the first two when they live in a texture of their own - which is the
        // case both when the file is two of them and when the viewer's load-time pack made two out of an uncompressed level 0. That
        // second texture is a BC5 for a fourth channel and a BC4 for a third one alone; the shader reads .rg of it either way and
        // uses only .r when C0 is 3, and a BC4 SRV returns (r, 0, 0, 1), so the green it would read is never part of phi.
        ID3D11ShaderResourceView* srvs[3] = { packed ? g.lat[0].bc_srv[0] : g.lat[0].srv, g.lat[1].srv,
                                              packed ? (g.lat[0].bc_n > 1 ? g.lat[0].bc_srv[1] : nullptr) : g.lat[0].srv_b };
        g.dec.sel[2] = packed ? g.lat[0].bc_n : (g.lat[0].srv_b ? 2 : 0);   // the textures level 0 is read from
        const int slot = filter_slot();   // key X: the trilinear state's anisotropic variant
        ID3D11SamplerState* samps[2] = { g_samplers[g.mips_on ? 1 : 0][slot], (g.lod_bias ? g_samplers1 : g_samplers)[g.mips_on ? 1 : 0][slot] };   // s0 level 0 (key M picks the MaxLOD = 0 set), s1 level 1 (key L: the LOD bias)
        set_uniforms();
        g_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        g_ctx->IASetInputLayout(g.layout);
        g_ctx->VSSetShader(g.vs, nullptr, 0);
        g_ctx->PSSetShader(g.ps, nullptr, 0);
        ID3D11Buffer* cbs[2] = { g_cbuffer, g_dec_cbuffer };   // b0 the scene, b1 the decoder
        g_ctx->VSSetConstantBuffers(0, 2, cbs);
        g_ctx->PSSetConstantBuffers(0, 2, cbs);
        g_ctx->PSSetShaderResources(0, 3, srvs);
        g_ctx->PSSetSamplers(0, 2, samps);
        UINT stride = 5*sizeof(float), offset = 0;
        if (g.cube) {
            g_ctx->IASetVertexBuffers(0, 1, &g_cube_vb, &stride, &offset);
            g_ctx->IASetIndexBuffer(g_cube_ib, DXGI_FORMAT_R32_UINT, 0);
            g_ctx->DrawIndexed((UINT)g_cube_index_count, 0, 0);
        } else {
            g_ctx->IASetVertexBuffers(0, 1, &g_quad_vb, &stride, &offset);
            g_ctx->IASetIndexBuffer(g_quad_ib, DXGI_FORMAT_R32_UINT, 0);
            g_ctx->DrawIndexed(6, 0, 0);
        }
        draw_debug_text();
        if (!shot_path.empty()) {   // the backbuffer to a 24-bit bottom-up .bmp BEFORE Present (a flip-model swap chain discards it after), then exit
            ID3D11Texture2D* bb = nullptr;
            if (FAILED(g_swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&bb)) || !bb) { fprintf(stderr, "ERROR: --shot could not get the backbuffer\n"); return 1; }
            D3D11_TEXTURE2D_DESC td{}; bb->GetDesc(&td); td.Usage = D3D11_USAGE_STAGING; td.BindFlags = 0; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ; td.MiscFlags = 0;
            ID3D11Texture2D* st = nullptr;
            if (FAILED(g_dev->CreateTexture2D(&td, nullptr, &st)) || !st) { fprintf(stderr, "ERROR: --shot could not create the staging texture\n"); safe_release(bb); return 1; }
            g_ctx->CopyResource(st, bb);
            D3D11_MAPPED_SUBRESOURCE map{};
            if (FAILED(g_ctx->Map(st, 0, D3D11_MAP_READ, 0, &map))) { fprintf(stderr, "ERROR: --shot could not map the staging texture\n"); safe_release(st); safe_release(bb); return 1; }
            {
                const int W = (int)td.Width, H = (int)td.Height, pitch = (W * 3 + 3) & ~3;
                std::vector<uint8_t> out((size_t)pitch * H, 0);
                for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { const uint8_t* sp = (const uint8_t*)map.pData + (size_t)y * map.RowPitch + (size_t)x * 4; uint8_t* op = &out[(size_t)(H - 1 - y) * pitch + (size_t)x * 3]; op[0] = sp[2]; op[1] = sp[1]; op[2] = sp[0]; }
                g_ctx->Unmap(st, 0);
                const uint32_t fsz = 54 + (uint32_t)out.size(); uint8_t hdr[54] = { 'B', 'M' }; uint32_t v;
                v = fsz; memcpy(hdr + 2, &v, 4); v = 54; memcpy(hdr + 10, &v, 4); v = 40; memcpy(hdr + 14, &v, 4); v = (uint32_t)W; memcpy(hdr + 18, &v, 4); v = (uint32_t)H; memcpy(hdr + 22, &v, 4);
                hdr[26] = 1; hdr[28] = 24; v = (uint32_t)out.size(); memcpy(hdr + 34, &v, 4);
                // A shot that was not written is a failed run, with the exit code to say so: a script that checks a
                // screenshot must not be told the frame exists when it does not.
                FILE* f = fopen(shot_path.c_str(), "wb");
                if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", shot_path.c_str()); safe_release(st); safe_release(bb); return 1; }
                const bool ok = fwrite(hdr, 1, 54, f) == 54 && fwrite(out.data(), 1, out.size(), f) == out.size();
                if (fclose(f) != 0 || !ok) { fprintf(stderr, "ERROR: writing %s failed\n", shot_path.c_str()); safe_release(st); safe_release(bb); return 1; }
                printf("wrote %s (%dx%d)\n", shot_path.c_str(), W, H);
            }
            safe_release(st); safe_release(bb);
            break;
        }
        g_swapchain->Present(1, 0);
    }
    printf("Done.\n");
    return 0;
}
