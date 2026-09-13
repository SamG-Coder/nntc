// image.h: the host-side image, its I/O, the edge-replicating pad and the 2x2 box filter that builds the source mip chain.
//
// An Image holds nc interleaved float channels in [0,1], one pixel after another, row by row from the top. A material of T
// textures is packed into one Image with nc = 3T: texture t owns channels 3t..3t+2. Nothing here knows about latents.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "stb_image.h"
#include "stb_image_write.h"

struct Image
{
    int w = 0;
    int h = 0;
    int nc = 0;
    std::vector<float> v;

    float* at(int x, int y) { return &v[((size_t)y * w + x) * nc]; }
    const float* at(int x, int y) const { return &v[((size_t)y * w + x) * nc]; }
};

// Open a path that may have come out of a material JSON.
//
// A JSON string is UTF-8 by definition, and on Windows the narrow CRT file calls read their argument in the process's
// ANSI code page - so a `"file": "café.png"` reaches fopen as bytes that name nothing, while the SAME file typed
// on the command line opens, because argv is already ANSI. That inconsistency is what this removes: on Windows the
// UTF-8 bytes are decoded to UTF-16 and the file is opened with _wfopen, everywhere else they are handed to fopen as
// they stand, which is what a UTF-8 filesystem wants.
//
// The decoding is spelled out here rather than taken from MultiByteToWideChar because this header is compiled by nvcc
// as part of the CUDA translation units, and <windows.h> there is a liability out of all proportion to twenty lines of
// UTF-8. A byte sequence that is not valid UTF-8 is handed to the narrow fopen unchanged: it is then either an ANSI
// path that works, or a path that does not exist, and the caller's own "cannot read" names it either way.
inline FILE* image_fopen_utf8(const std::string& path, const char* mode)
{
#ifdef _WIN32
    std::wstring wide;
    wide.reserve(path.size());
    size_t i = 0;
    bool ascii = true;
    while (i < path.size())
    {
        const unsigned char c = (unsigned char)path[i];
        unsigned int cp = 0;
        size_t extra = 0;
        if (c < 0x80) { cp = c; extra = 0; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1Fu; extra = 1; ascii = false; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0Fu; extra = 2; ascii = false; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07u; extra = 3; ascii = false; }
        else
            return fopen(path.c_str(), mode);   // a continuation or an invalid lead byte: not UTF-8
        if (i + extra >= path.size())
            return fopen(path.c_str(), mode);
        for (size_t k = 1; k <= extra; k++)
        {
            const unsigned char cc = (unsigned char)path[i + k];
            if ((cc & 0xC0) != 0x80)
                return fopen(path.c_str(), mode);
            cp = (cp << 6) | (cc & 0x3Fu);
        }
        i += extra + 1;
        if (cp >= 0x10000u)
        {
            cp -= 0x10000u;
            wide.push_back((wchar_t)(0xD800u + (cp >> 10)));
            wide.push_back((wchar_t)(0xDC00u + (cp & 0x3FFu)));
        }
        else
            wide.push_back((wchar_t)cp);
    }
    if (ascii)
        return fopen(path.c_str(), mode);   // nothing to convert, and the narrow call is the one every platform takes
    std::wstring wmode;
    for (const char* p = mode; *p; p++)
        wmode.push_back((wchar_t)(unsigned char)*p);
    return _wfopen(wide.c_str(), wmode.c_str());
#else
    return fopen(path.c_str(), mode);
#endif
}

// Load one image as three float channels in [0,1] (byte / 255), in any format stb_image reads.
//
// An input WITH AN ALPHA CHANNEL is IGNORED rather than refused: a texture of the material carries exactly three
// channels, so the alpha has nowhere to go, but the RGB underneath it is exactly what the caller meant to encode, and
// refusing the file only makes them convert it by hand first. What must not happen is that it goes unsaid, so
// `alpha_ignored` comes back true and the caller warns once about it, naming the file.
//
// The load asks for the FILE's own channels (req_comp 0) and this function does the reduction, because the channel
// count stb reports under a req_comp is not always the file's: two of the formats it reads decide their channel count
// rather than read it. stbi__gif_header hands back four for every GIF and stbi__psd_load four for every PSD, whatever
// the file holds, so a req_comp 3 load of an ordinary opaque GIF used to report an alpha channel that was never there
// and the warning fired on every one of them. With the file's own channels in hand the alpha can be looked at instead
// of guessed at: an alpha plane that is 255 everywhere carries no transparency, dropping it loses nothing, and nothing
// is said. The reduction itself is stb's: grey repeated across the three, and the fourth channel dropped.
//
// A 16-bit file still arrives through the 8-bit API, which reduces it to bytes exactly as it did before - req_comp
// changes which channels come back, not their depth.
inline bool image_load_rgb(const std::string& path, Image& img, bool& alpha_ignored)
{
    int w = 0, h = 0, comp = 0;
    alpha_ignored = false;
    FILE* f = image_fopen_utf8(path, "rb");
    if (!f)
        return false;
    unsigned char* p = stbi_load_from_file(f, &w, &h, &comp, 0);
    fclose(f);
    if (!p)
        return false;
    const size_t pixels = (size_t)w * h;
    if (comp == 2 || comp == 4)
        for (size_t i = 0; i < pixels; i++)
            if (p[i * (size_t)comp + (size_t)comp - 1] != 255)
            {
                alpha_ignored = true;
                break;
            }
    img.w = w;
    img.h = h;
    img.nc = 3;
    img.v.resize(pixels * 3);
    for (size_t i = 0; i < pixels; i++)
    {
        const unsigned char* s = p + i * (size_t)comp;
        const float r = (float)s[0] * (1.0f / 255.0f);
        const float g = comp >= 3 ? (float)s[1] * (1.0f / 255.0f) : r;
        const float b = comp >= 3 ? (float)s[2] * (1.0f / 255.0f) : r;
        img.v[i * 3 + 0] = r;
        img.v[i * 3 + 1] = g;
        img.v[i * 3 + 2] = b;
    }
    stbi_image_free(p);
    return true;
}

// Write one texture of an nc = 3T image as an 8-bit RGB PNG, rounding each channel to the nearest byte.
inline bool image_save_texture(const std::string& path, const Image& img, int texture, int crop_w, int crop_h)
{
    const int cw = crop_w > 0 ? std::min(crop_w, img.w) : img.w;
    const int ch = crop_h > 0 ? std::min(crop_h, img.h) : img.h;
    std::vector<uint8_t> out((size_t)cw * ch * 3);
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++)
            for (int c = 0; c < 3; c++)
            {
                const float f = img.at(x, y)[3 * texture + c];
                const float s = std::min(1.0f, std::max(0.0f, f)) * 255.0f;
                out[((size_t)y * cw + x) * 3 + c] = (uint8_t)std::lround(s);
            }
    return stbi_write_png(path.c_str(), cw, ch, 3, out.data(), cw * 3) != 0;
}

// Write an already-8-bit interleaved buffer (nc channels, one texture's triple taken out of it) as a PNG.
inline bool image_save_bytes(const std::string& path, const uint8_t* rgb, int w, int h, int nc, int texture,
                             int crop_w, int crop_h)
{
    const int cw = crop_w > 0 ? std::min(crop_w, w) : w;
    const int ch = crop_h > 0 ? std::min(crop_h, h) : h;
    std::vector<uint8_t> out((size_t)cw * ch * 3);
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++)
            for (int c = 0; c < 3; c++)
                out[((size_t)y * cw + x) * 3 + c] = rgb[((size_t)y * w + x) * nc + 3 * texture + c];
    return stbi_write_png(path.c_str(), cw, ch, 3, out.data(), cw * 3) != 0;
}

// The sRGB transfer function and its inverse, on floats, in the IEC 61966-2-1 form. They are here rather than taken
// from a library because the whole point of using them is that nothing goes through 8 bits: a source byte becomes a
// float once at load, and a texture whose deeper levels are derived in linear light is converted, filtered and
// converted back entirely in float.
//
// Neither of them is undefined on a negative input - both take the linear branch below their knee, so a small negative
// comes back a small negative - and the clamp that mips.cpp applies to every resized level is not there to protect
// them. It is there because the OBJECTIVE reads the chain raw: a level is a target the fit is measured against and the
// PNGs are written from, so it has to stay inside the byte range the source itself lives in, which a cubic's negative
// lobe leaves the moment the picture has an edge. stb's own float path does not clamp, so nothing else would.
inline float srgb_to_linear(float c)
{
    return c <= 0.04045f ? c * (1.0f / 12.92f) : std::pow((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

inline float linear_to_srgb(float c)
{
    // 1.055 * 1 - 0.055 is one ulp under 1 in float, so a saturated linear texel would come back as 0.99999994 and
    // not as the 1.0 it left as; the top of the range is returned exactly.
    return c <= 0.0031308f ? c * 12.92f : (c >= 1.0f ? 1.0f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
}

inline float saturate01(float c)
{
    // Written so that a nan lands on 0 rather than passing through: nothing upstream produces one today, and if
    // something ever did the chain must still stay inside [0,1].
    return c > 0.0f ? (c < 1.0f ? c : 1.0f) : 0.0f;
}

// One filtered texel of a tangent-space normal map, renormalised by the VIEWER'S OWN RULE (viewer/bin/nntc_view.hlsl,
// key V), spelled the same way here so that what the encoder fits the deep levels against is what a shader that
// normalises after the fetch would see there. A filtered normal is shorter than the one it came from - averaging two
// unit vectors that disagree does not give a unit vector - and re-lengthening it is what a consumer does.
//
// Only a texel that can BE a tangent-space normal is touched. A grey-ish texel unpacks to a near-zero vector with no
// direction to normalise towards, and a texel whose z is negative points into the surface, which no tangent-space
// normal does; both are left exactly as the filter left them. Object-space normals are out of scope: they are free to
// point anywhere, so this rule would refuse to touch half of one and mangle the rest.
inline void normal_renorm_rgb(float* rgb)
{
    const float nx = rgb[0] * 2.0f - 1.0f, ny = rgb[1] * 2.0f - 1.0f, nz = rgb[2] * 2.0f - 1.0f;
    const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (!(len > 0.5f) || !(nz > 0.0f))
        return;
    rgb[0] = saturate01(nx / len * 0.5f + 0.5f);
    rgb[1] = saturate01(ny / len * 0.5f + 0.5f);
    rgb[2] = saturate01(nz / len * 0.5f + 0.5f);
}

// One texture's three channels taken out of an nc = 3T image, and put back into one. The resizer works on a tightly
// packed RGB buffer, so a material is split a texture at a time, resized and re-interleaved.
inline void image_texture_extract(const Image& img, int texture, std::vector<float>& rgb)
{
    rgb.resize((size_t)img.w * img.h * 3);
    for (size_t p = 0; p < (size_t)img.w * img.h; p++)
        for (int c = 0; c < 3; c++)
            rgb[p * 3 + (size_t)c] = img.v[p * (size_t)img.nc + (size_t)(3 * texture + c)];
}

inline void image_texture_insert(Image& img, int texture, const std::vector<float>& rgb)
{
    for (size_t p = 0; p < (size_t)img.w * img.h; p++)
        for (int c = 0; c < 3; c++)
            img.v[p * (size_t)img.nc + (size_t)(3 * texture + c)] = rgb[p * 3 + (size_t)c];
}

// Extend an image to wp x hp by repeating its last column and its last row (the pad block-compressed formats want).
inline void image_pad(Image& img, int wp, int hp)
{
    if (wp == img.w && hp == img.h)
        return;
    Image out;
    out.w = wp;
    out.h = hp;
    out.nc = img.nc;
    out.v.resize((size_t)wp * hp * img.nc);
    for (int y = 0; y < hp; y++)
    {
        const int sy = std::min(y, img.h - 1);
        for (int x = 0; x < wp; x++)
        {
            const int sx = std::min(x, img.w - 1);
            for (int c = 0; c < img.nc; c++)
                out.v[((size_t)y * wp + x) * img.nc + c] = img.v[((size_t)sy * img.w + sx) * img.nc + c];
        }
    }
    img = out;
}

// Interleave T same-size RGB images into one nc = 3T image.
inline void image_pack(const std::vector<Image>& tex, Image& out)
{
    const int t_count = (int)tex.size();
    out.w = tex[0].w;
    out.h = tex[0].h;
    out.nc = 3 * t_count;
    out.v.resize((size_t)out.w * out.h * out.nc);
    for (size_t p = 0; p < (size_t)out.w * out.h; p++)
        for (int t = 0; t < t_count; t++)
            for (int c = 0; c < 3; c++)
                out.v[p * out.nc + 3 * t + c] = tex[t].v[p * 3 + c];
}

// The standard floor halving of a mip chain, never below one texel.
inline int mip_dim(int n)
{
    return n / 2 > 0 ? n / 2 : 1;
}

// out(x,y) = the average of the four source texels (2x,2y), (2x+1,2y), (2x,2y+1), (2x+1,2y+1). A one-wide or one-high
// plane has no second column or row, so it repeats the only one it has; an odd size drops its last column or row, which
// is exactly what floor halving means.
inline void image_box(const Image& in, Image& out)
{
    out.w = mip_dim(in.w);
    out.h = mip_dim(in.h);
    out.nc = in.nc;
    out.v.resize((size_t)out.w * out.h * out.nc);
    for (int y = 0; y < out.h; y++)
        for (int x = 0; x < out.w; x++)
        {
            const int xa = 2 * x < in.w ? 2 * x : in.w - 1, xb = 2 * x + 1 < in.w ? 2 * x + 1 : in.w - 1;
            const int ya = 2 * y < in.h ? 2 * y : in.h - 1, yb = 2 * y + 1 < in.h ? 2 * y + 1 : in.h - 1;
            const float* a = in.at(xa, ya);
            const float* b = in.at(xb, ya);
            const float* c = in.at(xa, yb);
            const float* d = in.at(xb, yb);
            float* o = out.at(x, y);
            for (int k = 0; k < in.nc; k++)
                o[k] = 0.25f * ((a[k] + b[k]) + (c[k] + d[k]));
        }
}

// The same 2x2 box as image_box, over ONE texture's three channels, from `in` into an `out` that is already sized.
// Every channel of image_box is averaged on its own, so the numbers this produces for a texture are bit for bit the
// numbers the whole-image path produces for it: a material in which one texture asks for a filter leaves the other
// textures' chains exactly where they were.
inline void image_box_texture(const Image& in, Image& out, int texture)
{
    for (int y = 0; y < out.h; y++)
        for (int x = 0; x < out.w; x++)
        {
            const int xa = 2 * x < in.w ? 2 * x : in.w - 1, xb = 2 * x + 1 < in.w ? 2 * x + 1 : in.w - 1;
            const int ya = 2 * y < in.h ? 2 * y : in.h - 1, yb = 2 * y + 1 < in.h ? 2 * y + 1 : in.h - 1;
            const float* a = in.at(xa, ya);
            const float* b = in.at(xb, ya);
            const float* c = in.at(xa, yb);
            const float* d = in.at(xb, yb);
            float* o = out.at(x, y);
            for (int k = 3 * texture; k < 3 * texture + 3; k++)
                o[k] = 0.25f * ((a[k] + b[k]) + (c[k] + d[k]));
        }
}

// PSNR of one texture over the first crop_w x crop_h pixels, both sides rounded to 8 bits first:
// mse = mean over the 3 channels of (round(255 a) - round(255 b))^2, psnr = 10 log10(255^2 / mse).
inline double image_psnr_texture(const Image& ref, const uint8_t* rec, int rec_w, int nc, int texture,
                                 int crop_w, int crop_h)
{
    double se = 0.0;
    size_t n = 0;
    for (int y = 0; y < crop_h; y++)
        for (int x = 0; x < crop_w; x++)
            for (int c = 0; c < 3; c++)
            {
                const float f = ref.at(x, y)[3 * texture + c];
                const int a = (int)std::lround(std::min(1.0f, std::max(0.0f, f)) * 255.0f);
                const int b = rec[((size_t)y * rec_w + x) * nc + 3 * texture + c];
                const double d = (double)(a - b);
                se += d * d;
                n++;
            }
    if (n == 0)
        return 0.0;
    const double mse = se / (double)n;
    return mse > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 100.0;
}
