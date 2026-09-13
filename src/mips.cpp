// mips.cpp: the stored chain.
//
// Output mip m is decoded from mip m of BOTH latents (the 1:1 rule), so the two chains are halved together: level 0's
// plane m is mip_dim applied m times to the padded extent, level 1's is mip_dim applied m times to a quarter of it.
// Floor halving means level 1's plane m is not exactly a quarter of level 0's once an odd size appears in either chain;
// the drift is sub-texel and is what the hardware's own chains do.
//
// The source chain is built once at start-up and plane m is fitted against src_m, so no latent is ever filtered into
// existence. It is also the ONLY place the encoder chooses its own ground truth, which is why the choice is per
// texture: an albedo is filtered in linear light, a tangent-space normal map is renormalised afterwards and a packed
// mask is neither, and a material carries all three at once.
//
// The default chain is the iterated 2x2 box - src_m = box2x2(src_{m-1}) over all 3T channels - and when no texture asks
// for anything else that is the whole of this file's work. A texture with a filter is derived DIRECTLY FROM THE PADDED
// BASE at every level instead, through the vendored resizer, which also changes the footprint: the iterated box drops
// the last column or row of an odd plane and so covers a sub-rectangle of the base, while a direct resize covers all
// of it (docs/DESIGN.md 4.2).

#include <cmath>

#include "model.h"
#include "options.h"
#include "stb_image_resize2.h"

void build_plane_sizes(Model& m, int mip_min, bool want_mips)
{
    const int levels = want_mips ? mip_count(m.width, m.height, mip_min) : 0;
    m.planes.assign((size_t)levels + 1, PlaneSize());
    m.planes[0].w0 = m.width;
    m.planes[0].h0 = m.height;
    m.planes[0].w1 = m.width / BLOCK;
    m.planes[0].h1 = m.height / BLOCK;
    for (int i = 1; i <= levels; i++)
    {
        m.planes[i].w0 = mip_dim(m.planes[i - 1].w0);
        m.planes[i].h0 = mip_dim(m.planes[i - 1].h0);
        m.planes[i].w1 = mip_dim(m.planes[i - 1].w1);
        m.planes[i].h1 = mip_dim(m.planes[i - 1].h1);
    }
}

// The filter name as the resizer's own enum. `default` never reaches here: it is the iterated box and does not go
// through the resizer at all.
static stbir_filter filter_of(const std::string& name)
{
    if (name == "box")
        return STBIR_FILTER_BOX;
    if (name == "catmullrom")
        return STBIR_FILTER_CATMULLROM;
    return STBIR_FILTER_MITCHELL;
}

bool build_source_chain(const Image& base, const std::vector<PlaneSize>& planes,
                        const std::vector<TextureSettings>& settings, std::vector<Image>& chain)
{
    const int levels = (int)planes.size() - 1;
    chain.assign((size_t)levels + 1, Image());
    chain[0] = base;

    // THE VANILLA PATH, byte for byte. A bare command line asks for no filter on any texture, and then the chain is
    // the one the tree has always built, over the whole interleaved image at once.
    bool filtered = false;
    for (const TextureSettings& s : settings)
        filtered = filtered || s.filter != "default";
    if (!filtered)
    {
        for (int i = 1; i <= levels; i++)
            image_box(chain[i - 1], chain[i]);
        return true;
    }

    std::vector<float> in, out;
    for (int i = 1; i <= levels; i++)
    {
        chain[i].w = planes[(size_t)i].w0;
        chain[i].h = planes[(size_t)i].h0;
        chain[i].nc = base.nc;
        chain[i].v.assign((size_t)chain[i].w * chain[i].h * chain[i].nc, 0.0f);
        for (size_t t = 0; t < settings.size(); t++)
        {
            const TextureSettings& s = settings[t];
            if (s.filter == "default")
            {
                image_box_texture(chain[i - 1], chain[i], (int)t);
                continue;
            }
            // Every level is resized from the BASE, not from the level above it: an iterated filter is a different
            // filter at every depth, and the deep planes are where the difference between a box and a windowed cubic
            // is worth having.
            image_texture_extract(chain[0], (int)t, in);
            if (s.srgb)
                for (float& v : in)
                    v = srgb_to_linear(v);
            out.assign((size_t)chain[i].w * chain[i].h * 3, 0.0f);
            if (!stbir_resize(in.data(), chain[0].w, chain[0].h, chain[0].w * 3 * (int)sizeof(float), out.data(),
                              chain[i].w, chain[i].h, chain[i].w * 3 * (int)sizeof(float), STBIR_RGB,
                              STBIR_TYPE_FLOAT, s.edge == "wrap" ? STBIR_EDGE_WRAP : STBIR_EDGE_CLAMP,
                              filter_of(s.filter)))
            {
                fprintf(stderr, "ERROR: the %s resize of texture %zu to %dx%d failed\n", s.filter.c_str(), t,
                        chain[i].w, chain[i].h);
                return false;
            }
            // THE CLAMP, always, and before anything else reads the level. Mitchell and Catmull-Rom have negative
            // lobes, so a resized value can land outside [0,1] wherever the source has an edge. What that costs is not
            // a nan - image.h's transfer functions take their linear branch on a small negative and hand one back -
            // but a TARGET outside the range the source itself lives in: the objective reads the chain raw, every
            // level's PSNR is taken against it and the `_src` PNGs are written from it, so a deep level has to stay in
            // the byte range the base is in. stb's float path does no clamping of its own. The source is in [0,1], so
            // this takes nothing away that was ever in the picture; --diag prints each level's range and says so.
            for (float& v : out)
                v = saturate01(v);
            if (s.srgb)
                for (float& v : out)
                    v = saturate01(linear_to_srgb(v));
            if (s.normal_map)
                for (size_t p = 0; p < out.size(); p += 3)
                    normal_renorm_rgb(&out[p]);
            image_texture_insert(chain[i], (int)t, out);
        }
    }
    return true;
}

// The share of the decoder's least-squares mass each plane gets, and the per-site weight that share implies.
//
// Blocks (b) and (c) are per plane and independent, so the only thing a mip weight can govern is how much of the one
// shared decoder W and b each level gets. Let N_m be plane m's pixel count and r_m its raw weight:
//
//     uniform  r_m = 1           every plane the same total mass, so a deep 8x8 plane's texel weighs N_0 / 64 times a
//                                base texel - which is why uniform is not the default
//     sqrt     r_m = sqrt(N_m)   halfway between the two, and the recipe every good run has used
//     pixels   r_m = N_m         every plane the same weight PER SITE, so the chain is one big image
//
// The base is held out of that split: it takes s_0 = 1 - mix, and the chain shares mix between its levels in
// proportion to r_m. Dividing by the plane's site count N_m (1 + K) turns a share into the weight one site carries,
// which is the omega that multiplies v v^T in the normal equations.
//
// A single stored plane has nothing to share, so its per-site weight is 1: E is then a plain weighted mean and the
// mix has no meaning.
void build_mip_weights(const std::vector<PlaneSize>& planes, const std::string& kind, double mix, int k,
                       std::vector<double>& share, std::vector<double>& per_site)
{
    const size_t n = planes.size();
    share.assign(n, 0.0);
    per_site.assign(n, 0.0);
    if (n == 1)
    {
        share[0] = 1.0;
        per_site[0] = 1.0;
        return;
    }
    std::vector<double> r(n, 0.0);
    double total = 0.0;
    for (size_t m = 1; m < n; m++)
    {
        const double pixels = (double)planes[m].w0 * (double)planes[m].h0;
        r[m] = kind == "uniform" ? 1.0 : (kind == "sqrt" ? std::sqrt(pixels) : pixels);
        total += r[m];
    }
    share[0] = 1.0 - mix;
    for (size_t m = 1; m < n; m++)
        share[m] = total > 0.0 ? mix * r[m] / total : 0.0;
    for (size_t m = 0; m < n; m++)
    {
        const double sites = (double)planes[m].w0 * (double)planes[m].h0 * (double)(1 + k);
        per_site[m] = sites > 0.0 ? share[m] / sites : 0.0;
    }
}
