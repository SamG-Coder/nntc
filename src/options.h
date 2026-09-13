// options.h: the command line as a struct.
//
// One material is one to six images of the same size, in any format stb_image reads; texture t contributes output
// channels 3t..3t+2, so nout = 3T.
// The representation is always two latent textures: level 0 at full resolution with C0 channels of bits0[c] bits each, and
// level 1 at a quarter of the resolution per axis with C1 channels of bits1 bits. The block factor between them is the
// constant 4 (model.h), so level 1's sampler always carries an LOD bias of 2.

#pragma once

#include <string>
#include <vector>

// Where a merged per-texture setting came from. The settings rows print it beside the value, because a material can be
// described in two places at once and a reader of the log should never have to guess which of them won.
enum class SettingSource
{
    Default,
    Json,
    CommandLine
};

inline const char* setting_source_text(SettingSource s)
{
    return s == SettingSource::Json ? "json" : (s == SettingSource::CommandLine ? "command line" : "default");
}

// One input texture's own settings, as they stand after the material JSON (when there is one) and the command line have
// been merged. Everything downstream reads this and not the raw option vectors, so there is one place a setting can
// come from and one place its origin is recorded.
struct TextureSettings
{
    std::string file;                      // the image, as it will be handed to stb_image
    std::string type;                      // a free label, echoed and never interpreted
    std::string filter = "default";        // default | box | mitchell | catmullrom: how the source chain is derived
    bool srgb = false;                     // derive the deep levels in linear light
    std::string edge = "clamp";            // clamp | wrap: the FILTER's edge mode, not the fit's sampler
    bool normal_map = false;               // renormalise every filtered level as a tangent-space normal
    double weight = 1.0;                   // the texture's weight in the objective
    double rgb_weights[3] = { 1.0, 1.0, 1.0 };   // its own channel weights, normalised to mean 1 per texture

    // Whether the MATERIAL JSON's entry carried the key at all, whatever its value. Presence decides one thing:
    // whether a command-line override has something to warn about, because a material that named a key and a flag
    // that replaced it are two people describing one texture whatever the two values were. Whether a DERIVED chain
    // was asked for is a question about the value and not about presence, so the implied filter below reads the
    // merged values and not these flags.
    bool json_filter = false, json_srgb = false, json_edge = false, json_normal_map = false, json_weight = false,
         json_rgb_weights = false;

    // The key that made the filter `box` when nobody named a filter at all. A value of `srgb` true, `edge` wrap or
    // `normal_map` true needs a derived chain, and the built-in iterated box derives nothing for it to apply to, so
    // the merge switches such a texture to the resizer's own box, which re-derives every level DIRECTLY FROM THE
    // BASE; that differs from the built-in iterated chain wherever an odd plane is halved (docs/DESIGN.md 4.2, the
    // footprint paragraph). It is recorded rather than inferred because the settings row has to name the key that
    // implied it, and only the merge knows which of srgb, edge, normal_map came first in that fixed precedence.
    std::string filter_implied_by;

    SettingSource src_type = SettingSource::Default;
    SettingSource src_filter = SettingSource::Default;
    SettingSource src_srgb = SettingSource::Default;
    SettingSource src_edge = SettingSource::Default;
    SettingSource src_normal_map = SettingSource::Default;
    SettingSource src_weight = SettingSource::Default;
    SettingSource src_rgb_weights = SettingSource::Default;
};

struct Options
{
    std::vector<std::string> inputs;      // one to six image paths, all the same size, in any format stb_image reads
    std::vector<TextureSettings> tex;     // the merged per-texture settings, one entry per input
    std::string prefix;                   // -o: a prefix, or a directory the first input's base name is placed in
    std::string desc;                     // the asset's descriptor: PREFIX_nntc.json, or the exact name -o gave when
                                          //   -o itself ended in .json. Never PREFIX.json, so that a prefix derived
                                          //   from an input can never name the input
    std::string json_path;                // the material JSON the inputs and their settings were read from, if any;
                                          //   empty on a bare command line, which is the vanilla path
    bool no_mkdir = false;                // --no-mkdir: refuse rather than create the output directory. -o DIR/ makes
                                          //   DIR by default, which is what a run in a script wants; a caller who
                                          //   would rather hear about a mistyped directory than find a new one asks
                                          //   for this, and the refusal comes before anything is written

    int c0 = 2;                           // --c0 N: level-0 channels, 1..4
    std::string l0 = "bc8";               // --l0 bc8|palette: how level 0 is solved and what its texel means.
                                          //   bc8      the DEFAULT: a CONTINUOUS 8-bit plane on a per-channel lo/hi
                                          //            grid, solved by the same sparse least squares level 1 is solved
                                          //            by, then BC4 / BC5-encoded and the pack refined block by block
                                          //            against the decoder's own error. --bits0 does not apply (it is
                                          //            8) and --bc0 must be 1, because the pack is the point.
                                          //   palette  the 1-4 bit index on the fixed [-1,1] palette, chosen by the
                                          //            exact per-texel search of block (c). The pack is lossless at
                                          //            1-3 bits, which is why the mode is kept.
    bool bc_refine_given = false, bc_refine_after_given = false, bc_outer_given = false;   // named on the command
                                          //   line, which --l0 palette refuses by name rather than ignoring
    int bc_refine = 4;                    // --bc-refine N: rounds of the per-block refinement of the pack (0 = off, the
                                          //   post-fit seed pack on its own)
    int bc_refine_after = 0;              // --bc-refine-after N: further rounds after level 1 and the decoder have been
                                          //   refitted against the packed plane
    int bc_outer = 2;                     // --bc-outer N: outer repack passes (refit, re-solve level 0, repack), each
                                          //   taken only if the SHIPPED objective falls (0 = off)
    std::vector<int> bits0 = { 3 };       // --bits0 B[,B,..]: bits per level-0 channel, each 1..4 (one applies to all)
    bool bits0_given = false;             // --bits0 was named on the command line, which --l0 bc8 refuses by name
                                          //   rather than silently ignoring
    int c1 = 4;                           // --c1 N: level-1 channels, 1..4
    int bits1 = 8;                        // --bits1 B: bits for every level-1 channel, 4..8
    int bc0 = 1;                          // --bc0 0|1|both: 1 writes level 0 as BC4 (one channel) or BC5 (two), and in
                                          //   two files above that - BC5 + BC4 for three channels, two BC5s for four;
                                          //   0 writes the uncompressed plane
    bool bc0_both = false;                // --bc0 both: also write the asset as PREFIX_u with level 0 uncompressed,
                                          //   from the SAME solve, so the two decode paths can be compared against each
                                          //   other without a second encode standing between them

    int mip_min = 8;                      // --mip-min S: halve the chain while level 0 stays >= S per axis
    int mips = 1;                         // --mips 0|1: 1 stores both chains under the 1:1 rule, 0 writes one level per texture
    int k = 4;                            // --k K: fixed subtexel sites per texel, K in {0,1,2,4}
    int rounds = 20;                      // --rounds R: the block-round budget
    double tol = 1e-4;                    // --tol X: stop when the relative drop in E is below X on two consecutive rounds
    int q1_start = 3;                     // --q1-start R0: the round at which level 1's grid is fitted and frozen and block
                                          //   (b) becomes quantised sweeps; 0 keeps level 1 continuous and rounds it once at
                                          //   the end
    int q1_sweeps = 4;                    // --q1-sweeps N: four-colour Gauss-Seidel sweeps per quantised block (b)
    std::string q1_range = "pct";         // --q1-range minmax|pct: level 1's grid range, the plane's own min/max or its
                                          //   0.1 % / 99.9 % percentiles per channel
    std::string init = "box";             // --init pca|box: level 1's initialisation. box is the default because the
                                          //   two split on the measured images - pca wins on one, box on another - and
                                          //   box is the one that needs no eigenbasis to explain
    std::string init0_scope = "base";     // --init0-scope base|chain: where the residual the seed's direction comes
                                          //   from is measured. base, the default, is the base plane alone; chain is
                                          //   every stored plane with the objective's own per-site weight, so that a
                                          //   texture whose full-resolution need only appears down the chain is
                                          //   visible when the channels are handed out. chain was measured and is NOT
                                          //   the default: it is worth a little on a multi-texture material (mg1-4 at
                                          //   --c1 3, E 6.281e-5 -> 6.214e-5) and costs a little on every single
                                          //   image (model10 45.65 -> 45.48 dB, chroma 43.67 -> 43.52), because a
                                          //   single image's deep planes want the same direction its base does.
    std::string init0 = "residual";       // --init0 residual|texture|luma: level 0's initialisation.
                                          //   residual  the DEFAULT: each channel in turn is the leading principal
                                          //             direction of what the decode still gets wrong, projected onto
                                          //             the plane and scaled to fill [-1,1], with the decoder refitted
                                          //             after each one so the next channel sees a residual the last
                                          //             one has already been given the chance to explain.
                                          //   texture   the same, except that the direction is taken from the single
                                          //             TEXTURE whose own residual carries the most weighted variance
                                          //             and is zero on every other texture, so a channel is dedicated
                                          //             to a picture rather than being a blend of all of them.
                                          //   luma      the control: channel 0 is the source luminance and every
                                          //             further channel starts at zero, with no direction at all.
    // --weights and --mip-filter are per-texture LISTS and both default to EMPTY, which means "every texture takes the
    // default". A non-empty list must give exactly one entry per input texture; there is deliberately no single-value
    // broadcast and no separate "was it given" flag, so the vector cannot disagree with a flag about whether the
    // command line spoke. (--bits0 keeps its own broadcast: it is per level-0 CHANNEL, and its count is itself a
    // layout decision, so one value standing for all of them says something a per-texture list cannot.)
    std::vector<double> weights;                  // --weights W[,W,..]: per-texture loss weight
    std::vector<std::string> mip_filter;          // --mip-filter F[,F,..]: per-texture source-chain filter
    std::vector<double> rgb_weights = { 1, 1, 1 };// --rgb-weights R,G,B: per-channel weight inside a texture, normalised to mean 1
    bool rgb_weights_given = false;       // --rgb-weights was named: ONE triple, applied to every texture. It is not a
                                          //   per-texture list, so a flag here cannot disagree with a count
    std::string mip_weight = "pixels";    // --mip-weight sqrt|pixels|uniform: how the chain shares the decoder's
                                          //   least-squares mass. pixels gives every plane the same per-sample weight
                                          //   and is the default because it won at every level on both measured
                                          //   images (docs/RESULTS.md, the mip-weight ablation)
    double mip_mix = 0.5;                 // --mip-mix X: the chain's total share of that mass
    double ridge = 1e-4;                  // --ridge X: level 1's proximal ridge, relative to mean(diag H)
    int sweeps = 2;                       // --sweeps N: level-0 coordinate sweeps when the joint enumeration is too large
    int device = 0;                       // --device N: the CUDA device index
    int png = 1;                          // --png 0|1: write the recon and source PNGs per level
    bool check = false;                   // --check: recompute E by brute force on the host and print both
    bool diag = false;                    // --diag: the per-plane diagnosis table - every texture's psnr at every
                                          //   stored level, and every latent channel's mean / sd / min / max and the
                                          //   share of its values sitting on an end of the stored grid, per plane
    bool quiet = false;                   // --quiet: NO PROGRESS AT ALL. The banner and its settings rows, the
                                          //   per-round lines, the level-0 init block, the grid freeze, the pack, the
                                          //   refinement and the outer passes are all suppressed. Warnings, errors,
                                          //   the `wrote` lines and the whole final report always print: a quiet run
                                          //   is a run with less progress on the screen, never one that says less
                                          //   about what it produced.
    bool help = false;                    // --help / --help-advanced: the usage was asked for, which is not a refusal
};
