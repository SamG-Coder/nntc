"""The release gate.

It builds nothing. It encodes tests/tiny.png with an already-built nntc_encode, decodes the asset again with the
independent Python reader, checks the objective against its own brute-force twin, packs level 0 with bc_check and
greps the tree for the things that must never appear in it.
    python tests/run_checks.py [--build-dir DIR]
"""

import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The vendored single-header libraries are carried verbatim and are never edited, so they are not searched.
VENDORED = {'stb_image.h', 'stb_image_write.h', 'stb_image_resize2.h', 'json.h', 'bcdec.h'}

# The patterns are assembled from pieces so that this file does not match itself.
FORBIDDEN = [
    ('a reference to the previous tree', re.compile('neural' + r'[0-9]')),
    ('an absolute Windows path', re.compile('C' + ':' + r'\\')),
    ('a line tag', re.compile(r'\[[A-Z]{2,5}\]')),
    ('an unfinished-work marker', re.compile('TO' + 'DO')),
    ('a placeholder marker', re.compile('XX' + 'X')),
    # The old program name coming back. The tree was renamed and every derived name took the second n with it, so the
    # one-n spelling followed by an underscore - in a path, a target or a sentence of prose - is a name that was missed
    # or one that has crept back. The lookbehind lets the current names through (nntc_encode, PREFIX_nntc.json), and
    # the format string that does still carry the old spelling is hyphenated (ntc-dds-1), not underscored, so it is
    # not this pattern's business.
    ('the old program name', re.compile('(?<![nN])' + 'ntc' + '_')),
]

# The eight bytes every PNG opens with, written out rather than escaped inline so that a tool that rewrites this
# file's line endings cannot corrupt the carriage return inside them.
PNG_MAGIC = bytes([137, 80, 78, 71, 13, 10, 26, 10])

# The absolute floor under every 'did E rise' comparison, the encoder's own E_NOISE: on a picture the representation
# reproduces exactly E is float rounding noise (1e-19 on an 8x8 of alternating columns), and below about 1e-10 the
# decoder solve's own ridge (1e-9 of its normal matrix's diagonal) decides its step rather than E; a relative test
# alone reads either as a rise. Deliberately loose: three orders of magnitude under the smallest real objective
# and a relative 1e-6; a real defect raises E by a fraction of itself.
E_NOISE = 1e-8
E_REL = 1e-6

SEARCHED_DIRS = ['src', 'shared', 'tools', 'tests', 'docs', 'viewer', 'viewer_vk']
SEARCHED_FILES = ['README.md', 'CMakeLists.txt', 'LICENSE', 'PRIOR_ART_DISCLOSURE.md']

# What the run reports at the end: one row per gate, in the order they were reached.
SUMMARY = []


def record(gate, detail):
    SUMMARY.append((gate, detail))


def out_asset(name, source='tiny'):
    """The -o argument for a check and the prefix its asset then lands at.

    -o NAME, with no extension and no file of that name, is a DIRECTORY and is created (nntc_encode --help), so the
    asset inside it takes the input's own base name: `-o out/tiny_bc` writes out/tiny_bc/tiny_lat0.dds and the rest.
    Every check names its own directory under out/, which also keeps one case's files away from another's.
    """
    return os.path.join('out', name), os.path.join('out', name, source)


def executable(build_dir, name):
    for candidate in (os.path.join(build_dir, 'Release', name + '.exe'),
                      os.path.join(build_dir, name + '.exe'),
                      os.path.join(build_dir, name)):
        if os.path.isfile(candidate):
            return candidate
    raise SystemExit('FAIL: %s was not found under %s; build first' % (name, build_dir))


def run(cmd):
    print('$ ' + ' '.join(cmd))
    # The executables print UTF-8 (their manifest makes the active code page UTF-8), so their output is decoded as
    # UTF-8 whatever the console's own code page is; a byte that is not UTF-8 is replaced, never fatal.
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc


def check_tree():
    failures = []
    paths = [os.path.join(ROOT, f) for f in SEARCHED_FILES]
    for d in SEARCHED_DIRS:
        for dirpath, dirnames, names in os.walk(os.path.join(ROOT, d)):
            dirnames[:] = [x for x in dirnames if x != '__pycache__']   # compiled Python is not source
            for n in names:
                if n not in VENDORED:
                    paths.append(os.path.join(dirpath, n))
    for path in paths:
        try:
            text = open(path, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        for label, pattern in FORBIDDEN:
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    failures.append('%s:%d: %s: %s' % (os.path.relpath(path, ROOT), i, label, line.strip()))
    return failures


# The round line carries the chain's per-level PSNR between `sampled psnr` and `moved0` whenever mips are stored, so
# the middle of the line is skipped rather than spelled out. A dot does not match a newline here, so this cannot run on
# into the next round's line.
ROUND = re.compile(r'^round\s+(\d+)\s+E ([0-9.e+-]+) ([0-9.e+-]+) ([0-9.e+-]+)\s+psnr ([0-9.e+-]+)\s+'
                   r'sampled psnr ([0-9.e+-]+).*?moved0 (\d+)\s+moved1 (\d+)', re.MULTILINE)


def loop_checks(text, tol):
    """The round loop's gates. Every block of every round minimises E in its own variables - blocks (b) and (c)
    exactly, block (a) subject to its ridge and its own acceptance test, so that E does not rise across it beyond the
    tolerance - so the three numbers each round prints must never rise, across rounds as well as inside one (the
    comparison below carries the loose tolerance that covers block (a)'s). The single exception is block
    (b) of the round that freezes level 1's grid: snapping the plane onto the grid applies a constraint rather than
    taking a step, and a constraint can only cost E. Every later block (b) is the quantised sweeps, which minimise the
    same objective over that grid and so may not raise E either - that is the gate on the quantised phase. The loop must
    also end, for one of the three stated reasons, and a stop on the tolerance must be backed by the drop the last round
    printed."""
    rounds = [[float(g) for g in hit.groups()] for hit in ROUND.finditer(text)]
    if not rounds:
        raise SystemExit('FAIL: the encoder printed no round lines')
    freeze = re.search(r'^grid frozen: round (\d+)', text, re.MULTILINE)
    frozen_round = int(freeze.group(1)) if freeze else 0
    previous = None
    quantised_blocks = 0
    for r in rounds:
        index = int(r[0])
        for block, e in zip('abc', r[1:4]):
            exempt = index == frozen_round and block == 'b'
            if previous is not None and e > previous * (1.0 + E_REL) + E_NOISE and not exempt:
                raise SystemExit('FAIL: E rose from %g to %g across block (%s) of round %d'
                                 % (previous, e, block, index))
            if frozen_round and index >= frozen_round and not exempt:
                quantised_blocks += 1
            previous = e
    print('  the loop: E never rises across the %d blocks of %d rounds (%.6e -> %.6e)'
          % (3 * len(rounds), len(rounds), rounds[0][1], rounds[-1][3]))
    if frozen_round:
        print('  the loop: the grid froze at round %d and E is monotone across the %d blocks after it'
              % (frozen_round, quantised_blocks))

    stop = re.search(r'rounds\s+(\d+) \(stopped on (tol|budget|no-move)\)', text)
    if not stop:
        raise SystemExit('FAIL: the report has no rounds line')
    used, reason = int(stop.group(1)), stop.group(2)
    if used != len(rounds):
        raise SystemExit('FAIL: the report says %d rounds but %d round lines were printed' % (used, len(rounds)))
    if reason == 'tol' and len(rounds) >= 2:
        drop = (rounds[-2][3] - rounds[-1][3]) / rounds[-1][3]
        if drop >= tol:
            raise SystemExit('FAIL: the loop stopped on tol after a relative drop of %g' % drop)
    print('  the loop: %d rounds, stopped on %s' % (used, reason))
    return rounds


def quantisation_checks(text, what):
    """The gates on level 1's grid: the plane the device held was exactly the grid value of every index it stored, and
    the writer re-quantised every level-1 byte it wrote back to the index it came from. The encoder refuses to write the
    asset if either fails, so this asserts that both were reached and that neither was empty."""
    on_grid = re.search(r'level 1 on grid: (\d+) values, every one exactly the grid value of its stored index', text)
    if not on_grid or int(on_grid.group(1)) == 0:
        raise SystemExit('FAIL: %s: the encoder did not report a level-1 plane on its grid' % what)
    written = re.search(r'level 1 grid: (\d+) stored indices, every one re-quantises to itself', text)
    if not written or int(written.group(1)) == 0:
        raise SystemExit('FAIL: %s: the writer did not report the zero-change assertion' % what)
    if int(written.group(1)) != int(on_grid.group(1)):
        raise SystemExit('FAIL: %s: %s values on the grid but %s indices written'
                         % (what, on_grid.group(1), written.group(1)))
    print('  the grid: %s: %s values, on the grid and re-quantising to themselves' % (what, written.group(1)))


# The REPORT's mip psnr line, which starts a line and ends in `dB`; the round lines carry the same levels in fp but
# neither start a line nor name a unit, so this cannot pick one of them up by accident.
MIP_PSNR = re.compile(r'^ *mip psnr\s+((?:M\d+ +[0-9.]+ *)+)dB', re.MULTILINE)


def mip_levels(text, what):
    """The per-level PSNR of a `mip psnr M1 .. MM` line, as a dict keyed by the level."""
    hit = MIP_PSNR.search(text)
    if not hit:
        raise SystemExit('FAIL: %s has no mip psnr line' % what)
    levels = {}
    for m, db in re.findall(r'M(\d+) ([0-9.]+)', hit.group(1)):
        levels[int(m)] = float(db)
    if not levels:
        raise SystemExit('FAIL: %s printed an empty mip psnr line' % what)
    return levels


def level_checks(encode_out, prefix):
    """The gates on the chain.

    Every plane of both latents is a parameter fitted against its own mip of the run's own source chain (the 1:1 rule), so
    the chain's levels have to be as real as the base's and as readable from the files alone. Three things are asserted:
    the encoder stored a chain at all; the independent Python decoder reproduces the encoder's own reconstruction at
    EVERY level to within one 8-bit step; and that decoder's per-level PSNR, computed from the .dds and the .json with
    nothing of the encoder trusted, is the number the encoder's own report printed, to 0.01 dB. The last one is what
    says the report is measuring the asset rather than some state the encoder happened to hold."""
    reported = mip_levels(encode_out, 'the encoder report')
    base = re.findall(r'psnr\s+texture \d+\s+([0-9.]+) dB', encode_out)
    if not base:
        raise SystemExit('FAIL: the encoder report has no per-texture base psnr line')
    reported_base = min(float(b) for b in base)

    proc = run([sys.executable, os.path.join('tools', 'dds_decode.py'), prefix, '--ref', prefix + '_recon'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: the independent decoder disagrees with the encoder at some level')
    worst = re.search(r'worst max \|diff\| over every level: (\d+)', proc.stdout)
    if not worst or int(worst.group(1)) > 1:
        raise SystemExit('FAIL: no round-trip result, or it is above one step')
    seen = set(int(m) for m in re.findall(r'^M(\d+) texture \d+:', proc.stdout, re.MULTILINE))
    if len(seen) != len(reported) + 1:
        raise SystemExit('FAIL: the report names %d chain levels but the decoder read %d levels in all'
                         % (len(reported), len(seen)))
    print('  the chain: max |diff| %s at every one of the %d stored levels' % (worst.group(1), len(seen)))

    proc = run([sys.executable, os.path.join('tools', 'dds_decode.py'), prefix, '--psnr-levels'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: dds_decode.py --psnr-levels returned %d' % proc.returncode)
    measured = mip_levels(proc.stdout, 'dds_decode.py --psnr-levels')
    hit = re.search(r'^psnr M0 ([0-9.]+) dB', proc.stdout, re.MULTILINE)
    if not hit:
        raise SystemExit('FAIL: dds_decode.py --psnr-levels printed no base psnr')
    if abs(float(hit.group(1)) - reported_base) > 0.01:
        raise SystemExit('FAIL: the encoder says %.2f dB at the base and the asset says %.2f dB'
                         % (reported_base, float(hit.group(1))))
    if set(measured) != set(reported):
        raise SystemExit('FAIL: the report names levels %s and the asset holds %s'
                         % (sorted(reported), sorted(measured)))
    for level in sorted(reported):
        if abs(measured[level] - reported[level]) > 0.01:
            raise SystemExit('FAIL: M%d: the encoder says %.2f dB and the asset says %.2f dB'
                             % (level, reported[level], measured[level]))
    print('  the chain: the encoder\'s psnr / mip psnr equals the asset\'s at every level (M0 %.2f, %s)'
          % (reported_base, ' '.join('M%d %.2f' % (l, reported[l]) for l in sorted(reported))))


# Not a gate: the checks encode the same small image every time, so the line below is a rough but free comparison from
# one build to the next. It is printed and nothing is asserted about it - a machine under load would fail an assertion
# that means nothing about the code.
TIMING = re.compile(r'^ *time +([0-9.]+) s total, \(a\) ([0-9.]+) ms / \(b\) ([0-9.]+) ms / \(c\) ([0-9.]+) ms'
                    r' / E ([0-9.]+) ms over (\d+) passes', re.MULTILINE)


def timing(text, what):
    hit = TIMING.search(text)
    if not hit:
        print('  timing: %s printed no time line' % what)
        return
    print('  timing: %s: %s s total, (a) %s / (b) %s / (c) %s / E %s ms over %s passes'
          % (what, hit.group(1), hit.group(2), hit.group(3), hit.group(4), hit.group(5), hit.group(6)))


RIDGE = re.compile(r'block \(a\) ridge: (\d+) standard, (\d+) reduced, (\d+) none, (\d+) refused')


def ridge_tally(text, what):
    """The report's block (a) ridge line, as the four counts. It says which rung of the ridge ladder each of the run's
    decoder solves ended on, and it is the only place the output says it. An ordinary image must be all-standard: the
    lower rungs are reached only when the normal matrix is near-singular, and a run that quietly starts taking them is
    a run whose decoder is no longer the one the same source always produced."""
    hit = RIDGE.search(text)
    if not hit:
        raise SystemExit('FAIL: %s printed no "block (a) ridge:" line' % what)
    return [int(g) for g in hit.groups()]


def number(text, pattern, what):
    hit = re.search(pattern, text)
    if not hit:
        raise SystemExit('FAIL: the encoder report has no %s line' % what)
    return [float(g) for g in hit.groups()]


def objective_checks(encode, prefix):
    """The objective gates: E against its brute-force twin, E against the plain centre mean squared error, that one
    call of the decoder's least squares never raises E, and that level 1's solve converges, agrees with a dense direct
    solve of the same system, leaves a zero gradient behind and never raises E either."""
    for k in ('4', '0'):
        args = [encode, os.path.join('tests', 'tiny.png'), '-o', prefix, '--l0', 'palette', '--c0', '2', '--bits0',
                '3', '--c1', '4', '--bits1', '8', '--k', k, '--check', '--png', '0']
        if k == '0':
            args += ['--mips', '0']
        proc = run(args)
        if proc.returncode != 0:
            raise SystemExit('FAIL: nntc_encode --check returned %d at --k %s' % (proc.returncode, k))
        out = proc.stdout
        loop_checks(out, 1e-4)

        rel = number(out, r'relative difference ([0-9.e+-]+), worst over E and every plane ([0-9.e+-]+)',
                     'E check')
        if max(rel) > 1e-9:
            raise SystemExit('FAIL: the two E paths differ by %g at --k %s' % (max(rel), k))
        print('  --k %s: the device E and the host brute force agree to %.1e' % (k, max(rel)))

        e = number(out, r'E\s+([0-9.e+-]+) before block \(a\), ([0-9.e+-]+) after', 'E')
        if e[1] > e[0]:
            raise SystemExit('FAIL: block (a) raised E from %g to %g at --k %s' % (e[0], e[1], k))
        print('  --k %s: block (a) took E from %.6e to %.6e' % (k, e[0], e[1]))

        eb = number(out, r'E block \(b\)\s+([0-9.e+-]+) after block \(a\), ([0-9.e+-]+) after block \(b\)',
                    'E block (b)')
        if eb[1] > eb[0] * (1.0 + E_REL) + E_NOISE:
            raise SystemExit('FAIL: block (b) raised E from %g to %g at --k %s' % (eb[0], eb[1], k))
        print('  --k %s: block (b) took E from %.6e to %.6e' % (k, eb[0], eb[1]))

        cg = number(out, r'block \(b\)\s+iterations (\d+)\s+residual ([0-9.e+-]+)', 'block (b)')
        if cg[1] >= 1e-6:
            raise SystemExit('FAIL: the level-1 conjugate gradients stopped at a relative residual of %g at --k %s'
                             % (cg[1], k))
        print('  --k %s: block (b) converged in %d iterations to a relative residual of %.1e' % (k, int(cg[0]), cg[1]))

        dense = number(out, r'block \(b\) dense max \|delta device - delta host\| [0-9.e+-]+ over a range of '
                            r'[0-9.e+-]+, relative ([0-9.e+-]+)  \(the bar is 1e-6, over the (\d+) plane',
                       'block (b) dense')
        if dense[0] >= 1e-6:
            raise SystemExit('FAIL: the device solve and the dense direct solve differ by %g of the range at --k %s'
                             % (dense[0], k))
        print('  --k %s: the dense direct solve agrees to %.1e of the correction\'s range, over %d plane(s)'
              % (k, dense[0], int(dense[1])))

        # The gradient probe runs on EVERY plane, not only the base, and the report carries the worst of them.
        fd = number(out, r'block \(b\) fd max \|dE/dx\| h / E over \d+ values of EACH of the (\d+) planes '
                         r'([0-9.e+-]+)', 'block (b) fd')
        if fd[1] >= 1e-5:
            raise SystemExit('FAIL: E still has a gradient of %g in level 1 after block (b) at --k %s' % (fd[1], k))
        print('  --k %s: the finite-difference gradient of E in level 1 is %.1e on the worst of %d planes'
              % (k, fd[1], int(fd[0])))

        if k == '0':
            same = number(out, r'E = centre\s+E ([0-9.e+-]+)\s+centre mse ([0-9.e+-]+)\s+'
                               r'relative difference ([0-9.e+-]+)', 'E = centre')
            if same[2] > 1e-12:
                raise SystemExit('FAIL: at --k 0 with one plane E %g is not the centre mean squared error %g'
                                 % (same[0], same[1]))
            print('  --k 0 --mips 0: E %.17g is the centre mean squared error %.17g' % (same[0], same[1]))



def round_trip(prefix, ref_prefix, what):
    """The independent Python decoder against the encoder's own reconstruction, at every stored level.

    On a block-compressed level 0 this is more than a round trip: dds_decode.py decodes the BC blocks itself and
    asserts, at 1-3 bits, that the byte it gets back is the quantisation index's own byte - so a level that came out
    of the block format changed is a failure there and not a quiet drift in the last bit here.
    """
    proc = run([sys.executable, os.path.join('tools', 'dds_decode.py'), prefix, '--ref', ref_prefix])
    worst = re.search(r'worst max \|diff\| over every level: (\d+)', proc.stdout)
    if proc.returncode != 0 or not worst or int(worst.group(1)) > 1:
        raise SystemExit('FAIL: %s does not round-trip within one step at every level' % what)
    print('  round trip: %s, max |diff| %s at every level' % (what, worst.group(1)))


def shot(view, asset, out, extra):
    """One --shot frame of the viewer, without the debug overlay (which names the file and its format, so two assets
    that must render the same would still differ inside it).

    There is no warm-up launch. Two things used to make a fresh --shot unsteady and both are gone. The swap chain was
    created at the constants 2560x1440 while the window Windows actually gave was smaller, so the first WM_SIZE resized
    the backbuffer and whether it reached the pump before or after the one frame --shot draws was a race; the viewer
    sizes the swap chain from GetClientRect and treats a resize to the size it already has as nothing. And the frame
    read the LIVE KEYBOARD - GetAsyncKeyState for the held keys, WM_KEYDOWN for the struck ones - while its own freshly
    created window was the foreground one, so a keystroke anywhere on the machine during the few milliseconds a --shot
    launch lives moved the camera or changed the filter for that frame; --shot now takes no input at all. shot_stable
    below is the gate on that.

    It returns the frame AND the viewer's stderr, because some of what the viewer has to say about a command line --
    a --tex outside the material, say -- is a warning and not a return code, and a check that greps the ENCODER's
    stdout for it is asserting nothing at all.
    """
    proc = run([view, asset, '--nooverlay', '--shot', out] + extra)
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_view --shot returned %d on %s' % (proc.returncode, asset))
    return open(os.path.join(ROOT, out), 'rb').read(), proc.stderr


def shot_stable(view, asset, out, extra, runs=5):
    """The same --shot launched `runs` times: every frame must be the same bytes.

    This is the basis of every frame comparison below, and it is a gate of its own because it failed on its own. Two
    assets that must render identically were compared by their frames, and the comparison failed in three suite runs of
    six - not because the assets differed but because one of the two launches had drawn a different frame: --shot read
    the keyboard, and the release gate runs while somebody is using the machine. A frame that is not a function of the
    command line and the asset alone cannot be the basis of any of the comparisons that follow, so it is asserted first
    and the difference is reported where it is, rather than as a difference between two assets."""
    first, _ = shot(view, asset, out, extra)
    for i in range(1, runs):
        again, _ = shot(view, asset, out, extra)
        if again != first:
            raise SystemExit('FAIL: two --shot launches of %s drew different frames (run %d of %d), so the '
                             'viewer frame is not a function of the command line and the asset alone'
                             % (asset, i + 1, runs))
    print('  the viewer: %d --shot launches of %s are byte-identical (%d bytes)'
          % (runs, os.path.basename(asset), len(first)))
    return first


def vulkan_viewer(build_dir):
    """The Vulkan viewer's executable, or None when this tree did not build one.

    It is None on a machine with no Vulkan SDK, where the CMake target is skipped and says so. It is NOT None off
    Windows any more: the target builds on Linux and its --shot needs no window, no surface and no desktop, so every
    arm that does not need a Direct3D frame to compare against runs there too (the GLFW window needs a desktop session,
    and no arm of this gate opens one). Every case that drives it reports SKIPPED rather than failing
    when it is missing or when the machine has no Vulkan device, which is what keeps the Direct3D coverage from
    depending on a second api being installed.
    """
    for candidate in (os.path.join(build_dir, 'Release', 'nntc_view_vk.exe'),
                      os.path.join(build_dir, 'nntc_view_vk.exe'),
                      os.path.join(build_dir, 'nntc_view_vk')):
        if os.path.isfile(candidate):
            return candidate
    return None


# The plain-GLSL decode, named explicitly wherever an arm written for stages 1 to 3 draws. Since stage 4 the viewer
# takes the cooperative-vector path BY DEFAULT on a device that offers it, and every assertion below - the pairs against
# the Direct3D viewer, the byte-identical launches, the flag toggles, the BC pack - is about the baseline path, which is
# the product. The cooperative-vector arms name --coopvec 1 for themselves. A device without the extension accepts
# --coopvec 0 as well: it is always honoured, and only --coopvec 1 can be refused.
#
# A BUILD without the extension reads the same way as a DEVICE without it, which is the whole point of the
# compile-time guard: viewer_vk/main.cpp prints the same 'cooperative vectors: not available - <reason>' line, with a
# reason naming its Vulkan headers, and refuses --coopvec 1 with the same ERROR and the same exit 1. Every arm below
# is keyed on those two shapes and not on the reason's words, so it needs nothing added for it; the one thing that is
# added is the sentence the skip prints, because 'this machine has no such GPU' and 'this build cannot have used one'
# are different facts about a run and a reader should not have to tell them apart from the reason string alone.
VK_PLAIN = ['--coopvec', '0']
# The column the Vulkan viewer draws its decode-path token at, and the same arithmetic its main.cpp does:
# OVL_W - 12 * 8 * FONT_SCALE, twelve characters in from the right-hand end of the 2560-pixel strip.
VK_COOP_COLUMN = 2560 - 12 * 8 * 2


def vk_shot(view, asset, out, extra, size=(640, 480)):
    """One headless --shot of the Vulkan viewer, the second arm of everything shot() drives.

    It returns the frame, the stderr and the stdout -- the viewer says some of what it has to say about a command line
    on each -- or None when the machine reports no Vulkan device, in which case the case that asked for it says
    skipped. The size is given explicitly and is small: this viewer creates no window, so it renders at whatever size
    it is asked for, and none of the checks that go through here compare against a Direct3D frame (the ones that do ask
    for the Direct3D frame's own size instead).
    """
    proc = run([view, asset, '--nooverlay', '--shot', out, '--size', str(size[0]), str(size[1])]
               + VK_PLAIN + extra)
    if proc.returncode != 0 and 'no Vulkan device' in proc.stderr:
        return None
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_view_vk --shot returned %d on %s: %r' % (proc.returncode, asset,
                                                                             proc.stderr[-300:]))
    return open(os.path.join(ROOT, out), 'rb').read(), proc.stderr, proc.stdout


def vk_shot_stable(view, asset, out, extra, runs=5, size=(640, 480)):
    """The same headless --shot launched `runs` times: every frame must be the same bytes.

    The claim is stronger here than for the Direct3D viewer and the reason is structural rather than a matter of care:
    this --shot creates no window and no surface at all, so there is no keystroke for it to receive and no compositor
    between it and the image it writes. Returns the frame, or None on a machine with no Vulkan device.
    """
    first = vk_shot(view, asset, out, extra, size)
    if first is None:
        return None
    for i in range(1, runs):
        again = vk_shot(view, asset, out, extra, size)
        if again is None or again[0] != first[0]:
            raise SystemExit('FAIL: two headless nntc_view_vk --shot launches of %s drew different frames (run %d of '
                             '%d)' % (asset, i + 1, runs))
    print('  the Vulkan viewer: %d --shot launches of %s are byte-identical (%d bytes)'
          % (runs, os.path.basename(asset), len(first[0])))
    return first[0]


def vk_skip_note(reason):
    """What a shared case's record() says when its Vulkan arm did not run.

    A skipped arm must never read as a passed one, so the summary carries the skip and its reason rather than falling
    silent about it -- and the cases whose Vulkan arm DID run say 'and both viewers' instead.
    """
    return ' (Vulkan arm skipped: %s)' % reason


def vk_six_texture_arm(build_dir, desc):
    """The Vulkan half of the six-texture case: six --tex frames that must be pairwise different, and a --tex past the
    end that must warn in the same words and fall back to texture 0.

    It is here rather than inside the case because it needs no Direct3D frame, so it runs on Linux as well, where the
    Direct3D half is skipped. It returns the text the case's record() appends.
    """
    vk = vulkan_viewer(build_dir)
    if not vk:
        print('  the Vulkan viewer: skipped, nntc_view_vk was not built')
        return vk_skip_note('nntc_view_vk was not built')
    frames = {}
    for t in range(6):
        got = vk_shot(vk, desc, os.path.join('out', 'tiny_six_vk%d.bmp' % t), ['--tex', str(t)])
        if got is None:
            print('  the Vulkan viewer: skipped, no Vulkan device on this machine')
            return vk_skip_note('this machine reports no Vulkan device')
        if 'WARNING: --tex' in got[1]:
            raise SystemExit('FAIL: --tex %d is inside a six-texture material and must not warn on the Vulkan '
                             'viewer: %r' % (t, got[1].strip()))
        frames[t] = got[0]
    for a in range(6):
        for b in range(a + 1, 6):
            if frames[a] == frames[b]:
                raise SystemExit('FAIL: the Vulkan viewer drew the same frame for --tex %d and --tex %d, so its '
                                 'shader is not selecting the output triple' % (a, b))
    past = vk_shot(vk, desc, os.path.join('out', 'tiny_six_vk6.bmp'), ['--tex', '6'])
    if past is None or 'WARNING: --tex 6 is outside 0..5' not in past[1]:
        raise SystemExit('FAIL: --tex 6 on a six-texture material must warn on the Vulkan viewer\'s stderr, not %r'
                         % (past and past[1].strip()))
    if past[0] != frames[0]:
        raise SystemExit('FAIL: a --tex past the end must fall back to texture 0 on the Vulkan viewer too')
    print('  the Vulkan viewer: the six --tex frames are pairwise different and --tex 6 warns and falls back to 0 '
          '(%d bytes each)' % len(frames[0]))
    return ' on both viewers'


def two_file_layout(prefix, c0):
    """The file layout of a three- or four-channel level 0, from the JSON and the files on disk.

    BC5 carries two channels, so three or four of them take two files. FOUR are two BC5s (channels 0-1 and 2-3, 16
    bits per texel). THREE are a BC5 of channels 0-1 and a BC4 holding channel 2 alone, 12 bits per texel: a second
    BC5 there would spend a quarter of the plane on a padding channel nothing reads, which is what this asserts is no
    longer happening. Each entry of "files" is an object carrying its own format and channel count, because the two
    files no longer share either, and the second file's byte count must be half the first's when it is a BC4.
    """
    meta = json.load(open(os.path.join(ROOT, prefix + '_nntc.json')))
    tex = meta['textures'][0]
    files = tex.get('files')
    if not files or len(files) != 2 or 'file' in tex:
        raise SystemExit('FAIL: a %d-channel level 0 must be two files named in "files" and no "file"' % c0)
    want = [(83, 2), (83, 2) if c0 == 4 else (80, 1)]
    for entry, (dxgi, stored) in zip(files, want):
        if not isinstance(entry, dict):
            raise SystemExit('FAIL: each "files" entry must be an object with its own format, not %r' % (entry,))
        if entry.get('dxgi_format_id') != dxgi or entry.get('channels_stored') != stored:
            raise SystemExit('FAIL: %s must be DXGI %d storing %d channel(s), the JSON says %s / %s'
                             % (entry.get('file'), dxgi, stored, entry.get('dxgi_format_id'),
                                entry.get('channels_stored')))
    if tex['dxgi_format_id'] != want[0][0] or tex['channels_stored'] != want[0][1]:
        raise SystemExit("FAIL: the entry's own dxgi_format_id / channels_stored must describe the first file")
    if tex.get('bc_palette') != 'standard':
        raise SystemExit('FAIL: the JSON must record which BC palette was fitted against')
    sizes = []
    for entry in files:
        path = os.path.join(ROOT, os.path.dirname(prefix), entry['file'])
        if not os.path.isfile(path):
            raise SystemExit('FAIL: the JSON names %s, which is not there' % entry['file'])
        sizes.append(os.path.getsize(path))
    # Both files are the 148-byte header and then the same block grid, at 16 bytes a block for a BC5 and 8 for a BC4.
    payload = [s - 148 for s in sizes]
    expect = payload[0] if c0 == 4 else payload[0] // 2
    if payload[1] != expect or payload[0] % 2:
        raise SystemExit('FAIL: the two files hold %d and %d bytes of blocks, expected %d for the second'
                         % (payload[0], payload[1], expect))
    print('  two files: %s (%s, %d channel(s), %d bytes) and %s (%s, %d channel(s), %d bytes), bc_palette standard'
          % (files[0]['file'], files[0]['dxgi_format'], files[0]['channels_stored'], sizes[0],
             files[1]['file'], files[1]['dxgi_format'], files[1]['channels_stored'], sizes[1]))


def bc_checks(encode, build_dir):
    """The gates on level 0 as BC4 / BC5 in the .dds.

    Three things have to hold. The asset must round-trip through a decoder that shares no code with the encoder, with
    the quantisation index exact at 1-3 bits. Three or four level-0 channels must come out as two files -- a BC5 and a
    BC4 at three, two BC5s at four -- named in "files" and read as one plane again. And the viewer must render the
    block-compressed asset exactly as it renders
    the same plane packed at load - byte for byte, since at these bit depths the two are the same bytes; that is the
    check that the file's blocks and the viewer's own pack are one thing and not two.

    --bc0 both writes one solve twice, block-compressed and uncompressed, which is what makes the last check possible:
    two separate encodes of the same image are not bit-identical, so a comparison across two runs would mean nothing.
    """
    odir, compressed = out_asset('tiny_bc')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--l0', 'palette', '--bits0', '3',
                '--bc0', 'both'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_encode --bc0 both returned %d' % proc.returncode)
    if 'BC5_UNORM' not in proc.stdout:
        raise SystemExit('FAIL: two level-0 channels must be written as one BC5')
    if 'level 0 pack: lossless' not in proc.stdout:
        raise SystemExit('FAIL: a 3-bit level 0 must pack losslessly')
    print('  the pack: two channels at 3 bits, one BC5 file, lossless')
    # Each asset round-trips against ITS OWN reconstruction. The two are not the same picture: a block-compressed
    # level 0 decodes index k to k / (2^bits - 1) of full scale and an uncompressed one to rep(k) / 255, which at 3
    # bits are different numbers, so the encoder writes the twin's own recon PNGs from the twin's own plane.
    round_trip(compressed, compressed + '_recon', 'the --bc0 1 asset')
    round_trip(compressed + '_u', compressed + '_u_recon', 'the --bc0 0 asset of the same solve')

    # Three channels: a BC5 and a BC4, named in "files" rather than in "file". --bc0 both so that the second file has
    # an uncompressed plane to be compared against, which is the check that used to be skipped.
    odir, three = out_asset('tiny_bc3')
    three_u = three + '_u'
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--l0', 'palette', '--c0', '3',
                '--bits0', '3', '--bc0', 'both'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_encode --c0 3 returned %d' % proc.returncode)
    loop_checks(proc.stdout, 1e-4)
    two_file_layout(three, 3)
    round_trip(three, three + '_recon', 'the two-file level 0')

    # Four bits, where the pack is LOSSY: eight palette entries per block against sixteen values. The asset the encoder
    # reports on must be the packed plane, so --psnr-levels read from the files alone has to agree with the report's
    # own psnr and mip psnr line, decibel for decibel.
    odir, four = out_asset('tiny_bc4')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--l0', 'palette', '--bits0', '4',
                '--bc0', 'both'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_encode --bits0 4 --bc0 both returned %d' % proc.returncode)
    if 'level 0 pack: lossy' not in proc.stdout:
        raise SystemExit('FAIL: a 4-bit level 0 must pack lossily')
    if 'the report below is of the PACKED plane' not in proc.stdout:
        raise SystemExit('FAIL: the encoder must say that the report is of the packed plane')
    reported = mip_levels(proc.stdout, 'the 4-bit report')
    base = re.findall(r'psnr\s+texture \d+\s+([0-9.]+) dB', proc.stdout)
    reported_base = min(float(b) for b in base)
    dec = run([sys.executable, os.path.join('tools', 'dds_decode.py'), four, '--psnr-levels'])
    if dec.returncode != 0:
        raise SystemExit('FAIL: dds_decode.py --psnr-levels returned %d on the 4-bit asset' % dec.returncode)
    measured = mip_levels(dec.stdout, 'the 4-bit asset')
    hit = re.search(r'^psnr M0 ([0-9.]+) dB', dec.stdout, re.MULTILINE)
    if not hit or abs(float(hit.group(1)) - reported_base) > 0.01:
        raise SystemExit('FAIL: at 4 bits the encoder says %.2f dB at the base and the asset says %s'
                         % (reported_base, hit.group(1) if hit else 'nothing'))
    for level in sorted(reported):
        if abs(measured[level] - reported[level]) > 0.01:
            raise SystemExit('FAIL: at 4 bits M%d: the encoder says %.2f dB and the asset says %.2f dB'
                             % (level, reported[level], measured[level]))
    round_trip(four, four + '_recon', 'the lossy 4-bit pack')
    print('  4 bits: the pack is lossy and the report is of the packed plane (M0 %.2f, %s), equal to the asset\'s'
          % (reported_base, ' '.join('M%d %.2f' % (l, reported[l]) for l in sorted(reported))))

    if os.name != 'nt':
        print('  bc_check and the viewer: skipped (Windows only)')
        return
    check = executable(build_dir, 'bc_check')
    proc = run([check, compressed + '_nntc.json', compressed + '_u_nntc.json'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: bc_check returned %d on the block-compressed asset' % proc.returncode)
    if 'the index round trip' not in proc.stdout or 'lossless' not in proc.stdout:
        raise SystemExit('FAIL: bc_check did not validate the index round trip')
    if not re.search(r'the decoded bytes vs the uncompressed plane\s*: max \|diff\| 0 / 255', proc.stdout):
        raise SystemExit('FAIL: the blocks in the file do not decode to the uncompressed plane of the same solve')
    print('  bc_check: the file\'s blocks decode to the uncompressed plane exactly, index lossless')
    proc = run([check, three + '_nntc.json', three_u + '_nntc.json'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: bc_check returned %d on the two-file asset' % proc.returncode)
    # The lossless assertion is per CHANNEL, so a three-channel level 0 - a BC5 of channels 0-1 and a BC4 of channel 2
    # alone, no padding channel anywhere - must produce three of those lines, one of them for channel 2, which lives in
    # the second file and used to be skipped entirely.
    lossless_lines = re.findall(r'channel (\d+), (\d+) bits: the index round trip .*lossless', proc.stdout)
    if len(lossless_lines) != 3 or [int(c) for c, _ in lossless_lines] != [0, 1, 2]:
        raise SystemExit('FAIL: a three-channel level 0 must assert the index round trip for channels 0, 1 and 2, '
                         'not %s' % lossless_lines)
    if len(re.findall(r'the decoded bytes vs the uncompressed plane\s*: max \|diff\| 0 / 255', proc.stdout)) != 2:
        raise SystemExit('FAIL: both files of the two-file level 0 must be compared against the uncompressed plane')
    print('  bc_check: the two-file level 0 validates channels 0, 1 and 2, both files against the uncompressed plane')

    # The frame comparisons, and first the thing they rest on: that one asset renders to the same bytes every launch.
    view = executable(build_dir, 'nntc_view')
    a = shot_stable(view, compressed + '_nntc.json', os.path.join('out', 'tiny_bc_file.bmp'), [])
    b, _ = shot(view, compressed + '_u_nntc.json', os.path.join('out', 'tiny_bc_load.bmp'), ['--bc'])
    if a != b:
        raise SystemExit('FAIL: the viewer renders the BC asset differently from the same plane packed at load')
    print('  the viewer: the BC-in-file frame is byte-identical to the packed-at-load frame (%d bytes)' % len(a))

    # And the same for the THREE-channel asset, whose level 0 is two files of different formats bound to t0 and t2: the
    # file's own BC5 + BC4 against the pack the viewer makes of the uncompressed twin of the same solve. It exercises
    # the two-texture binding and the shader's "use .r of t2 at three channels" path, which the two-channel case does
    # not reach at all.
    a3, _ = shot(view, three + '_nntc.json', os.path.join('out', 'tiny_bc3_file.bmp'), [])
    b3, _ = shot(view, three_u + '_nntc.json', os.path.join('out', 'tiny_bc3_load.bmp'), ['--bc'])
    if a3 != b3:
        raise SystemExit('FAIL: the viewer renders the two-file BC level 0 differently from the same plane packed at '
                         'load')
    print('  the viewer: the three-channel BC5 + BC4 frame is byte-identical to the packed-at-load frame (%d bytes)'
          % len(a3))


def bc8_ship_checks(text, what):
    """The outer repack loop, the last free refit, and the restore -- read off the shipped `E` and nothing else.

    The encoder ends a bc8 run with one more block (b) and block (a) over a level 0 that does not move, and the line
    it prints for that is the one place the whole tail of the run can be checked from. Its STARTING `E` is whatever
    the loop left, which must be the last ACCEPTED pass's `E`, or -- when no pass was accepted, so a rejected pass
    restored the previous state -- the `E` the loop started from. That equality IS the restore gate: a restore that
    dropped a plane, an index, a block, a grid or the decoder would leave the objective somewhere else. Its ENDING
    `E` must then be the report's own `E shipped`, to the digit, and at or below where it started, because two exact
    minimisations over a fixed plane cannot raise it.
    """
    steps = re.search(r'E [0-9.e+-]+ psnr [0-9.]+ before the pack, E [0-9.e+-]+ psnr [0-9.]+ after it, '
                      r'E ([0-9.e+-]+) psnr [0-9.]+ after the refit', text)
    if not steps:
        raise SystemExit('FAIL: %s printed no post-fit refit line' % what)
    pre_loop = float(steps.group(1))
    # --bc-refine-after runs more refinement passes after that refit, and they are what the loop then starts from.
    after = re.search(r'level 0 refine: \d+ further pass\w* after the refit: E [0-9.e+-]+ psnr [0-9.]+ -> '
                      r'E ([0-9.e+-]+)', text)
    if after:
        pre_loop = float(after.group(1))
    accepted = re.findall(r'level 0 outer : pass \d+ ACCEPTED: shipped E [0-9.e+-]+ psnr [0-9.]+ -> E ([0-9.e+-]+)',
                          text)
    rejected = re.findall(r'level 0 outer : pass (\d+) rejected', text)
    expected = float(accepted[-1]) if accepted else pre_loop

    final = re.search(r'level 0 final : one more refit of level 1 and the decoder against the packed plane: '
                      r'shipped E ([0-9.e+-]+) psnr [0-9.]+ -> E ([0-9.e+-]+) psnr', text)
    if not final:
        raise SystemExit('FAIL: %s printed no final refit line' % what)
    started, ended = float(final.group(1)), float(final.group(2))
    if abs(started - expected) > 1e-12 * max(1.0, abs(expected)):
        raise SystemExit('FAIL: %s: the loop left E %g and the last %s pass says %g -- the restore did not put the '
                         'previous state back' % (what, started, 'accepted' if accepted else 'pre-loop', expected))
    if ended > started * (1.0 + E_REL) + E_NOISE:
        raise SystemExit('FAIL: %s: the last refit raised E from %g to %g, which its two minimisations forbid'
                         % (what, started, ended))
    shipped = re.search(r'^ *E shipped\s+([0-9.e+-]+)', text, re.MULTILINE)
    if not shipped:
        raise SystemExit('FAIL: %s has no E shipped line' % what)
    if abs(float(shipped.group(1)) - ended) > 1e-12 * max(1.0, abs(ended)):
        raise SystemExit('FAIL: %s: the last refit ended at E %g and the report ships %s'
                         % (what, ended, shipped.group(1)))
    print('  %s: %d outer pass%s accepted, %d rejected; the loop left E %.6e, the last free refit took it to %.6e, '
          'and that is E shipped' % (what, len(accepted), '' if len(accepted) == 1 else 'es', len(rejected), started,
                                     ended))
    return ended


def bc8_checks(encode, build_dir):
    """The gates on --l0 bc8: level 0 solved as a continuous 8-bit plane and BC4 / BC5-encoded once after the last
    round, with one refit of level 1 and the decoder against the packed plane.

    What has to hold:

      - the loop is monotone, exactly as the palette mode's is. Block (c') is a minimisation of the same objective in
        level 0's values (continuous while its grid is open, the exact grid argmin once it is frozen), so nothing in
        the round may raise E;
      - the POST-FIT step is the one place E may rise, and only there: the pack is a constraint being applied, so E
        after it is at or above E before it, and the refit that follows is two minimisations, so E after the refit is
        at or below E after the pack. Both numbers are reported beside the pack's own psnr. The pack's error is in the
        BYTE units of the latent and the objective is in the output's, so there is no arithmetic that turns one into a
        bound on the other; what is asserted is the direction of each step, which is what each step promises;
      - the shipped asset round-trips through the independent Python decoder within one step at every level, and its
        --psnr-levels agrees with the report to a hundredth of a decibel, so the report describes the PACKED plane;
      - the viewer opens it and draws a frame. There is no uncompressed twin to be byte-identical to under this mode,
        because at 8 bits the pack is lossy by construction and the two planes are genuinely different pictures;
      - --bits0 and --bc0 0 are refused BY NAME rather than ignored.
    """
    odir, prefix = out_asset('tiny_bc8')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--l0', 'bc8', '--c0', '2', '--c1', '4'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_encode --l0 bc8 returned %d' % proc.returncode)
    loop_checks(proc.stdout, 1e-4)
    quantisation_checks(proc.stdout, '--l0 bc8')

    meta = json.load(open(os.path.join(ROOT, prefix + '_nntc.json')))
    lat0 = meta['textures'][0]
    if lat0['bits_per_channel'] != [8, 8]:
        raise SystemExit('FAIL: --l0 bc8 must store level 0 at 8 bits per channel, not %s' % lat0['bits_per_channel'])
    if lat0['dequantise']['kind'] != 'range' or lat0['dequantise']['levels'] != 255:
        raise SystemExit('FAIL: --l0 bc8 must publish level 0 as kind range with 255 levels')
    if 'palette' in lat0['dequantise']:
        raise SystemExit('FAIL: --l0 bc8 publishes no level-0 palette; there is no index on one')
    if lat0.get('bc_palette') != 'standard' or lat0['dxgi_format_id'] != 83:
        raise SystemExit('FAIL: two bc8 channels must be one BC5 on the standard palette')
    if len(lat0['dequantise']['lo']) < 2 or len(lat0['dequantise']['hi']) < 2:
        raise SystemExit('FAIL: level 0 must publish lo and hi per channel')
    print('  the JSON: level 0 at 8 bits, kind range, 255 levels, lo/hi per channel, BC5, no palette')

    # The pack and the refit, in the order their own arithmetic promises.
    pack = re.search(r'packing psnr ([0-9.]+) dB against the continuous plane', proc.stdout)
    steps = re.search(r'E ([0-9.e+-]+) psnr ([0-9.]+) before the pack, E ([0-9.e+-]+) psnr ([0-9.]+) after it, '
                      r'E ([0-9.e+-]+) psnr ([0-9.]+) after the refit', proc.stdout)
    if not pack or not steps:
        raise SystemExit('FAIL: --l0 bc8 must report the packing psnr and E before the pack, after it and after the '
                         'refit')
    e_pre, psnr_pre, e_packed, psnr_packed, e_refit, psnr_refit = (float(g) for g in steps.groups())
    if e_packed < e_pre * (1.0 - 1e-9):
        raise SystemExit('FAIL: the pack lowered E from %g to %g, which a constraint cannot do' % (e_pre, e_packed))
    if e_refit > e_packed * (1.0 + 1e-9):
        raise SystemExit('FAIL: the refit raised E from %g to %g, which its two minimisations forbid'
                         % (e_packed, e_refit))
    print('  the pack: %.2f dB against the continuous plane; E %.6e -> %.6e (the pack) -> %.6e (the refit), '
          'psnr %.2f -> %.2f -> %.2f' % (float(pack.group(1)), e_pre, e_packed, e_refit, psnr_pre, psnr_packed,
                                         psnr_refit))

    # THE REFINEMENT. It starts from the stb_dxt-style pack the line above measured and takes a block's candidate only
    # when that block's own share of E strictly falls, so E after it can only be at or below E after the seed pack -
    # equality being the case where nothing improved. Every pass also reports the decrease it accepted, which is a sum
    # of negatives and therefore cannot be negative when it is printed as a fall.
    refined = re.search(r'E ([0-9.e+-]+) psnr ([0-9.]+) after the seed pack, E ([0-9.e+-]+) psnr ([0-9.]+) after the '
                        r'refinement', proc.stdout)
    if not refined:
        raise SystemExit('FAIL: --l0 bc8 must report E after the seed pack and after the refinement')
    e_seed, _, e_refined, _ = (float(g) for g in refined.groups())
    if abs(e_seed - e_packed) > 1e-12 * max(1.0, e_packed):
        raise SystemExit('FAIL: the seed pack of the refinement is not the pack the report measured (%g vs %g)'
                         % (e_seed, e_packed))
    if e_refined > e_seed * (1.0 + 1e-9):
        raise SystemExit('FAIL: the refinement raised E from %g to %g, which accepting only strict decreases forbids'
                         % (e_seed, e_refined))
    passes = re.findall(r"pass (\d+): (\d+) of (\d+) blocks improved, the blocks' own E fell by ([0-9.e+-]+)",
                        proc.stdout)
    if not passes:
        raise SystemExit('FAIL: the refinement must report its passes')
    for _, improved, blocks, fell in passes:
        if int(improved) > int(blocks) or float(fell) < 0.0:
            raise SystemExit('FAIL: a refinement pass reported %s of %s blocks improved and a fall of %s'
                             % (improved, blocks, fell))
    print('  the refinement: %d pass%s, E %.6e -> %.6e, %s of %s blocks improved on the first'
          % (len(passes), '' if len(passes) == 1 else 'es', e_seed, e_refined, passes[0][1], passes[0][2]))

    # THE OUTER REPACK LOOP. Every pass is taken only if the SHIPPED E falls, and a pass that does not is the last one.
    for pass_no, before, after in re.findall(r'level 0 outer : pass (\d+) ACCEPTED: shipped E ([0-9.e+-]+) psnr '
                                             r'[0-9.]+ -> E ([0-9.e+-]+)', proc.stdout):
        if float(after) >= float(before):
            raise SystemExit('FAIL: outer pass %s was accepted with E %s -> %s' % (pass_no, before, after))
    rejected = re.findall(r'level 0 outer : pass (\d+) rejected', proc.stdout)
    accepted = re.findall(r'level 0 outer : pass (\d+) ACCEPTED', proc.stdout)
    if rejected and int(rejected[0]) != len(accepted) + 1:
        raise SystemExit('FAIL: the outer loop must stop at its first rejected pass')
    # What the loop, the restore and the last free refit left, against the report's own E shipped.
    e_shipped = bc8_ship_checks(proc.stdout, 'the bc8 defaults')
    if e_shipped > e_refined * (1.0 + 1e-9):
        raise SystemExit('FAIL: the refit, the outer loop and the last refit raised E from %g to %g, which their '
                         'minimisations and their accept-only-if-lower rule forbid' % (e_refined, e_shipped))

    round_trip(prefix, prefix + '_recon', 'the bc8 asset')

    # The report is of the packed plane, so the asset read from the files alone must say the same numbers.
    reported = mip_levels(proc.stdout, 'the bc8 report')
    base = re.findall(r'psnr\s+texture \d+\s+([0-9.]+) dB', proc.stdout)
    reported_base = min(float(b) for b in base)
    dec = run([sys.executable, os.path.join('tools', 'dds_decode.py'), prefix, '--psnr-levels'])
    if dec.returncode != 0:
        raise SystemExit('FAIL: dds_decode.py --psnr-levels returned %d on the bc8 asset' % dec.returncode)
    measured = mip_levels(dec.stdout, 'the bc8 asset')
    hit = re.search(r'^psnr M0 ([0-9.]+) dB', dec.stdout, re.MULTILINE)
    if not hit or abs(float(hit.group(1)) - reported_base) > 0.01:
        raise SystemExit('FAIL: bc8: the encoder says %.2f dB at the base and the asset says %s'
                         % (reported_base, hit.group(1) if hit else 'nothing'))
    for level in sorted(reported):
        if abs(measured[level] - reported[level]) > 0.01:
            raise SystemExit('FAIL: bc8 M%d: the encoder says %.2f dB and the asset says %.2f dB'
                             % (level, reported[level], measured[level]))
    print('  the asset: --psnr-levels equals the report (M0 %.2f, %s)'
          % (reported_base, ' '.join('M%d %.2f' % (l, reported[l]) for l in sorted(reported))))

    # --grid on a bc8 asset used to have nothing to check: the mode publishes no palette, so the palette statement was
    # vacuous. It now asserts the `range` entry's own shape as well -- 255 levels, 8 bits on every used channel, lo and
    # hi per channel with hi above lo -- and prints the line below per level, which is what this greps for.
    dec = run([sys.executable, os.path.join('tools', 'dds_decode.py'), prefix, '--grid'])
    if dec.returncode != 0 or 'no entry publishes a palette' not in dec.stdout:
        raise SystemExit('FAIL: the bc8 asset must publish no palette and still pass --grid')
    ranges = re.findall(r'^level (\d+): kind range, (\d+) levels, bits ([0-9,]+),', dec.stdout, re.MULTILINE)
    if len(ranges) != 2 or ranges[0] != ('0', '255', '8,8'):
        raise SystemExit('FAIL: --grid must assert level 0 as kind range with 255 levels at 8 bits, not %s' % ranges)
    print('  --grid: both levels are kind range and their levels / bits / lo / hi are asserted, not just printed')

    # The two flags the mode fixes, refused by name rather than ignored.
    for args, what in ((['--l0', 'bc8', '--bits0', '4'], '--bits0 under bc8'),
                       (['--l0', 'bc8', '--bc0', '0'], '--bc0 0 under bc8')):
        odir_bad, _ = out_asset('tiny_bc8_refuse')
        bad = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir_bad] + args)
        if bad.returncode == 0 or 'does not apply under --l0 bc8' not in bad.stderr:
            raise SystemExit('FAIL: %s must be refused by name' % what)
    print('  the refusals: --bits0 and --bc0 0 are refused by name under --l0 bc8')

    bc8_coverage_checks(encode, build_dir)

    if os.name != 'nt':
        print('  the viewer: skipped (Windows only)')
        return
    view = executable(build_dir, 'nntc_view')
    frame, _ = shot(view, prefix + '_nntc.json', os.path.join('out', 'tiny_bc8_shot.bmp'), [])
    raw, _ = shot(view, prefix + '_nntc.json', os.path.join('out', 'tiny_bc8_raw0.bmp'), ['--raw0'])
    if frame == raw:
        raise SystemExit('FAIL: the bc8 decode frame is the same bytes as --raw0, so the decoder drew nothing')
    print('  the viewer: opens the bc8 asset and its decode differs from its own --raw0 frame (%d bytes)'
          % len(frame))


def bc8_coverage_checks(encode, build_dir):
    """The bc8 layouts and paths the gate ran on nothing but --c0 2.

    Four of them, each round-tripped through the independent decoder and each put through the shipped-E gate above, so
    the outer loop's restore is exercised on every one:

      - --c0 1, which is a single BC4 file;
      - --c0 3, which is a BC5 of channels 0-1 and a BC4 of channel 2 alone -- the layout this stage introduced, and
        the one place a second file has a different block size from the first;
      - --bc-refine-after 1, the refinement run again after the refit, which moves what the outer loop starts from;
      - --q1-start 0, the control path, where BOTH steps of an outer pass used to be re-fits of a range rather than
        minimisations: the refit is now level 1's quantised sweeps on the grid it already carries, and step (2) is the
        continuous block (c') followed by a snap onto level 0's existing grid, so a pass is weighed against a plane
        whose grid it did not move;
      - a crop whose DEEPEST stored plane is not a multiple of four (40x40 gives 40, 20, 10, 5), so a partial edge
        block exists at the bottom of the chain and the pack, the refinement's four colours and the block error all
        have to handle it.

    Every one of these passed by hand; none of them was a gate.
    """
    import shutil
    import tempfile

    for name, args, what in (
            ('tiny_bc8_c1', ['--c0', '1'], '--l0 bc8 --c0 1 (one BC4)'),
            ('tiny_bc8_c3', ['--c0', '3'], '--l0 bc8 --c0 3 (BC5 + BC4)'),
            ('tiny_bc8_after', ['--c0', '2', '--bc-refine-after', '1'], '--l0 bc8 --bc-refine-after 1'),
            ('tiny_bc8_q0', ['--c0', '2', '--q1-start', '0'], '--l0 bc8 --q1-start 0')):
        odir, prefix = out_asset(name)
        proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--l0', 'bc8'] + args)
        if proc.returncode != 0:
            raise SystemExit('FAIL: nntc_encode %s returned %d' % (what, proc.returncode))
        loop_checks(proc.stdout, 1e-4)
        bc8_ship_checks(proc.stdout, what)
        meta = json.load(open(os.path.join(ROOT, prefix + '_nntc.json')))
        tex = meta['textures'][0]
        if name == 'tiny_bc8_c1':
            if tex['dxgi_format_id'] != 80 or tex['channels_stored'] != 1 or 'files' in tex:
                raise SystemExit('FAIL: a one-channel bc8 level 0 must be one BC4 file storing one channel')
        if name == 'tiny_bc8_c3':
            two_file_layout(prefix, 3)
            # The point of the layout: the SECOND file is a BC4 of one channel and not a second BC5.
            if tex['files'][1]['dxgi_format_id'] != 80 or tex['files'][1]['channels_stored'] != 1:
                raise SystemExit('FAIL: the second file of a three-channel level 0 must be a BC4 storing one channel')
        round_trip(prefix, prefix + '_recon', what)
        print('  %s: solves, ships and round-trips' % what)

    # A crop whose deepest plane is not a multiple of four.
    from PIL import Image as PILImage
    tmp = tempfile.mkdtemp(prefix='nntc_odd_')
    try:
        odd_png = os.path.join(tmp, 'odd.png')
        PILImage.open(os.path.join(ROOT, 'tests', 'tiny.png')).convert('RGB').crop((0, 0, 40, 40)).save(odd_png)
        odir, prefix = out_asset('tiny_bc8_odd', 'odd')
        proc = run([encode, odd_png, '-o', odir, '--l0', 'bc8', '--c0', '2'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: nntc_encode on a 40x40 bc8 input returned %d' % proc.returncode)
        loop_checks(proc.stdout, 1e-4)
        meta = json.load(open(os.path.join(ROOT, prefix + '_nntc.json')))
        deepest = meta['textures'][0]['mip_sizes'][-1]
        bc8_ship_checks(proc.stdout, '--l0 bc8 on a chain ending at %dx%d' % (deepest[0], deepest[1]))
        if deepest[0] % 4 == 0 and deepest[1] % 4 == 0:
            raise SystemExit('FAIL: the odd-size case must end on a plane that is not a multiple of 4, not %s'
                             % (deepest,))
        round_trip(prefix, prefix + '_recon', 'a bc8 chain whose deepest plane is %dx%d' % (deepest[0], deepest[1]))
        print('  --l0 bc8, 40x40: the chain ends at %dx%d, a partial edge block, and the asset round-trips'
              % (deepest[0], deepest[1]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # THE REJECTION. The outer loop's restore path is reached only when a pass fails to lower the shipped E, which the
    # defaults never do on tiny.png. With the refinement turned off the seed pack of a second pass is no better than
    # the refined one it is compared against, so pass 1 is rejected and the previous state is put back -- and the
    # `level 0 final` line's starting E must then be the E the loop started from, to the digit, which is the whole of
    # the restore gate (bc8_ship_checks above).
    odir, prefix = out_asset('tiny_bc8_reject')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--l0', 'bc8', '--c0', '2',
                '--bc-refine', '0', '--bc-outer', '2'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_encode --bc-refine 0 --bc-outer 2 returned %d' % proc.returncode)
    loop_checks(proc.stdout, 1e-4)
    if 'level 0 outer : pass 1 rejected' not in proc.stdout:
        raise SystemExit('FAIL: --bc-refine 0 --bc-outer 2 must reject its first pass and exercise the restore')
    if 'ACCEPTED' in proc.stdout:
        raise SystemExit('FAIL: no pass of the rejection case may be accepted, or the restore is not what is tested')
    bc8_ship_checks(proc.stdout, 'the rejected outer pass')
    round_trip(prefix, prefix + '_recon', 'the asset a rejected outer pass restored')
    print('  the restore: pass 1 rejected, the pre-loop state put back, and the restored asset round-trips')


def grid_checks(encode):
    """The gate on A.1: the grid the encoder FITS is the grid the shipped format DECODES to.

    Three bit depths where the two rules used to disagree. At 3 bits an uncompressed level-0 byte is k k k, which is
    36 / 255 for k = 1 and not 1 / 7; at 5 and 7 bits a level-1 byte is the index replicated the same way. The encoder
    now fits on the byte for an uncompressed plane and on the BC palette for a block-compressed one, so the independent
    decoder -- which dequantises exactly as a sampler does, with no index shift anywhere in the arithmetic -- must
    reproduce the encoder's own reconstruction with NOTHING left over, at every level, on both twins of one solve.
    """
    for name, args in (('--l0 palette --bits0 3 --bc0 both',
                        ['--l0', 'palette', '--bits0', '3', '--bc0', 'both']),
                       ('--bits1 5', ['--bits1', '5']),
                       ('--bits1 7', ['--bits1', '7'])):
        odir, prefix = out_asset('tiny_grid')
        proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir] + args)
        if proc.returncode != 0:
            raise SystemExit('FAIL: nntc_encode %s returned %d' % (name, proc.returncode))
        prefixes = [(prefix, prefix + '_recon')]
        if 'both' in args:
            prefixes.append((prefix + '_u', prefix + '_u_recon'))
        for asset, ref in prefixes:
            # The grid itself, from the .json alone and with no image in it: every published value is the value a
            # sampler returns. That is the whole of A.1, and it is exact - unlike the image round trip below, which
            # carries the encoder's fp32 decode against this reader's fp64 one and is allowed its one 8-bit step.
            dec = run([sys.executable, os.path.join('tools', 'dds_decode.py'), asset, '--grid'])
            if dec.returncode != 0:
                raise SystemExit('FAIL: %s (%s): the published grid is not the sampled grid' % (asset, name))
            round_trip(asset, ref, '%s: %s' % (name, os.path.basename(asset)))
    record('the fit grid', 'the shipped grid decodes exactly as fitted at 3 bits (both formats), 5 and 7 bits')


def coverage_checks(encode, build_dir):
    """The layouts the gate did not exercise: a padded input, one level-0 channel (BC4), four of them (two BC5 files
    and a lossless assertion per channel), and a single stored level through the independent decoder."""
    import shutil
    import tempfile

    # A padded input: 45x30 is a multiple of neither 4 nor anything else, so the encoder pads it by edge replication to
    # 48x32, warns, and reports every PSNR over the original 45x30 extent. It is a crop of tiny.png made in a temporary
    # directory, so nothing is added to tests/.
    from PIL import Image as PILImage
    tmp = tempfile.mkdtemp(prefix='nntc_pad_')
    try:
        pad_png = os.path.join(tmp, 'pad.png')
        PILImage.open(os.path.join(ROOT, 'tests', 'tiny.png')).convert('RGB').crop((0, 0, 45, 30)).save(pad_png)
        odir, prefix = out_asset('tiny_pad', 'pad')
        proc = run([encode, pad_png, '-o', odir])
        if proc.returncode != 0:
            raise SystemExit('FAIL: nntc_encode on a padded input returned %d' % proc.returncode)
        if 'WARNING: the input is 45x30' not in proc.stdout:
            raise SystemExit('FAIL: a padded input must say so')
        meta = json.load(open(os.path.join(ROOT, prefix + '_nntc.json')))
        if not meta['source'].get('padded') or meta['source']['padded_width'] != 48 or meta['source']['height'] != 30:
            raise SystemExit('FAIL: the JSON must record both the original and the padded size')
        round_trip(prefix, prefix + '_recon', 'the padded 45x30 input')
        print('  padding: 45x30 padded to 48x32, both sizes in the JSON, the asset round-trips')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # One level-0 channel: BC4, half a byte a texel, one file.
    odir, one = out_asset('tiny_c1')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--l0', 'palette', '--c0', '1',
                '--bits0', '3'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_encode --c0 1 returned %d' % proc.returncode)
    meta = json.load(open(os.path.join(ROOT, one + '_nntc.json')))
    if meta['textures'][0]['dxgi_format_id'] != 80 or meta['textures'][0]['channels_stored'] != 1:
        raise SystemExit('FAIL: a one-channel level 0 must be one BC4 storing one channel')
    round_trip(one, one + '_recon', 'the BC4 level 0')
    print('  --c0 1: one BC4 file, one channel stored, the asset round-trips')

    # Four level-0 channels: two BC5 files with no padding channel at all, so every one of the four must carry the
    # lossless assertion.
    odir, four = out_asset('tiny_c4')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--l0', 'palette', '--c0', '4',
                '--bits0', '2', '--bc0', 'both'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_encode --c0 4 returned %d' % proc.returncode)
    two_file_layout(four, 4)
    round_trip(four, four + '_recon', 'the four-channel level 0')
    if os.name == 'nt':
        check = executable(build_dir, 'bc_check')
        proc = run([check, four + '_nntc.json', four + '_u_nntc.json'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: bc_check returned %d on the four-channel asset' % proc.returncode)
        lossless = re.findall(r'channel (\d+), \d+ bits: the index round trip .*lossless', proc.stdout)
        if [int(c) for c in lossless] != [0, 1, 2, 3]:
            raise SystemExit('FAIL: all four level-0 channels must carry the lossless assertion, not %s' % lossless)
        print('  --c0 4: two BC5 files, the index round trip asserted for every one of the four channels')

    # One stored level: --mips 0 writes a single level, which carries neither DDSD_MIPMAPCOUNT nor the MIPMAP caps bit,
    # and the Python reader asserts both of those header fields.
    odir, flat = out_asset('tiny_m0')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--mips', '0'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_encode --mips 0 returned %d' % proc.returncode)
    meta = json.load(open(os.path.join(ROOT, flat + '_nntc.json')))
    if meta['textures'][0]['mip_count'] != 1 or meta['textures'][1]['mip_count'] != 1:
        raise SystemExit('FAIL: --mips 0 must store one level per texture')
    round_trip(flat, flat + '_recon', 'the single-level asset')
    print('  --mips 0: one stored level per texture, the header says so and the decoder reads it')
    record('the coverage cases', 'a padded input, --c0 1 (BC4), --c0 4 (two BC5s), --mips 0')


def ill_conditioned_checks(encode):
    """A picture that drives block (a)'s normal matrix nearly singular, which is where its ridge used to decide the
    step instead of the objective.

    One white pixel on black. Level 1's channels converge to within a fraction of their span of constant, the columns
    of A become nearly dependent, and the exact minimiser's norm runs past a hundred - at which point a ridge of 1e-9
    times the mean diagonal is no longer negligible against the solution and the minimiser of the RIDGED problem is not
    the minimiser of E. Before the ladder in solve_decoder.cu the run raised E across block (a) from round 9 on: E
    1.875599e-06 after round 8, 2.112583e-06 after round 9's block (a), a 13 % rise, repeated every round to the end.

    The image is written here rather than carried in tests/ because it is three lines of PIL and nothing else in the
    tree wants it. Three extents, because the defect showed at all of them, and the default layout, because that is the
    one an asset ships at.
    """
    import shutil
    import tempfile
    from PIL import Image as PILImage
    tmp = tempfile.mkdtemp(prefix='nntc_dot_')
    try:
        for width, height in ((100, 36), (64, 64), (128, 64)):
            dot = PILImage.new('RGB', (width, height), (0, 0, 0))
            dot.putpixel((50 % width, 18 % height), (255, 255, 255))
            path = os.path.join(tmp, 'dot_%dx%d.png' % (width, height))
            dot.save(path)
            odir, _ = out_asset('tiny_dot_%dx%d' % (width, height), 'dot_%dx%d' % (width, height))
            proc = run([encode, path, '-o', odir, '--png', '0'])
            if proc.returncode != 0:
                raise SystemExit('FAIL: the %dx%d dot returned %d' % (width, height, proc.returncode))
            if 'raised E' in proc.stdout:
                raise SystemExit('FAIL: the %dx%d dot raised E across a block, which every block forbids: %s'
                                 % (width, height, [l for l in proc.stdout.splitlines() if 'raised E' in l]))
            rounds = loop_checks(proc.stdout, 1e-4)
            # The ladder's own arm, on the one image that reaches it: the line has to be there and to add up, and the
            # run has to stay free of an E rise whichever rungs it took. Which rungs those are is NOT asserted - it is
            # a property of a near-singular matrix and not of the encoder - but it is printed, because a change in the
            # mix is the first thing to look at if this gate ever starts failing.
            tally = ridge_tally(proc.stdout, 'the %dx%d dot' % (width, height))
            if sum(tally) <= 0:
                raise SystemExit('FAIL: the %dx%d dot counted no block (a) calls' % (width, height))
            print('  the %dx%d dot: %d rounds, no block raised E (%.6e -> %.6e), block (a) ridge %d/%d/%d/%d'
                  % (width, height, len(rounds), rounds[0][1], rounds[-1][3],
                     tally[0], tally[1], tally[2], tally[3]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record('a near-singular block (a)',
           'one white pixel on black at three extents: E monotone, no ridge-driven rise, the ridge tally reported')


def init0_checks(encode):
    """The gates on --init0.

    UNDER --l0 bc8 (the default) the residual seed is a sequence of minimisations with one free step between them: the
    channel is written into a plane whose column is ZERO, which block (a) can only have seen as zero, so writing it
    cannot move E, and the refit that follows is block (a), which does not raise it beyond its tolerance. The whole
    seed is therefore monotone in E, channel by channel, and the report prints both ends of every step - which is what
    is checked here, alongside the loop's own monotonicity afterwards and a round trip of the asset either init made.

    UNDER --l0 palette IT IS NOT, and the gate says so rather than asserting the stronger thing on both. levels is
    2^bits - 1 and therefore odd at every depth the palette mode allows, so no index stands for the value 0: an
    unseeded channel is a small non-zero constant (a seventh of the span at 3 bits), block (a) has spent weight on it,
    and writing the seeded channel over it can and does raise E - by 63 % of it on tiny.png at --c0 4 --bits0 2 before
    the refit takes it back. So the palette arm asserts what is still true there: every REFIT lowers E, and the loop
    that follows is monotone and round-trips.
    """
    inits = {}
    for name in ('residual', 'texture', 'luma'):
        odir, prefix = out_asset('tiny_init0_' + name)
        proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--c0', '2', '--init0', name])
        if proc.returncode != 0:
            raise SystemExit('FAIL: nntc_encode --init0 %s returned %d' % (name, proc.returncode))
        loop_checks(proc.stdout, 1e-4)
        round_trip(prefix, prefix + '_recon', '--init0 ' + name)
        inits[name] = proc.stdout

    # One texture has one 3x3 covariance block and it IS the whole covariance, so --init0 texture cannot differ from
    # --init0 residual there; the gate says so rather than leaving the reader to believe it. On a material of several
    # textures the two do differ, which is the whole point of the arm, and that is measured in docs/RESULTS.md.
    for name in ('_lat0.dds', '_lat1.dds', '_nntc.json'):
        a_bytes = open(os.path.join(ROOT, 'out', 'tiny_init0_residual', 'tiny' + name), 'rb').read()
        b_bytes = open(os.path.join(ROOT, 'out', 'tiny_init0_texture', 'tiny' + name), 'rb').read()
        if a_bytes != b_bytes:
            raise SystemExit('FAIL: --init0 texture differs from --init0 residual on a SINGLE texture, where the one '
                             'texture block is the whole covariance and the two must choose the same direction')
    print('  --init0 texture: byte-identical to --init0 residual on one texture, as the covariance makes it')

    # --init0-scope: the chain arm reads the residual of every plane instead of the base plane's, so it runs and
    # round-trips like any other; and naming it beside --init0 luma, which takes no residual at all, is refused by
    # name rather than quietly doing nothing.
    odir, prefix = out_asset('tiny_init0_chain')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--c0', '2', '--init0-scope', 'chain'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_encode --init0-scope chain returned %d' % proc.returncode)
    loop_checks(proc.stdout, 1e-4)
    round_trip(prefix, prefix + '_recon', '--init0-scope chain')
    if '--init0-scope chain' not in proc.stdout:
        raise SystemExit('FAIL: the report does not say which scope the seed used')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--init0', 'luma', '--init0-scope', 'chain'])
    if proc.returncode == 0 or 'takes no residual' not in proc.stderr:
        raise SystemExit('FAIL: --init0 luma --init0-scope chain was not refused by name')
    print('  --init0-scope: chain solves and round-trips; naming it beside --init0 luma is refused')

    # The seed's own steps, in the order the channels were added: E may not rise across the write and may not rise
    # across the refit, so the whole chain of numbers is non-increasing.
    steps = re.findall(r'E ([0-9.e+-]+) -> ([0-9.e+-]+) across the refit', inits['residual'])
    if len(steps) != 2:
        raise SystemExit('FAIL: --init0 residual printed %d seeded channels, expected 2' % len(steps))
    previous = None
    for before, after in steps:
        before, after = float(before), float(after)
        if previous is not None and before > previous * (1.0 + 1e-9):
            raise SystemExit('FAIL: E rose from %g to %g between two seeded level-0 channels' % (previous, before))
        if after > before * (1.0 + 1e-9):
            raise SystemExit('FAIL: the refit after a seeded level-0 channel raised E from %g to %g' % (before, after))
        previous = after
    print('  --init0 residual: E monotone across the %d seeded channels (%.6e -> %.6e)'
          % (len(steps), float(steps[0][0]), previous))

    # The palette mode's own arm of the same seed, at the layout the claim broke on. Everything the bc8 arm asserts
    # except the one step the palette grid cannot support: the write is not free there, so only the refit's direction
    # is asserted, and the run still has to solve, stay monotone through the loop and round-trip.
    odir, prefix = out_asset('tiny_init0_palette')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--c0', '4', '--l0', 'palette', '--bits0', '2'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_encode --l0 palette --init0 residual returned %d' % proc.returncode)
    loop_checks(proc.stdout, 1e-4)
    round_trip(prefix, prefix + '_recon', '--init0 residual under --l0 palette')
    pal_steps = re.findall(r'E ([0-9.e+-]+) -> ([0-9.e+-]+) across the refit', proc.stdout)
    if len(pal_steps) != 4:
        raise SystemExit('FAIL: --init0 residual under --l0 palette printed %d seeded channels, expected 4'
                         % len(pal_steps))
    for before, after in pal_steps:
        if float(after) > float(before) * (1.0 + 1e-9):
            raise SystemExit('FAIL: the refit after a seeded level-0 channel raised E from %s to %s under --l0 palette'
                             % (before, after))
    print('  --init0 residual under --l0 palette: %d channels seeded, every refit lowers E (the write itself need not, '
          'since no palette index is zero)' % len(pal_steps))

    # The search that CHOOSES the snap, which the palette mode runs after each seeded channel and its refit. Both of
    # its steps are minimisations - block (c) in level 0's own indices, then block (a) - so E may not rise across the
    # pair, and it has to be printed once per seeded channel.
    search_steps = re.findall(r'E ([0-9.e+-]+) -> ([0-9.e+-]+) across the search that chooses the snap', proc.stdout)
    if len(search_steps) != len(pal_steps):
        raise SystemExit('FAIL: --l0 palette printed %d snap searches for %d seeded channels'
                         % (len(search_steps), len(pal_steps)))
    for before, after in search_steps:
        if float(after) > float(before) * (1.0 + 1e-9):
            raise SystemExit('FAIL: the search after a seeded level-0 channel raised E from %s to %s' % (before, after))
    print('  --init0 residual under --l0 palette: the search that chooses each snap lowers E on all %d channels'
          % len(search_steps))

    # THE GATE THE v0.9f DEFECT BROKE. A second full-resolution channel cannot honestly ship worse than one, and under
    # --l0 palette --bits0 4 - the one depth whose BC4 / BC5 pack is lossy - it did: on model10 --c0 2 shipped 36.38 dB
    # against --c0 1's 39.00 and on model23 31.72 against 36.69. The cause was the seed (docs/DESIGN.md 3.4), and this
    # is the gate that says it has not come back. E shipped is the objective on the values the .dds actually holds, so
    # the comparison is over the pack as well as over the solve, which is where the defect lived.
    palette_e = {}
    for channels in ('1', '2'):
        odir, prefix = out_asset('tiny_palette_c' + channels)
        proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--c0', channels, '--l0', 'palette',
                    '--bits0', '4'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: nntc_encode --l0 palette --bits0 4 --c0 %s returned %d'
                             % (channels, proc.returncode))
        loop_checks(proc.stdout, 1e-4)
        round_trip(prefix, prefix + '_recon', '--l0 palette --bits0 4 --c0 ' + channels)
        found = re.search(r'E shipped\s+([0-9.e+-]+)', proc.stdout)
        if not found:
            raise SystemExit('FAIL: --l0 palette --c0 %s printed no shipped E' % channels)
        palette_e[channels] = float(found.group(1))
    if palette_e['2'] > palette_e['1']:
        raise SystemExit('FAIL: --l0 palette --bits0 4 --c0 2 shipped E %.6e, WORSE than --c0 1\'s %.6e; a second '
                         'full-resolution channel may not cost' % (palette_e['2'], palette_e['1']))
    print('  --l0 palette --bits0 4: --c0 2 ships E %.6e against --c0 1\'s %.6e, so the second channel buys'
          % (palette_e['2'], palette_e['1']))

    # The control really is the old behaviour: no direction is taken from anything, because channel 0 is the
    # luminance and every further channel starts at the palette value nearest zero.
    if "of the residual's variance" in inits['luma'] or 'level 0 init --init0 luma' not in inits['luma']:
        raise SystemExit('FAIL: --init0 luma did not run as the control')
    print('  --init0 luma: the control runs and round-trips, with no direction taken from the residual')


def crop_inputs(directory, count, size):
    """`count` same-size crops of tests/tiny.png, written into `directory` as c0.png, c1.png, ...

    A material needs several images of one size and the tree carries exactly one test image, so the crops are made
    here rather than added to tests/. There is one offset more than the six-texture cap, because the
    refusal of a seventh texture needs a seventh picture to be refused about. The offsets differ so that the textures are genuinely different pictures: a
    material of six copies of one crop would be solved by a decoder that repeats one row six times, which is not the
    layout any of these cases is about.
    """
    from PIL import Image as PILImage
    offsets = [(0, 0), (32, 0), (0, 32), (32, 32), (16, 16), (8, 24), (24, 8)]
    src = PILImage.open(os.path.join(ROOT, 'tests', 'tiny.png')).convert('RGB')
    paths = []
    for i in range(count):
        x, y = offsets[i]
        path = os.path.join(directory, 'c%d.png' % i)
        src.crop((x, y, x + size, y + size)).save(path)
        paths.append(path)
    return paths


def list_rule_checks(encode):
    """The per-texture list rule: --weights and --mip-filter take exactly one entry per texture, or none at all.

    Both directions are refused, because both are typing errors and a caller who wrote one of them cannot tell from a
    bare count which number the encoder thought it was reading -- so the message names the flag, the count given and
    the texture count. The third case is the one the rule must NOT break: two textures and no list at all, where every
    texture takes the default and the run goes through.
    """
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix='nntc_list_')
    try:
        crops = crop_inputs(tmp, 2, 32)
        for flag, value, given in (('--weights', '2', 1), ('--weights', '2,1,1', 3),
                                   ('--mip-filter', 'default', 1), ('--mip-filter', 'default,box,box', 3)):
            odir, _ = out_asset('tiny_list', 'c0')
            bad = run([encode] + crops + ['-o', odir, flag, value, '--png', '0'])
            want = '%s gives %d value' % (flag, given)
            if bad.returncode == 0 or want not in bad.stderr or 'for 2 textures' not in bad.stderr:
                raise SystemExit('FAIL: %s %s on two textures must be refused with the flag and both counts, not %r'
                                 % (flag, value, bad.stderr.strip()))
        print('  the list rule: --weights and --mip-filter are refused too short and too long, with both counts named')

        odir, prefix = out_asset('tiny_list_none', 'c0')
        proc = run([encode] + crops + ['-o', odir, '--png', '0'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: two inputs and no --weights returned %d' % proc.returncode)
        if '  weight 1 (default)' not in proc.stdout:
            raise SystemExit('FAIL: an unnamed --weights must leave every texture at the default, and the settings '
                             'rows must say so')
        print('  the list rule: two textures and no --weights solves, every texture at the default')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record('the per-texture list rule', '--weights / --mip-filter refused at any count but one per texture or none')


def six_texture_checks(encode, build_dir):
    """Six textures: the cap the encoder, the kernels' per-output arrays and the viewer's decoder now share.

    Six crops of tiny.png are one material of nout 18. The layout does not change with the texture count -- the same
    four plus four latent channels carry all six -- so what is asserted is that the whole path holds: the solve is
    monotone, the asset the independent Python decoder reads back agrees with the report at every level, and the
    viewer's shader decodes eighteen outputs and can be asked for the last of them (--tex 5).

    The same command line is then run again with --quiet, which is the gate on its redefinition: no banner, no
    settings rows and no round lines, and the report all the same.
    """
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix='nntc_six_')
    try:
        crops = crop_inputs(tmp, 6, 32)
        odir, prefix = out_asset('tiny_six', 'c0')
        proc = run([encode] + crops + ['-o', odir])
        if proc.returncode != 0:
            raise SystemExit('FAIL: a six-texture material returned %d' % proc.returncode)
        if 'nntc_encode: 6 textures' not in proc.stdout or ', nout 18' not in proc.stdout:
            raise SystemExit('FAIL: six inputs must be one material of six textures and nout 18')
        loop_checks(proc.stdout, 1e-4)
        level_checks(proc.stdout, prefix)
        base = re.findall(r'psnr\s+texture (\d+)\s+([0-9.]+) dB', proc.stdout)
        if [int(t) for t, _ in base] != list(range(6)):
            raise SystemExit('FAIL: the report must carry a psnr line for each of the six textures, not %s' % base)
        meta = json.load(open(os.path.join(ROOT, prefix + '_nntc.json')))
        if meta['decode']['textures_out'] != 6 or meta['decoder']['nout'] != 18:
            raise SystemExit('FAIL: the JSON must publish six output textures and nout 18')
        print('  six textures: nout 18, the loop is monotone and the asset agrees with the report at every level')

        for mode, extra in (('bc8', []), ('palette', ['--l0', 'palette', '--bits0', '3'])):
            checked = run([encode] + crops + ['-o', os.path.join('out', 'tiny_six_check_' + mode), '--png', '0',
                                              '--check', '--rounds', '3'] + extra)
            if checked.returncode != 0:
                raise SystemExit('FAIL: six textures under --check --l0 %s returned %d' % (mode, checked.returncode))
            rel = number(checked.stdout,
                         r'relative difference ([0-9.e+-]+), worst over E and every plane ([0-9.e+-]+)', 'E check')
            if max(rel) > 1e-9:
                raise SystemExit('FAIL: at eighteen outputs the device E and the host brute force differ by %g under '
                                 '--l0 %s' % (max(rel), mode))
            fd = number(checked.stdout, r'block \(b\) fd max \|dE/dx\| h / E over \d+ values of EACH of the (\d+) '
                                        r'planes ([0-9.e+-]+)', 'block (b) fd')
            if fd[1] >= 1e-5:
                raise SystemExit('FAIL: at eighteen outputs block (b) leaves a gradient of %g under --l0 %s'
                                 % (fd[1], mode))
            print('  six textures: under --l0 %s the device E agrees with the host brute force to %.1e and block (b) '
                  'leaves a gradient of %.1e over %d planes' % (mode, max(rel), fd[1], int(fd[0])))

        # Five textures, nout 15: an odd texture count that neither latent's channel count divides, run for its own
        # sake because every other case in the tree is two, four or six.
        five = run([encode] + crops[:5] + ['-o', os.path.join('out', 'tiny_five')])
        if five.returncode != 0:
            raise SystemExit('FAIL: a five-texture material returned %d' % five.returncode)
        if 'nntc_encode: 5 textures' not in five.stdout or ', nout 15' not in five.stdout:
            raise SystemExit('FAIL: five inputs must be one material of five textures and nout 15')
        loop_checks(five.stdout, 1e-4)
        level_checks(five.stdout, os.path.join('out', 'tiny_five', 'c0'))
        print('  five textures: nout 15 solves and its asset agrees with the report at every level')

        # Determinism AT SIX TEXTURES. The plain determinism check runs on one texture, and what a larger nout changes
        # is the shape of every reduction block (a) makes; a reduction whose order drifted with the output count would
        # pass there and fail here.
        det = [os.path.join('out', 'tiny_six_det_%s' % side) for side in ('a', 'b')]
        for d in det:
            proc = run([encode] + crops + ['-o', d, '--png', '0', '--quiet'])
            if proc.returncode != 0:
                raise SystemExit('FAIL: a six-texture determinism run returned %d' % proc.returncode)
        for suffix in ('_lat0.dds', '_lat1.dds', '_nntc.json'):
            a = open(os.path.join(ROOT, det[0], 'c0' + suffix), 'rb').read()
            b = open(os.path.join(ROOT, det[1], 'c0' + suffix), 'rb').read()
            if a != b:
                raise SystemExit('FAIL: two six-texture encodes differ in %s' % suffix)
        print('  six textures: two encodes of the same six inputs are byte-identical')

        quiet = run([encode] + crops + ['-o', os.path.join('out', 'tiny_six_quiet'), '--png', '0', '--quiet'])
        if quiet.returncode != 0:
            raise SystemExit('FAIL: --quiet returned %d' % quiet.returncode)
        if 'nntc_encode: 6 textures' in quiet.stdout or re.search(r'^round\s+\d+', quiet.stdout, re.MULTILINE):
            raise SystemExit('FAIL: --quiet must print neither the banner and its settings rows nor the round lines')
        # --quiet means quiet: nothing but WARNING and ERROR lines. This run has neither, so its stdout must be empty.
        stray = [line for line in quiet.stdout.splitlines() if line.strip() and not line.startswith('WARNING')]
        if stray:
            raise SystemExit('FAIL: --quiet must print nothing but warnings and errors, not %r' % stray[:3])
        if quiet.stderr.strip():
            raise SystemExit('FAIL: --quiet printed to stderr on a clean run: %r' % quiet.stderr[:200])
        print('  --quiet: nothing at all on a clean run')

        if os.name != 'nt':
            # No Direct3D viewer here, but the Vulkan one builds on Linux and its --shot needs no desktop, so the
            # frame arm of this case still runs - against itself rather than against a second api.
            print('  the viewer: skipped (the Direct3D viewer is Windows only)')
            vk_note = vk_six_texture_arm(build_dir, prefix + '_nntc.json')
            record('six textures', 'nout 18 solves and round-trips; --quiet drops every progress line and '
                                   'keeps the whole report' +
                                   (', and the six --tex frames are pairwise different on the Vulkan viewer'
                                    if vk_note == ' on both viewers' else vk_note))
            return
        view = executable(build_dir, 'nntc_view')
        frames = {}
        for t in range(6):
            frame, err = shot(view, prefix + '_nntc.json', os.path.join('out', 'tiny_six_shot%d.bmp' % t), ['--tex', str(t)])
            if 'WARNING: --tex' in err:
                raise SystemExit('FAIL: --tex %d is inside a six-texture material and must not warn: %r'
                                 % (t, err.strip()))
            frames[t] = frame
        for a in range(6):
            for b in range(a + 1, 6):
                if frames[a] == frames[b]:
                    raise SystemExit('FAIL: --tex %d and --tex %d drew the same frame, so the shader is not selecting '
                                     'the output triple' % (a, b))
        past, err = shot(view, prefix + '_nntc.json', os.path.join('out', 'tiny_six_shot6.bmp'), ['--tex', '6'])
        if 'WARNING: --tex 6 is outside 0..5' not in err:
            raise SystemExit('FAIL: --tex 6 on a six-texture material must warn on stderr, not %r' % err.strip())
        if past != frames[0]:
            raise SystemExit('FAIL: a --tex past the end must fall back to texture 0 and draw its frame')
        print('  the viewer: the six --tex frames are pairwise different and --tex 6 warns and falls back to 0 '
              '(%d bytes each)' % len(frames[0]))

        # The same on Vulkan, as a second arm of the same case: the shader that selects the output triple is a
        # transliteration of the other one, so the six frames must be pairwise different there too and a --tex past
        # the end must warn in the same words and fall back to texture 0.
        vk_note = vk_six_texture_arm(build_dir, prefix + '_nntc.json')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record('six textures', 'nout 18 agrees with the host brute force under both level-0 modes, is deterministic, '
                           'renders one distinct frame per texture' + vk_note + '; nout 15 solves; --quiet drops '
                           'every progress line and keeps the whole report')


def mip_filter_checks(encode):
    """The source chain's per-texture filters.

    The chain is the one place the encoder chooses its own ground truth, so what is asserted here is that each name
    reaches the resizer, that `default` is the path the tree has always taken - byte for byte, both as an unnamed
    default and as a named one - and that a texture left on `default` inside a material that filters another texture
    still gets exactly the iterated box it would have got on its own. A filter with no chain to derive is not an
    error: it is idle, and the asset is the one the run without it writes.
    """
    sources = {}
    for name in ('default', 'box', 'mitchell', 'catmullrom'):
        odir, prefix = out_asset('tiny_filter_' + name)
        proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--mip-filter', name])
        if proc.returncode != 0:
            raise SystemExit('FAIL: --mip-filter %s returned %d' % (name, proc.returncode))
        if ('filter %s (command line)' % name) not in proc.stdout:
            raise SystemExit('FAIL: the settings rows must say the command line chose --mip-filter %s' % name)
        loop_checks(proc.stdout, 1e-4)
        level_checks(proc.stdout, prefix)
        sources[name] = open(os.path.join(ROOT, prefix + '_src_M1.png'), 'rb').read()
    print('  the filters: all four names solve and their assets agree with the report at every level')

    # `default` is the untouched path: named or not, it is the same asset, and every other name is a different chain.
    odir, plain = out_asset('tiny_filter_none')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir])
    if proc.returncode != 0:
        raise SystemExit('FAIL: the unfiltered run returned %d' % proc.returncode)
    for name in ('_lat0.dds', '_lat1.dds', '_nntc.json'):
        a = open(os.path.join(ROOT, plain + name), 'rb').read()
        b = open(os.path.join(ROOT, os.path.join('out', 'tiny_filter_default'), 'tiny' + name), 'rb').read()
        if a != b:
            raise SystemExit('FAIL: --mip-filter default differs from no flag at all in %s' % name)
    base = open(os.path.join(ROOT, plain + '_src_M1.png'), 'rb').read()
    if base != sources['default']:
        raise SystemExit('FAIL: --mip-filter default did not reproduce the box chain')
    # `box` is deliberately not asserted to DIFFER here: tiny.png is 64x64, so every level is an exact integer ratio,
    # and a box over 2x2 source texels is the same mean whether it is taken once directly or iterated -- which is why
    # gate 9's renormalisation case uses it. The odd chains below are where the two part company, and they say where.
    # The two cubics are the ones a mis-wired name would show up in at any size.
    for name in ('mitchell', 'catmullrom'):
        if sources[name] == base:
            raise SystemExit('FAIL: --mip-filter %s produced the box chain, so the name reached nothing' % name)
    if sources['mitchell'] == sources['catmullrom']:
        raise SystemExit('FAIL: mitchell and catmullrom produced the same chain, so one of the names is mis-wired')
    print('  the filters: default is byte-identical to no flag, and mitchell and catmullrom each differ from it and '
          'from each other')

    # A MIXED LIST. Texture 0 is left on default inside a material whose texture 1 is filtered, and its own chain has
    # to be the one it would have had alone -- the box is per channel, so a texture's numbers cannot depend on what
    # the texture beside it asked for.
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix='nntc_mix_')
    try:
        crops = crop_inputs(tmp, 2, 32)
        odir, both_default = out_asset('tiny_filter_pair', 'c0')
        proc = run([encode] + crops + ['-o', odir])
        if proc.returncode != 0:
            raise SystemExit('FAIL: the two-texture unfiltered run returned %d' % proc.returncode)
        odir, mixed = out_asset('tiny_filter_mixed', 'c0')
        proc = run([encode] + crops + ['-o', odir, '--mip-filter', 'default,mitchell'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: --mip-filter default,mitchell returned %d' % proc.returncode)
        loop_checks(proc.stdout, 1e-4)
        # M0 is the base, which no filter touches; the chain levels are the ones a per-texture split can get wrong.
        made = sorted(m for m in re.findall(r'_src_t0_M(\d+)\.png',
                                            ' '.join(os.listdir(os.path.join(ROOT, os.path.dirname(both_default)))))
                      if int(m) >= 1)
        if len(made) < 2:
            raise SystemExit('FAIL: the two-texture chain wrote %d source levels, expected the whole chain' % len(made))
        for m in made:
            a = open(os.path.join(ROOT, both_default + '_src_t0_M%s.png' % m), 'rb').read()
            b = open(os.path.join(ROOT, mixed + '_src_t0_M%s.png' % m), 'rb').read()
            if a != b:
                raise SystemExit('FAIL: at M%s a texture left on default changed chain because the texture beside it '
                                 'was filtered' % m)
            a1 = open(os.path.join(ROOT, both_default + '_src_t1_M%s.png' % m), 'rb').read()
            b1 = open(os.path.join(ROOT, mixed + '_src_t1_M%s.png' % m), 'rb').read()
            if a1 == b1:
                raise SystemExit('FAIL: at M%s the filtered texture of a mixed list got the box chain' % m)
        print('  the filters: in a mixed list the default texture keeps the box chain byte for byte at every one of '
              'the %d chain levels, and the filtered one does not' % len(made))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # The refusals. A filter derives the levels BELOW the base, so a run that stores none has nothing to apply it to,
    # by either of the two routes to a chain of one; and a name the resizer does not have is refused with the list.
    odir, _ = out_asset('tiny_filter_bad')
    bad = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--mip-filter', 'lanczos', '--png', '0'])
    if bad.returncode == 0 or 'default, box, mitchell and catmullrom' not in bad.stderr:
        raise SystemExit('FAIL: an unknown filter name must be refused with the four names listed')
    # A filter is a setting and the chain is the encoder's decision from the image's size, so a filter on a run that
    # has no level below the base (--mips 0, or a source --mip-min stops at the base) is NOT an error: the texture gets
    # no mipmaps and the setting is idle. Both must encode, with one stored level, and match the same run without the
    # filter named, since the filter touched nothing.
    for tag, flags in (('mips0', ['--mips', '0']), ('min64', ['--mip-min', '64'])):
        a_dir, a_pre = out_asset('tiny_filter_idle_' + tag)
        b_dir, b_pre = out_asset('tiny_filter_idle_' + tag + '_ref')
        ok = run([encode, os.path.join('tests', 'tiny.png'), '-o', a_dir, '--mip-filter', 'mitchell', '--png', '0'] + flags)
        if ok.returncode != 0:
            raise SystemExit('FAIL: a filter on a run with no chain level (%s) must be accepted, not %r'
                             % (' '.join(flags), ok.stderr.strip()))
        if '1 stored level' not in ok.stdout:
            raise SystemExit('FAIL: the idle-filter run (%s) must store one level' % ' '.join(flags))
        ref = run([encode, os.path.join('tests', 'tiny.png'), '-o', b_dir, '--png', '0'] + flags)
        if ref.returncode != 0:
            raise SystemExit('FAIL: the reference run for the idle filter (%s) returned %d' % (' '.join(flags), ref.returncode))
        for suffix in ('_lat0.dds', '_lat1.dds'):
            if open(a_pre + suffix, 'rb').read() != open(b_pre + suffix, 'rb').read():
                raise SystemExit('FAIL: an idle filter (%s) must leave %s byte-identical' % (' '.join(flags), suffix))
    print('  the filters: an unknown name is refused with the four names listed; a filter with --mips 0 or on a source '
          'with no chain level is accepted and idle, the .dds byte-identical to the run without it')

    # THE FOOTPRINT. The iterated box drops the last column and row of an odd plane, so it covers a sub-rectangle of
    # the base; a direct resize covers all of it. 44x44 at --mip-min 4 halves to 22, 11 and 5, so the odd plane is
    # reached twice over -- and 44 is a multiple of 4, so no padding stands between the two chains either.
    tmp = tempfile.mkdtemp(prefix='nntc_odd44_')
    try:
        from PIL import Image as PILImage
        odd = os.path.join(tmp, 'odd44.png')
        PILImage.open(os.path.join(ROOT, 'tests', 'tiny.png')).convert('RGB').crop((0, 0, 44, 44)).save(odd)
        planes, levels = {}, {}
        for name in ('default', 'box'):
            odir, prefix = out_asset('tiny_44_' + name, 'odd44')
            proc = run([encode, odd, '-o', odir, '--mip-min', '4', '--mip-filter', name])
            if proc.returncode != 0:
                raise SystemExit('FAIL: the 44x44 run at --mip-filter %s returned %d' % (name, proc.returncode))
            if 'WARNING: the input is' in proc.stdout:
                raise SystemExit('FAIL: 44x44 is a multiple of 4 and must not be padded')
            loop_checks(proc.stdout, 1e-4)
            level_checks(proc.stdout, prefix)
            planes[name] = [int(w) for w, _ in re.findall(r'^    M\d+  level 0 (\d+)x(\d+)', proc.stdout, re.M)]
            if planes[name] != [44, 22, 11, 5]:
                raise SystemExit('FAIL: the 44x44 chain must be 44, 22, 11, 5 at --mip-min 4, not %s' % planes[name])
            for m in (1, 2, 3):
                levels[(name, m)] = open(os.path.join(ROOT, prefix + '_src_M%d.png' % m), 'rb').read()
        # WHERE THE FOOTPRINT SHOWS AND WHERE IT DOES NOT. 44 -> 22 and 22 -> 11 are exact 2:1 and 4:1 ratios from the
        # base, so the iterated box and a direct box resize are the same mean and the two files are the same bytes.
        # 11 -> 5 is not: the iterated box drops the eleventh column and row, so it covers 10 of the 11 and the direct
        # resize covers all of them. That is the whole of DESIGN 4.2's footprint note, in three files.
        for m in (1, 2):
            if levels[('default', m)] != levels[('box', m)]:
                raise SystemExit('FAIL: at the exact ratio of M%d the iterated box and a direct box resize must be the '
                                 'same bytes' % m)
        if levels[('default', 3)] == levels[('box', 3)]:
            raise SystemExit('FAIL: M3 of the 44 / 22 / 11 / 5 chain is the odd halving, where the iterated box covers '
                             'a sub-rectangle of the base and a direct resize covers all of it; the two must differ')
        # AND THE CUBICS SOLVE ON THE ODD CHAIN TOO. The direct resize to an odd plane is the case the resizer is
        # driven hardest by, and the two names had never been run on one.
        for name in ('mitchell', 'catmullrom'):
            odir, prefix = out_asset('tiny_44_' + name, 'odd44')
            proc = run([encode, odd, '-o', odir, '--mip-min', '4', '--mip-filter', name])
            if proc.returncode != 0:
                raise SystemExit('FAIL: the 44x44 run at --mip-filter %s returned %d' % (name, proc.returncode))
            loop_checks(proc.stdout, 1e-4)
            level_checks(proc.stdout, prefix)
        print('  the filters: on the 44 / 22 / 11 / 5 chain default and box agree byte for byte at M1 and M2 and '
              'differ at the odd M3; mitchell and catmullrom solve there too')

        # THE SAME AT THE DEFAULT --mip-min, on the chain DESIGN 4.2 names: 68 -> 34 -> 17 -> 8. M1 is an exact 2:1 and
        # is the same bytes; M3 is the odd halving and is a different picture. M2 is an exact 4:1 and comes out the
        # same picture to within one step of one texel -- the two paths sum the same sixteen values in a different
        # order, and one texel of 289 lands on the other side of a rounding. Asserting it EXACTLY equal would be
        # asserting an order of floating-point addition that nothing promises.
        wide = os.path.join(tmp, 'odd68.png')
        PILImage.open(os.path.join(ROOT, 'tests', 'tiny.png')).convert('RGB').crop((0, 0, 68, 68)).save(wide)
        wide_levels = {}
        for name in ('default', 'box'):
            odir, prefix = out_asset('tiny_68_' + name, 'odd68')
            proc = run([encode, wide, '-o', odir, '--mip-filter', name])
            if proc.returncode != 0:
                raise SystemExit('FAIL: the 68x68 run at --mip-filter %s returned %d' % (name, proc.returncode))
            got = [int(w) for w, _ in re.findall(r'^    M\d+  level 0 (\d+)x(\d+)', proc.stdout, re.M)]
            if got != [68, 34, 17, 8]:
                raise SystemExit('FAIL: the 68x68 chain must be 68, 34, 17, 8 at the default --mip-min, not %s' % got)
            loop_checks(proc.stdout, 1e-4)
            level_checks(proc.stdout, prefix)
            for m in (1, 2, 3):
                wide_levels[(name, m)] = PILImage.open(
                    os.path.join(ROOT, prefix + '_src_M%d.png' % m)).convert('RGB')

        def worst_step(a, b):
            return max(max(abs(x - y) for x, y in zip(p, q))
                       for p, q in zip(list(a.getdata()), list(b.getdata())))

        if worst_step(wide_levels[('default', 1)], wide_levels[('box', 1)]) != 0:
            raise SystemExit('FAIL: M1 of the 68 chain is an exact 2:1 and must be the same bytes both ways')
        if worst_step(wide_levels[('default', 2)], wide_levels[('box', 2)]) > 1:
            raise SystemExit('FAIL: M2 of the 68 chain is an exact 4:1 and must agree within one step')
        if worst_step(wide_levels[('default', 3)], wide_levels[('box', 3)]) <= 1:
            raise SystemExit('FAIL: M3 of the 68 chain is the odd halving and must be a different picture')
        print('  the filters: 68 / 34 / 17 / 8 solves both ways, M1 identical, M2 within one step, M3 the footprint')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record('the source filters', 'four names, default byte-identical to no flag, a mixed list at every chain '
                                 'level, the refusals, and the odd 44 / 22 / 11 / 5 and 68 / 34 / 17 / 8 '
                                 'chains where the iterated box and a direct resize part company')


def write_rgb8(path, width, height, pixel):
    """An 8-bit RGB PNG, written here rather than carried in the tree so that what it holds is stated in code.

    `pixel(x, y)` returns the (r, g, b) triple. The writer is zlib and struct, like write_grey16 below, so the gate
    does not depend on what a particular Pillow can write.
    """
    import binascii
    import zlib
    raw = bytearray()
    for y in range(height):
        raw.append(0)   # filter type 0, one per scanline
        for x in range(width):
            raw += bytes(pixel(x, y))

    def chunk(tag, payload):
        return (struct.pack('>I', len(payload)) + tag + payload
                + struct.pack('>I', binascii.crc32(tag + payload) & 0xFFFFFFFF))

    header = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)   # 8 bits, colour type 2 (RGB), no interlace
    png = (PNG_MAGIC + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(bytes(raw), 9))
           + chunk(b'IEND', b''))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'wb').write(png)


def write_material(path, entries):
    """One material JSON for a check: the array of texture entries, in texture order."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(entries, f, indent=2)
    return path


def material_checks(encode, build_dir):
    """The material JSON, and the two settings only it can set.

    `nntc_encode material.json` is the other way in: one array in texture order, every entry a `file` and whatever the
    command line has no per-texture spelling for. What is asserted here is the whole of that path -- the keys reaching
    the settings, a relative file resolving against the MATERIAL rather than the working directory, the asset taking
    the material's own name, the refusals, the command line overriding a key and saying so, and `source.inputs` in the
    asset's JSON -- and then the two settings the command line cannot reach: srgb, which derives the deeper levels in
    linear light, and normal_map, which renormalises them.
    """
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix='nntc_mat_')
    try:
        crops = crop_inputs(tmp, 2, 32)
        # Every key at once, and `file` written RELATIVE so that it has to be resolved against the material's own
        # directory -- the two travel together, and a material that only works from one working directory mostly does
        # not work.
        full = write_material(os.path.join(tmp, 'full.json'), [
            {'file': 'c0.png', 'type': 'albedo', 'filter': 'mitchell', 'srgb': True, 'edge': 'wrap', 'weight': 2,
             'rgb_weights': [1, 1, 0.5]},
            {'file': 'c1.png', 'type': 'normal', 'filter': 'catmullrom', 'normal_map': True}])
        odir = os.path.join('out', 'tiny_material')
        prefix = os.path.join(odir, 'full')   # the asset takes the MATERIAL's stem, not the first texture's
        proc = run([encode, full, '-o', odir])
        if proc.returncode != 0:
            raise SystemExit('FAIL: a material JSON returned %d' % proc.returncode)
        for want in ("type 'albedo' (json)", 'weight 2 (json)', 'rgb weights 1 1 0.5 (json)',
                     'filter mitchell (json)', 'srgb yes (json)', 'edge wrap (json)', 'normal map yes (json)'):
            if want not in proc.stdout:
                raise SystemExit('FAIL: the settings rows must carry %r from the material' % want)
        if not os.path.isfile(os.path.join(ROOT, prefix + '_nntc.json')):
            raise SystemExit("FAIL: the asset must be named after the material's stem (full_nntc.json)")
        loop_checks(proc.stdout, 1e-4)
        level_checks(proc.stdout, prefix)
        meta = json.load(open(os.path.join(ROOT, prefix + '_nntc.json')))
        inputs = meta['source'].get('inputs')
        if not inputs or len(inputs) != 2:
            raise SystemExit('FAIL: the asset JSON must carry source.inputs, one entry per input')
        if inputs[0] != {'file': 'c0.png', 'type': 'albedo', 'filter': 'mitchell', 'srgb': True, 'edge': 'wrap',
                         'normal_map': False, 'weight': 2, 'rgb_weights': [1, 1, 0.5]}:
            raise SystemExit('FAIL: source.inputs[0] does not describe the input that was encoded: %r' % (inputs[0],))
        if not inputs[1]['normal_map'] or inputs[1]['filter'] != 'catmullrom':
            raise SystemExit('FAIL: source.inputs[1] does not describe the normal map that was encoded')
        print('  the material: every key reaches the settings, the asset takes the material\'s name and its JSON '
              'carries source.inputs')

        # THE PRECEDENCE. The command line replaces the key and says which value it threw away; the row then says the
        # command line, not the material.
        over = run([encode, full, '-o', os.path.join('out', 'tiny_material_over'), '--png', '0', '--weights', '1,1'])
        if over.returncode != 0:
            raise SystemExit('FAIL: --weights over a material returned %d' % over.returncode)
        if 'the material sets weight 2 and --weights sets 1; the command line wins' not in over.stdout:
            raise SystemExit('FAIL: an overridden material key must print a WARNING on stdout naming both values')
        if 'weight 1 (command line)' not in over.stdout:
            raise SystemExit('FAIL: the settings rows must attribute an overridden weight to the command line')
        # A key the material did NOT set is not an override and must not warn.
        if over.stdout.count('the command line wins') != 1:
            raise SystemExit('FAIL: only the key the material actually set may warn')
        print('  the material: --weights overrides it, warns on stdout naming both values, and the row says so')

        # The refusals, each by name.
        bad = write_material(os.path.join(tmp, 'unknown.json'), [{'file': 'c0.png', 'normalmap': True}])
        proc = run([encode, bad, '-o', os.path.join('out', 'tiny_material_bad'), '--png', '0'])
        if proc.returncode == 0 or "'normalmap' is not a key of a texture entry" not in proc.stderr:
            raise SystemExit('FAIL: an unknown material key must be refused by name')
        # An EXPLICIT `default` beside srgb is a contradiction and stays refused. Naming NO filter at all is the other
        # case entirely and is asserted by implied_box_check below, so the entry has to say `default` out loud.
        bad = write_material(os.path.join(tmp, 'srgbdef.json'),
                             [{'file': 'c0.png', 'filter': 'default', 'srgb': True}])
        proc = run([encode, bad, '-o', os.path.join('out', 'tiny_material_bad'), '--png', '0'])
        if proc.returncode == 0 or 'which derives nothing' not in proc.stderr:
            raise SystemExit("FAIL: srgb beside an explicit default filter must be refused")
        # The same combination, but produced by the COMMAND LINE overriding the material's filter, which the message
        # has to distinguish: the material is not the one that asked for it.
        proc = run([encode, full, '-o', os.path.join('out', 'tiny_material_bad'), '--png', '0',
                    '--mip-filter', 'default,default'])
        if proc.returncode == 0 or "that filter came from --mip-filter, which replaced the material's" not in proc.stderr:
            raise SystemExit('FAIL: when --mip-filter caused the default-with-srgb combination, the message must say '
                             'so')
        # AND IT MUST NOT CLAIM A REPLACEMENT THAT DID NOT HAPPEN. The same refusal over a material that named no
        # filter of its own has nothing to have replaced, so that half of the message is dropped.
        plain = write_material(os.path.join(tmp, 'srgbnofilter.json'), [{'file': 'c0.png', 'srgb': True}])
        proc = run([encode, plain, '-o', os.path.join('out', 'tiny_material_bad'), '--png', '0',
                    '--mip-filter', 'default'])
        if proc.returncode == 0 or 'that filter came from --mip-filter' not in proc.stderr:
            raise SystemExit('FAIL: --mip-filter default over a filterless srgb material must still be refused')
        if "replaced the material's" in proc.stderr:
            raise SystemExit('FAIL: the refusal must not say the material was replaced when it named no filter')
        proc = run([encode, full, crops[0], '-o', os.path.join('out', 'tiny_material_bad'), '--png', '0'])
        if proc.returncode == 0 or 'cannot stand beside image arguments' not in proc.stderr:
            raise SystemExit('FAIL: a material and image arguments together must be refused')
        proc = run([encode, full, full, '-o', os.path.join('out', 'tiny_material_bad'), '--png', '0'])
        if proc.returncode == 0 or 'two material JSONs' not in proc.stderr:
            raise SystemExit('FAIL: two materials in one run must be refused')
        empty = write_material(os.path.join(tmp, 'empty.json'), [])
        proc = run([encode, empty, '-o', os.path.join('out', 'tiny_material_bad'), '--png', '0'])
        if proc.returncode == 0 or 'empty array' not in proc.stderr:
            raise SystemExit('FAIL: an empty material array must be refused')
        bad = write_material(os.path.join(tmp, 'both.json'),
                             [{'file': 'c0.png', 'filter': 'mitchell', 'srgb': True, 'normal_map': True}])
        proc = run([encode, bad, '-o', os.path.join('out', 'tiny_material_bad'), '--png', '0'])
        if proc.returncode == 0 or 'cannot also ask for' not in proc.stderr:
            raise SystemExit('FAIL: normal_map with srgb must be refused')
        bad = write_material(os.path.join(tmp, 'weight.json'), [{'file': 'c0.png', 'weight': -1}])
        proc = run([encode, bad, '-o', os.path.join('out', 'tiny_material_bad'), '--png', '0'])
        if proc.returncode == 0 or '--weights must not be negative' not in proc.stderr:
            raise SystemExit("FAIL: a material's negative weight must be refused by the same message a flag's is")
        bad = write_material(os.path.join(tmp, 'rgb.json'), [{'file': 'c0.png', 'rgb_weights': [0, 0, 0]}])
        proc = run([encode, bad, '-o', os.path.join('out', 'tiny_material_bad'), '--png', '0'])
        if proc.returncode == 0 or '--rgb-weights must not be all zero' not in proc.stderr:
            raise SystemExit("FAIL: a material's all-zero rgb weights must be refused by the same message")
        print('  the material: an unknown key, srgb beside an explicit default (from either side), a mixed command '
              'line, two '
              'materials, an empty array, normal_map with srgb, and the weight checks are all refused by name')

        # --quiet over a material: no banner, no rows, no round lines, the report all the same.
        quiet = run([encode, full, '-o', os.path.join('out', 'tiny_material_quiet'), '--png', '0', '--quiet'])
        if quiet.returncode != 0:
            raise SystemExit('FAIL: --quiet over a material returned %d' % quiet.returncode)
        if 'nntc_encode: 2 textures' in quiet.stdout or re.search(r'^round\s+\d+', quiet.stdout, re.MULTILINE):
            raise SystemExit('FAIL: --quiet over a material must print neither the banner nor the round lines')
        if '\nreport\n' in quiet.stdout or 'wrote ' in quiet.stdout:
            raise SystemExit('FAIL: --quiet must print neither the report nor the wrote lines')
        print('  the material: --quiet prints no banner, no round lines, no wrote lines and no report')

        # THE INDEPENDENT DECODER AND THE VIEWER on an asset written from a material.
        dec = run([sys.executable, os.path.join('tools', 'dds_decode.py'), prefix, '--grid'])
        if dec.returncode != 0:
            raise SystemExit("FAIL: dds_decode.py --grid failed on a material's asset")
        print("  the material: dds_decode.py --grid passes on the material's asset")
        if os.name == 'nt':
            view = executable(build_dir, 'nntc_view')
            frame, _ = shot(view, prefix + '_nntc.json', os.path.join('out', 'tiny_material_shot.bmp'), ['--tex', '1'])
            raw, _ = shot(view, prefix + '_nntc.json', os.path.join('out', 'tiny_material_raw0.bmp'), ['--tex', '1',
                                                                                                 '--raw0'])
            if frame == raw:
                raise SystemExit("FAIL: the material asset's decode frame is the same bytes as its --raw0 frame, so "
                                 'the decoder drew nothing')
            print("  the material: the viewer opens it, draws texture 1 and its decode differs from its own --raw0 "
                  'frame (%d bytes)' % len(frame))

        # SRGB IS A DIFFERENT CHAIN. One texture, one filter, the only difference being the light the deeper levels
        # were derived in -- so a source PNG that came out the same would mean the key reached nothing.
        chains = {}
        for name, entry in (('plain', {'file': 'c0.png'}),
                            ('mitchell', {'file': 'c0.png', 'filter': 'mitchell'}),
                            ('srgb', {'file': 'c0.png', 'filter': 'mitchell', 'srgb': True})):
            path = write_material(os.path.join(tmp, name + '.json'), [entry])
            odir = os.path.join('out', 'tiny_srgb_' + name)
            proc = run([encode, path, '-o', odir])
            if proc.returncode != 0:
                raise SystemExit('FAIL: the %s material returned %d' % (name, proc.returncode))
            chains[name] = open(os.path.join(ROOT, odir, name + '_src_M1.png'), 'rb').read()
        if chains['plain'] == chains['mitchell'] or chains['mitchell'] == chains['srgb']:
            raise SystemExit('FAIL: default, mitchell and mitchell + srgb must each derive a different chain')
        print('  the material: default, mitchell and mitchell + srgb each derive a different source chain')

        # DETERMINISM on the two paths the plain determinism check does not reach: the resizer with the sRGB
        # conversions around it, and the renormalisation after it.
        for name, entry in (('srgb', {'file': 'c0.png', 'filter': 'mitchell', 'srgb': True}),
                            ('normal', {'file': 'c0.png', 'filter': 'mitchell', 'normal_map': True})):
            path = write_material(os.path.join(tmp, 'det_' + name + '.json'), [entry])
            dirs = [os.path.join('out', 'det_%s_%s' % (name, side)) for side in ('a', 'b')]
            for d in dirs:
                proc = run([encode, path, '-o', d, '--png', '0'])
                if proc.returncode != 0:
                    raise SystemExit('FAIL: the %s determinism run returned %d' % (name, proc.returncode))
            for suffix in ('_lat0.dds', '_lat1.dds', '_nntc.json'):
                a = open(os.path.join(ROOT, dirs[0], 'det_' + name + suffix), 'rb').read()
                b = open(os.path.join(ROOT, dirs[1], 'det_' + name + suffix), 'rb').read()
                if a != b:
                    raise SystemExit('FAIL: two %s encodes differ in %s' % (name, suffix))
        print('  the material: mitchell + srgb and mitchell + normal_map are each byte-identical over two encodes')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    normal_renorm_check(encode)
    implied_box_check(encode)
    record('the material JSON', 'every key, the precedence and its warnings, the refusals, source.inputs, --grid, the '
                                'viewer, and determinism under srgb and normal_map')


def normal_renorm_check(encode):
    """The normal-map renormalisation, on a picture whose answer is known before the run.

    An 8x8 RGB source of four 4x4 uniform blocks, filtered with `box` at an exact 2:1 ratio, so M1 is 4x4 and every
    one of its texels is the plain mean of four identical texels -- the block's own colour. Three of the blocks are
    ones the rule must NOT touch: grey unpacks to a zero-length vector with no direction to normalise towards, black
    and (200,140,30) unpack to vectors pointing into the surface, which no tangent-space normal does. The fourth is an
    ordinary normal, and after the rule its unpacked vector must be of unit length.

    The third block used to be (128,128,0), which unpacks to (0.004, 0.004, -1.000) -- ALREADY of unit length, so a
    rule that had lost its z test entirely would have left it exactly where the filter did and passed. (200,140,30) is
    0.958 long and points into the surface, so the z test is the only thing that can exclude it.

    The three excluded blocks are checked against the SAME run without `normal_map`, so what is asserted is that the
    rule left them exactly where the filter did and not merely that they look plausible.
    """
    import shutil
    import tempfile
    from PIL import Image as PILImage
    blocks = [(128, 128, 128), (0, 0, 0), (200, 140, 30), (200, 140, 220)]
    tmp = tempfile.mkdtemp(prefix='nntc_norm_')
    try:
        src = os.path.join(tmp, 'n.png')
        write_rgb8(src, 8, 8, lambda x, y: blocks[(y // 4) * 2 + x // 4])
        levels = {}
        for name, entry in (('flat', {'file': 'n.png', 'filter': 'box'}),
                            ('renorm', {'file': 'n.png', 'filter': 'box', 'normal_map': True})):
            path = write_material(os.path.join(tmp, name + '.json'), [entry])
            odir = os.path.join('out', 'tiny_renorm_' + name)
            proc = run([encode, path, '-o', odir, '--mip-min', '4', '--png', '1'])
            if proc.returncode != 0:
                raise SystemExit('FAIL: the %s renormalisation run returned %d' % (name, proc.returncode))
            img = PILImage.open(os.path.join(ROOT, odir, name + '_src_M1.png')).convert('RGB')
            if img.size != (4, 4):
                raise SystemExit('FAIL: M1 of an 8x8 source at --mip-min 4 must be 4x4, not %s' % (img.size,))
            levels[name] = img
        # The three blocks the rule excludes, texel by texel, against the unrenormalised run.
        for by in range(2):
            for bx in range(2):
                if (by * 2 + bx) == 3:
                    continue
                for y in range(by * 2, by * 2 + 2):
                    for x in range(bx * 2, bx * 2 + 2):
                        if levels['renorm'].getpixel((x, y)) != levels['flat'].getpixel((x, y)):
                            raise SystemExit('FAIL: the renormalisation touched the %s block, which has no direction '
                                             'to normalise towards' % (blocks[by * 2 + bx],))
        # The fourth block: an ordinary normal, of unit length after the rule and not before it.
        before = levels['flat'].getpixel((2, 2))
        after = levels['renorm'].getpixel((2, 2))
        def length(rgb):
            v = [c / 255.0 * 2.0 - 1.0 for c in rgb]
            return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5
        if abs(length(after) - 1.0) > 0.01:
            raise SystemExit('FAIL: the renormalised normal is %.4f long, not 1' % length(after))
        if abs(length(before) - 1.0) <= 0.01:
            raise SystemExit('FAIL: the test normal was already unit length, so the check proves nothing')
        excluded = length(levels['flat'].getpixel((0, 2)))   # the z < 0 block: blocks[2], rows 2-3, columns 0-1
        if abs(excluded - 1.0) <= 0.01:
            raise SystemExit('FAIL: the z < 0 block is already unit length, so leaving it alone proves nothing')
        print('  the renormalisation: grey, black and a z < 0 block of length %.4f are untouched; the ordinary normal '
              'goes from %.4f to %.4f long' % (excluded, length(before), length(after)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record('the normal renormalisation', 'grey, black and a z < 0 block that is NOT already unit length are left '
                                         'where the filter left them; the ordinary normal comes back unit length')


def implied_box_check(encode):
    """The filter `srgb`, `edge` and `normal_map` imply when the entry names none.

    What asks for a derivation is the VALUE and not the key: `srgb` true, `edge` wrap, `normal_map` true. Each needs a
    DERIVED chain, and the built-in iterated box derives nothing, so an entry that carries one of those values and no
    filter at all is switched to stb's `box`, derived directly from the base. What is asserted is that the implied run
    is the explicit `"filter": "box"` run BYTE FOR BYTE in its source planes, which is the only place a different
    filter could hide, plus the two things that make the switch visible: the settings row naming the key that implied
    it, and `source.inputs` recording the effective box rather than the `default` nobody got. Then the cases that must
    NOT imply anything: a filter the entry named, a filter a flag named, and a key carrying its INERT value, which
    asks for nothing and leaves the vanilla chain. And last the footprint: the implied box is the built-in chain's
    picture only where the ratio is exact, so on an odd chain the two really do differ.
    """
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix='nntc_implied_')
    try:
        crop_inputs(tmp, 1, 32)
        for key, implied, explicit in (
                ('srgb', {'file': 'c0.png', 'srgb': True},
                 {'file': 'c0.png', 'filter': 'box', 'srgb': True}),
                ('normal_map', {'file': 'c0.png', 'normal_map': True},
                 {'file': 'c0.png', 'filter': 'box', 'normal_map': True}),
                ('edge', {'file': 'c0.png', 'edge': 'wrap'},
                 {'file': 'c0.png', 'filter': 'box', 'edge': 'wrap'})):
            planes = {}
            for side, entry in (('imp', implied), ('exp', explicit)):
                name = '%s_%s' % (key, side)
                path = write_material(os.path.join(tmp, name + '.json'), [entry])
                odir = os.path.join('out', 'tiny_implied_' + name)
                proc = run([encode, path, '-o', odir])
                if proc.returncode != 0:
                    raise SystemExit('FAIL: the %s %s run returned %d' % (key, side, proc.returncode))
                if side == 'imp':
                    want = 'filter box (implied by %s)' % key
                    if want not in proc.stdout:
                        raise SystemExit('FAIL: the settings row of an implied box must say %r' % want)
                    meta = json.load(open(os.path.join(ROOT, odir, name + '_nntc.json')))
                    if meta['source']['inputs'][0]['filter'] != 'box':
                        raise SystemExit('FAIL: source.inputs must record the effective filter, which is box')
                planes[side] = open(os.path.join(ROOT, odir, name + '_src_M1.png'), 'rb').read()
            if planes['imp'] != planes['exp']:
                raise SystemExit("FAIL: %s's implied box derived a different chain from an explicit one" % key)
        # A filter the entry DID name is not overridden by a key that would have implied one.
        path = write_material(os.path.join(tmp, 'keep.json'),
                              [{'file': 'c0.png', 'filter': 'mitchell', 'srgb': True}])
        proc = run([encode, path, '-o', os.path.join('out', 'tiny_implied_keep'), '--png', '0'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: mitchell beside srgb returned %d' % proc.returncode)
        if 'filter mitchell (json)' not in proc.stdout or 'implied by' in proc.stdout:
            raise SystemExit('FAIL: srgb must imply nothing when the entry named a filter of its own')
        # Nor does a filter the COMMAND LINE named. The material named none of its own, so nothing of the material's
        # was overridden and there is no override warning to print either.
        path = write_material(os.path.join(tmp, 'cli.json'), [{'file': 'c0.png', 'srgb': True}])
        proc = run([encode, path, '-o', os.path.join('out', 'tiny_implied_cli'), '--png', '0',
                    '--mip-filter', 'mitchell'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: --mip-filter mitchell over an srgb material returned %d' % proc.returncode)
        if 'filter mitchell (command line)' not in proc.stdout or 'implied by' in proc.stdout:
            raise SystemExit('FAIL: --mip-filter must keep its own filter over an srgb material')
        if 'the material sets filter' in proc.stdout:
            raise SystemExit('FAIL: nothing of the material was overridden, so no filter warning may print')

        # THE INERT VALUES ASK FOR NOTHING. A key is not a request: `"srgb": false`, `"edge": "clamp"` and
        # `"normal_map": false` name the behaviour the encoder has anyway, so they imply no filter and contradict
        # none -- an explicit `default` beside one of them is accepted, where the same key set to its asking value
        # would be the refused contradiction. The row still shows the value and its json source, the filter stays
        # `default`, and the latents are the bare material's to the byte: the vanilla chain, not a re-derived one.
        bare = write_material(os.path.join(tmp, 'inert_ref.json'), [{'file': 'c0.png'}])
        ref_dir = os.path.join('out', 'tiny_implied_inert_ref')
        proc = run([encode, bare, '-o', ref_dir, '--png', '0'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: the bare material behind the inert-value cases returned %d' % proc.returncode)
        for tag, entry, row in (('srgbfalse', {'file': 'c0.png', 'srgb': False}, 'srgb no (json)'),
                                ('edgeclamp', {'file': 'c0.png', 'edge': 'clamp'}, 'edge clamp (json)'),
                                ('normalfalse', {'file': 'c0.png', 'normal_map': False}, 'normal map no (json)'),
                                ('inertdefault', {'file': 'c0.png', 'filter': 'default', 'srgb': False},
                                 'filter default (json)')):
            path = write_material(os.path.join(tmp, tag + '.json'), [entry])
            odir = os.path.join('out', 'tiny_implied_' + tag)
            proc = run([encode, path, '-o', odir, '--png', '0'])
            if proc.returncode != 0:
                raise SystemExit('FAIL: the inert %s entry must be accepted, not %r' % (tag, proc.stderr.strip()))
            if row not in proc.stdout or 'implied by' in proc.stdout:
                raise SystemExit('FAIL: an inert value must imply no filter and must still print %r' % row)
            if tag != 'inertdefault' and 'filter default (default)' not in proc.stdout:
                raise SystemExit('FAIL: an inert value beside no filter must leave filter default (default)')
            for suffix in ('_lat0.dds', '_lat1.dds'):
                if (open(os.path.join(ROOT, odir, tag + suffix), 'rb').read() !=
                        open(os.path.join(ROOT, ref_dir, 'inert_ref' + suffix), 'rb').read()):
                    raise SystemExit('FAIL: an inert %s must leave the vanilla chain, %s and all' % (tag, suffix))

        # An implied box with no level below the base is idle like any other filter: --mips 0 stores the base alone,
        # the box has nothing to derive, and the latents are the bare material's.
        for tag, entry in (('mips0_imp', {'file': 'c0.png', 'srgb': True}), ('mips0_ref', {'file': 'c0.png'})):
            path = write_material(os.path.join(tmp, tag + '.json'), [entry])
            proc = run([encode, path, '-o', os.path.join('out', 'tiny_implied_' + tag), '--png', '0', '--mips', '0'])
            if proc.returncode != 0:
                raise SystemExit('FAIL: an implied box under --mips 0 (%s) returned %d' % (tag, proc.returncode))
        for suffix in ('_lat0.dds', '_lat1.dds'):
            if (open(os.path.join(ROOT, 'out', 'tiny_implied_mips0_imp', 'mips0_imp' + suffix), 'rb').read() !=
                    open(os.path.join(ROOT, 'out', 'tiny_implied_mips0_ref', 'mips0_ref' + suffix), 'rb').read()):
                raise SystemExit('FAIL: an implied box with no chain level must leave %s byte-identical' % suffix)

        # THE FOOTPRINT, THROUGH THE MATERIAL. The implied box re-derives every level FROM THE BASE, which is the
        # built-in iterated box's picture only where the ratio is exact. 44x44 at --mip-min 4 gives 44, 22, 11, 5:
        # M1 and M2 are exact and agree, M3 is the odd halving and does not. There is no such thing as an implied box
        # without a key -- no key, no implication -- so what the implied chain is compared against for the difference
        # is the BARE material's built-in one.
        tmp44 = tempfile.mkdtemp(prefix='nntc_implied44_')
        try:
            crop_inputs(tmp44, 1, 44)
            odd = {}
            for tag, entry in (('imp', {'file': 'c0.png', 'srgb': True}),
                               ('exp', {'file': 'c0.png', 'filter': 'box', 'srgb': True}),
                               ('wrap', {'file': 'c0.png', 'edge': 'wrap'}),
                               ('bare', {'file': 'c0.png'})):
                path = write_material(os.path.join(tmp44, tag + '.json'), [entry])
                odir = os.path.join('out', 'tiny_implied44_' + tag)
                proc = run([encode, path, '-o', odir, '--mip-min', '4'])
                if proc.returncode != 0:
                    raise SystemExit('FAIL: the 44x44 material run %s returned %d' % (tag, proc.returncode))
                for m in (1, 2, 3):
                    odd[(tag, m)] = open(os.path.join(ROOT, odir, '%s_src_M%d.png' % (tag, m)), 'rb').read()
            for m in (1, 2):
                if odd[('imp', m)] != odd[('exp', m)]:
                    raise SystemExit('FAIL: on the odd 44 chain an implied box must derive M%d exactly as the '
                                     'explicit box does' % m)
            if odd[('wrap', 1)] != odd[('bare', 1)]:
                raise SystemExit('FAIL: at the exact ratio of M1 the implied box and the built-in chain must agree')
            if odd[('wrap', 3)] == odd[('bare', 3)]:
                raise SystemExit('FAIL: at the odd halving of M3 the implied box re-derives from the base and must '
                                 'differ from the built-in iterated chain')
        finally:
            shutil.rmtree(tmp44, ignore_errors=True)
        print('  the material: srgb, normal_map and edge each imply filter box when the entry names no filter, byte '
              'for byte the explicit box, and imply nothing when it does, when a flag names one, or when the value '
              'is the inert one; on the odd 44 / 22 / 11 / 5 chain the implied box parts company with the built-in '
              'one at M3')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record('the implied box filter', 'srgb, normal_map and edge beside no filter derive the explicit box chain byte '
                                     'for byte, the row names the key, source.inputs says box; a named filter, a '
                                     '--mip-filter and an inert value each imply nothing; and the implied box parts '
                                     'company with the built-in chain at the odd halving of the 44 chain')


def material_detail_checks(encode, build_dir):
    """The material JSON's arithmetic, its precedence warnings, its escaping, its refusals and its -o forms.

    material_checks above asserts that the path WORKS. What is asserted here is what it computes: a known answer for
    the sRGB chain, an edge mode that reaches the resizer, per-texture rgb weights that are the same numbers the one
    global triple produces, the exact text and count of every override warning, a `type` full of characters that must
    be escaped surviving a round trip through the asset's own JSON, and the dozen refusals the path had no gate on.
    """
    import shutil
    import tempfile
    from PIL import Image as PILImage
    tmp = tempfile.mkdtemp(prefix='nntc_matdet_')
    try:
        crops = crop_inputs(tmp, 7, 32)

        # (18) THE sRGB CHAIN'S KNOWN ANSWER. Eight columns alternating 0 and 255, box filtered at an exact 2:1: every
        # M1 texel is the mean of one black and one white column. In the source's own encoding that mean is 0.5, which
        # is the byte 128. In LINEAR light it is the mean of 0 and 1, which is 0.5 linear, which is sRGB 0.7354 and the
        # byte 188. There is no tolerance in this: the two numbers are what the transfer function is, and a chain that
        # went through 8 bits anywhere, or applied the curve once instead of twice, cannot produce both.
        cols = os.path.join(tmp, 'cols.png')
        write_rgb8(cols, 8, 8, lambda x, y: (255, 255, 255) if (x & 1) else (0, 0, 0))
        for name, entry, want in (('plain', {'file': 'cols.png', 'filter': 'box'}, (128, 128, 128)),
                                  ('srgb', {'file': 'cols.png', 'filter': 'box', 'srgb': True}, (188, 188, 188))):
            path = write_material(os.path.join(tmp, name + '.json'), [entry])
            odir = os.path.join('out', 'mat_srgb_' + name)
            proc = run([encode, path, '-o', odir, '--mip-min', '4', '--quiet'])
            if proc.returncode != 0:
                raise SystemExit('FAIL: the %s column material returned %d' % (name, proc.returncode))
            img = PILImage.open(os.path.join(ROOT, odir, name + '_src_M1.png')).convert('RGB')
            got = set(img.getdata())
            if img.size != (4, 4) or got != {want}:
                raise SystemExit('FAIL: the %s chain of alternating columns must be %s everywhere at M1, not %s'
                                 % (name, want, sorted(got)))
        print('  the material: a box over alternating columns is 128 in the source encoding and 188 in linear light, '
              'exactly')

        # (19) THE EDGE MODE REACHES THE RESIZER. clamp and wrap differ only at the border, and a cubic's support is
        # two texels wide, so one filtered level of one crop is enough to tell them apart -- and a name that reached
        # nothing would give the same bytes twice.
        edges = {}
        for name, edge in (('clamp', 'clamp'), ('wrap', 'wrap')):
            path = write_material(os.path.join(tmp, 'edge_' + name + '.json'),
                                  [{'file': 'c0.png', 'filter': 'mitchell', 'edge': edge}])
            odir = os.path.join('out', 'mat_edge_' + name)
            proc = run([encode, path, '-o', odir, '--mip-min', '4', '--quiet'])
            if proc.returncode != 0:
                raise SystemExit('FAIL: the %s edge material returned %d' % (name, proc.returncode))
            edges[name] = open(os.path.join(ROOT, odir, 'edge_' + name + '_src_M1.png'), 'rb').read()
        if edges['clamp'] == edges['wrap']:
            raise SystemExit('FAIL: edge clamp and edge wrap derived the same chain, so the key reached nothing')
        print('  the material: edge clamp and edge wrap derive different chains under mitchell')

        # (21) PER-TEXTURE rgb WEIGHTS ARE THE SAME ARITHMETIC AS THE GLOBAL TRIPLE. The plan's own promise is that one
        # global triple normalised per texture is EXACTLY what the single global normalisation used to produce, so a
        # material that spells the same triple on every texture has to solve to the same bytes as the flag does.
        same = write_material(os.path.join(tmp, 'rgbsame.json'),
                              [{'file': 'c0.png', 'rgb_weights': [1, 1, 0.5]},
                               {'file': 'c1.png', 'rgb_weights': [1, 1, 0.5]}])
        flag_dir = os.path.join('out', 'mat_rgb_flag')
        json_dir = os.path.join('out', 'mat_rgb_json')
        a = run([encode, crops[0], crops[1], '-o', flag_dir, '--png', '0', '--quiet', '--rgb-weights', '1,1,0.5'])
        b = run([encode, same, '-o', json_dir, '--png', '0', '--quiet'])
        if a.returncode != 0 or b.returncode != 0:
            raise SystemExit('FAIL: the rgb-weight pair returned %d / %d' % (a.returncode, b.returncode))
        for suffix in ('_lat0.dds', '_lat1.dds'):
            x = open(os.path.join(ROOT, flag_dir, 'c0' + suffix), 'rb').read()
            y = open(os.path.join(ROOT, json_dir, 'rgbsame' + suffix), 'rb').read()
            if x != y:
                raise SystemExit('FAIL: the same rgb weights from the flag and from the material differ in %s'
                                 % suffix)
        # And a triple on ONE texture is a different problem from no triple at all, or the key would be doing nothing.
        one = write_material(os.path.join(tmp, 'rgbone.json'),
                             [{'file': 'c0.png', 'rgb_weights': [1, 1, 0.5]}, {'file': 'c1.png'}])
        none = write_material(os.path.join(tmp, 'rgbnone.json'), [{'file': 'c0.png'}, {'file': 'c1.png'}])
        ships = []
        for path, name in ((one, 'rgbone'), (none, 'rgbnone')):
            proc = run([encode, path, '-o', os.path.join('out', 'mat_' + name), '--png', '0'])
            if proc.returncode != 0:
                raise SystemExit('FAIL: the %s material returned %d' % (name, proc.returncode))
            ships.append(number(proc.stdout, r'E shipped\s+([0-9.e+-]+)', 'E shipped')[0])
        if ships[0] == ships[1]:
            raise SystemExit('FAIL: rgb_weights on one texture of two must change the objective, and E shipped is '
                             '%.9e either way' % ships[0])
        print('  the material: per-texture rgb weights reproduce the global triple byte for byte, and a triple on one '
              'texture moves E shipped (%.6e against %.6e)' % (ships[0], ships[1]))

        # (22) THE OVERRIDE WARNINGS, by their exact text and their exact count. One per texture that CARRIED the key,
        # and none for a texture that did not: a warning that fired for every texture would say the material had set
        # something it never set.
        both = write_material(os.path.join(tmp, 'ovr.json'),
                              [{'file': 'c0.png', 'filter': 'mitchell', 'rgb_weights': [1, 1, 0.5]},
                               {'file': 'c1.png', 'filter': 'mitchell', 'rgb_weights': [1, 1, 0.5]}])
        proc = run([encode, both, '-o', os.path.join('out', 'mat_ovr'), '--png', '0', '--quiet',
                    '--mip-filter', 'catmullrom,catmullrom', '--rgb-weights', '1,1,1'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: the override run returned %d' % proc.returncode)
        for t, name in ((0, 'c0.png'), (1, 'c1.png')):
            want = ('WARNING: texture %d (%s): the material sets filter mitchell and --mip-filter sets catmullrom; '
                    'the command line wins' % (t, name))
            if want not in proc.stdout:
                raise SystemExit('FAIL: the filter override must warn for texture %d by name, not %r'
                                 % (t, proc.stdout))
            want = ('WARNING: texture %d (%s): the material sets rgb weights 1 1 0.5 and --rgb-weights sets 1 1 1; '
                    'the command line wins' % (t, name))
            if want not in proc.stdout:
                raise SystemExit('FAIL: the rgb-weight override must warn for texture %d by name' % t)
        if proc.stdout.count('the command line wins') != 4:
            raise SystemExit('FAIL: two keys on two textures are four overrides, and the run printed %d'
                             % proc.stdout.count('the command line wins'))
        half = write_material(os.path.join(tmp, 'ovrhalf.json'),
                              [{'file': 'c0.png', 'filter': 'mitchell'}, {'file': 'c1.png'}])
        proc = run([encode, half, '-o', os.path.join('out', 'mat_ovr_half'), '--png', '0', '--quiet',
                    '--mip-filter', 'catmullrom,catmullrom'])
        if proc.returncode != 0 or proc.stdout.count('the command line wins') != 1:
            raise SystemExit('FAIL: only the texture that carried the key may warn, and the run printed %d warnings'
                             % proc.stdout.count('the command line wins'))
        print('  the material: every override warns once per texture that carried the key, by its exact text (4 and 1)')

        # (23) THE ESCAPING. `type` is free text echoed into the asset's own JSON, and the writer concatenates text
        # rather than serialising it, so it carries an escaper of its own. A quote, a backslash and a tab are the three
        # characters that break a hand-written writer, and what they must survive is a round trip: the asset has to
        # PARSE, and the string that comes back has to be the string that went in.
        nasty = 'a"b\\c\td'
        esc = write_material(os.path.join(tmp, 'escape.json'), [{'file': 'c0.png', 'type': nasty}])
        odir = os.path.join('out', 'mat_escape')
        proc = run([encode, esc, '-o', odir, '--quiet'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: a material whose type needs escaping returned %d' % proc.returncode)
        meta = json.load(open(os.path.join(ROOT, odir, 'escape_nntc.json')))
        got = meta['source']['inputs'][0]['type']
        if got != nasty:
            raise SystemExit('FAIL: the asset JSON gives the type back as %r, not %r' % (got, nasty))
        if os.name == 'nt':
            view = executable(build_dir, 'nntc_view')
            frame, _ = shot(view, os.path.join(odir, 'escape_nntc.json'), os.path.join('out', 'mat_escape.bmp'), [])
            if not frame:
                raise SystemExit('FAIL: the viewer could not open an asset whose JSON carries an escaped string')
        print('  the material: a type of %r round-trips through the asset JSON and the viewer opens it' % nasty)

        # (24) THE REFUSALS THAT HAD NO GATE. Every one of them is a run that would otherwise have gone through with a
        # key doing nothing, or with a number the objective cannot use.
        bad_dir = os.path.join('out', 'mat_refuse')
        seven = write_material(os.path.join(tmp, 'seven.json'), [{'file': 'c%d.png' % i} for i in range(7)])
        cases = [
            ('seven images on the command line', [encode] + crops + ['-o', bad_dir, '--png', '0'],
             'one to 6 input images are needed (7 given)'),
            ('a material of seven entries', [encode, seven, '-o', bad_dir, '--png', '0'],
             'has 7 entries and a material is one to 6 textures'),
        ]
        entries = [
            ('an edge mode that is not one', {'file': 'c0.png', 'filter': 'mitchell', 'edge': 'mirror'},
             "'edge' is 'mirror' and must be clamp or wrap"),
            ('a filter that is not a name', {'file': 'c0.png', 'filter': 'lanczos'},
             'which is not a filter name'),
            # Each of these names `default` out loud AND asks for a derivation by VALUE: it is the CONTRADICTION that
            # is refused, and leaving the filter out instead is the implied box of implied_box_check. A key set to its
            # inert value asks for nothing and is refused nowhere; implied_box_check accepts those.
            ('edge wrap beside an explicit default filter', {'file': 'c0.png', 'filter': 'default', 'edge': 'wrap'},
             "sets 'edge', which only a derived source chain can honour"),
            ('normal_map beside an explicit default filter',
             {'file': 'c0.png', 'filter': 'default', 'normal_map': True},
             "sets 'normal_map', which only a derived source chain can honour"),
            ('an entry with no file', {'type': 'albedo'}, "no 'file'"),
            ('a file that is not a string', {'file': 3}, "'file' must be a non-empty string"),
            ('a weight that is not a number', {'file': 'c0.png', 'weight': '2'}, "'weight' must be a number"),
            ('rgb weights that are not three numbers', {'file': 'c0.png', 'rgb_weights': [1, 1]},
             "'rgb_weights' must be an array of three numbers"),
            ('srgb that is not a boolean', {'file': 'c0.png', 'filter': 'mitchell', 'srgb': 'yes'},
             "'srgb' must be true or false"),
            ('the only weight being zero', {'file': 'c0.png', 'weight': 0},
             '--weights must not be all zero'),
            ('a negative rgb weight', {'file': 'c0.png', 'rgb_weights': [1, -1, 1]},
             '--rgb-weights must not be negative'),
            ('a file that is not on disk', {'file': 'nope.png'}, 'cannot read'),
        ]
        for i, (what, entry, want) in enumerate(entries):
            path = write_material(os.path.join(tmp, 'refuse%d.json' % i), [entry])
            cases.append((what, [encode, path, '-o', bad_dir, '--png', '0'], want))
        not_array = os.path.join(tmp, 'obj.json')
        open(not_array, 'wb').write(b'{"file": "c0.png"}')
        cases.append(('a material that is not an array', [encode, not_array, '-o', bad_dir, '--png', '0'],
                      'must be one JSON array of texture entries'))
        pair = write_material(os.path.join(tmp, 'pair.json'), [{'file': 'c0.png'}, {'file': 'c1.png'}])
        cases.append(('an empty entry inside --mip-filter',
                      [encode, crops[0], crops[1], '-o', bad_dir, '--png', '0', '--mip-filter', 'box,,box'],
                      "--mip-filter 'box,,box' is not a comma-separated list of filter names"))
        cases.append(('--weights counted against a material',
                      [encode, pair, '-o', bad_dir, '--png', '0', '--weights', '1'],
                      '--weights gives 1 value for 2 textures'))
        for what, cmd, want in cases:
            bad = run(cmd)
            if bad.returncode == 0 or want not in bad.stderr:
                raise SystemExit('FAIL: %s must be refused with %r, not %r' % (what, want, bad.stderr.strip()))
        refusals = len(cases)
        print('  the material: %d refusals, each by its own message' % refusals)

        # (25) THE -o FORMS, and the vanilla identity. `-o dir/x.nntc` is a PREFIX, extension and all, so the asset is
        # x.nntc_lat0.dds beside x.nntc_nntc.json; `-o dir/` is a directory and the asset takes the MATERIAL's stem
        # inside it; `-o dir/name.json` IS the descriptor, and the .dds files take that name without the extension.
        vanilla = write_material(os.path.join(tmp, 'm_vanilla.json'), [{'file': 'c0.png'}, {'file': 'c1.png'}])
        pref = os.path.join('out', 'mat_prefix', 'asset.nntc')
        proc = run([encode, vanilla, '-o', pref, '--png', '0', '--quiet'])
        if proc.returncode != 0 or not os.path.isfile(os.path.join(ROOT, pref + '_lat0.dds')):
            raise SystemExit('FAIL: -o with an extension must be the prefix itself, so asset.nntc_lat0.dds')
        if not os.path.isfile(os.path.join(ROOT, pref + '_nntc.json')):
            raise SystemExit('FAIL: the descriptor of an explicit prefix is PREFIX_nntc.json')
        trailing = os.path.join('out', 'mat_trailing') + os.sep
        proc = run([encode, vanilla, '-o', trailing, '--png', '0', '--quiet'])
        if proc.returncode != 0 or not os.path.isfile(os.path.join(ROOT, trailing, 'm_vanilla_lat0.dds')):
            raise SystemExit('FAIL: -o with a trailing separator must be a directory holding MATERIAL_lat0.dds')
        if not os.path.isfile(os.path.join(ROOT, trailing, 'm_vanilla_nntc.json')):
            raise SystemExit('FAIL: -o with a trailing separator must hold MATERIAL_nntc.json')
        named = os.path.join('out', 'mat_named', 'chosen.json')
        proc = run([encode, vanilla, '-o', named, '--png', '0', '--quiet'])
        if proc.returncode != 0 or not os.path.isfile(os.path.join(ROOT, named)):
            raise SystemExit('FAIL: -o NAME.json must write the descriptor under exactly that name')
        if not os.path.isfile(os.path.join(ROOT, 'out', 'mat_named', 'chosen_lat0.dds')):
            raise SystemExit('FAIL: -o NAME.json must write chosen_lat0.dds beside chosen.json')
        if os.path.isfile(os.path.join(ROOT, 'out', 'mat_named', 'chosen_nntc.json')):
            raise SystemExit('FAIL: -o NAME.json must not also write a _nntc descriptor')
        if os.name == 'nt':
            view = executable(build_dir, 'nntc_view')
            frame, _ = shot(view, named, os.path.join('out', 'mat_named.bmp'), [])
            if not frame:
                raise SystemExit('FAIL: the viewer could not open the descriptor -o named outright')
        # The fallback in tools/dds_decode.py: PREFIX.json is what an asset written before the _nntc suffix carries,
        # and `-o NAME.json` writes exactly that shape, so this asset is the old naming and must still read.
        proc = run([sys.executable, os.path.join('tools', 'dds_decode.py'),
                    os.path.join('out', 'mat_named', 'chosen'), '--grid'])
        if proc.returncode != 0 or 'note:' not in proc.stdout:
            raise SystemExit('FAIL: dds_decode.py must fall back to PREFIX.json with a note (%d, %r)'
                             % (proc.returncode, proc.stdout[-200:]))

        # THE VANILLA IDENTITY. A material whose entries carry nothing but `file` asks for exactly what the bare
        # command line asks for, so it must solve to exactly the same latents. The descriptors differ only in the three
        # file names the prefix makes, and nowhere else.
        cli_dir = os.path.join('out', 'mat_vanilla_cli')
        json_dir2 = os.path.join('out', 'mat_vanilla_json')
        for cmd, where in (([encode, crops[0], crops[1], '-o', cli_dir], cli_dir),
                           ([encode, vanilla, '-o', json_dir2], json_dir2)):
            proc = run(cmd + ['--png', '0', '--quiet'])
            if proc.returncode != 0:
                raise SystemExit('FAIL: the vanilla identity run into %s returned %d' % (where, proc.returncode))
        for suffix in ('_lat0.dds', '_lat1.dds'):
            x = open(os.path.join(ROOT, cli_dir, 'c0' + suffix), 'rb').read()
            y = open(os.path.join(ROOT, json_dir2, 'm_vanilla' + suffix), 'rb').read()
            if x != y:
                raise SystemExit('FAIL: a material of bare `file` keys is not the bare command line: %s differs'
                                 % suffix)
        left = io.open(os.path.join(ROOT, cli_dir, 'c0_nntc.json'), encoding='utf-8').read().replace('c0_lat', 'A_lat')
        right = io.open(os.path.join(ROOT, json_dir2, 'm_vanilla_nntc.json'),
                        encoding='utf-8').read().replace('m_vanilla_lat', 'A_lat')
        if left != right:
            raise SystemExit('FAIL: the two vanilla descriptors differ by more than the asset names')
        print('  the material: -o prefix.ext, -o dir/ and -o name.json all land where they say, the descriptor takes '
              'the _nntc suffix except where -o named it, dds_decode.py falls back to the old name, and a bare-`file` '
              'material is the bare command line byte for byte')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record('the material JSON in detail', 'the sRGB known answer (128 against 188), the edge mode, per-texture rgb '
                                          'weights against the global triple, every override warning by text and '
                                          'count, an escaped type through the asset JSON, %d refusals, the three -o '
                                          'forms with the _nntc descriptor and its fallback, and the vanilla identity'
                                          % refusals)


def review_fix_checks(encode, build_dir):
    """The cases the review of v0.10a-d found, one arm per finding.

    They are gathered in one function because they share their scratch material and because what they have in common is
    the point of every one of them: a run that PRODUCED something -- a deleted file, a nan objective, a silent exit --
    where it should have refused, or said nothing where the caller needed a sentence.
    """
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix='nntc_review_')
    try:
        crops = crop_inputs(tmp, 6, 32)
        beside = write_material(os.path.join(tmp, 'beside.json'), [{'file': 'c0.png'}, {'file': 'c1.png'}])

        # (1) THE MATERIAL IS NOT THE ASSET'S DESCRIPTOR. The descriptor is PREFIX_nntc.json, so a material encoded
        # into its own directory (-o naming the directory the material is in) no longer names it, and the run goes
        # through: the material is read back afterwards to prove it was not touched, and the descriptor lands under
        # the _nntc name next to it. With no -o at all the asset lands in the CURRENT DIRECTORY (the gate's root), so
        # that run is checked there and its three files taken away again.
        for extra in ([], ['-o', tmp + os.sep]):
            good = run([encode, beside, '--png', '0', '--quiet'] + extra)
            if good.returncode != 0:
                raise SystemExit('FAIL: a material encoded with no -o or into its own directory must encode (%d)'
                                 % good.returncode)
            entries = json.load(open(beside))
            if not isinstance(entries, list) or len(entries) != 2:
                raise SystemExit('FAIL: the run overwrote the material itself')
            where = tmp if extra else ROOT
            if not os.path.isfile(os.path.join(where, 'beside_nntc.json')):
                raise SystemExit('FAIL: the descriptor must be beside_nntc.json in %s' % ('the directory of the material' if extra else 'the current directory'))
            if not extra:
                for name in ('beside_lat0.dds', 'beside_lat1.dds', 'beside_nntc.json'):
                    os.remove(os.path.join(ROOT, name))
        # The refusal is the backstop for the spelling that CAN still name the material: -o giving a .json is the
        # descriptor outright, so `-o material.json` would write the asset over the material that described the run.
        bad = run([encode, beside, '-o', beside, '--png', '0'])
        if bad.returncode == 0 or 'which is the material' not in bad.stderr:
            raise SystemExit('FAIL: -o naming the material itself must be refused, not %r' % bad.stderr.strip())
        entries = json.load(open(beside))
        if not isinstance(entries, list) or len(entries) != 2:
            raise SystemExit('FAIL: the refused run overwrote the material itself')
        ok = run([encode, beside, '-o', os.path.join('out', 'review_elsewhere'), '--png', '0', '--quiet'])
        if ok.returncode != 0:
            raise SystemExit('FAIL: the same material into a different directory must encode (%d)' % ok.returncode)
        print('  the review: a material encoded with no -o writes beside_nntc.json in the current directory, into its own '
              'directory beside itself, and survives both; -o naming the material itself is still refused; -o elsewhere runs')

        # (2) A WEIGHT NO FLOAT CHANNEL WEIGHT CAN HOLD. 1e300 is a perfectly good double and an infinity as a float,
        # and the run used to go all the way to a written asset with every progress line reading psnr 100.00. The
        # owner's rule is a CLAMP onto the sane range, not a refusal: the run must succeed, warn once naming the
        # texture and the value, and the settings row must show the clamped weight.
        huge = write_material(os.path.join(tmp, 'huge.json'), [{'file': 'c0.png', 'weight': 1e300}])
        for cmd in ([encode, huge, '-o', os.path.join('out', 'review_huge'), '--png', '0'],
                    [encode, crops[0], crops[1], '-o', os.path.join('out', 'review_huge'), '--png', '0',
                     '--weights', '1,1e300']):
            proc = run(cmd)
            if proc.returncode != 0:
                raise SystemExit('FAIL: a weight of 1e300 must be clamped and the run must go on, not %r'
                                 % proc.stderr.strip())
            if proc.stdout.count('is above the sane range and is clamped to 128') != 1:
                raise SystemExit('FAIL: the clamp of a 1e300 weight must warn exactly once')
            if not re.search(r'weight 128 \((json|command line)\)', proc.stdout):
                raise SystemExit('FAIL: the settings row must show the clamped weight 128')
            if 'psnr 100.00' in proc.stdout or 'nan' in proc.stdout.lower().replace('nan(ind)', 'nan'):
                raise SystemExit('FAIL: the clamped run must not carry a nan objective')
        small = run([encode, crops[0], crops[1], '-o', os.path.join('out', 'review_huge'), '--png', '0',
                     '--weights', '1,1e-9'])
        if small.returncode != 0 or 'is raised to 1e-06' not in small.stdout:
            raise SystemExit('FAIL: a positive weight below what a float holds must be raised with a warning')
        print('  the review: a weight of 1e300 is clamped to 128 with one warning, from the material and from '
              '--weights, and 1e-9 is raised to 1e-06')

        # (3) --weights USED TO EXIT 1 WITHOUT A WORD, while the --mip-filter arm nine lines below it named its value.
        for value in ('abc', '', '1,', ',1', 'inf', 'nan', '1e999'):
            bad = run([encode, crops[0], '-o', os.path.join('out', 'review_bad'), '--png', '0', '--weights', value])
            if bad.returncode == 0 or 'ERROR' not in bad.stderr or '--weights' not in bad.stderr:
                raise SystemExit('FAIL: --weights %r must be refused with a message naming the flag, not %r'
                                 % (value, bad.stderr.strip()))
        print('  the review: --weights is refused by name on seven spellings that used to exit 1 in silence')

        # (4) A NON-ASCII FILE NAME OUT OF A MATERIAL. A JSON string is UTF-8 and the narrow CRT calls read the ANSI
        # code page, so the very file that opened as a positional could not be opened when a material named it.
        # Windows only: elsewhere the bytes go to the filesystem as they stand and there was never anything to fix.
        if os.name == 'nt':
            accent = os.path.join(tmp, u'caf\u00e9.png')
            shutil.copyfile(crops[0], accent)
            uni = os.path.join(tmp, 'unicode.json')
            with io.open(uni, 'w', encoding='utf-8') as f:
                f.write(json.dumps([{'file': u'caf\u00e9.png'}], ensure_ascii=False))
            proc = run([encode, uni, '-o', os.path.join('out', 'review_unicode'), '--png', '0', '--quiet'])
            if proc.returncode != 0:
                raise SystemExit('FAIL: a material naming a non-ASCII file must encode it (%d): %s'
                                 % (proc.returncode, proc.stderr.strip()))
            print('  the review: a material naming a non-ASCII file encodes it')
            # AND ON THE COMMAND LINE. argv arrives through the narrow entry point, which reads the process's active
            # code page; the embedded manifest declares it UTF-8, so the accented name survives from the shell to fopen
            # and back out in the wrote line. The gate passes the argument as a Python str, which subprocess hands to
            # CreateProcessW; the C runtime then narrows it to the active code page, UTF-8 under the manifest.
            proc = run([encode, accent, '-o', os.path.join('out', 'review_unicode_cli'), '--png', '0'])
            if proc.returncode != 0:
                raise SystemExit('FAIL: a non-ASCII file named on the command line must encode (%d): %s'
                                 % (proc.returncode, proc.stderr.strip()))
            if u'caf\u00e9' not in proc.stdout:
                raise SystemExit('FAIL: the wrote line must carry the accented name as UTF-8, not %r' % proc.stdout[-200:])
            print('  the review: a non-ASCII file named on the command line encodes, and its name prints as UTF-8')
        else:
            print('  the review: the non-ASCII file name is a Windows case and is skipped')

        # (5) --quiet MEANS NO PROGRESS. It used to leave about thirty lines behind: the level-0 init block, the grid
        # freeze, the pack, the refinement, the outer passes and two stray blank lines.
        quiet = run([encode, crops[0], crops[1], '-o', os.path.join('out', 'review_quiet'), '--png', '0', '--quiet'])
        if quiet.returncode != 0:
            raise SystemExit('FAIL: the --quiet run returned %d' % quiet.returncode)
        noisy = [l for l in quiet.stdout.splitlines() if re.match(r'^(level 0|grid frozen|round|nntc_encode:)', l)]
        if noisy:
            raise SystemExit('FAIL: --quiet still printed %d progress line(s), the first being %r'
                             % (len(noisy), noisy[0]))
        if '\nreport\n' in quiet.stdout or 'wrote ' in quiet.stdout:
            raise SystemExit('FAIL: --quiet must print neither the report nor the wrote lines')
        # The two things --quiet must NOT take with it: the warning about the input, and the warning about the
        # settings. Neither is progress, and a run that quietly padded or quietly overrode would be the same silence
        # every other case here is about.
        pad = os.path.join(tmp, 'pad45.png')
        write_rgb8(pad, 45, 30, lambda x, y: ((x * 5) & 255, (y * 9) & 255, ((x + y) * 3) & 255))
        proc = run([encode, pad, '-o', os.path.join('out', 'review_quiet_pad'), '--png', '0', '--quiet'])
        if proc.returncode != 0 or 'WARNING: the input is 45x30' not in proc.stdout:
            raise SystemExit('FAIL: --quiet must still print the padding WARNING')
        over = write_material(os.path.join(tmp, 'over.json'), [{'file': 'c0.png', 'weight': 2}])
        proc = run([encode, over, '-o', os.path.join('out', 'review_quiet_over'), '--png', '0', '--quiet',
                    '--weights', '1'])
        if proc.returncode != 0 or 'the command line wins' not in proc.stdout:
            raise SystemExit('FAIL: --quiet must still print the override WARNING')
        print('  the review: --quiet prints nothing but the WARNINGs')

        # (6) A BYTE-ORDER MARK AND A PARSE POSITION. Several Windows editors write the mark when they save UTF-8; a
        # trailing comma is the everyday hand-edit. Both used to come back as the same "must be one JSON array".
        bom = os.path.join(tmp, 'bom.json')
        open(bom, 'wb').write(b'\xef\xbb\xbf' + json.dumps([{'file': 'c0.png'}]).encode('utf-8'))
        proc = run([encode, bom, '-o', os.path.join('out', 'review_bom'), '--png', '0', '--quiet'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: a material with a UTF-8 byte-order mark must encode (%d)' % proc.returncode)
        comma = os.path.join(tmp, 'comma.json')
        open(comma, 'wb').write(b'[\r\n  {"file": "c0.png"},\r\n]\r\n')
        bad = run([encode, comma, '-o', os.path.join('out', 'review_bad'), '--png', '0'])
        if bad.returncode == 0 or not re.search(r'does not parse: .*\(line \d+, column \d+\)', bad.stderr):
            raise SystemExit('FAIL: a trailing comma must be refused with a line and a column, not %r'
                             % bad.stderr.strip())
        print('  the review: a byte-order mark encodes and a trailing comma is refused with its line and column')

        # (7) A REPEATED KEY. Last-wins made the entry below encode c0.png without a word about nope.png.
        dup = os.path.join(tmp, 'dup.json')
        open(dup, 'wb').write(b'[{"file": "nope.png", "file": "c0.png"}]')
        bad = run([encode, dup, '-o', os.path.join('out', 'review_bad'), '--png', '0'])
        if bad.returncode == 0 or "'file' appears twice" not in bad.stderr:
            raise SystemExit('FAIL: a repeated key in one entry must be refused by name, not %r' % bad.stderr.strip())
        print('  the review: a key repeated inside one entry is refused by name')

        # (8) THE SEED DIRECTION AT SIX TEXTURES. The line was bounded by the old four-texture cap, so outputs 12..17
        # printed as +0.0000 whatever the direction the seed had actually taken.
        proc = run([encode] + crops + ['-o', os.path.join('out', 'review_six_dir'), '--png', '0', '--init0',
                                       'residual', '--rounds', '1'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: the six-texture --init0 residual run returned %d' % proc.returncode)
        rows = re.findall(r'^\s+e =((?:\s[-+]\d+\.\d+)+)\s*$', proc.stdout, re.M)
        if not rows:
            raise SystemExit('FAIL: the seed printed no direction line')
        for row in rows:
            values = row.split()
            if len(values) != 18:
                raise SystemExit('FAIL: a six-texture seed direction has 18 entries, the line printed %d'
                                 % len(values))
            if all(float(v) == 0.0 for v in values[12:]):
                raise SystemExit('FAIL: outputs 12..17 of the seed direction are all zero, which is the truncation '
                                 'this case exists for')
        print('  the review: the seed direction prints all 18 outputs at six textures (%d lines)' % len(rows))

        # (10) THE CLAMP IS OBSERVABLE. A one-pixel black and white stripe is the worst case there is for a cubic's
        # negative lobe, and --diag now prints the range of every source level, so the clamp can be seen rather than
        # asserted in a comment. The comment it replaces was wrong about why the clamp is there.
        stripe = os.path.join(tmp, 'stripe.png')
        write_rgb8(stripe, 64, 64, lambda x, y: (255, 255, 255) if (x & 1) else (0, 0, 0))
        proc = run([encode, stripe, '-o', os.path.join('out', 'review_stripe'), '--png', '0', '--mip-filter',
                    'mitchell', '--diag'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: the stripe run returned %d' % proc.returncode)
        ranges = re.findall(r'^      M(\d+)\s+\d+x\d+\s+tex0 \[(-?\d+\.\d+), (-?\d+\.\d+)\]', proc.stdout, re.M)
        if len(ranges) < 2:
            raise SystemExit('FAIL: --diag printed %d source-range rows, expected the whole chain' % len(ranges))
        for level, lo, hi in ranges:
            if float(lo) < 0.0 or float(hi) > 1.0:
                raise SystemExit('FAIL: the mitchell chain of a stripe reaches [%s, %s] at M%s, so the clamp is not '
                                 'happening' % (lo, hi, level))
        shipped = number(proc.stdout, r'E shipped\s+([0-9.e+-]+)', 'the stripe run')[0]
        if shipped != shipped or shipped in (float('inf'), float('-inf')):
            raise SystemExit('FAIL: the stripe run shipped a non-finite E (%r)' % shipped)
        header = proc.stdout.split('diag: per-texture psnr', 1)[1][:400]
        if 'filter mitchell, srgb no, normal map no' not in header:
            raise SystemExit("FAIL: the --diag header must name each texture's chain the way the psnr lines do")
        print('  the review: every level of a mitchell stripe chain stays inside [0,1] and --diag says so (%d levels)'
              % len(ranges))

        # (11) THE PATHS IN A MESSAGE. A `file` resolved against a material's directory came back as
        # `json\..\img\nope.png`, with two platforms' separators in the one path and a component that cancels out.
        sub = os.path.join(tmp, 'sub')
        os.makedirs(sub, exist_ok=True)
        missing = write_material(os.path.join(sub, 'missing.json'), [{'file': '../nope.png'}])
        bad = run([encode, missing, '-o', os.path.join('out', 'review_bad'), '--png', '0'])
        named = re.search(r"cannot read '([^']*)'", bad.stderr)
        if bad.returncode == 0 or not named:
            raise SystemExit('FAIL: a material naming a file that is not there must be refused by path, not %r'
                             % bad.stderr.strip())
        shown = named.group(1)
        if '..' in shown or ('/' in shown and '\\' in shown) or not shown.endswith('nope.png'):
            raise SystemExit('FAIL: the refusal names %r, which is not the resolved path in one separator' % shown)
        print('  the review: a missing file is named by its resolved path (%s)' % shown)

        # (13) --help-advanced SAYS WHAT --rgb-weights IS: one triple over every texture, and an override of a
        # material's own per-texture triples rather than a peer of them.
        helped = run([encode, '--help-advanced'])
        if helped.returncode != 0 or 'ONE triple, applied to EVERY texture' not in helped.stdout:
            raise SystemExit('FAIL: --help-advanced must say that --rgb-weights is one triple over every texture')
        if "material JSON's rgb_weights" not in helped.stdout:
            raise SystemExit('FAIL: --help-advanced must say that --rgb-weights overrides a material with a warning')
        print('  the review: --help-advanced describes --rgb-weights as one triple that overrides a material')

        # (12) THE VIEWER'S USAGE. It listed two of the fourteen flags the argument loop accepts, and --help itself
        # fell through to "ignoring unknown option" and then to a usage printed with an exit code of 1.
        if os.name == 'nt':
            view = executable(build_dir, 'nntc_view')
            usage = run([view, '--help'])
            if usage.returncode != 0:
                raise SystemExit('FAIL: nntc_view --help must exit 0, not %d' % usage.returncode)
            if 'ignoring unknown option' in usage.stderr:
                raise SystemExit('FAIL: nntc_view --help must be a flag of its own, not an unknown option')
            for flag in ('--shot', '--tex', '--cube', '--z', '--yaw', '--pitch', '--nomips', '--nobias', '--noaniso',
                         '--renorm', '--raw0', '--raw1', '--bc', '--nooverlay'):
                if flag not in usage.stdout:
                    raise SystemExit('FAIL: nntc_view --help does not name %s' % flag)
            print('  the review: nntc_view --help exits 0 and names all fourteen flags')
        else:
            print('  the review: the Direct3D viewer usage is a Windows case and is skipped')
        # And the Vulkan viewer's usage. WHAT IS PINNED, exactly: both usages exit 0, both name the same fourteen
        # flags in the same spellings, and the Vulkan one names those fourteen plus its own two, --size and --device.
        # That is a check on the spellings and on the list being complete - a flag the argument loop accepts but the
        # usage does not name can otherwise be found only by reading the source - and it is not a check that the two
        # texts read alike, which no comparison of two --help outputs could make.
        vk = vulkan_viewer(build_dir)
        vk_note = vk_skip_note('nntc_view_vk was not built')
        if vk:
            vk_usage = run([vk, '--help'])
            if vk_usage.returncode != 0:
                raise SystemExit('FAIL: nntc_view_vk --help must exit 0, not %d' % vk_usage.returncode)
            for flag in ('--shot', '--tex', '--cube', '--z', '--yaw', '--pitch', '--nomips', '--nobias',
                         '--noaniso', '--renorm', '--raw0', '--raw1', '--bc', '--nooverlay', '--size', '--device'):
                if flag not in vk_usage.stdout:
                    raise SystemExit('FAIL: nntc_view_vk --help does not name %s' % flag)
            vk_note = ''
            print('  the review: nntc_view_vk --help exits 0 and names the same fourteen flags and its own two')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record('the review fixes', 'the material is not overwritten by the _nntc descriptor and the refusal still fires '
                               'for -o material.json, a float-overflowing weight, a silent --weights, a '
                               'UTF-8 file name, --quiet, a byte-order mark and a parse position, a repeated key, the '
                               'seed direction at 18 outputs, the clamp under --diag, the resolved paths, and the '
                               'usages of ' + ('both viewers' if os.name == 'nt' and not vk_note else
                                               'the viewers that are here') + vk_note)


def write_grey16(path, width, height):
    """A 16-bit greyscale PNG, written here rather than carried in the tree so that what it holds is stated.

    Sample (x, y) is (x * 1024 + y * 37) & 0xFFFF, which sweeps the whole 16-bit range several times over 64 texels:
    its TOP BYTE - the byte stb_image keeps, and therefore the only thing the encoder ever sees of it - is a smooth
    ramp, while the low byte the encoder discards runs fast. An image whose two halves differ like that is what
    separates a reader that takes the top byte from one that takes the low one or clips.

    The writer is zlib and struct, so the gate does not depend on what a particular Pillow can write.
    """
    import binascii
    import zlib
    raw = bytearray()
    for y in range(height):
        raw.append(0)   # filter type 0, one per scanline
        for x in range(width):
            v = (x * 1024 + y * 37) & 0xFFFF
            raw += struct.pack('>H', v)

    def chunk(tag, payload):
        return (struct.pack('>I', len(payload)) + tag + payload
                + struct.pack('>I', binascii.crc32(tag + payload) & 0xFFFFFFFF))

    # bit depth 16, colour type 0 (greyscale), deflate, no filter, no interlace.
    header = struct.pack('>IIBBBBB', width, height, 16, 0, 0, 0, 0)
    png = (PNG_MAGIC + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(bytes(raw), 9))
           + chunk(b'IEND', b''))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'wb').write(png)


def diag_and_16bit_checks(encode):
    """--diag's per-plane table against the report it stands beside, and a 16-bit input read the way stb reads it.

    TWO GATES, ONE RUN, on a two-texture material whose second texture is a 16-bit greyscale PNG:

      the per-plane table  --diag prints every texture's psnr at every stored level. The report's own lines are a
                           projection of that table - `psnr texture t` is the base row and `mip psnr M<m>` is the
                           worst texture of row m - so the two must agree exactly, and a table computed from a
                           different reconstruction, a different crop or a different source mip would not.
      the 16-bit input     stb_image reduces a 16-bit sample to ITS TOP BYTE, so the encoder is fitted against the top
                           byte and its psnr is of that. tools/dds_decode.py has to read the same source the same way
                           or the two numbers are of two different images: Pillow's own .convert("RGB") of a 16-bit
                           greyscale clips every sample past 255 to white, which does not merely differ, it is
                           meaningless. This is the gate on load_rgb8.
    """
    odir, prefix = out_asset('diag16', source='tiny')
    grey = os.path.join(ROOT, 'out', 'diag16_src', 'grey16.png')
    write_grey16(grey, 64, 64)
    proc = run([encode, os.path.join('tests', 'tiny.png'), grey, '-o', odir, '--diag'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: the --diag / 16-bit encode returned %d' % proc.returncode)
    text = proc.stdout

    rows = re.findall(r'^      M(\d+)\s+\d+x\d+((?:\s+-?\d+\.\d+){2})\s+[0-9.e+-]+$', text, re.M)
    if len(rows) < 2:
        raise SystemExit('FAIL: --diag printed %d per-plane rows, expected the whole chain' % len(rows))
    table = [[float(v) for v in body.split()] for _, body in rows]

    base = [float(v) for v in re.findall(r'psnr\s+texture \d+\s+(\d+\.\d+) dB', text)]
    if len(base) != 2:
        raise SystemExit('FAIL: the report named %d textures, expected 2' % len(base))
    for t, (a, b) in enumerate(zip(table[0], base)):
        if abs(a - b) > 0.005:
            raise SystemExit('FAIL: --diag row M0 texture %d is %.2f dB but the report says %.2f' % (t, a, b))

    chain = re.search(r'^  mip psnr\s+((?:M\d+ \d+\.\d+\s*)+)dB', text, re.M)
    if not chain:
        raise SystemExit('FAIL: the report printed no mip psnr line to check the --diag table against')
    quoted = [float(v) for v in re.findall(r'M\d+ (\d+\.\d+)', chain.group(1))]
    if len(quoted) != len(table) - 1:
        raise SystemExit('FAIL: the mip psnr line quotes %d levels and --diag printed %d chain rows'
                         % (len(quoted), len(table) - 1))
    for i, q in enumerate(quoted):
        worst = min(table[i + 1])
        if abs(worst - q) > 0.005:
            raise SystemExit('FAIL: --diag row M%d is worst %.2f dB but the mip psnr line quotes %.2f'
                             % (i + 1, worst, q))
    print('  --diag: the per-plane table matches the report at every level (%d planes, 2 textures)' % len(table))

    proc = run([sys.executable, os.path.join('tools', 'dds_decode.py'), prefix, '--psnr',
                os.path.join('tests', 'tiny.png'), grey])
    if proc.returncode != 0:
        raise SystemExit('FAIL: dds_decode --psnr returned %d on the 16-bit material' % proc.returncode)
    read = [float(v) for v in re.findall(r'psnr (\d+\.\d+) dB', proc.stdout)]
    if len(read) != 2:
        raise SystemExit('FAIL: dds_decode --psnr reported %d textures, expected 2' % len(read))
    for t, (a, b) in enumerate(zip(read, base)):
        if abs(a - b) > 0.01:
            raise SystemExit('FAIL: dds_decode reads texture %d at %.2f dB and the encoder reports %.2f; a 16-bit '
                             'source must be reduced to its top byte, as stb_image does' % (t, a, b))
    print('  16-bit source: dds_decode --psnr agrees with the report on both textures (%.2f, %.2f dB)'
          % (read[0], read[1]))
    record('--diag and a 16-bit source', 'the per-plane table matches the report; dds_decode reduces 16 bits the way '
                                         'stb does')


ALPHA_WARNING = 'has an alpha channel; it is ignored (the tool encodes 24-bit RGB)'


def alpha_uses(text):
    """The lines of an encode's stdout that are the alpha warning. A list rather than a count, so a check that wants
    the file it names can look at it."""
    return [line for line in text.splitlines() if line.startswith('WARNING: ') and ALPHA_WARNING in line]


def alpha_input_checks(encode):
    """An input carrying an alpha channel: the alpha is IGNORED WITH A WARNING, not refused.

    A texture of the material is three channels, so there is nowhere for a fourth to go; but the RGB under it is what
    the caller meant to encode, and refusing the file only makes them convert it by hand first. What the run must not
    do is stay quiet, so there is exactly one WARNING per such file, on stdout, naming it.

    Four things are gated, on a copy of tiny.png given a NON-CONSTANT alpha plane - constant alpha would pass a loader
    that wrongly premultiplied, and this one would not:

      the warning        printed once for the RGBA file and not at all for the same picture without alpha.
      the identity       the asset from the RGBA file is BYTE-IDENTICAL to the asset from the RGB one. This is the
                         real assertion: the alpha is not merely tolerated, it has no effect whatever on what is
                         solved or stored, which a loader that premultiplied or that read the alpha as a channel
                         could not manage.
      grey+alpha         a two-channel LA PNG is the same case (the grey is repeated and the alpha dropped) and warns.
      the material path  a material JSON reaches the same loader, so a file it names warns exactly as a file on the
                         command line does - and the warning survives --quiet, like every other WARNING.

    And two files that carry an alpha plane and no transparency, which must go through in SILENCE, because nothing is
    lost when the plane is opaque everywhere:

      an opaque RGBA     the same picture with alpha 255 at every pixel.
      a GIF87a           stb_image's GIF reader reports four channels for every GIF whatever the file holds
                         (stbi__gif_header), as its PSD reader does for every PSD, so a loader that trusted the
                         reported channel count warned on every GIF anyone ever encoded. The alpha values are what
                         decides, not the count.
    """
    import tempfile
    from PIL import Image as PILImage

    tmp = tempfile.mkdtemp(prefix='nntc_alpha_')
    try:
        base = PILImage.open(os.path.join(ROOT, 'tests', 'tiny.png')).convert('RGB')
        w, h = base.size
        alpha = PILImage.new('L', (w, h))
        alpha.putdata([(x * 7 + y * 3) & 0xFF for y in range(h) for x in range(w)])
        rgb_png = os.path.join(tmp, 'rgb.png')
        rgba_png = os.path.join(tmp, 'rgba.png')
        la_png = os.path.join(tmp, 'greya.png')
        base.save(rgb_png)
        with_alpha = base.copy()
        with_alpha.putalpha(alpha)
        with_alpha.save(rgba_png)
        PILImage.merge('LA', [base.convert('L'), alpha]).save(la_png)

        odir_rgb, pref_rgb = out_asset('alpha_rgb', 'rgb')
        odir_rgba, pref_rgba = out_asset('alpha_rgba', 'rgba')
        plain = run([encode, rgb_png, '-o', odir_rgb, '--png', '0'])
        if plain.returncode != 0:
            raise SystemExit('FAIL: the encode of the alpha-free picture returned %d' % plain.returncode)
        if alpha_uses(plain.stdout):
            raise SystemExit('FAIL: a picture with no alpha channel must not warn about one')
        four = run([encode, rgba_png, '-o', odir_rgba, '--png', '0'])
        if four.returncode != 0:
            raise SystemExit('FAIL: an RGBA input must ENCODE, not be refused; it returned %d' % four.returncode)
        want = 'WARNING: %s %s' % (rgba_png, ALPHA_WARNING)
        if four.stdout.count(want) != 1 or len(alpha_uses(four.stdout)) != 1:
            raise SystemExit('FAIL: an RGBA input must warn exactly once, naming the file: %r, not %r'
                             % (want, alpha_uses(four.stdout)))

        names = ['_lat0.dds', '_lat1.dds']
        total = 0
        for name in names:
            a = open(os.path.join(ROOT, pref_rgb + name), 'rb').read()
            b = open(os.path.join(ROOT, pref_rgba + name), 'rb').read()
            if a != b:
                raise SystemExit('FAIL: the RGBA asset differs from the RGB one in %s (%d and %d bytes); the alpha '
                                 'must not reach the solve at all' % (name, len(a), len(b)))
            total += len(a)
        # The descriptors differ only where the asset's own base name is in them. That used to be checked by replacing
        # every 'rgba' in the whole text with 'rgb', which would have swallowed a real difference anywhere the two
        # letters happened to meet - a value, a key, a sentence of the sampling note. The names live in exactly two
        # places, the `file` of each source input and the `file` of each texture, so those are taken out of both
        # descriptors and the rest of the two objects must be equal with nothing substituted at all.
        x = json.load(open(os.path.join(ROOT, pref_rgb + '_nntc.json')))
        y = json.load(open(os.path.join(ROOT, pref_rgba + '_nntc.json')))

        def take_names(node, out):
            """Remove every 'file' / 'files' value from the object, in place, and collect them in order."""
            if isinstance(node, dict):
                for key in ('file', 'files'):
                    if key in node:
                        out.append(node.pop(key))
                for value in node.values():
                    take_names(value, out)
            elif isinstance(node, list):
                for value in node:
                    take_names(value, out)
            return out

        names_rgb = take_names(x, [])
        names_rgba = take_names(y, [])
        if x != y:
            raise SystemExit('FAIL: the two descriptors differ somewhere other than the file names they carry')
        if len(names_rgb) != len(names_rgba) or not names_rgb:
            raise SystemExit('FAIL: the two descriptors name %d and %d files' % (len(names_rgb), len(names_rgba)))
        for a_name, b_name in zip(names_rgb, names_rgba):
            if a_name != str(b_name).replace('rgba', 'rgb'):
                raise SystemExit('FAIL: the descriptors name %r and %r, which is not the same file under the two '
                                 'assets\' own base names' % (a_name, b_name))
        print('  the alpha channel: an RGBA input encodes with one warning, and its asset is byte-identical to the '
              'same picture without alpha (%d bytes over %d files)' % (total, len(names)))

        odir_la, _ = out_asset('alpha_la', 'greya')
        two = run([encode, la_png, '-o', odir_la, '--png', '0'])
        if two.returncode != 0:
            raise SystemExit('FAIL: a grey+alpha input must encode; it returned %d' % two.returncode)
        want = 'WARNING: %s %s' % (la_png, ALPHA_WARNING)
        if two.stdout.count(want) != 1 or len(alpha_uses(two.stdout)) != 1:
            raise SystemExit('FAIL: a grey+alpha input must warn exactly once, naming the file, not %r'
                            % alpha_uses(two.stdout))
        print('  grey+alpha: the two-channel picture encodes with the same one warning')

        mat = write_material(os.path.join(tmp, 'mat.json'), [{'file': 'rgba.png'}, {'file': 'rgb.png'}])
        odir_mat, _ = out_asset('alpha_material', 'mat')
        proc = run([encode, mat, '-o', odir_mat, '--png', '0', '--quiet'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: a material naming an RGBA texture returned %d' % proc.returncode)
        lines = alpha_uses(proc.stdout)
        if len(lines) != 1 or 'rgba.png' not in lines[0] or ALPHA_WARNING not in lines[0]:
            raise SystemExit('FAIL: the material path must warn once about its RGBA texture and not about the other, '
                             'even under --quiet, not %r' % lines)
        print('  the material path: the same loader, the same one warning, under --quiet')

        opaque_png = os.path.join(tmp, 'opaque.png')
        gif_file = os.path.join(tmp, 'plain.gif')
        opaque = base.copy()
        opaque.putalpha(PILImage.new('L', (w, h), 255))
        opaque.save(opaque_png)
        base.convert('P').save(gif_file)
        head = open(gif_file, 'rb').read(6)
        if head != b'GIF87a':
            raise SystemExit('FAIL: the silent-GIF check needs a GIF87a without transparency, not %r' % head)
        for path, what in ((opaque_png, 'an RGBA whose alpha is 255 everywhere'), (gif_file, 'a GIF87a')):
            odir_quiet, _ = out_asset('alpha_silent', os.path.splitext(os.path.basename(path))[0])
            quiet = run([encode, path, '-o', odir_quiet, '--png', '0'])
            if quiet.returncode != 0:
                raise SystemExit('FAIL: %s must encode; it returned %d' % (what, quiet.returncode))
            if alpha_uses(quiet.stdout):
                raise SystemExit('FAIL: %s carries no transparency and must not warn, not %r'
                                 % (what, alpha_uses(quiet.stdout)))
        print('  no transparency, no warning: an opaque RGBA and a GIF87a both encode in silence')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record('an alpha channel', 'ignored with one warning per file, on the command line and in a material; the asset '
                               'is byte-identical to the same picture without alpha')


def bare_command_check(encode):
    """`nntc_encode tests/tiny.png` with nothing else: the everyday command line. It must write the asset in the
    CURRENT DIRECTORY (the gate runs every command from the tree's root) under the input's own base name, with the
    layout the defaults promise, and then be cleaned up again so the root carries nothing it did not before."""
    # What was in the root BEFORE the run: only the files this command adds are removed afterwards, so a scratch file
    # someone left there survives a check run rather than being deleted by it.
    before = set(os.listdir(ROOT))
    made = []
    try:
        proc = run([encode, os.path.join('tests', 'tiny.png')])
        if proc.returncode != 0:
            raise SystemExit('FAIL: nntc_encode with no flags returned %d' % proc.returncode)
        expected = ['tiny_lat0.dds', 'tiny_lat1.dds', 'tiny_nntc.json']
        for name in expected:
            path = os.path.join(ROOT, name)
            if not os.path.isfile(path):
                raise SystemExit('FAIL: the bare command did not write %s in the current directory' % name)
        if os.path.exists(os.path.join(ROOT, 'tests', 'tiny_nntc.json')):
            raise SystemExit('FAIL: the bare command wrote beside the input instead of in the current directory')
        for name in os.listdir(ROOT):
            if name not in before:
                made.append(os.path.join(ROOT, name))
        meta = json.load(open(os.path.join(ROOT, 'tiny_nntc.json')))
        if meta['textures'][0]['dxgi_format_id'] != 83 or meta['decoder']['C0'] != 2 or meta['decoder']['C1'] != 4:
            raise SystemExit('FAIL: the defaults are not C0 2 / C1 4 with a BC5 level 0')
        # The default level-0 mode is --l0 bc8: a continuous 8-bit plane on a per-channel range, packed and refined.
        if meta['textures'][0]['bits_per_channel'] != [8, 8] or meta['textures'][0]['dequantise']['kind'] != 'range':
            raise SystemExit('FAIL: the default level 0 is not the continuous 8-bit plane of --l0 bc8')
        if 'analysis by synthesis' not in proc.stdout:
            raise SystemExit('FAIL: the default run must refine the pack')
        print('  the bare command: tiny_lat0.dds, tiny_lat1.dds and tiny_nntc.json in the current directory, BC5 level 0 '
              'at 8 bits, the pack refined')
    finally:
        for path in made:
            if os.path.isfile(path):
                os.remove(path)

def determinism_check(encode):
    """Two encodes of one image with one command line must produce byte-identical files.

    There is no seed in the encoder, so the only thing that could make a run differ from itself is the order its
    floating-point reductions landed in - and a decoder that differs in the last bit moves a level-0 argmin somewhere,
    which moves a texel, which changes the file. Every reduction whose result is used is therefore a fixed-grid
    two-stage reduction, and this is the gate that says so. It runs on the default command line, so it exercises block
    (a)'s normal equations, the objective, level 1's conjugate gradients and sweeps, and the pca-free box init.
    """
    names = ['_lat0.dds', '_lat1.dds', '_nntc.json']
    dirs = [os.path.join('out', 'det_a'), os.path.join('out', 'det_b')]
    prefixes = [os.path.join(d, 'tiny') for d in dirs]
    for d in dirs:
        proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', d, '--png', '0'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: nntc_encode returned %d during the determinism check' % proc.returncode)
    total = 0
    for name in names:
        a = open(os.path.join(ROOT, prefixes[0] + name), 'rb').read()
        b = open(os.path.join(ROOT, prefixes[1] + name), 'rb').read()
        if a != b:
            raise SystemExit('FAIL: two encodes of tests/tiny.png differ in %s (%d and %d bytes)'
                             % (name, len(a), len(b)))
        total += len(a)
    print('  determinism: two encodes of tests/tiny.png are byte-identical (%d bytes over %d files)'
          % (total, len(names)))
    record('determinism', 'two encodes byte-identical, %d bytes' % total)


def old_format_check(encode, build_dir):
    """Assets written before the project was renamed say "ntc-dds-1" in their descriptor. They are the same format,
    so both readers accept the older string with a note rather than refusing a file they can read. Both readers
    open the .dds files as well as the descriptor, so a fresh asset has its format string rewritten to the older
    name and is then read by each."""
    odir, prefix = out_asset('tiny_oldformat')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--png', '0', '--quiet'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: the encode for the old-format case returned %d' % proc.returncode)
    desc = prefix + '_nntc.json'
    text = open(desc, 'rb').read().decode()
    if text.count('"format": "nntc-dds-1"') != 1:
        raise SystemExit('FAIL: the fresh descriptor must carry the format string exactly once')
    open(desc, 'wb').write(text.replace('"format": "nntc-dds-1"', '"format": "ntc-dds-1"').encode())
    proc = run([sys.executable, os.path.join('tools', 'dds_decode.py'), prefix, '--grid'])
    if proc.returncode != 0 or 'older name of nntc-dds-1' not in proc.stdout:
        raise SystemExit('FAIL: dds_decode must read a rewritten older format string, not %r' % proc.stdout[-300:])
    if os.name == 'nt':
        # run() rather than shot(): the note is on STDOUT, with the rest of the load's commentary, and shot() hands
        # back the stderr a warning would arrive on.
        view = executable(build_dir, 'nntc_view')
        proc = run([view, desc, '--nooverlay', '--shot', os.path.join('out', 'tiny_oldformat', 'shot.bmp')])
        if proc.returncode != 0:
            raise SystemExit('FAIL: the viewer returned %d on the older format string' % proc.returncode)
        if 'older name of nntc-dds-1' not in proc.stdout:
            raise SystemExit('FAIL: the viewer must open the older format string with a note on stdout, not %r'
                             % proc.stdout[-300:])
    # The Vulkan viewer reads the same descriptor through the same header and owes the same note in the same words: a
    # file one viewer opens and the other refuses would make every frame comparison meaningless. It is outside the
    # Windows guard above because it needs no Direct3D frame and its --shot needs no desktop, so it runs on Linux too.
    vk = vulkan_viewer(build_dir)
    vk_note = vk_skip_note('nntc_view_vk was not built')
    vk_ran = False
    if vk:
        got = vk_shot(vk, desc, os.path.join('out', 'tiny_oldformat', 'shot_vk.bmp'), [])
        if got is None:
            vk_note = vk_skip_note('this machine reports no Vulkan device')
        elif 'older name of nntc-dds-1' not in got[2]:
            raise SystemExit('FAIL: the Vulkan viewer must open the older format string with a note on stdout, not %r'
                             % got[2][-300:])
        else:
            vk_note = ''
            vk_ran = True
    d3d_ran = os.name == 'nt'
    if d3d_ran and vk_ran:
        who = ' and both viewers'
    else:
        who = (' and the Direct3D viewer' if d3d_ran else '') + (' and the Vulkan viewer' if vk_ran else '')
    record('the older format string', 'ntc-dds-1 is read as nntc-dds-1 with a note by the Python reader' + who
           + vk_note)
    print('  the older format string: both readers accept ntc-dds-1 with a note')


def argument_and_status_checks(encode):
    """What the programs do with an argument they cannot honour: one line, no traceback, and a status of 1.

    Three things are gated here. --no-mkdir turns the directory -o creates by default into a refusal, and the refusal
    has to come before anything is written. dds_decode takes the asset's PREFIX, but the descriptor path is the name
    the user has in front of them - it is what the viewer and bc_check take - so that spelling is accepted too, and a
    prefix with no descriptor under it is one line rather than a traceback. And every failure the encoder can be made
    to reach exits 1, which is what a caller tests.
    """
    tiny = os.path.join('tests', 'tiny.png')

    # --no-mkdir: the refusal names the directory, and the directory is still not there afterwards.
    missing = os.path.join('out', 'no_mkdir_case', 'deeper')
    if os.path.isdir(os.path.join(ROOT, missing)):
        shutil.rmtree(os.path.join(ROOT, missing))
    proc = run([encode, tiny, '-o', missing + os.sep, '--no-mkdir', '--png', '0', '--quiet'])
    if proc.returncode != 1:
        raise SystemExit('FAIL: --no-mkdir over a missing directory returned %d, not 1' % proc.returncode)
    if '--no-mkdir' not in proc.stderr or 'does not exist' not in proc.stderr:
        raise SystemExit('FAIL: the --no-mkdir refusal must name the directory and the flag, not %r'
                         % proc.stderr[-300:])
    if os.path.isdir(os.path.join(ROOT, missing)):
        raise SystemExit('FAIL: --no-mkdir refused and created the directory anyway')
    proc = run([encode, tiny, '-o', missing + os.sep, '--png', '0', '--quiet'])
    if proc.returncode != 0 or not os.path.isdir(os.path.join(ROOT, missing)):
        raise SystemExit('FAIL: the same -o without --no-mkdir must create the directory and succeed')
    print('  --no-mkdir: a missing output directory is refused by name, and made without the flag')

    # THE BARE NAME. `-o out/name` with no extension is a DIRECTORY, and under --no-mkdir it is refused as one; the
    # refusal says which reading it took, because a caller who meant a prefix has no other way to tell.
    bare = os.path.join('out', 'no_mkdir_bare')
    if os.path.isdir(os.path.join(ROOT, bare)):
        shutil.rmtree(os.path.join(ROOT, bare))
    proc = run([encode, tiny, '-o', bare, '--no-mkdir', '--png', '0', '--quiet'])
    if proc.returncode != 1 or 'a bare name is a directory' not in proc.stderr:
        raise SystemExit('FAIL: the --no-mkdir refusal of a bare name must say it read the name as a directory, '
                         'not %r' % proc.stderr[-300:])
    print('  --no-mkdir: the refusal of a bare name says it was read as a directory')

    # dds_decode: the prefix and the descriptor path are the same asset, and a prefix with nothing under it is a line.
    prefix = os.path.join(missing, 'tiny')
    tool = os.path.join('tools', 'dds_decode.py')
    by_prefix = run([sys.executable, tool, prefix, '--grid'])
    by_descriptor = run([sys.executable, tool, prefix + '_nntc.json', '--grid'])
    if by_prefix.returncode != 0 or by_descriptor.returncode != 0:
        raise SystemExit('FAIL: dds_decode --grid must accept the prefix and the descriptor path alike')
    if by_prefix.stdout != by_descriptor.stdout:
        raise SystemExit('FAIL: dds_decode must read the same asset from the prefix and from the descriptor path')
    absent = run([sys.executable, tool, os.path.join('out', 'no_such_asset_here'), '--grid'])
    if absent.returncode != 1:
        raise SystemExit('FAIL: dds_decode over a missing prefix returned %d, not 1' % absent.returncode)
    if 'Traceback' in absent.stderr:
        raise SystemExit('FAIL: dds_decode over a missing prefix must not raise')
    if not any(line.startswith('ERROR:') and 'no_such_asset_here' in line for line in absent.stdout.splitlines()):
        raise SystemExit('FAIL: dds_decode must say in one line what it looked for, not %r' % absent.stdout[-300:])
    print('  dds_decode: the prefix and the descriptor path read the same asset; a missing one is one line')

    # A descriptor named outright is opened as given. The note is about a PREFIX.json this tool went looking for, so a
    # file the caller pointed straight at must not draw it.
    named_dir = os.path.join('out', 'named_descriptor')
    named_json = os.path.join(named_dir, 'named.json')
    if os.path.isdir(os.path.join(ROOT, named_dir)):
        shutil.rmtree(os.path.join(ROOT, named_dir))
    proc = run([encode, tiny, '-o', named_json, '--png', '0', '--quiet'])
    if proc.returncode != 0 or not os.path.isfile(os.path.join(ROOT, named_json)):
        raise SystemExit('FAIL: -o NAME.json must write that descriptor (%d)' % proc.returncode)
    direct = run([sys.executable, tool, named_json, '--grid'])
    if direct.returncode != 0:
        raise SystemExit('FAIL: dds_decode over a descriptor named outright returned %d' % direct.returncode)
    if 'note:' in direct.stdout and 'instead' in direct.stdout:
        raise SystemExit('FAIL: a descriptor named outright must be opened with no fallback note, not %r'
                         % direct.stdout[:300])
    # --psnr-levels needs the run's own _src_ chain, which --png 0 does not write: one line, no traceback.
    levels = run([sys.executable, tool, named_json, '--psnr-levels'])
    if levels.returncode != 1 or 'Traceback' in levels.stderr:
        raise SystemExit('FAIL: --psnr-levels without the source chain must be one line and a status of 1, not %d %r'
                         % (levels.returncode, levels.stderr[-300:]))
    said = [line for line in levels.stdout.splitlines() if line.startswith('ERROR:')]
    if len(said) != 1 or '_src_' not in said[0] or '--png 1' not in said[0]:
        raise SystemExit('FAIL: --psnr-levels must name the missing source PNG and --png 1, not %r' % said)
    print('  dds_decode: a descriptor named outright opens silently; --psnr-levels without the chain is one line')

    # NO -o AT ALL: the prefix is the input's own directory, which exists if and only if the input does, so a missing
    # input under a missing directory must be reported as the input it is - and must leave that directory uncreated,
    # with and without --no-mkdir.
    gone = os.path.join('out', 'no_such_dir_case')
    if os.path.isdir(os.path.join(ROOT, gone)):
        shutil.rmtree(os.path.join(ROOT, gone))
    for extra in ([], ['--no-mkdir']):
        proc = run([encode, os.path.join(gone, 'x.png'), '--png', '0', '--quiet'] + extra)
        if proc.returncode != 1:
            raise SystemExit('FAIL: a bare run on a missing input returned %d, not 1' % proc.returncode)
        if 'cannot read' not in proc.stderr:
            raise SystemExit('FAIL: a missing input must be reported as an input it cannot read, not %r'
                             % proc.stderr[-300:])
        if os.path.exists(os.path.join(ROOT, gone)):
            raise SystemExit('FAIL: a refused bare run created %s' % gone)
    print('  no -o: a missing input under a missing directory says "cannot read" and creates nothing')

    # Exit codes: an unreadable input, an unknown flag and a refused material all end at 1.
    bad_json = os.path.join(ROOT, 'out', 'no_mkdir_case', 'broken.json')
    open(bad_json, 'w').write('{ "textures": [ { "file": ')
    cases = [('an unreadable input', [encode, os.path.join('tests', 'no_such_image.png'), '--quiet']),
             ('an unknown flag', [encode, tiny, '--no-such-flag']),
             ('a malformed material', [encode, os.path.join('out', 'no_mkdir_case', 'broken.json'), '--quiet'])]
    for what, cmd in cases:
        proc = run(cmd)
        if proc.returncode != 1:
            raise SystemExit('FAIL: %s returned %d, not 1' % (what, proc.returncode))
    print('  exit codes: an unreadable input, an unknown flag and a malformed material all return 1')
    record('arguments and status', '--no-mkdir refuses a missing directory; dds_decode takes the prefix or the '
                                   'descriptor; every refusal returns 1')


def never_delete_check(encode):
    """The encoder never deletes a file (the owner's rule): a stale sibling under the prefix - a leftover deeper level's
    PNG, the other layout's .dds - stays where it is, and the descriptor is the authority on which files are the
    asset's. Both kinds are planted before a run and must survive it byte for byte."""
    odir, prefix = out_asset('tiny_never_delete')
    os.makedirs(odir, exist_ok=True)
    planted = {prefix + '_recon_M9.png': b'not a png, and not ours to remove',
               prefix + '_lat0a.dds': b'a stale two-file level 0 beside a one-file run',
               prefix + '_src_t3_M2.png': b'a stale material level'}
    for name, body in planted.items():
        open(name, 'wb').write(body)
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--quiet'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: the never-delete run returned %d' % proc.returncode)
    for name, body in planted.items():
        if not os.path.exists(name) or open(name, 'rb').read() != body:
            raise SystemExit('FAIL: the encoder removed or rewrote a stale sibling it did not write: %s' % name)
    record('never deleting', 'stale siblings under the prefix (a deeper level PNG, the .dds of the other layout, a material '
           'level) survive a run untouched')
    print('  never deleting: three planted stale siblings survived the run byte for byte')


def viewer_refusal_checks(encode, build_dir):
    """The viewer refuses what the encoder refuses: an unknown flag, a malformed number, a flag with no value, and a
    --shot it could not write all exit 1 with an ERROR line, and a refused --shot writes no file. --tex outside the
    material stays a warning with texture 0 shown, which the six-texture case asserts.

    The Direct3D half is Windows only; the Vulkan half is not, and runs wherever that viewer was built."""
    odir, prefix = out_asset('tiny_view_refuse')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--png', '0', '--quiet'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: the encode for the viewer refusals returned %d' % proc.returncode)
    desc = prefix + '_nntc.json'
    shot = os.path.join(odir, 'frame.bmp')
    d3d_done = os.name == 'nt'
    if not d3d_done:
        print('  the viewer refusals: the Direct3D viewer is Windows only; its arm is skipped')
    else:
        view = executable(build_dir, 'nntc_view')
        for label, extra, phrase in (('an unknown flag', ['--bogus'], 'unknown option'),
                                     ('a malformed --tex', ['--tex', 'nope'], '--tex needs'),
                                     ('a malformed --z', ['--z', '1.5x'], '--z needs'),
                                     ('a --shot with no name', [], '--shot needs')):
            cmd = [view, desc, '--nooverlay', '--shot', shot] + extra if label != 'a --shot with no name' else [view, desc, '--shot']
            bad = run(cmd)
            if bad.returncode != 1 or phrase not in bad.stderr:
                raise SystemExit('FAIL: the viewer must refuse %s with exit 1 and %r, not %d %r'
                                 % (label, phrase, bad.returncode, bad.stderr[-200:]))
        missing = os.path.join(odir, 'no_such_dir', 'frame.bmp')
        bad = run([view, desc, '--nooverlay', '--shot', missing])
        if bad.returncode != 1 or 'cannot write' not in bad.stderr or os.path.exists(missing):
            raise SystemExit('FAIL: a --shot into a missing directory must exit 1 and write nothing, not %d %r'
                             % (bad.returncode, bad.stderr[-200:]))
    # The Vulkan viewer's argument loop is the same loop with the same spellings and the same two parsers, so it owes
    # the same five refusals with the same phrases. It is a second arm of this case rather than a case of its own
    # because what is being asserted is that the two programs refuse ALIKE.
    vk = vulkan_viewer(build_dir)
    vk_done = False
    vk_note = vk_skip_note('nntc_view_vk was not built')
    if vk:
        vk_shot_path = os.path.join(odir, 'frame_vk.bmp')
        probe = run([vk, desc, '--nooverlay', '--shot', vk_shot_path, '--size', '64', '64'])
        if probe.returncode != 0 and 'no Vulkan device' in probe.stderr:
            vk_note = vk_skip_note('this machine reports no Vulkan device')
            print('  the Vulkan viewer refusals: skipped, no Vulkan device on this machine')
        elif probe.returncode != 0:
            raise SystemExit('FAIL: the Vulkan viewer could not draw the asset the refusals are checked on: %r'
                             % probe.stderr[-300:])
        else:
            for label, extra, phrase in (('an unknown flag', ['--bogus'], 'unknown option'),
                                         ('a malformed --tex', ['--tex', 'nope'], '--tex needs'),
                                         ('a malformed --z', ['--z', '1.5x'], '--z needs'),
                                         ('a --shot with no name', [], '--shot needs')):
                cmd = ([vk, desc, '--nooverlay', '--shot', vk_shot_path] + extra if label != 'a --shot with no name'
                       else [vk, desc, '--shot'])
                bad = run(cmd)
                if bad.returncode != 1 or phrase not in bad.stderr:
                    raise SystemExit('FAIL: the Vulkan viewer must refuse %s with exit 1 and %r, not %d %r'
                                     % (label, phrase, bad.returncode, bad.stderr[-200:]))
            vk_missing = os.path.join(odir, 'no_such_dir_vk', 'frame.bmp')
            bad = run([vk, desc, '--nooverlay', '--shot', vk_missing])
            if bad.returncode != 1 or 'cannot write' not in bad.stderr or os.path.exists(vk_missing):
                raise SystemExit('FAIL: a Vulkan --shot into a missing directory must exit 1 and write nothing, not '
                                 '%d %r' % (bad.returncode, bad.stderr[-200:]))
            # And the two flags this viewer has that the other one does not, which no other case reaches: a --size
            # with one value where two are needed, a --size below the 16 the loop accepts, a --device that is not a
            # number at all (atoi would have read it as device 0), and a --device past the end, which must say how
            # many devices the machine has rather than only that the index is wrong.
            for label, extra, phrase in (('a --size with one value', ['--size', '640'], '--size needs'),
                                         ('a --size below the minimum', ['--size', '8', '8'], '--size needs'),
                                         ('a --device that is not a number', ['--device', 'nope'], '--device needs'),
                                         ('a --device past the end', ['--device', '99'], 'this machine has')):
                bad = run([vk, desc, '--nooverlay', '--shot', vk_shot_path] + extra)
                if bad.returncode != 1 or phrase not in bad.stderr:
                    raise SystemExit('FAIL: the Vulkan viewer must refuse %s with exit 1 and %r, not %d %r'
                                     % (label, phrase, bad.returncode, bad.stderr[-200:]))
            vk_done = True
            vk_note = ''
            print('  the Vulkan viewer refusals: the same five, with the same phrases and the same exit code, and '
                  'its own four on --size and --device')
    record('the viewer refusals', 'an unknown flag, a malformed number, a valueless --shot and an unwritable --shot all '
           'exit 1; the refused shot writes no file'
           + (' -- and the same five on the Vulkan viewer, plus its own --size and --device refusals' if vk_done
              else vk_note)
           + ('' if d3d_done else ' (the Direct3D arm is Windows only)'))
    print('  the viewer refusals: exit 1 on an unknown flag, a malformed number, a valueless --shot, an unwritable shot')


def frame_size(path):
    """The width and height of a .bmp the viewers wrote, from its header.

    The depth is asserted here rather than assumed, because every caller of this goes on to treat the file as a 24-bit
    bottom-up frame: rows of three bytes with a four-byte pitch, which is what both viewers write and what
    tools/frame_diff.py and bmp_rows below both read. A viewer that started writing 32-bit frames would otherwise be
    caught only by a comparison failing somewhere else, saying the wrong thing about why.
    """
    data = open(os.path.join(ROOT, path), 'rb').read()
    if data[:2] != b'BM' or len(data) < 54:
        raise SystemExit('FAIL: %s is not a .bmp' % path)
    planes, depth = struct.unpack_from('<HH', data, 26)
    if planes != 1 or depth != 24:
        raise SystemExit('FAIL: %s has %d plane(s) at %d bits; the viewers write 24-bit single-plane frames'
                         % (path, planes, depth))
    return struct.unpack_from('<ii', data, 18)


def bmp_rows(path, first, last, columns=None):
    """Rows `first` up to `last` of a .bmp the viewers wrote, TOP-FIRST, as one block of bytes.

    `columns`, when it is given, keeps only that many pixels from the left of each row. The overlay comparison between
    the two viewers needs it since stage 4: the Vulkan strip carries a decode-path token at a fixed column near its
    right-hand end that the Direct3D viewer has no state for and cannot grow, so the two strips are asserted identical
    over the columns they SHARE rather than over the whole width.

    The file itself is bottom-up, so row y from the top is the row at (height - 1 - y) in the file, and every row is
    padded to a multiple of four bytes which is not part of the picture. The overlay case compares the strip's own
    rows with the rows under it, and doing that on raw file bytes would compare a strip at the top of one frame with
    whatever is at the bottom of the other.
    """
    data = open(os.path.join(ROOT, path), 'rb').read()
    if data[:2] != b'BM' or len(data) < 54:
        raise SystemExit('FAIL: %s is not a .bmp' % path)
    offset = struct.unpack_from('<I', data, 10)[0]
    width, height = struct.unpack_from('<ii', data, 18)
    if first < 0 or last > height or first > last:
        raise SystemExit('FAIL: rows %d..%d asked of %s, which is %d rows tall' % (first, last, path, height))
    pitch = (width * 3 + 3) & ~3
    keep = width * 3 if columns is None else min(width, columns) * 3
    return b''.join(data[offset + (height - 1 - y) * pitch:offset + (height - 1 - y) * pitch + keep]
                    for y in range(first, last))


def frame_diff(a, b, max_diff, min_psnr, what):
    """tools/frame_diff.py on two frames, refusing when either threshold is missed.

    The comparison is the script's and not this file's on purpose: it is the same tool a reader runs by hand when a
    frame looks wrong, so the number in a gate failure and the number on the command line are produced by one piece of
    code. It returns the line the script printed, so a passing run can say what the difference actually was rather
    than only that it was allowed.
    """
    proc = run([sys.executable, os.path.join('tools', 'frame_diff.py'), a, b,
                '--max-diff', str(max_diff), '--min-psnr', str(min_psnr)])
    if proc.returncode != 0:
        raise SystemExit('FAIL: %s: the two viewers\' frames differ more than allowed: %s'
                         % (what, proc.stderr.strip()[-300:] or proc.stdout.strip()[-300:]))
    psnr = re.search(r'PSNR ([0-9.]+|inf)', proc.stdout)
    worst = re.findall(r'max \|diff\| (\d+)', proc.stdout)
    return '%s: max %s, PSNR %s dB' % (what, max(int(w) for w in worst) if worst else '?',
                                       psnr.group(1) if psnr else '?')


def vulkan_viewer_checks(encode, build_dir):
    """The Vulkan viewer against the Direct3D one, on the same assets, frame by frame.

    Stage 2 of docs/VULKAN_VIEWER_PLAN.md draws the asset, so the question this case asks is no longer "did it clear"
    but "is it the same picture". Both viewers render the same asset headless with no overlay and tools/frame_diff.py
    reports the largest absolute difference per channel and the PSNR between the two frames.

    THE TWO FRAMES ARE NOT EXPECTED TO BE IDENTICAL and that must not be written into a gate: the rasteriser's fill
    rule, the bilinear weight precision (both APIs specify a minimum number of fractional bits, not an exact value),
    the block-compression palette evaluation and anisotropic tap placement are all free to differ. What is asserted is
    that the difference is small enough to be one of those rather than a bug -- a wrong UV orientation, a missed y
    flip, a dropped LOD bias or a feature vector in the wrong order would every one of them be far larger.

    The two thresholds are deliberately different and the reason is not a tolerance being widened until it passes. The
    default device is the one the Direct3D viewer also draws on, so the comparison there is between two APIs on ONE
    sampler and is tight. --device 1 is a different vendor's sampler, so the comparison there is between two filters
    that round their weights differently, and it is looser by exactly that much.

    Two things about the sizes. The Vulkan --shot creates no window, so it renders at whatever size it is given;
    the Direct3D one renders into its WINDOW, which Windows clamps to the desktop's work area. So the Direct3D frame is
    taken first and the Vulkan one is asked for that size, because two frames of different shapes are two different
    projections. And the filter toggles are exercised at --z -50 rather than at the default camera: a 64x64 asset on a
    2560-wide frame is MAGNIFIED at the default distance, where mips, the level-1 bias and anisotropy all have nothing
    to do, so a toggle that changed the frame there would mean something was wrong.

    WHICH CAMERA A PAIR IS TAKEN AT decides what the pair can catch. At the default camera the quad is magnified and
    the two frames come out byte-identical here, so those pairs test the y flip and the UV orientation and nothing
    else - their thresholds are never approached, let alone exercised. The pairs at --z -50 --noaniso are where mips
    and the level-1 bias are live and where a difference would actually be a finding, so the states that matter are
    compared there as well.

    The whole case is SKIPPED, never failed, when the executable was not built (a tree with no Vulkan SDK) or when the
    machine has no Vulkan device (a headless build agent, a remote session with no driver). Gate coverage of the
    Direct3D viewer cannot regress either way, because none of it is touched here. Off Windows there is no Direct3D
    viewer to compare against, so every pair is skipped with that reason and the arms that need no second viewer -
    the toggles, the flags, --bc, the overlay, the refusals, --help and the five identical launches - run as they do
    here.
    """
    started = time.perf_counter()
    view = vulkan_viewer(build_dir)
    if not view:
        record('the Vulkan viewer', 'skipped (nntc_view_vk was not built; this tree has no Vulkan SDK)')
        print('  the Vulkan viewer: skipped, nntc_view_vk is not under %s' % build_dir)
        return
    cross = os.name == 'nt'   # the Direct3D viewer, and so every cross-viewer pair below, is Windows only
    d3d = executable(build_dir, 'nntc_view') if cross else None
    if not cross:
        print('  the Vulkan viewer: the cross-viewer pairs are skipped, the Direct3D viewer is Windows only')

    odir, prefix = out_asset('tiny_vk')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--png', '0', '--quiet'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: the encode for the Vulkan viewer case returned %d' % proc.returncode)
    desc = prefix + '_nntc.json'

    # The first launch is also the probe: a machine with no driver says so and the case is skipped, not failed.
    single_vk = os.path.join('out', 'vk_tiny_t0.bmp')
    probe = run([view, desc, '--nooverlay', '--shot', single_vk] + VK_PLAIN)
    if probe.returncode != 0 and 'no Vulkan device' in probe.stderr:
        record('the Vulkan viewer', 'skipped (this machine reports no Vulkan device)')
        print('  the Vulkan viewer: skipped, no Vulkan device on this machine')
        return
    if probe.returncode != 0:
        raise SystemExit('FAIL: nntc_view_vk --shot returned %d: %r' % (probe.returncode, probe.stderr[-300:]))
    # That probe asked for no --size, so it also pins the default: 2560x1440, the Direct3D viewer's window size, and
    # 24 bits, which frame_size asserts. Every comparison in this file is of two frames of one shape, and the default
    # is the shape a reader gets when they run the viewer by hand.
    if frame_size(single_vk) != (2560, 1440):
        raise SystemExit('FAIL: nntc_view_vk --shot with no --size must be 2560x1440, not %dx%d'
                         % frame_size(single_vk))
    print('  the Vulkan viewer: --shot with no --size is a 2560x1440 24-bit frame')
    # The two portability lines, which are start-up and not a flag: the depth attachment's format is chosen from what
    # the device reports rather than assumed to be D32_SFLOAT, and the 1.3 features the viewer draws every frame with
    # are queried by name rather than inferred from the version number. Both are asserted because a silent regression
    # in either is a program that runs here and refuses to draw on someone else's machine.
    depth = re.search(r'^depth format: (D32_SFLOAT|X8_D24_UNORM_PACK32|D16_UNORM)$', probe.stdout, re.MULTILINE)
    if not depth:
        raise SystemExit('FAIL: nntc_view_vk printed no chosen depth format at start-up: %r' % probe.stdout[:400])
    if not re.search(r'^features: dynamicRendering and synchronization2 both reported$', probe.stdout, re.MULTILINE):
        raise SystemExit('FAIL: nntc_view_vk printed no Vulkan 1.3 feature check at start-up: %r' % probe.stdout[:400])
    print('  the Vulkan viewer: the depth format is chosen from the device (%s) and the 1.3 features are queried by '
          'name' % depth.group(1))
    # The devices this viewer can actually draw on, which is not every line it printed: an unusable device's line
    # carries a disqualifier after two spaces and a dash ("no graphics queue", "a graphics queue, but no family that
    # can present to this surface", "below Vulkan 1.3", "no VK_KHR_swapchain"),
    # and asking for one with --device is a refusal rather than a second measurement. The name is kept so that the
    # summary can say which part the second comparison was made on.
    devices = {}
    chosen = re.search(r'^drawing on device (\d+):', probe.stdout, re.MULTILINE)
    chosen_device = chosen.group(1) if chosen else '-1'
    for m in re.finditer(r'^device (\d+): (.*)$', probe.stdout, re.MULTILINE):
        if ' - ' not in m.group(2):
            devices[m.group(1)] = m.group(2).strip()

    # THE REFUSAL THIS MACHINE CANNOT REACH, checked by looking for the hardware rather than by assuming it is absent.
    # The encoder's default level 0 is a BC4 / BC5 pair, and a device that does not report textureCompressionBC cannot
    # sample it at all; the viewer refuses such an asset from the DESCRIPTOR, before any .dds is opened, naming the
    # format and the encode that would produce an uncompressed level 0 instead. Every device on this machine reports
    # the feature, so there is nothing here to refuse - and the honest form of that is to look for a device that lacks
    # it, assert the refusal on the first one found, and SKIP WITH THE REASON when there is none, rather than to write
    # a check that silently tests nothing. The viewer says so in one 'note:' line at start-up, which is what is read.
    bc_probe = os.path.join('out', 'vk_bc_probe.bmp')
    no_bc = None
    for index in sorted(devices, key=int):
        seen = run([view, desc, '--nooverlay', '--shot', bc_probe, '--size', '64', '64', '--device', index] + VK_PLAIN)
        if 'does not report textureCompressionBC' in seen.stdout + seen.stderr:
            no_bc = index
            break
    if no_bc is None:
        bc_note = ('the BC-asset refusal skipped, every device on this machine reports textureCompressionBC, so it is '
                   'unreachable here')
    else:
        bad = run([view, desc, '--nooverlay', '--shot', bc_probe, '--size', '64', '64', '--device', no_bc] + VK_PLAIN)
        named = 'BC4_UNORM_BLOCK' in bad.stderr or 'BC5_UNORM_BLOCK' in bad.stderr
        if bad.returncode != 1 or not named or '--bc0 0' not in bad.stderr:
            raise SystemExit('FAIL: a BC asset on a device with no textureCompressionBC must exit 1 naming the format '
                             'and the uncompressed encode, not %d %r' % (bad.returncode, bad.stderr[-300:]))
        if '.dds' in bad.stdout:
            raise SystemExit('FAIL: the BC refusal must come from the descriptor, before any .dds is opened: %r'
                             % bad.stdout[-300:])
        bc_note = ('the BC-asset refusal names the format and the uncompressed encode, before any .dds is opened '
                   '(device %s)' % no_bc)
    print('  the Vulkan viewer: %s' % bc_note)

    def pair(asset, tag, extra, max_diff=8, min_psnr=40.0, device=None):
        """One asset drawn by both viewers at one camera, and the two frames compared."""
        if not cross:
            raise SystemExit('FAIL: a cross-viewer pair was asked for where there is no Direct3D viewer')
        a = os.path.join('out', 'vk_%s_d3d.bmp' % tag)
        b = os.path.join('out', 'vk_%s_vk.bmp' % tag)
        shot(d3d, asset, a, extra)
        w, h = frame_size(a)
        cmd = [view, asset, '--nooverlay', '--shot', b, '--size', str(w), str(h)] + VK_PLAIN + extra
        if device is not None:
            cmd += ['--device', str(device)]
        made = run(cmd)
        if made.returncode != 0:
            raise SystemExit('FAIL: nntc_view_vk --shot returned %d on %s: %r' % (made.returncode, asset,
                                                                                  made.stderr[-300:]))
        return frame_diff(a, b, max_diff, min_psnr, tag)

    # The default-camera pairs. They are the y flip and the UV orientation and not much more: the quad is magnified
    # here, so every sampler state that could differ has nothing to do and the two frames come out byte-identical on
    # this machine. The pairs below at --z -50 are the ones whose thresholds mean anything.
    lines = [pair(desc, 'tiny_t0', ['--tex', '0'])] if cross else []

    # A material, so that a second output triple is decoded and shown: --tex 1 reads outv[3..5] of the same shader.
    import tempfile
    tmp = tempfile.mkdtemp(prefix='nntc_vk_')
    try:
        crops = crop_inputs(tmp, 2, 32)
        modir, mprefix = out_asset('tiny_vk_mat', 'c0')
        proc = run([encode] + crops + ['-o', modir, '--png', '0', '--quiet'])
        if proc.returncode != 0:
            raise SystemExit('FAIL: the two-texture encode for the Vulkan viewer case returned %d' % proc.returncode)
        mdesc = mprefix + '_nntc.json'
        if cross:
            lines.append(pair(mdesc, 'mat_t0', ['--tex', '0']))
            lines.append(pair(mdesc, 'mat_t1', ['--tex', '1']))
        # Five identical launches on this stage's other asset too, so the claim covers a material and not only a
        # single image: a frame that is not a function of the command line and the asset alone cannot be the basis of
        # any of the comparisons above.
        vk_shot_stable(view, mdesc, os.path.join('out', 'vk_mat_stable.bmp'), ['--tex', '1'])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # A two-file level 0: a BC5 of channels 0-1 and a BC4 of channel 2 alone, which is the path where the shader reads
    # a third texture and sel.z says so. A viewer that ignored the second file would draw a recognisable, wrong picture,
    # so the comparison against the Direct3D frame is what asserts it, not the return code.
    codir, cprefix = out_asset('tiny_vk_c0_3')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', codir, '--l0', 'bc8', '--c0', '3', '--png', '0',
                '--quiet'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: the --c0 3 encode for the Vulkan viewer case returned %d' % proc.returncode)
    if cross:
        lines.append(pair(cprefix + '_nntc.json', 'c0_3', ['--tex', '0']))

        # THE MINIFIED CAMERA, where the sampler states are live. --noaniso because anisotropic tap placement is one
        # of the things the plan's section 1.14 says is free to differ between two APIs; everything else about the
        # fetch is not, so the threshold here is 2 out of 255 rather than the 8 the default-camera pairs carry. The
        # second pair drops level 1's LOD bias as well, which is the state the format's section 5 is about: if the
        # bias reached one viewer's sampler and not the other's, THIS is the comparison that says so.
        lines.append(pair(desc, 'far_noaniso', ['--tex', '0', '--z', '-50', '--noaniso'], max_diff=2))
        lines.append(pair(desc, 'far_nobias', ['--tex', '0', '--z', '-50', '--noaniso', '--nobias'], max_diff=2))
        # And the two Vulkan frames must differ FROM EACH OTHER. Two pairs that both passed would prove nothing if
        # --nobias had quietly done nothing at this camera: the pair would then be comparing one state twice.
        biased = open(os.path.join(ROOT, 'out', 'vk_far_noaniso_vk.bmp'), 'rb').read()
        unbiased = open(os.path.join(ROOT, 'out', 'vk_far_nobias_vk.bmp'), 'rb').read()
        if biased == unbiased:
            raise SystemExit('FAIL: at --z -50 --noaniso the Vulkan frames with and without level 1\'s LOD bias are '
                             'the same bytes, so the two pairs above compare one state twice')
        print('  the Vulkan viewer: at --z -50 --noaniso the frames with and without --nobias differ from each other')

    # The three filter toggles, at a distance where each one has something to do. Each must CHANGE the frame: a flag
    # that is parsed and then reaches no sampler is worse than one that is refused.
    far = ['--z', '-50']
    base = os.path.join('out', 'vk_far_base.bmp')
    if run([view, desc, '--nooverlay', '--shot', base] + VK_PLAIN + far).returncode != 0:
        raise SystemExit('FAIL: nntc_view_vk --shot at --z -50 failed')
    base_bytes = open(os.path.join(ROOT, base), 'rb').read()
    for flag in ('--nomips', '--nobias', '--noaniso'):
        changed = os.path.join('out', 'vk_far%s.bmp' % flag.replace('-', '_'))
        if run([view, desc, '--nooverlay', '--shot', changed] + VK_PLAIN + far + [flag]).returncode != 0:
            raise SystemExit('FAIL: nntc_view_vk --shot with %s failed' % flag)
        if open(os.path.join(ROOT, changed), 'rb').read() == base_bytes:
            raise SystemExit('FAIL: %s did not change the Vulkan viewer\'s frame, so it reaches no sampler' % flag)
    print('  the Vulkan viewer: --nomips, --nobias and --noaniso each change the frame at --z -50')

    # Stage 3's flags, each of which must reach something. --raw0 and --raw1 replace the decode with a latent as
    # stored and --renorm renormalises the shown triple, so all three change the frame of any asset.
    base3 = vk_shot(view, desc, os.path.join('out', 'vk_flags_base.bmp'), [])[0]
    for flag in ('--raw0', '--raw1', '--renorm'):
        changed = vk_shot(view, desc, os.path.join('out', 'vk_flags%s.bmp' % flag.replace('-', '_')), [flag])[0]
        if changed == base3:
            raise SystemExit('FAIL: %s did not change the Vulkan viewer\'s frame, so it reaches no shader constant'
                             % flag)
    print('  the Vulkan viewer: --raw0, --raw1 and --renorm each change the frame')

    # "It changes the frame" says only that the flag reached something. What it reached has to be the SAME something
    # the other viewer's flag reaches, and only a comparison against that viewer says so: --raw0 showing level 1, a
    # --renorm that renormalised the wrong triple or a --cube whose faces were wound the other way would every one of
    # them change the frame and be wrong. Each is compared with anisotropy off, for the reason the far pairs give.
    if cross:
        for flag in ('--raw0', '--raw1', '--renorm', '--cube'):
            lines.append(pair(desc, 'flag' + flag.replace('-', '_'), ['--noaniso', flag], max_diff=2))

    # Key 4 and its flag, both ways round. On the encoder's default level 0 the file already holds optimised BC
    # blocks, there is nothing for the viewer to pack and --bc must do NOTHING; on an uncompressed level 0 the pack is
    # made at load and --bc binds it, which is a different picture. Both halves matter: a --bc that silently did
    # nothing everywhere would pass the first check alone.
    if vk_shot(view, desc, os.path.join('out', 'vk_bc_file.bmp'), ['--bc'])[0] != base3:
        raise SystemExit('FAIL: --bc changed the frame of an asset whose level 0 is already block-compressed, where '
                         'the Vulkan viewer has no pack of its own to bind')
    updir, uprefix = out_asset('tiny_vk_unc')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', updir, '--l0', 'palette', '--bc0', '0', '--png', '0',
                '--quiet'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: the uncompressed-level-0 encode for the Vulkan viewer case returned %d'
                         % proc.returncode)
    udesc = uprefix + '_nntc.json'
    plain = vk_shot(view, udesc, os.path.join('out', 'vk_unc_plain.bmp'), [])
    packed = vk_shot(view, udesc, os.path.join('out', 'vk_unc_bc.bmp'), ['--bc'])
    if 'packed at load' not in packed[2]:
        raise SystemExit('FAIL: the Vulkan viewer must pack an uncompressed level 0 at load and say so: %r'
                         % packed[2][-300:])
    if plain[0] == packed[0]:
        raise SystemExit('FAIL: --bc did not change the frame of an uncompressed level 0, so the load-time pack is '
                         'not bound')
    print('  the Vulkan viewer: --bc binds the load-time pack of an uncompressed level 0 and does nothing on a BC one')

    # And the pack itself, across the two viewers: both make it at load from the same shared/bc_pack.h over the same
    # uncompressed plane, so the BLOCKS are the same bytes and the only thing left to differ is the sampler. With
    # anisotropy off the two frames are identical, so that is what is asserted - max 0, not a tolerance - and a
    # difference of any size would mean one of them packed something else.
    if cross:
        lines.append(pair(udesc, 'bc_pack', ['--noaniso', '--bc'], max_diff=0))

    # The overlay. --nooverlay is what every comparison above relies on, so the gate has to know the strip was there
    # to leave off: the same frame with and without it must differ.
    with_strip = os.path.join('out', 'vk_overlay_on.bmp')
    without = os.path.join('out', 'vk_overlay_off.bmp')
    made = run([view, desc, '--shot', with_strip, '--size', '640', '480'] + VK_PLAIN)
    if made.returncode != 0:
        raise SystemExit('FAIL: nntc_view_vk --shot with the overlay returned %d: %r' % (made.returncode,
                                                                                         made.stderr[-300:]))
    if vk_shot(view, desc, without, [])[0] == open(os.path.join(ROOT, with_strip), 'rb').read():
        raise SystemExit('FAIL: --nooverlay drew the same frame as a shot with the overlay, so the debug strip is '
                         'not being drawn at all')
    print('  the Vulkan viewer: the overlay is drawn, and --nooverlay leaves it off')

    # THE STRIP ITSELF, across the two viewers, which is the check the comment here used to promise and not make. The
    # overlay is the same font, the same scale, the same layout and the same words rasterised by the same code, so its
    # 84 rows are asserted BYTE-IDENTICAL between the viewers - at the Direct3D frame's own size, since the strip is a
    # fixed number of pixels and a frame of another shape puts different pixels under it. Below those rows each frame
    # must equal that viewer's own --nooverlay frame, which is what says the strip is drawn OVER the scene and changes
    # nothing under it - and it is why every comparison in this case may use --nooverlay in the first place.
    if cross:
        ov_d3d = os.path.join('out', 'vk_overlay_d3d.bmp')
        ov_vk = os.path.join('out', 'vk_overlay_vk.bmp')
        no_d3d = os.path.join('out', 'vk_overlay_d3d_off.bmp')
        no_vk = os.path.join('out', 'vk_overlay_vk_off.bmp')
        made = run([d3d, desc, '--shot', ov_d3d, '--noaniso'])
        if made.returncode != 0:
            raise SystemExit('FAIL: nntc_view --shot with the overlay returned %d' % made.returncode)
        w, h = frame_size(ov_d3d)
        for cmd in ([d3d, desc, '--nooverlay', '--noaniso', '--shot', no_d3d],
                    [view, desc, '--noaniso', '--shot', ov_vk, '--size', str(w), str(h)] + VK_PLAIN,
                    [view, desc, '--nooverlay', '--noaniso', '--shot', no_vk, '--size', str(w), str(h)] + VK_PLAIN):
            made = run(cmd)
            if made.returncode != 0:
                raise SystemExit('FAIL: an overlay-case --shot returned %d: %r' % (made.returncode,
                                                                                   made.stderr[-300:]))
        if frame_size(ov_vk) != frame_size(ov_d3d):
            raise SystemExit('FAIL: the two overlay frames are not the same shape, so their rows cannot be compared')
        # Over the columns the two strips SHARE. The Vulkan one carries its decode-path token at VK_COOP_COLUMN,
        # which is past the end of anything the four shared lines print and which the Direct3D viewer has no state
        # for; everything to the left of it is the same font, the same scale, the same layout and the same words
        # rasterised by the same code, so it is asserted byte-identical exactly as it always was.
        if bmp_rows(ov_d3d, 0, 84, VK_COOP_COLUMN) != bmp_rows(ov_vk, 0, 84, VK_COOP_COLUMN):
            raise SystemExit('FAIL: the two viewers\' 84 overlay rows are not byte-identical over the columns they '
                             'share, so the debug strip is not the same strip')
        for tag, on, off in (('Direct3D', ov_d3d, no_d3d), ('Vulkan', ov_vk, no_vk)):
            if bmp_rows(on, 84, h) != bmp_rows(off, 84, h):
                raise SystemExit('FAIL: below its 84 overlay rows the %s frame differs from its own --nooverlay '
                                 'frame, so the strip is changing the scene under it' % tag)
        print('  the Vulkan viewer: the overlay\'s 84 rows are byte-identical to the Direct3D viewer\'s, and below '
              'them each frame equals its own --nooverlay frame')


    # STAGE 4: THE COOPERATIVE-VECTOR DECODE PATH, and the whole of what a gate can assert about it.
    #
    # It is optional at run time by construction - one vendor, no portable successor - so every line below is SKIPPED
    # with the device's own reason rather than failed when the query did not pass, exactly as the whole case is skipped
    # on a machine with no Vulkan at all. The arms are four:
    #
    #   the PICTURE, which is the point: the same asset drawn by the two pipelines must be the same picture. The
    #   threshold is 4 out of 255 and 50 dB, and it is not a tolerance chosen to pass - it is what 11 bits of mantissa
    #   in the weights can do to an 8-bit output. An fp16 conversion error, a layout mismatch, a wrong M or K or a
    #   padding bug would all be far larger, and would look like "a bit off" if the threshold were widened instead;
    #
    #   the REFUSAL: --coopvec 1 on a device that has no such extension exits 1 with an ERROR line. A run that quietly
    #   fell back would be a measurement of the other path under the name of this one;
    #
    #   the MEASUREMENT: --bench runs on both paths and prints a number for each. The numbers themselves are not
    #   asserted - a gate that pinned a frame time would fail on a busy machine and say nothing about correctness -
    #   but that both paths can be timed at all is;
    #
    #   the OVERLAY: the state line's decode-path token, which is the only thing about the two frames' debug strips
    #   that differs. It is asserted at 2560x1440 because the strip is 2560 pixels wide at 1:1 and a narrower frame
    #   clips its right-hand end, where the token is.
    cv_line = next((l for l in probe.stdout.splitlines() if l.startswith('cooperative vectors: ')), '')
    cv_here = cv_line.startswith('cooperative vectors: available')
    cv_asset = os.path.join('examples', 'm1_m4_c0_3_c1_4_nntc.json')
    cv_reason = cv_line.split(' - ', 1)[1] if ' - ' in cv_line else 'the device query did not pass'
    # The refusal a build compiled against headers older than 1.4.307 prints. It is not a failure and it is not a
    # property of the machine: the arms below are skipped exactly as they are on a device without the extension, and
    # the summary says which of the two it was. (On Linux, where this build is the usual one, the gate as a whole
    # needs the encoder and does not run - but this arm's Linux behaviour is stated here rather than left to be
    # discovered.)
    cv_compiled_out = 'predate VK_NV_cooperative_vector' in cv_reason
    # WHICH device the stage-4 arms run on. The default device is the first discrete GPU that can present, and on a
    # machine where that part has no cooperative vectors and a second one does - a laptop whose discrete GPU is not
    # the NVIDIA part, say - skipping the whole stage would be reporting the machine's device order rather than the
    # viewer. So where the default cannot, every other usable device is PROBED with the flag itself, and the first
    # that draws a frame carries the arms; only when none can is the stage skipped, with the default's own reason.
    cv_dev = []
    cv_device = chosen_device
    # A build with the extension compiled out cannot take the path on ANY device, so probing the other devices with
    # --coopvec 1 would be asking every one of them the same question the build has already answered.
    if not cv_here and not cv_compiled_out and os.path.isfile(os.path.join(ROOT, cv_asset)):
        for index in devices:
            if index == str(chosen_device):
                continue
            probe_cv = run([view, cv_asset, '--nooverlay', '--shot', os.path.join('out', 'vk_coop_probe.bmp'),
                            '--size', '64', '64', '--device', index, '--coopvec', '1'])
            if probe_cv.returncode == 0:
                cv_here, cv_dev, cv_device = True, ['--device', index], index
                print('  the Vulkan viewer: device %s (%s) carries the cooperative-vector arms, because the default '
                      'device cannot (%s)' % (index, devices[index], cv_reason))
                break
    cv_note = ''
    if not cv_here and cv_compiled_out:
        cv_note = '; the cooperative-vector arms skipped (this build has them compiled out: %s)' % cv_reason
        print('  the Vulkan viewer: the cooperative-vector arms are skipped, this build has them compiled out - %s'
              % cv_reason)
    elif not cv_here:
        cv_note = '; the cooperative-vector arms skipped (%s)' % cv_reason
        print('  the Vulkan viewer: the cooperative-vector arms are skipped, %s' % cv_reason)
    elif not os.path.isfile(os.path.join(ROOT, cv_asset)):
        cv_note = '; the cooperative-vector arms skipped (%s is not in this tree)' % cv_asset
        print('  the Vulkan viewer: the cooperative-vector arms are skipped, %s is not in this tree' % cv_asset)
    else:
        cv_frames = {}
        for flag in ('1', '0'):
            out = os.path.join('out', 'vk_coop_%s.bmp' % flag)
            made = run([view, cv_asset, '--nooverlay', '--noaniso', '--shot', out, '--size', '640', '480',
                        '--coopvec', flag] + cv_dev)
            if made.returncode != 0:
                raise SystemExit('FAIL: nntc_view_vk --coopvec %s --shot returned %d: %r'
                                 % (flag, made.returncode, made.stderr[-300:]))
            cv_frames[flag] = (out, made.stdout)
        # The two runs must have taken DIFFERENT paths, and the viewer says which at start-up. Without this the
        # comparison below could be one path against itself and would pass for the wrong reason.
        if 'decode: cooperative vectors' not in cv_frames['1'][1] or 'decode: plain' not in cv_frames['0'][1]:
            raise SystemExit('FAIL: --coopvec 1 and --coopvec 0 did not report the two decode paths: %r / %r'
                             % (cv_frames['1'][1][-200:], cv_frames['0'][1][-200:]))
        print('  the Vulkan viewer, the two decode paths on %s: %s'
              % (os.path.basename(cv_asset),
                 frame_diff(cv_frames['1'][0], cv_frames['0'][0], 4, 50.0, 'cooperative vectors vs plain')))

        # The refusal, on a device that has no cooperative vectors. On this machine that is device 1, the integrated
        # part; where every usable device has the extension there is nothing to refuse and the arm says so.
        refused = False
        for index in devices:
            if index == str(cv_device):
                continue
            bad_cv = run([view, cv_asset, '--nooverlay', '--shot', os.path.join('out', 'vk_coop_refuse.bmp'),
                          '--size', '64', '64', '--device', index, '--coopvec', '1'])
            if bad_cv.returncode == 0:
                continue        # that device has the extension too, so there is nothing to refuse there
            # Exit 1 for something OTHER than the flag is that device's own business and not this arm's: a part whose
            # driver will not open the asset at all, or one whose depth formats or 1.3 features this viewer refuses,
            # says so on stderr and is passed over. What this arm is looking for is a device that runs everything
            # else and refuses THIS flag, which is the only thing that proves the refusal is not a silent fall back.
            if bad_cv.returncode != 1 or 'ERROR: --coopvec 1' not in bad_cv.stderr:
                first = next((l for l in bad_cv.stderr.splitlines() if l.startswith('ERROR')), '').strip()
                print('  the Vulkan viewer: device %s is passed over for the --coopvec 1 refusal, it exits %d for '
                      'another reason (%s)' % (index, bad_cv.returncode, first or 'no ERROR line'))
                continue
            refused = True
            print('  the Vulkan viewer: --coopvec 1 on device %s (%s) is refused with an ERROR and exit 1'
                  % (index, devices[index]))
            break
        if not refused:
            print('  the Vulkan viewer: the --coopvec 1 refusal is skipped, every usable device here has the '
                  'extension')

        # --bench on both paths. What is asserted is that each run exits 0 and prints its two microsecond figures;
        # the figures themselves belong in viewer_vk/README.md, where they are quoted as measured.
        for flag in ('1', '0'):
            timed = run([view, cv_asset, '--nooverlay', '--bench', '20', '--size', '640', '480',
                         '--coopvec', flag] + cv_dev)
            if timed.returncode != 0:
                raise SystemExit('FAIL: --bench 20 --coopvec %s returned %d: %r'
                                 % (flag, timed.returncode, timed.stderr[-300:]))
            if not re.search(r'^bench: 20 frames at 640x480, decode .*: mean [0-9.]+ us, min [0-9.]+ us',
                             timed.stdout, re.MULTILINE):
                raise SystemExit('FAIL: --bench 20 --coopvec %s printed no mean and minimum: %r'
                                 % (flag, timed.stdout[-300:]))
        print('  the Vulkan viewer: --bench 20 times both decode paths and prints a mean and a minimum for each')

        # The strip. Left of the token the two frames' 84 overlay rows are the same bytes - the state line is the same
        # words either way - and the whole strip is not, which is the token itself.
        strips = []
        for flag in ('1', '0'):
            out = os.path.join('out', 'vk_coop_ovl_%s.bmp' % flag)
            made = run([view, cv_asset, '--noaniso', '--shot', out, '--size', '2560', '1440',
                        '--coopvec', flag] + cv_dev)
            if made.returncode != 0:
                raise SystemExit('FAIL: the overlay --shot with --coopvec %s returned %d' % (flag, made.returncode))
            strips.append(out)
        if bmp_rows(strips[0], 0, 84, VK_COOP_COLUMN) != bmp_rows(strips[1], 0, 84, VK_COOP_COLUMN):
            raise SystemExit('FAIL: the two decode paths\' overlay strips differ left of the decode-path token, so '
                             'something other than that token changed')
        if bmp_rows(strips[0], 0, 84) == bmp_rows(strips[1], 0, 84):
            raise SystemExit('FAIL: the overlay strip is the same bytes on both decode paths, so its Coop token is '
                             'not being drawn')
        print('  the Vulkan viewer: the overlay strip differs between the two decode paths in its Coop token alone')
        cv_note = ('; the two decode paths are the same picture on %s, --coopvec 1 is refused where the extension is '
                   'absent, --bench times both, and the strip differs in its Coop token alone'
                   % os.path.basename(cv_asset))

    # The second device, when there is one. This is the point of --device: the baseline path is the product and must
    # run on a non-NVIDIA part, and on this machine device 1 is the integrated Radeon. Only a USABLE device is asked
    # for - an unusable one is a refusal, not a second measurement - and its name goes into the summary, because
    # "on both devices" says nothing about which part the number came from.
    second = None
    if cross and '1' in devices:
        second = devices['1']
        lines.append(pair(desc, 'tiny_dev1', ['--tex', '0'], max_diff=16, min_psnr=40.0, device=1))
    elif cross:
        print('  the Vulkan viewer: --device 1 skipped, this machine has no second usable Vulkan device')

    for line in lines:
        print('  the Vulkan viewer vs the Direct3D one, %s' % line)

    # Five launches, five identical files. It is a stronger claim here than for the other viewer, because --shot
    # creates no window and no surface at all, so there is no keystroke to receive and no compositor in the way.
    if run([view, desc, '--nooverlay', '--shot', single_vk] + VK_PLAIN).returncode != 0:
        raise SystemExit('FAIL: nntc_view_vk --shot failed on the first of the five identical launches')
    identical = open(os.path.join(ROOT, single_vk), 'rb').read()
    for i in range(1, 5):
        again = run([view, desc, '--nooverlay', '--shot', single_vk] + VK_PLAIN)
        if again.returncode != 0:
            raise SystemExit('FAIL: nntc_view_vk --shot returned %d on launch %d' % (again.returncode, i + 1))
        if open(os.path.join(ROOT, single_vk), 'rb').read() != identical:
            raise SystemExit('FAIL: two headless nntc_view_vk --shot launches drew different frames (run %d of 5)'
                             % (i + 1))
    print('  the Vulkan viewer: 5 --shot launches are byte-identical (%d bytes)' % len(identical))

    if run([view, '--help']).returncode != 0:
        raise SystemExit('FAIL: nntc_view_vk --help must exit 0')
    bad = run([view, '--bogus'])
    if bad.returncode != 1 or 'unknown option' not in bad.stderr:
        raise SystemExit('FAIL: nntc_view_vk must refuse an unknown flag with exit 1, not %d %r'
                         % (bad.returncode, bad.stderr[-200:]))

    # A SHADER THAT DOES NOT COMPILE, at startup. Since the viewer reads its two GLSL files from beside the executable
    # and from nowhere else, the only way to hand it a broken one is to break the copy beside a copy of the
    # executable: both shaders and the exe go into a scratch directory, view.frag there is overwritten with something
    # that is not GLSL, and the run must print SHADER ERROR, exit 1 and write no frame - not a blank picture and an
    # exit code of 0. Key R with a broken shader takes the same compile through the same function, and it cannot be
    # driven from here at all (there is no window to press R in), so it stays a by-hand check.
    bad_dir = os.path.join('out', 'vk_badshader')
    os.makedirs(os.path.join(ROOT, bad_dir), exist_ok=True)
    src_dir = os.path.dirname(os.path.join(ROOT, view))
    exe_name = os.path.basename(view)
    for name in (exe_name, 'view.vert', 'view.frag'):
        shutil.copyfile(os.path.join(src_dir, name), os.path.join(ROOT, bad_dir, name))
    with open(os.path.join(ROOT, bad_dir, 'view.frag'), 'w') as f:
        f.write('this is not a shader, and the point is that it is not\n')
    bad_frame = os.path.join(bad_dir, 'frame.bmp')
    if os.path.exists(os.path.join(ROOT, bad_frame)):
        os.remove(os.path.join(ROOT, bad_frame))
    bad = run([os.path.join(bad_dir, exe_name), desc, '--nooverlay', '--shot', bad_frame, '--size', '64', '64']
              + VK_PLAIN)
    if bad.returncode != 1 or 'SHADER ERROR' not in bad.stderr or os.path.exists(os.path.join(ROOT, bad_frame)):
        raise SystemExit('FAIL: a broken view.frag beside the executable must exit 1 with SHADER ERROR and write no '
                         'frame, not %d %r' % (bad.returncode, bad.stderr[-300:]))
    print('  the Vulkan viewer: a broken view.frag beside the executable is SHADER ERROR, exit 1 and no frame')

    # AND THE OTHER HALF OF THAT RULE: the good copy beside the executable is read even when the WORKING DIRECTORY has
    # a broken view.frag of its own. The case above proves a broken shader is not ignored; this one proves which of
    # two candidate files is the one that gets compiled, which is the part a change to exe_dir could silently break.
    # The copied executable from the badshader directory cannot serve - its own view.frag is the broken one - so the
    # run is the ordinary executable, started from a scratch directory that holds two files named like the shaders and
    # containing nothing that compiles. Every path is absolute, because the working directory is no longer the tree.
    cwd_dir = os.path.join(ROOT, 'out', 'vk_cwd_decoy')
    os.makedirs(cwd_dir, exist_ok=True)
    for name in ('view.vert', 'view.frag'):
        with open(os.path.join(cwd_dir, name), 'w') as f:
            f.write('this file is in the working directory and must never be the one that is compiled\n')
    cwd_frame = os.path.join(cwd_dir, 'frame.bmp')
    if os.path.exists(cwd_frame):
        os.remove(cwd_frame)
    from_cwd = subprocess.run([os.path.abspath(os.path.join(ROOT, view)), os.path.abspath(os.path.join(ROOT, desc)),
                               '--nooverlay', '--shot', cwd_frame, '--size', '64', '64'] + VK_PLAIN,
                              cwd=cwd_dir, capture_output=True, text=True, encoding='utf-8', errors='replace')
    compiled = re.search(r'^shaders compiled: (.*)$', from_cwd.stdout, re.MULTILINE)
    exe_here = os.path.dirname(os.path.abspath(os.path.join(ROOT, view)))
    if from_cwd.returncode != 0 or not compiled or not os.path.exists(cwd_frame):
        raise SystemExit('FAIL: the viewer started from a directory holding a decoy view.frag must still compile the '
                         'shaders beside its executable, not %d %r' % (from_cwd.returncode, from_cwd.stderr[-300:]))
    for one in compiled.group(1).split(', '):
        if os.path.dirname(os.path.abspath(one)) != exe_here:
            raise SystemExit('FAIL: the viewer compiled %r, which is not beside its executable (%s)' % (one, exe_here))
    print('  the Vulkan viewer: started from a directory with a decoy view.frag, the shaders beside the executable '
          'are still the ones compiled')

    seconds = time.perf_counter() - started
    record('the Vulkan viewer',
           ('the same picture as the Direct3D viewer on a single image, a two-texture material (--tex 0 and 1) and a '
            'two-file level 0 at the default camera, and at --z -50 --noaniso with and without level 1\'s bias, where '
            'the two frames differ from each other; --raw0, --raw1, --renorm, --cube and the load-time BC pack each '
            'compared against that viewer too; the overlay\'s 84 rows byte-identical between the viewers and the rows '
            'below them equal to each viewer\'s own --nooverlay frame'
            + (', and the same asset on %s through --device 1' % second if second else '')
            if cross else
            'the cross-viewer pairs skipped (the Direct3D viewer is Windows only)')
           + cv_note
           + '; the filter toggles, --raw0, --raw1 and --renorm each change the frame, --bc binds the load-time pack '
             'of an uncompressed level 0 and does nothing on a BC one, the overlay is drawn and --nooverlay leaves it '
             'off, --shot with no --size is 2560x1440, five launches byte-identical, a broken shader exits 1, the '
             'shaders are read from beside the executable and not from the working directory, --help exits 0 and an '
             'unknown flag exits 1; ' + bc_note + ' (%.0f s)' % seconds)
    print('  the Vulkan viewer: --help exits 0 and an unknown flag exits 1; the Vulkan arms took %.0f s' % seconds)


def absolute_path_check(encode, build_dir):
    """A descriptor may name its .dds files by an absolute path (a hand-authored one; the encoder writes base names),
    and every reader takes an absolute path as written and a relative one from the descriptor's directory. The gate
    rewrites a fresh descriptor's names to absolute paths, moves the descriptor to another directory, and opens it
    with the Python decoder, the viewer and bc_check."""
    odir, prefix = out_asset('tiny_abs_paths')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--png', '0', '--quiet'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: the encode for the absolute-path case returned %d' % proc.returncode)
    desc = prefix + '_nntc.json'
    meta = json.load(open(desc))
    for t in meta['textures']:
        if 'file' in t:
            t['file'] = os.path.abspath(os.path.join(ROOT, odir, t['file']))
        for f in t.get('files', []):
            f['file'] = os.path.abspath(os.path.join(ROOT, odir, f['file']))
    elsewhere = os.path.join('out', 'tiny_abs_paths_elsewhere')
    os.makedirs(elsewhere, exist_ok=True)
    moved = os.path.join(elsewhere, 'moved_nntc.json')
    with open(moved, 'w') as f:
        json.dump(meta, f, indent=1)
    proc = run([sys.executable, os.path.join('tools', 'dds_decode.py'), moved, '--grid'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: the Python decoder must open a descriptor naming absolute paths, not %r' % proc.stdout[-300:])
    if os.name == 'nt':
        view = executable(build_dir, 'nntc_view')
        frame, err = shot(view, moved, os.path.join(elsewhere, 'shot.bmp'), [])
        check = executable(build_dir, 'bc_check')
        proc = run([check, moved])
        if proc.returncode != 0:
            raise SystemExit('FAIL: bc_check must open a descriptor naming absolute paths, not %r' % proc.stderr[-300:])
    record('absolute paths in a descriptor', 'taken as written by the Python decoder' + (', the viewer and bc_check' if os.name == 'nt' else '')
           + '; relative names stay beside the descriptor')
    print('  absolute paths in a descriptor: every reader takes them as written')


def main():
    build_dir = os.path.join(ROOT, 'build')
    if '--build-dir' in sys.argv:
        build_dir = sys.argv[sys.argv.index('--build-dir') + 1]

    encode = executable(build_dir, 'nntc_encode')
    bits0 = 3
    odir, prefix = out_asset('tiny')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--l0', 'palette',
                '--c0', '2', '--bits0', str(bits0), '--c1', '4', '--bits1', '8'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_encode returned %d' % proc.returncode)
    rounds = loop_checks(proc.stdout, 1e-4)
    record('the round loop', 'E monotone across %d blocks of %d rounds' % (3 * len(rounds), len(rounds)))
    timing(proc.stdout, 'tiny.png, the default layout')

    tally = ridge_tally(proc.stdout, 'tiny.png')
    if tally[0] <= 0 or tally[1] or tally[2] or tally[3]:
        raise SystemExit('FAIL: tiny.png took a block (a) rung below the shipped ridge: %s' % (tally,))
    print('  block (a) ridge: %d standard, none reduced, none unridged, none refused' % tally[0])
    record('block (a)\'s ridge ladder', 'tiny.png takes the shipped ridge on every one of its %d decoder solves'
           % tally[0])

    quantisation_checks(proc.stdout, 'the default --q1-start')
    level_checks(proc.stdout, prefix)
    record('level 1 on its grid', 'every stored index re-quantises to itself')
    record('the mip chain', "the independent decoder matches the encoder's psnr at every level")

    # The control path, at the bit depth where the two differ most: level 1 continuous through the loop and rounded
    # once at the end. It has to reach the grid and round-trip exactly as the quantised phase does, since one writer
    # and one JSON describe both.
    odir, control = out_asset('tiny_q0')
    proc = run([encode, os.path.join('tests', 'tiny.png'), '-o', odir, '--l0', 'palette',
                '--c0', '2', '--bits0', str(bits0), '--c1', '4', '--bits1', '4', '--q1-start', '0'])
    if proc.returncode != 0:
        raise SystemExit('FAIL: nntc_encode --q1-start 0 returned %d' % proc.returncode)
    loop_checks(proc.stdout, 1e-4)
    quantisation_checks(proc.stdout, '--q1-start 0')
    proc = run([sys.executable, os.path.join('tools', 'dds_decode.py'), control, '--ref', control + '_recon'])
    worst = re.search(r'worst max \|diff\| over every level: (\d+)', proc.stdout)
    if proc.returncode != 0 or not worst or int(worst.group(1)) > 1:
        raise SystemExit('FAIL: the --q1-start 0 asset does not round-trip within one step')
    print('  round trip: --q1-start 0 max |diff| %s at every level' % worst.group(1))
    record('--q1-start 0', 'the control path solves, stores and round-trips')

    bc_checks(encode, build_dir)
    record('level 0 as BC4 / BC5',
           'lossless at 3 bits, BC5 + BC4 at 3 channels, five --shot launches of one asset byte-identical, and '
           'both the two-channel and the three-channel file frames identical to the packed-at-load frame'
           if os.name == 'nt' else 'lossless at 3 bits, BC5 + BC4 at 3 channels')
    bc8_checks(encode, build_dir)
    record('--l0 bc8', 'the loop is monotone, the refinement and the outer loop only lower E, the asset agrees '
                       'with the report')
    record('the bc8 coverage', '--c0 1, --c0 3 (BC5 + BC4), --bc-refine-after, --q1-start 0, a chain with a partial '
                               'edge block, and a run whose outer pass is rejected and restored')

    grid_checks(encode)
    init0_checks(encode)
    record('--init0', 'under bc8 the residual seed is monotone channel by channel; under --l0 palette every refit '
                      'and every snap search lowers E and --c0 2 ships below --c0 1 at 4 bits; residual, '
                      'texture, luma and both scopes round-trip')
    coverage_checks(encode, build_dir)
    ill_conditioned_checks(encode)
    list_rule_checks(encode)
    mip_filter_checks(encode)
    material_checks(encode, build_dir)
    material_detail_checks(encode, build_dir)
    review_fix_checks(encode, build_dir)
    six_texture_checks(encode, build_dir)
    bare_command_check(encode)
    old_format_check(encode, build_dir)
    viewer_refusal_checks(encode, build_dir)
    vulkan_viewer_checks(encode, build_dir)
    absolute_path_check(encode, build_dir)
    never_delete_check(encode)
    argument_and_status_checks(encode)
    record('the bare command line', 'writes the three files in the current directory, with the default layout')

    determinism_check(encode)
    diag_and_16bit_checks(encode)
    alpha_input_checks(encode)

    objective_checks(encode, os.path.join('out', 'tiny_check'))
    record('the objective', 'device E == the host brute force, and every block lowers it')
    record('block (b)', 'converged, dense-solve agreement, zero gradient left behind')

    failures = check_tree()
    for f in failures:
        print('FAIL: ' + f)
    if failures:
        raise SystemExit('FAIL: %d forbidden pattern(s) in the tree' % len(failures))
    print('  tree: clean')
    record('the tree', 'no forbidden pattern in %s' % ', '.join(SEARCHED_DIRS + SEARCHED_FILES))

    width = max(len(g) for g, _ in SUMMARY)
    print('\n%-*s   %s' % (width, 'gate', 'result'))
    print('%s   %s' % ('-' * width, '-' * 60))
    for gate, detail in SUMMARY:
        print('%-*s   %s' % (width, gate, detail))
    # Where the run left its assets. It is about 200 MB of them, which is worth saying once rather than letting a
    # caller discover it: out/ is gitignored apart from the logs that are force-added as records of a run.
    print('\nthe assets of this run are under %s (about 200 MB); out/ is gitignored except its tracked logs'
          % os.path.join(ROOT, 'out'))
    print('\nAll checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
