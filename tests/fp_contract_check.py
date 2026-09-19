#!/usr/bin/env python3
"""Build tests/fp_contract_check.cpp with every toolchain this machine has, run each, and diff the dumps.

    python tests/fp_contract_check.py [-o OUTDIR]

WHAT IT ANSWERS. model.h's level1_value and refine_bc.cu's bc0_value each compute `lo + frac * span` in double and
round once to float, and the device spells the two operations out separately so that the host agrees with it bit for
bit. Whether the HOST agrees is a compiler setting: CMakeLists.txt puts -ffp-contract=off on the gcc and clang branch,
and the MSVC branch relies on /fp:precise not contracting. That reliance is safe on x64, where a double-precision FMA
needs /arch:AVX2 to be emitted at all, and is UNTESTED on MSVC ARM64, where fmadd is a baseline instruction. This
script is how it gets tested: run it on each machine and diff the dumps.

ON THE ARM LAPTOP, which is the run that matters and which cannot be done from an x64 box: open an ARM64 Native Tools
command prompt (or x64_arm64 cross tools), and run this script from the tree's root. It picks the cl.exe on that
prompt's PATH, so the dump it writes is an ARM64 one. Compare `msvc.txt` with the x64 machine's `msvc.txt`: the two
must be identical, line for line, and the `hash` line at the end is the quick way to look. If they differ, add
/fp:contract- to CMakeLists.txt's MSVC branch and run this again.

Each toolchain is built with the flags the encoder itself is built with, because the question is about the shipped
configuration and not about what some other set of flags would do. gcc and clang are additionally built a second time
with contraction turned ON, which is what the -ffp-contract=off in CMakeLists.txt exists to prevent: on a machine where
that second dump differs from the first, the option is load-bearing and the diff shows exactly how.
"""

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'tests', 'fp_contract_check.cpp')
INC = os.path.join(ROOT, 'src')


def have(exe):
    return shutil.which(exe) is not None


def build_and_run(name, argv, outdir, exe_name, refusal_is_a_finding=False):
    """Compile argv (which must already name the source and the output), run the result, keep its stdout.

    A refusal from the program itself is the answer for a build that deliberately allows contraction, so those runs
    report rather than stop. A refusal from a build that is meant to pass stops the script."""
    print('  %-14s %s' % (name, ' '.join(os.path.basename(a) if os.sep in a else a for a in argv)))
    r = subprocess.run(argv, cwd=outdir, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit('ERROR: %s did not compile fp_contract_check.cpp' % name)
    if r.stderr.strip():
        print('    the compiler said:\n%s' % r.stderr.strip())
    exe = os.path.join(outdir, exe_name)
    run = subprocess.run([exe], capture_output=True, text=True)
    path = os.path.join(outdir, name + '.txt')
    with open(path, 'w', newline='\n') as f:
        f.write(run.stdout)
    for line in run.stdout.splitlines():
        if line.startswith('hash ') or line.startswith('witnesses '):
            print('    %s' % line)
    if run.returncode != 0:
        print('    it refused: %s' % run.stderr.strip().splitlines()[0][:160])
        if not refusal_is_a_finding:
            raise SystemExit('ERROR: the %s build of fp_contract_check refused, which means this toolchain builds an '
                             'encoder that will refuse correct assets' % name)
    return path, run.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=os.path.join(ROOT, 'out_stepA', 'fp_contract'),
                    help='where the dumps go (default out_stepA/fp_contract)')
    args = ap.parse_args()
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    dumps = []
    skipped = []
    # image.h reaches for fopen and _wfopen, which the Microsoft CRT deprecates; the encoder's own build defines this
    # for the same reason and it says nothing about floating point.
    crt = ['_CRT_SECURE_NO_WARNINGS'] if os.name == 'nt' else []

    # MSVC, with the encoder's own options: /W4 and nothing about floating point, which is the reliance under test.
    if os.name == 'nt' and have('cl'):
        dumps.append(('msvc', build_and_run(
            'msvc', ['cl', '/nologo', '/O2', '/std:c++17', '/EHsc', '/W4'] + ['/D' + d for d in crt]
            + ['/I' + INC, SRC, '/Fe:msvc.exe'], outdir, 'msvc.exe')[0]))
    else:
        skipped.append('MSVC (no cl.exe on this PATH: run from a Native Tools prompt)')

    # gcc and clang, with the encoder's own options, and then again with contraction allowed so that the value of the
    # option is visible rather than assumed.
    for cxx in ('g++', 'clang++'):
        if not have(cxx):
            skipped.append('%s (not on this PATH)' % cxx)
            continue
        base = cxx.replace('+', 'p')
        common = ['-O2', '-std=c++17', '-Wall', '-Wextra'] + ['-D' + d for d in crt] + ['-I' + INC, SRC]
        dumps.append((base, build_and_run(
            base, [cxx] + common + ['-ffp-contract=off', '-o', base], outdir, base)[0]))
        build_and_run(base + '_contract_on',
                      [cxx] + common + ['-ffp-contract=fast', '-march=native', '-o', base + '_contract_on'],
                      outdir, base + '_contract_on', refusal_is_a_finding=True)

    if not dumps:
        raise SystemExit('ERROR: no C++ compiler was found, so nothing was checked')

    # The comparison. Every dump must be identical to the first.
    print('')
    ref_name, ref_path = dumps[0]
    ref = open(ref_path, encoding='utf-8').read().splitlines()
    bad = 0
    for name, path in dumps[1:]:
        other = open(path, encoding='utf-8').read().splitlines()
        if other == ref:
            print('  %s == %s' % (name, ref_name))
            continue
        bad += 1
        diffs = [(i, a, b) for i, (a, b) in enumerate(zip(ref, other), 1) if a != b]
        print('  %s DIFFERS from %s on %d line(s) of %d' % (name, ref_name, len(diffs) + abs(len(ref) - len(other)),
                                                            len(ref)))
        for i, a, b in diffs[:10]:
            print('    line %d\n      %s %s\n      %s %s' % (i, ref_name, a, name, b))

    # The contraction-on builds are reported rather than asserted. They are the calibration: a build that ALLOWS
    # contraction should fail the witnesses, and if it does not then this machine's target has no double FMA to emit
    # and the check has proved nothing here beyond the dumps agreeing.
    for name, _ in dumps:
        on = os.path.join(outdir, name + '_contract_on.txt')
        if os.path.exists(on):
            same = open(on, encoding='utf-8').read().splitlines() == open(
                os.path.join(outdir, name + '.txt'), encoding='utf-8').read().splitlines()
            print('  %s with contraction ALLOWED: %s' % (
                name, 'identical, so this target emits no double FMA and the witnesses prove nothing HERE'
                if same else 'DIFFERENT, so the option is load-bearing and the witnesses can see it'))

    for s in skipped:
        print('  skipped: %s' % s)
    print('')
    if bad:
        print('FAIL: %d toolchain(s) do not agree with %s' % (bad, ref_name))
        return 1
    print('All dumps agree (%d toolchain%s compared).' % (len(dumps), '' if len(dumps) == 1 else 's'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
