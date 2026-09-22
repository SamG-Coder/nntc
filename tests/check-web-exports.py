"""Verify browser ZIP exports with upstream's independent Python decoder.

Run npm run test:ui first, then python tests/check-web-exports.py.
Requires numpy and Pillow (the existing decoder's dependencies).
"""
import pathlib
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import dds_decode

for archive, sources in [('m1.zip', ['m1.png']), ('m1-m4.zip', ['m1.png', 'm2.png', 'm3.png', 'm4.png'])]:
    with tempfile.TemporaryDirectory() as temporary:
        directory = pathlib.Path(temporary)
        with zipfile.ZipFile(ROOT / 'out' / 'web-tests' / archive) as bundle:
            assert bundle.testzip() is None, 'ZIP CRC failure'
            for name in bundle.namelist():
                assert pathlib.PurePosixPath(name).name == name and '\\' not in name, 'Expected flat ZIP entries'
                (directory / name).write_bytes(bundle.read(name))
        descriptor = directory / 'm1_nntc.json'
        # Exercise upstream's header, format and every mip decoder, not just our JS reader.
        sys.argv = ['dds_decode.py', str(descriptor), '--psnr'] + [str(ROOT / 'examples' / name) for name in sources]
        assert dds_decode.main() == 0
