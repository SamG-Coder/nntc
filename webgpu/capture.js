// capture.js: a rendered frame turned into bytes a person or a script can keep.
//
// The .bmp is the one that matters. ../viewer/main.cpp:1162-1186 writes its --shot frame as a 24-bit BOTTOM-UP .bmp
// with a (W * 3 + 3) & ~3 row pitch and BGR triples, and tools/frame_diff.py reads that; a frame from this page has
// to be the same file in the same layout or it cannot be compared against the three native viewers at all. So this
// is a transliteration of those twenty lines and not a fresh design.
//
// WHICH BYTES ARRIVE DEPENDS ON THE CANVAS FORMAT, which is why the capture says what it got instead of assuming.
// getPreferredCanvasFormat answers bgra8unorm on Windows and rgba8unorm elsewhere; the .bmp wants blue, green, red
// in that order, so one of the two needs its red and blue swapped and the other does not. Guessing here would make
// a picture that is right on one machine and colour-swapped on another, which is exactly the kind of fault a frame
// comparison is supposed to catch rather than contain.
//
// The .png is for a person: an OffscreenCanvas 2D context and convertToBlob, which is the browser's own encoder.
// It is top-down and RGBA, so it is the other order from the .bmp in both respects.

const BMP_HEADER_BYTES = 54;

// True when the four bytes of a pixel arrive as blue, green, red, alpha.
function isBGRA(format) {
    return String(format || '').startsWith('bgra');
}

// A 24-bit bottom-up .bmp, byte for byte the file the native viewers write.
export function bmpBytes(shot) {
    const { width, height, pixels } = shot;
    const swap = !isBGRA(shot.format);          // rgba8unorm: red and blue change places on the way out
    const pitch = (width * 3 + 3) & ~3;         // every row padded to a multiple of four bytes
    const body = new Uint8Array(pitch * height);
    for (let y = 0; y < height; y++) {
        const src = y * width * 4;
        const dst = (height - 1 - y) * pitch;   // BOTTOM-UP: the first row of the file is the last row of the image
        for (let x = 0; x < width; x++) {
            const s = src + x * 4, d = dst + x * 3;
            body[d] = pixels[s + (swap ? 2 : 0)];
            body[d + 1] = pixels[s + 1];
            body[d + 2] = pixels[s + (swap ? 0 : 2)];
        }
    }
    const out = new Uint8Array(BMP_HEADER_BYTES + body.length);
    const view = new DataView(out.buffer);
    out[0] = 0x42; out[1] = 0x4D;                                   // 'BM'
    view.setUint32(2, out.length, true);                            // the whole file
    view.setUint32(10, BMP_HEADER_BYTES, true);                     // where the pixels start
    view.setUint32(14, 40, true);                                   // BITMAPINFOHEADER
    view.setInt32(18, width, true);
    view.setInt32(22, height, true);                                // positive: bottom-up, as above
    view.setUint16(26, 1, true);                                    // one plane
    view.setUint16(28, 24, true);                                   // bits per pixel
    view.setUint32(34, body.length, true);                          // the pixels' size
    out.set(body, BMP_HEADER_BYTES);
    return out;
}

// The same frame as a .png, through the browser's own encoder. Top-down and RGBA, so the rows are not reversed and
// the swap is the other way round from the .bmp's.
export async function pngBlob(shot) {
    const { width, height, pixels } = shot;
    const rgba = new Uint8ClampedArray(width * height * 4);
    const swap = isBGRA(shot.format);
    for (let i = 0; i < width * height; i++) {
        const s = i * 4;
        rgba[s] = pixels[s + (swap ? 2 : 0)];
        rgba[s + 1] = pixels[s + 1];
        rgba[s + 2] = pixels[s + (swap ? 0 : 2)];
        rgba[s + 3] = 255;   // the frame is opaque; the alpha the render left is not part of the picture
    }
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
    return await canvas.convertToBlob({ type: 'image/png' });
}

// Hand a file to the person. A page cannot write to disk; an anchor with `download` is the whole of what it can do,
// and the object url is released once the click has been taken, or it holds the bytes until the tab closes.
export function offerDownload(name, blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// The name a saved frame gets: the material it is of, and the size, so a folder of them can be told apart without
// opening them. A descriptor's name has the part every one of them carries taken off it, as the pulldown does.
export function shotName(materialName, width, height, extension) {
    const base = String(materialName || 'frame')
        .replace(/^.*[\\/]/, '')
        .replace(/_nntc\.json$/i, '')
        .replace(/\.json$/i, '');
    return base + '_' + width + 'x' + height + '.' + extension;
}
