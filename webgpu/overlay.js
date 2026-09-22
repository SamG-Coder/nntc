// overlay.js: the drawn debug strip - two lines, over the scene, in the native viewers' own font.
//
// The owner's rule (docs/WEBGPU_VIEWER_PLAN.md section 6.3): the drawn overlay carries ONLY what changes fast enough
// that a DOM element updated per frame would be wasteful, or what has to be read while the camera is moving. That is
// the frame rate and the camera, and nothing else. Every mode and status readout the native strip carries lives in the
// left panel instead, where the control that sets it is.
//
// And it is DRAWN, not HTML over the canvas: "the overlay [is] rendered via draws referencing a basic font texture,
// just like now. No HTML over the 3D output." So this is the native path, transliterated - blit_char from
// ../viewer/main.cpp (FONT_SCALE 2, black at alpha 180, white glyphs) into an RGBA byte buffer, one rgba8unorm
// texture, and a second pipeline drawing one alpha-blended quad at the top-left after the scene.
//
// Captures render with the strip OFF, as --nooverlay does: the frame rate changes from frame to frame, so a capture
// with it on could never be compared byte for byte.

import { FONT8X8 } from './font8x8.js';

export const OVL_W = 1000;      // wide enough for the camera line at 16 px per character (47 characters)
export const OVL_H = 44;        // two lines at LINE_ADV, plus the native strip's two-pixel top margin
const FONT_SCALE = 2;
const LINE_ADV = 20;

// printf's %+5.1f and friends, because the camera line is the native one verbatim and its widths are part of it.
function fixed(value, width, decimals, sign) {
    let s = Math.abs(value).toFixed(decimals);
    s = (value < 0 ? '-' : (sign ? '+' : '')) + s;
    while (s.length < width) s = ' ' + s;
    return s;
}

export function cameraLine(state) {
    return 'X:' + fixed(state.x, 5, 1, true) + ' Y:' + fixed(state.y, 5, 1, true)
         + ' Z:' + fixed(state.z, 5, 1, false) + ' Yaw:' + fixed(state.yaw, 6, 1, true)
         + ' Pitch:' + fixed(state.pitch, 6, 1, true);
}

export function fpsLine(fps, ms) {
    if (fps === null) return 'idle';
    return fixed(fps, 2, 0, false) + ' fps  ' + fixed(ms, 4, 1, false) + ' ms';
}

export class Overlay {
    // THE STRIP IS BUILT THROUGH create(), NOT THROUGH new, and the reason is the same one renderer.js's loadShader
    // has: createRenderPipeline never throws. It marks the pipeline invalid and returns it, and this pipeline draws
    // in the SAME PASS as the scene, so an invalid one takes the picture down with it. The text that reaches
    // createShaderModule is not always WGSL either - a 404 on overlay.wgsl hands it the server's HTML error page,
    // which compiles into nothing at all - so the compilation messages are read and the pipeline is built with the
    // form of the call that REJECTS. Nothing here is half-built: create() either returns a strip that can draw or
    // throws with the reason, and main.js puts that reason on the page.
    static async create(device, colorFormat, depthFormat, shaderCode) {
        const overlay = new Overlay(device);
        await overlay.buildPipeline(colorFormat, depthFormat, shaderCode);
        return overlay;
    }

    constructor(device) {
        this.device = device;
        this.buf = new Uint8Array(OVL_W * OVL_H * 4);
        this.lines = ['', ''];
        this.dirty = true;
        this.texture = device.createTexture({
            label: 'overlay strip',
            size: [OVL_W, OVL_H, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
        this.vbuf = device.createBuffer({ size: 16 * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        this.ibuf = device.createBuffer({ size: 6 * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.ibuf, 0, new Uint32Array([0, 1, 2, 0, 2, 3]));
        this.verts = new Float32Array(16);
    }

    async buildPipeline(colorFormat, depthFormat, shaderCode) {
        const device = this.device;
        const module = device.createShaderModule({ label: 'overlay.wgsl', code: shaderCode });
        const info = await module.getCompilationInfo();
        for (const m of info.messages) {
            if (m.type === 'error') throw new Error('overlay.wgsl:' + m.lineNum + ': ' + m.message);
        }
        const descriptor = {
            label: 'overlay',
            layout: 'auto',
            vertex: {
                module: module, entryPoint: 'VSMain',
                buffers: [{
                    arrayStride: 16,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x2' },
                    ],
                }],
            },
            fragment: {
                module: module, entryPoint: 'PSMain',
                targets: [{
                    format: colorFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' },
                    },
                }],
            },
            primitive: { topology: 'triangle-list' },
            // The strip sits over whatever was drawn: the native viewer turns the depth test off for it, and a
            // pipeline in a pass that has a depth attachment must still declare the state it is not using.
            depthStencil: { format: depthFormat, depthWriteEnabled: false, depthCompare: 'always' },
        };
        let pipeline;
        try {
            pipeline = await device.createRenderPipelineAsync(descriptor);
        } catch (e) {
            throw new Error('overlay.wgsl compiles but will not make a pipeline: ' + (e.message || e));
        }
        this.pipeline = pipeline;
        this.bindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.texture.createView() },
                { binding: 1, resource: this.sampler },
            ],
        });
    }

    setLines(line0, line1) {
        if (this.lines[0] === line0 && this.lines[1] === line1) return;
        this.lines[0] = line0;
        this.lines[1] = line1;
        this.dirty = true;
    }

    blitChar(px, py, ch) {
        let c = ch.charCodeAt(0);
        if (c < 32 || c > 127) c = 46;   // '.', as the native blit_char does
        const glyph = (c - 32) * 8;
        for (let y = 0; y < 8; y++) {
            const row = FONT8X8[glyph + y];
            for (let x = 0; x < 8; x++) {
                if (!((row >> x) & 1)) continue;
                for (let sy = 0; sy < FONT_SCALE; sy++) {
                    for (let sx = 0; sx < FONT_SCALE; sx++) {
                        const X = px + x * FONT_SCALE + sx, Y = py + y * FONT_SCALE + sy;
                        if (X < 0 || X >= OVL_W || Y < 0 || Y >= OVL_H) continue;
                        const d = (Y * OVL_W + X) * 4;
                        this.buf[d] = 255; this.buf[d + 1] = 255; this.buf[d + 2] = 255; this.buf[d + 3] = 255;
                    }
                }
            }
        }
    }

    update() {
        if (!this.dirty) return;
        for (let i = 0; i < this.buf.length; i += 4) {
            this.buf[i] = 0; this.buf[i + 1] = 0; this.buf[i + 2] = 0; this.buf[i + 3] = 180;
        }
        for (let li = 0; li < 2; li++) {
            let x = 4;
            const y = 2 + li * LINE_ADV;
            for (const ch of this.lines[li]) {
                this.blitChar(x, y, ch);
                x += 8 * FONT_SCALE;
            }
        }
        this.device.queue.writeTexture(
            { texture: this.texture }, this.buf, { bytesPerRow: OVL_W * 4, rowsPerImage: OVL_H },
            [OVL_W, OVL_H, 1]);
        this.dirty = false;
    }

    // The strip drawn into an open render pass, after the scene. The quad is placed in clip space from the target's
    // size, so the strip is the same number of PIXELS whatever the canvas is - exactly as the native one is.
    draw(pass, targetW, targetH) {
        this.update();
        const w = OVL_W / targetW * 2.0, h = OVL_H / targetH * 2.0;
        this.verts.set([
            -1.0, 1.0, 0.0, 0.0,
            -1.0 + w, 1.0, 1.0, 0.0,
            -1.0 + w, 1.0 - h, 1.0, 1.0,
            -1.0, 1.0 - h, 0.0, 1.0,
        ]);
        this.device.queue.writeBuffer(this.vbuf, 0, this.verts);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.setVertexBuffer(0, this.vbuf);
        pass.setIndexBuffer(this.ibuf, 'uint32');
        pass.drawIndexed(6);
    }
}
