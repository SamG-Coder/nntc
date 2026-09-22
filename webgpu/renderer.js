// renderer.js: the WebGPU side - the adapter, the textures, the samplers, the 2032-byte uniform block, the quad and
// the cube, and the Direct3D 11 viewer's camera.
//
// Everything here is the native viewers' state, carried over rather than reinvented, because the whole argument of
// this tree is that independent implementations draw the same picture: ../viewer/main.cpp's eight sampler states
// (4.3 of the plan), its camera (FOV 90, z = -3 at reset, near 0.001, far 100, the projection built row-major and
// transposed once at write time), its quad whose aspect is level 0's W / H, and its cube.
//
// TWO THINGS A READER SHOULD CHECK FIRST, because getting either wrong gives a plausible picture that is wrong:
//
//   * LEVEL 1'S GRADIENTS ARE SCALED BY 2^lod_bias_level1 (const0.z), and that scaling is in the shader, not in a
//     sampler. WebGPU's sampler has no LOD bias field at all, so this is the only form the encoder's 1:1 mip rule can
//     take here. Without it level 1 is read two mips finer than it was fitted and the colour goes splotchy from mip 1
//     down - while still looking right at 1:1.
//   * THE UNIFORM OFFSETS. The block is the Direct3D viewer's two constant buffers laid end to end, 2032 bytes, and
//     WGSL's uniform layout rules give the same offsets as std140 and as Direct3D's 16-byte rules. They are asserted
//     below before the first draw rather than trusted.

// ---------------------------------------------------------------------------
// 4x4 matrices, row-major with the column-vector convention, as ../viewer/main.cpp builds them; one transpose at
// uniform-write time, so the sixteen floats in the buffer are the matrix's COLUMNS - which is what WGSL's mat4x4<f32>
// reads them as, and what the Vulkan viewer already relies on.
// ---------------------------------------------------------------------------
function matIdentity() { const r = new Float32Array(16); for (let i = 0; i < 4; i++) r[i * 4 + i] = 1; return r; }
function matMul(a, b) {
    const r = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += a[i * 4 + k] * b[k * 4 + j];
        r[i * 4 + j] = s;
    }
    return r;
}
function matPerspective(fovDeg, aspect, znear, zfar) {
    const m = new Float32Array(16);
    const f = 1.0 / Math.tan((fovDeg * Math.PI / 180.0) / 2.0);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = zfar / (znear - zfar);
    m[11] = (zfar * znear) / (znear - zfar);
    m[14] = -1.0;
    return m;
}
function matTranslate(x, y, z) { const m = matIdentity(); m[3] = x; m[7] = y; m[11] = z; return m; }
function matRotY(deg) {
    const m = matIdentity(), r = deg * Math.PI / 180.0, c = Math.cos(r), s = Math.sin(r);
    m[0] = c; m[2] = s; m[8] = -s; m[10] = c; return m;
}
function matRotX(deg) {
    const m = matIdentity(), r = deg * Math.PI / 180.0, c = Math.cos(r), s = Math.sin(r);
    m[5] = c; m[6] = -s; m[9] = s; m[10] = c; return m;
}
function matTranspose(a) {
    const r = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) r[i * 4 + j] = a[j * 4 + i];
    return r;
}

// The camera's own constants, ../viewer/main.cpp:97 and 175.
export const FOV_DEGREES = 90.0;
export const Z_MIN = 0.40, Z_MAX = -50.0;
export const Z_SPEED = 1.0, XY_SPEED = 0.75, ROT_SPEED = 90.0;
export const ANISO_MAX = 8;

// The uniform block, in floats. Asserted against 2032 bytes below.
const U_MVP = 0, U_TEXSIZE = 16, U_LODINFO = 20, U_CONST0 = 24, U_CONST1 = 28;
// The pulldown's values in the order the shader reads them: index 0 is every channel, 1..4 are the four singly.
export const CHANNELS = ['rgb', 'r', 'g', 'b', 'a'];
const U_LO0 = 32, U_HI0 = 36, U_LO1 = 40, U_HI1 = 44, U_DIMS = 48, U_SEL = 52, U_W = 56, U_BIAS = 488;
const U_FLOATS = 508;

// The quad's vertices, ../viewer/main.cpp:300. UV origin (0, 0) = top-left = the first uploaded row; the .dds levels
// are top-row-first, so v runs 0..1 top to bottom with no flip.
function quadVerts(aspect) {
    let hw, hh;
    if (aspect >= 1.0) { hw = 1.0; hh = 1.0 / aspect; } else { hw = aspect; hh = 1.0; }
    return new Float32Array([
        -hw, -hh, 0.0, 0.0, 1.0, hw, -hh, 0.0, 1.0, 1.0,
        hw, hh, 0.0, 1.0, 0.0, -hw, hh, 0.0, 0.0, 0.0,
    ]);
}
const CUBE_VERTS = (() => {
    const h = 0.5;
    return new Float32Array([
        -h, -h, h, 0, 1, h, -h, h, 1, 1, h, h, h, 1, 0, -h, h, h, 0, 0,
        h, -h, -h, 0, 1, -h, -h, -h, 1, 1, -h, h, -h, 1, 0, h, h, -h, 0, 0,
        h, -h, h, 0, 1, h, -h, -h, 1, 1, h, h, -h, 1, 0, h, h, h, 0, 0,
        -h, -h, -h, 0, 1, -h, -h, h, 1, 1, -h, h, h, 1, 0, -h, h, -h, 0, 0,
        -h, h, h, 0, 1, h, h, h, 1, 1, h, h, -h, 1, 0, -h, h, -h, 0, 0,
        -h, -h, -h, 0, 1, h, -h, -h, 1, 1, h, -h, h, 1, 0, -h, -h, h, 0, 0,
    ]);
})();
// A UV SPHERE, generated rather than written out: the quad and the cube are ten and twenty-four vertices and are
// literals like the native viewer's, but a sphere at any useful tessellation is not something to type.
//
// The seam is DUPLICATED, which is the whole subtlety. The ring of vertices where u would wrap from 1 back to 0 is
// emitted twice, once with u = 1 and once with u = 0, because a vertex carries one UV and a shared seam vertex
// would make the last column of quads interpolate backwards across the whole texture - a visible smear of the
// entire image squeezed into one column. The poles are rings rather than single points for the same reason: a pole
// vertex would need a different u for every triangle meeting it.
//
// v runs 0 at the north pole to 1 at the south, matching the quad's top-row-first convention, so a texture lands on
// the sphere the same way up as it lands on the quad.
const SPHERE_SEGMENTS = 48;   // around
const SPHERE_RINGS = 24;      // pole to pole

const SPHERE = (() => {
    const verts = [];
    const idx = [];
    const r = 0.6;   // a little under the cube's half-diagonal, so C does not change the apparent size much
    for (let ring = 0; ring <= SPHERE_RINGS; ring++) {
        const v = ring / SPHERE_RINGS;
        const phi = v * Math.PI;                      // 0 at the north pole
        const y = Math.cos(phi), rs = Math.sin(phi);
        for (let seg = 0; seg <= SPHERE_SEGMENTS; seg++) {
            const u = seg / SPHERE_SEGMENTS;
            // + PI puts u = 0 at -z, so the seam is BEHIND the sphere at the default yaw and the texture's
            // middle faces the viewer. Without it the two duplicated seam columns meet dead centre, and the
            // picture's left and right edges are what you look at first - where the quad and the cube's front
            // face both show u running 0..1 across the view.
            const theta = u * 2 * Math.PI + Math.PI;
            verts.push(r * rs * Math.sin(theta), r * y, r * rs * Math.cos(theta), u, v);
        }
    }
    const row = SPHERE_SEGMENTS + 1;
    for (let ring = 0; ring < SPHERE_RINGS; ring++) {
        for (let seg = 0; seg < SPHERE_SEGMENTS; seg++) {
            const a = ring * row + seg, b = a + row;
            idx.push(a, b, a + 1, a + 1, b, b + 1);
        }
    }
    return { verts: new Float32Array(verts), idx: new Uint32Array(idx) };
})();

const CUBE_INDICES = (() => {
    const idx = [];
    for (let i = 0; i < 6; i++) { const b = i * 4; idx.push(b, b + 1, b + 2, b, b + 2, b + 3); }
    return new Uint32Array(idx);
})();

export class Renderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.asset = null;
        this.textures = [];
        this.width = 0;
        this.height = 0;
        // Where a failure that no call can throw out of goes: a lost device and an uncaptured validation error.
        // main.js points it at the panel's status line; until it does, and for anything this class is used from
        // outside the page, the console is the fallback rather than silence.
        this.onError = null;
    }

    report(message) {
        if (this.onError) this.onError(message); else console.error(message);
    }

    // The adapter, the device and the canvas context. `wantUnpack` is the ?unpack field: true forces the CPU unpack on
    // any device, false asks for hardware BC and is REFUSED BY NAME on a device that has none - the shape of the
    // native viewers' --coopvec 0|1 and --linalg 0|1, which refuse rather than quietly falling back - and null (the
    // default) means hardware BC where the feature exists and the unpack where it does not.
    async init(wantUnpack) {
        if (!navigator.gpu) {
            throw new Error('This browser has no WebGPU here (or this page is not served over HTTPS or localhost). '
                            + 'WebGPU is in Chrome and Edge 113+, Firefox 141+ on Windows and 145+ on Apple-silicon '
                            + 'Macs, and Safari 26+.');
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter: the browser has the API but would not give this page a GPU.');
        this.adapter = adapter;
        this.hasBC = adapter.features.has('texture-compression-bc');
        if (wantUnpack === false && !this.hasBC) {
            throw new Error('unpack=0 asks for hardware BC and this adapter has none '
                            + '(texture-compression-bc is not among its features).');
        }
        this.unpack = wantUnpack === null ? !this.hasBC : wantUnpack;
        // The feature is asked for whenever the adapter has it, WHATEVER the unpack field says: the field decides what
        // is uploaded, not what the device can do, so the checkbox can switch paths without a new device and without
        // losing the camera.
        const required = this.hasBC ? ['texture-compression-bc'] : [];
        // maxTextureDimension2D defaults to 8192 on the DEVICE however large the adapter's is, and ../shared/dds.h
        // accepts up to 16384, so the adapter's own limit is asked for and an asset past it is refused by name in
        // createTextureForFile rather than turning into a validation error nobody can read.
        this.device = await adapter.requestDevice({
            requiredFeatures: required,
            requiredLimits: { maxTextureDimension2D: adapter.limits.maxTextureDimension2D },
        });
        // A LOST DEVICE FREEZES THE PICTURE, and a frozen picture with nothing said reads as a hung page. The reason
        // goes in the status line, where every other failure goes: 'destroyed' is the page's own doing (a navigation,
        // or the tab being discarded) and needs no sentence, and anything else - a driver reset, a GPU removed, a
        // TDR - is worth naming, because the only cure is reloading the page and that is what it says.
        this.device.lost.then((info) => {
            if (info.reason === 'destroyed') return;
            const why = info.message ? ': ' + String(info.message).replace(/\s*\.\s*$/, '') : '';
            this.report('The WebGPU device was lost (' + info.reason + ')' + why
                        + '. The picture is frozen; reload the page to get a new one.');
        });
        // Every validation error WebGPU raises outside an error scope. Without this it is a console line and nothing
        // else, which is how a refused texture upload drew a wrong picture in silence. The message is Dawn's own,
        // because it names the call and the value that was wrong better than a paraphrase could.
        this.device.addEventListener('uncapturederror', (e) => {
            const said = e && e.error && e.error.message ? e.error.message : String(e && e.error);
            this.report('WebGPU refused a call: ' + said);
        });
        this.maxTextureDimension2D = this.device.limits.maxTextureDimension2D;
        this.info = adapter.info || {};
        this.context = this.canvas.getContext('webgpu');
        // Never an -srgb view: the latents are not colours and the decoder's output is written as it is
        // (../viewer/main.cpp:1014). getPreferredCanvasFormat returns bgra8unorm or rgba8unorm, neither of which is.
        this.colorFormat = navigator.gpu.getPreferredCanvasFormat();
        this.colorSpace = 'srgb';
        this.configureContext();
        this.depthFormat = 'depth24plus';
        this.uniformData = new Float32Array(U_FLOATS);
        this.uniformInts = new Int32Array(this.uniformData.buffer);
        if (this.uniformData.byteLength !== 2032 || U_W * 4 !== 224 || U_BIAS * 4 !== 1952) {
            throw new Error('the uniform block is ' + this.uniformData.byteLength + ' bytes with W at ' + (U_W * 4)
                            + ' and bias at ' + (U_BIAS * 4) + '; it must be 2032, 224 and 1952');
        }
        this.uniformBuffer = this.device.createBuffer({
            label: 'constants', size: 2032,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.createSamplers();
        this.quadIndex = this.device.createBuffer({ size: 24, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(this.quadIndex, 0, new Uint32Array([0, 1, 2, 0, 2, 3]));
        this.quadVertex = this.device.createBuffer({
            size: 4 * 5 * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        this.cubeVertex = this.device.createBuffer({
            size: CUBE_VERTS.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(this.cubeVertex, 0, CUBE_VERTS);
        this.cubeIndex = this.device.createBuffer({
            size: CUBE_INDICES.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(this.cubeIndex, 0, CUBE_INDICES);
        this.sphereVertex = this.device.createBuffer({
            size: SPHERE.verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(this.sphereVertex, 0, SPHERE.verts);
        this.sphereIndex = this.device.createBuffer({
            size: SPHERE.idx.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(this.sphereIndex, 0, SPHERE.idx);
        // A 1x1 stand-in for lat0b when level 0 is one file: the binding exists in the shader either way, and the
        // shader reads it only when sel.z is 2.
        this.dummy = this.device.createTexture({
            label: 'lat0b stand-in', size: [1, 1, 1], format: 'r8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.device.queue.writeTexture({ texture: this.dummy }, new Uint8Array(1), { bytesPerRow: 1 }, [1, 1, 1]);
    }

    configureContext() {
        this.context.configure({
            device: this.device, format: this.colorFormat, alphaMode: 'opaque', colorSpace: this.colorSpace,
        });
    }

    // The canvas colour space (the plan's 6.5), which is WebGL's drawingBufferColorSpace under another name: a
    // DISPLAY-SIDE relabelling that changes no pixel. It reaches nothing that is read back, because a saved frame
    // comes from the texture's own numbers. A browser or a display that will not take 'display-p3' is a message and
    // a fall back to 'srgb', which is what KTX2 Studio's control does, rather than a silent nothing.
    setColorSpace(space) {
        const previous = this.colorSpace;
        this.colorSpace = space;
        try {
            this.configureContext();
        } catch (e) {
            this.colorSpace = previous;
            this.configureContext();
            return 'this browser would not configure the canvas as ' + space + ': ' + (e.message || e)
                   + ' - it stays ' + previous + '.';
        }
        return '';
    }

    // The eight states of ../viewer/main.cpp:1051-1064, one for one: [mips on / off] x [point, bilinear, trilinear,
    // anisotropic]. MaxAnisotropy 8, clamp on every axis, no LOD bias anywhere (there is no such field here), and the
    // mips-off row is lodMaxClamp 0 - the sampler held at mip 0, nothing reloaded.
    createSamplers() {
        this.samplers = [[], []];
        const filters = [
            { mag: 'nearest', min: 'nearest', mip: 'nearest', aniso: 1 },
            { mag: 'linear', min: 'linear', mip: 'nearest', aniso: 1 },
            { mag: 'linear', min: 'linear', mip: 'linear', aniso: 1 },
            { mag: 'linear', min: 'linear', mip: 'linear', aniso: ANISO_MAX },
        ];
        for (let mips = 0; mips < 2; mips++) {
            for (const f of filters) {
                this.samplers[mips].push(this.device.createSampler({
                    addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge',
                    magFilter: f.mag, minFilter: f.min, mipmapFilter: f.mip,
                    maxAnisotropy: f.aniso, lodMinClamp: 0, lodMaxClamp: mips ? 32 : 0,
                }));
            }
        }
    }

    // The shader, compiled AND made into a pipeline before anything of it is kept. `Shift+R` promises what
    // ../viewer_d3d12/README.md promises of its own `R` - "both previous pipeline state objects are kept" when the
    // reload fails - and a shader that compiles is not the same thing as a shader that can be drawn with: rename
    // PSMain and every message is a warning, createRenderPipeline returns an INVALID pipeline rather than throwing,
    // every pass built from it is invalid, and the view goes black with an empty error line. So the pipeline is
    // built asynchronously, which is the form of the call that REJECTS, into a local; this.pipeline is assigned only
    // once it exists, and a failure leaves the previous one - and its bind groups - exactly where they were.
    async loadShader(url) {
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) throw new Error('cannot fetch ' + url + ' (' + response.status + ')');
        const code = await response.text();
        const module = this.device.createShaderModule({ label: url, code: code });
        const info = await module.getCompilationInfo();
        for (const m of info.messages) {
            if (m.type === 'error') throw new Error(url + ':' + m.lineNum + ': ' + m.message);
        }
        const descriptor = {
            label: 'scene',
            layout: 'auto',
            vertex: {
                module: module, entryPoint: 'VSMain',
                buffers: [{
                    arrayStride: 20,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x3' },
                        { shaderLocation: 1, offset: 12, format: 'float32x2' },
                    ],
                }],
            },
            fragment: { module: module, entryPoint: 'PSMain', targets: [{ format: this.colorFormat }] },
            // Cull nothing, as the native rasterizer state does, so the cube's inside faces draw too.
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            depthStencil: { format: this.depthFormat, depthWriteEnabled: true, depthCompare: 'less' },
        };
        let pipeline;
        try {
            pipeline = await this.device.createRenderPipelineAsync(descriptor);
        } catch (e) {
            throw new Error(url + ' compiles but will not make a pipeline: ' + (e.message || e));
        }
        this.pipeline = pipeline;
        this.bindGroups = null;   // an 'auto' layout belongs to its pipeline, so the bind groups are rebuilt with it
        if (this.asset) this.buildBindGroup();
        return code;
    }

    // One stored .dds file as a GPU texture with its whole chain. A block-compressed level's bytes go up as they are
    // (bytesPerRow is the BLOCK row's byte count, which writeTexture has no 256-byte alignment rule about); an
    // unpacked one goes up as w * channels. Every stored level is uploaded, because the sampler will fetch every one.
    //
    // THE COPY EXTENT OF A BLOCK-COMPRESSED LEVEL IS ITS PHYSICAL SIZE, NOT ITS LOGICAL ONE, and getting that wrong
    // is silent. WebGPU requires a compressed copy's width and height to be multiples of the block size, so a mip
    // whose extent is not - the 90x50, 45x25 and 22x12 levels of ../examples/npot360x200, which the encoder's
    // default --mip-min 8 produces from ANY non-power-of-two source (360 -> 180 -> 90, 1000 -> 500 -> 250 -> 125) -
    // is REFUSED outright with "copySize.width (90) is not a multiple of compressed texture format block width (4)".
    // Only the BASE has to be block-aligned, and descriptor.js refuses one that is not; the chain below it is free
    // to leave alignment and routinely does, which is why this is the ordinary case and not an exotic one. The level
    // then stays zero-initialised and the shader samples zeros there (s0 = 0, so z0 = lo0: a flat wrong colour at
    // every distance that reads it) while nothing at all is said. The stored bytes already cover the whole blocks -
    // ceil(w/4) * ceil(h/4) of them, which is what level.pitch and rowsPerImage above count - so rounding the extent
    // up to the block grid copies exactly those bytes and nothing more. The uncompressed path keeps the logical
    // extent, which is what it stores.
    createTextureForFile(file) {
        const img = file.img;
        if (img.W > this.maxTextureDimension2D || img.H > this.maxTextureDimension2D) {
            throw new Error("'" + file.name + "' is " + img.W + 'x' + img.H + ' and this adapter'
                            + ' will not make a texture larger than ' + this.maxTextureDimension2D);
        }
        const texture = this.device.createTexture({
            label: file.name,
            size: [img.W, img.H, 1],
            mipLevelCount: img.mips,
            format: file.boundFormat,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        for (let i = 0; i < img.mips; i++) {
            const level = img.levels[i];
            let data, bytesPerRow, rowsPerImage;
            let copyW = level.w, copyH = level.h;
            if (file.unpacked) {
                data = file.levelBytes[i];
                bytesPerRow = level.w * img.channels;
                rowsPerImage = level.h;
            } else {
                data = img.bytes.subarray(level.offset, level.offset + level.size);
                bytesPerRow = level.pitch;
                rowsPerImage = img.bc ? ((level.h + 3) >> 2) : level.h;
                if (img.bc) { copyW = (level.w + 3) & ~3; copyH = (level.h + 3) & ~3; }
            }
            this.device.queue.writeTexture(
                { texture: texture, mipLevel: i },
                data, { bytesPerRow: bytesPerRow, rowsPerImage: rowsPerImage },
                [copyW, copyH, 1]);
        }
        return texture;
    }

    // Everything the asset decides: the textures, the quad's aspect, and the decoder half of the uniform block.
    //
    // EACH FILE'S UPLOAD RUNS INSIDE ITS OWN VALIDATION ERROR SCOPE, which is why this is async. WebGPU's validation
    // is not a layer that can be switched on: every call is checked, always. What a page can get wrong is not
    // LISTENING - a createTexture or a writeTexture that is refused raises no JavaScript exception, it marks the
    // object invalid and writes one line to a console nobody is reading. That is how the compressed copy extent
    // above stayed hidden: the asset loaded, the panel said BC5 (GPU), and one mip was zeros.
    //
    // The scope is per FILE rather than one around the loop, because attribution is most of the value: "the upload
    // of npot360x200_lat0.dds was refused, copySize.width (90) is not a multiple of 4" names the plane to look at,
    // where one message for the whole asset names only that something went wrong. Inside the scope a refusal
    // becomes what every other refusal in this page already is - a thrown Error, the previous material left on
    // screen, and the sentence in the status line.
    async setAsset(asset) {
        // The new textures are made BEFORE the old ones are released, so that a refusal here - an asset past the
        // adapter's size limit is the one that can happen - leaves the material on screen where it was, which is the
        // rule every other refusal in this page follows.
        const made = [];
        try {
            for (const L of asset.levels) for (const f of L.files) {
                this.device.pushErrorScope('validation');
                let texture = null, thrown = null;
                try {
                    texture = this.createTextureForFile(f);
                } catch (e) {
                    thrown = e;
                }
                const refused = await this.device.popErrorScope();
                if (texture) made.push(texture);
                if (thrown) throw thrown;
                if (refused) {
                    const said = "'" + asset.name + "': WebGPU refused the upload of " + f.name + ' - '
                                 + refused.message;
                    // Through report() as well as thrown: the throw is what leaves the previous material on screen,
                    // and report() is what ?strict=1 listens to, so a refusal caught in a scope stops the page
                    // exactly as an uncaptured one does rather than being the one kind that slips through.
                    this.report(said);
                    throw new Error(said);
                }
            }
        } catch (e) {
            for (const t of made) t.destroy();
            throw e;
        }
        for (const t of this.textures) t.destroy();
        this.textures = made;
        // t0 level 0, t1 level 1, t2 level 0's channels past the first two when they live in a texture of their own.
        const l0 = asset.levels[0], l1 = asset.levels[1];
        this.viewLat0 = made[0].createView();
        this.viewLat0b = l0.files.length > 1 ? made[1].createView() : this.dummy.createView();
        this.viewLat1 = made[l0.files.length].createView();
        this.level0Textures = l0.files.length > 1 ? 2 : 0;   // sel.z, as ../viewer/main.cpp writes it
        this.asset = asset;
        this.device.queue.writeBuffer(this.quadVertex, 0, quadVerts(l0.W / l0.H));
        // The decoder constants, written once per asset: the dequantisation, the shape, the weights and the biases.
        const u = this.uniformData, ui = this.uniformInts;
        u.fill(0, U_LO0, U_FLOATS);
        for (let c = 0; c < 4; c++) {
            u[U_LO0 + c] = l0.lo[c]; u[U_HI0 + c] = l0.hi[c];
            u[U_LO1 + c] = l1.lo[c]; u[U_HI1 + c] = l1.hi[c];
        }
        const d = asset.decoder;
        ui[U_DIMS] = d.C0; ui[U_DIMS + 1] = d.C1; ui[U_DIMS + 2] = d.nin; ui[U_DIMS + 3] = d.nout;
        ui[U_SEL] = d.mask; ui[U_SEL + 1] = 0; ui[U_SEL + 2] = this.level0Textures; ui[U_SEL + 3] = 0;
        // W row-major: row r, column c at W[r * 6 + c / 4][c % 4], which is float U_W + (r * 6 + (c >> 2)) * 4 + (c & 3).
        for (let r = 0; r < d.nout; r++) {
            for (let c = 0; c < d.nin; c++) {
                u[U_W + (r * 6 + (c >> 2)) * 4 + (c & 3)] = d.weights[r * d.nin + c];
            }
        }
        for (let r = 0; r < d.nout; r++) u[U_BIAS + (r >> 2) * 4 + (r & 3)] = d.bias[r];
        this.buildBindGroup();
    }

    buildBindGroup() {
        this.bindGroups = [[], []];
        for (let mips = 0; mips < 2; mips++) {
            for (let f = 0; f < 4; f++) {
                this.bindGroups[mips].push(this.device.createBindGroup({
                    layout: this.pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.uniformBuffer } },
                        { binding: 1, resource: this.viewLat0 },
                        { binding: 2, resource: this.viewLat1 },
                        { binding: 3, resource: this.viewLat0b },
                        // s0 level 0, s1 level 1: the SAME state, as ../viewer/main.cpp:1120 binds them. Level 1's
                        // LOD shift is const0.z, not a sampler field, so it composes with every filter for free.
                        { binding: 4, resource: this.samplers[mips][f] },
                        { binding: 5, resource: this.samplers[mips][f] },
                    ],
                }));
            }
        }
    }

    // The canvas's backing size, from its CSS size times devicePixelRatio: the canvas is sized by script and never
    // scaled by CSS.
    resize(width, height) {
        width = Math.max(1, Math.min(width, this.maxTextureDimension2D));
        height = Math.max(1, Math.min(height, this.maxTextureDimension2D));
        if (width === this.width && height === this.height) return false;
        this.width = width;
        this.height = height;
        this.canvas.width = width;
        this.canvas.height = height;
        if (this.depthTexture) this.depthTexture.destroy();
        this.depthTexture = this.device.createTexture({
            label: 'depth', size: [width, height, 1], format: this.depthFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        return true;
    }

    // The scene half of the uniform block, written every frame, exactly as set_uniforms does.
    // `place` is the all-textures row: an X offset in world units and which texture this object shows. Left
    // out, it is the single object the viewer has always drawn, at the camera's own position and state.texShown.
    setUniforms(state, place) {
        const u = this.uniformData, ui = this.uniformInts, a = this.asset;
        const proj = matPerspective(FOV_DEGREES, this.width / this.height, 0.001, 100.0);
        // THE OFFSET GOES OUTSIDE THE ROTATIONS, so each object turns in place and the row stays spread across
        // the view. Inside them the row is one rigid group that swings about its own centre, and a quarter turn -
        // which the orbit reaches every three seconds - leaves it edge-on with the objects hiding one another,
        // which defeats the point of drawing them side by side.
        let model = matMul(matRotY(state.yaw), matRotX(state.pitch));
        if (place) model = matMul(matTranslate(place.x, 0, 0), model);
        model = matMul(matTranslate(state.x, state.y, state.z), model);
        const mvp = matTranspose(matMul(proj, model));
        u.set(mvp, U_MVP);
        u[U_TEXSIZE] = a.levels[0].W; u[U_TEXSIZE + 1] = a.levels[0].H;
        u[U_TEXSIZE + 2] = a.levels[1].W; u[U_TEXSIZE + 3] = a.levels[1].H;
        u[U_LODINFO] = a.levels[0].mips - 1; u[U_LODINFO + 1] = a.levels[1].mips - 1;
        u[U_CONST0] = state.raw0 ? 1 : 0;
        u[U_CONST0 + 1] = state.raw1 ? 1 : 0;
        // const0.z: the factor the shader multiplies level 1's UV derivatives by before its sample. 2^lod_bias_level1
        // raises the hardware's LOD by that many mips, which is the encoder's 1:1 rule; off leaves the gradients
        // alone, which is the control. Written here every frame, so nothing can leave a zero scale in the slot.
        u[U_CONST0 + 2] = state.lodBias ? Math.pow(2, a.lodBiasLevel1) : 1.0;
        u[U_CONST0 + 3] = state.renorm ? 1 : 0;
        // const1.x: which channel of an as-stored view to draw. Written every frame like the rest, so the two
        // raw views and this control can never disagree about what is on screen.
        u[U_CONST1] = CHANNELS.indexOf(state.channel) < 0 ? 0 : CHANNELS.indexOf(state.channel);
        u[U_CONST1 + 1] = 0; u[U_CONST1 + 2] = 0; u[U_CONST1 + 3] = 0;
        ui[U_SEL + 1] = place ? place.tex : state.texShown;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
    }


    // WHERE THE OBJECTS GO. One when the row is off, and one per output texture when it is on, spread along X
    // about the middle so the camera keeps pointing at the group rather than at its first member.
    //
    // `N` ROTATES THE ASSIGNMENT rather than picking one of them: with the row on there is no single "shown"
    // texture to cycle, so state.texShown becomes the offset, and texture (i + texShown) % count lands on object
    // i. Pressing N therefore slides the whole material along the row, which is what makes it possible to compare
    // any two textures side by side.
    places(state) {
        const count = this.asset ? this.asset.texturesOut : 1;
        if (!state.allTextures || count < 2) return [null];
        const gap = state.geom === 'quad' ? 2.4 : 1.6;
        const out = [];
        for (let i = 0; i < count; i++) {
            out.push({ x: (i - (count - 1) / 2) * gap, tex: (i + state.texShown) % count });
        }
        return out;
    }

    // THE SCENE DRAW, and there is ONE of it. The canvas frame and the captured frame must be the same picture or
    // the gate arm that compares a capture against a native viewer proves nothing about what is on screen: an edit
    // to a pipeline, a bind-group slot or a draw call that reached only one of two copies would leave the check
    // green while the canvas diverged, which is the very failure the arm exists to catch.
    drawScene(pass, state) {
        const slot = (state.filterMode === 2 && state.aniso) ? 3 : state.filterMode;
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroups[state.mipsOn ? 1 : 0][slot]);
        if (state.geom === 'cube') {
            pass.setVertexBuffer(0, this.cubeVertex);
            pass.setIndexBuffer(this.cubeIndex, 'uint32');
            pass.drawIndexed(CUBE_INDICES.length);
        } else if (state.geom === 'sphere') {
            pass.setVertexBuffer(0, this.sphereVertex);
            pass.setIndexBuffer(this.sphereIndex, 'uint32');
            pass.drawIndexed(SPHERE.idx.length);
        } else {
            pass.setVertexBuffer(0, this.quadVertex);
            pass.setIndexBuffer(this.quadIndex, 'uint32');
            pass.drawIndexed(6);
        }
    }

    // ------------------------------------------------------------------
    // THE CAPTURED FRAME (the plan's 8.1)
    //
    // A frame drawn away from the canvas, at a size of its own, with NO OVERLAY - which is what
    // `nntc_view --shot --nooverlay` produces, and the only thing worth comparing it against. The canvas keeps
    // whatever it was showing; nothing here touches it.
    //
    // IT RENDERS IN THE CANVAS'S OWN FORMAT rather than choosing rgba8unorm, because the pipeline was built for
    // that format and a colour attachment whose format disagrees with the pipeline is a validation error.
    // getPreferredCanvasFormat answers bgra8unorm on Windows, and BGRA is the order a .bmp wants anyway, so the
    // caller is told which it got rather than handed a silent guess.
    //
    // copyTextureToBuffer wants its rows padded to 256 bytes, which is a device rule and not a choice, so the
    // padding is taken off here and the caller gets width * 4 with nothing between the rows.
    async capture(state, width, height) {
        if (!this.asset || !this.pipeline || !this.bindGroups) throw new Error('nothing is loaded to capture');
        width = Math.max(1, Math.min(Math.round(width), this.maxTextureDimension2D));
        height = Math.max(1, Math.min(Math.round(height), this.maxTextureDimension2D));
        const target = this.device.createTexture({
            label: 'capture', size: [width, height, 1], format: this.colorFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        const depth = this.device.createTexture({
            label: 'capture depth', size: [width, height, 1], format: this.depthFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        const rowBytes = width * 4;
        const padded = (rowBytes + 255) & ~255;
        const readback = this.device.createBuffer({
            label: 'capture readback', size: padded * height,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        // The projection comes from this.width/this.height, so the capture borrows them for the length of one
        // frame: a 2560x1421 shot must have 2560x1421's aspect and not the panel-sized canvas's.
        const wasW = this.width, wasH = this.height;
        this.width = width; this.height = height;
        try {
            // The same loop render() uses, for the same reason, so a capture of the all-textures row is the row.
            const view = target.createView();
            const places = this.places(state);
            for (let i = 0; i < places.length; i++) {
                this.setUniforms(state, places[i]);
                const first = i === 0;
                const draw = this.device.createCommandEncoder();
                const pass = draw.beginRenderPass({
                    colorAttachments: [{
                        view: view,
                        clearValue: { r: 0.2, g: 0.2, b: 0.2, a: 1.0 },
                        loadOp: first ? 'clear' : 'load', storeOp: 'store',
                    }],
                    depthStencilAttachment: {
                        view: depth.createView(),
                        depthClearValue: 1.0,
                        depthLoadOp: first ? 'clear' : 'load', depthStoreOp: 'store',
                    },
                });
                this.drawScene(pass, state);
                pass.end();
                this.device.queue.submit([draw.finish()]);
            }
            const encoder = this.device.createCommandEncoder();
            encoder.copyTextureToBuffer({ texture: target },
                                        { buffer: readback, bytesPerRow: padded, rowsPerImage: height },
                                        { width: width, height: height, depthOrArrayLayers: 1 });
            this.device.queue.submit([encoder.finish()]);
            // BOTH WAITS, in this order: the work has to finish before the buffer can be mapped, and the map has
            // to resolve before there are bytes to read. Skipping either hands back a frame from before the draw
            // some of the time, which is the worst kind of capture - one that is usually right.
            await this.device.queue.onSubmittedWorkDone();
            await readback.mapAsync(GPUMapMode.READ);
            const src = new Uint8Array(readback.getMappedRange());
            const out = new Uint8Array(rowBytes * height);
            for (let y = 0; y < height; y++) {
                out.set(src.subarray(y * padded, y * padded + rowBytes), y * rowBytes);
            }
            readback.unmap();
            return { width: width, height: height, format: this.colorFormat, pixels: out };
        } finally {
            // RESTORE ONLY IF NOBODY ELSE MOVED IT. A resize can land during the two awaits above - the observer
            // fires on a window or divider drag - and it writes this.width, the canvas and the depth texture to
            // the new size. Putting the old numbers back then would leave every later frame building its
            // projection, and placing its overlay, from a size the canvas no longer has: a stretched picture that
            // lasts until the next size CHANGE, since the observer does not fire twice for the same size.
            if (this.width === width && this.height === height) { this.width = wasW; this.height = wasH; }
            readback.destroy();
            depth.destroy();
            target.destroy();
            // The uniform block is still holding the capture's projection. The next frame overwrites it, but a
            // page that captured and then did not redraw would keep it, so put the canvas's back now.
            this.setUniforms(state);
        }
    }

    // One frame. `overlay` is the Overlay or null; it draws into the same pass after the scene, as the native viewers
    // draw their strip.
    render(state, overlay) {
        if (!this.asset || !this.pipeline || !this.bindGroups || !this.depthTexture) return;
        // ONE PASS PER OBJECT, because the uniform block holds one object's matrices and one texture index and
        // queue.writeBuffer is ordered against submits rather than against draws inside a pass. With the row off
        // this is exactly the one pass it always was; with it on it is one per texture, which is at most a
        // handful. The first clears the target, the rest load it, so the objects share one depth buffer and
        // occlude each other correctly.
        const view = this.context.getCurrentTexture().createView();
        const places = this.places(state);
        for (let i = 0; i < places.length; i++) {
            this.setUniforms(state, places[i]);
            const first = i === 0;
            const encoder = this.device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: view,
                    clearValue: { r: 0.2, g: 0.2, b: 0.2, a: 1.0 },   // the native clear colour
                    loadOp: first ? 'clear' : 'load', storeOp: 'store',
                }],
                depthStencilAttachment: {
                    view: this.depthTexture.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: first ? 'clear' : 'load', depthStoreOp: 'store',
                },
            });
            this.drawScene(pass, state);
            // The strip goes over the LAST object only, or it would be drawn once per object into the same pixels.
            if (overlay && i === places.length - 1) overlay.draw(pass, this.width, this.height);
            pass.end();
            this.device.queue.submit([encoder.finish()]);
        }
    }
}
