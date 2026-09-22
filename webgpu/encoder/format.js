// SPDX-License-Identifier: Apache-2.0
import {dxgiName} from '../dds.js';

export function writeDDS(width, height, dxgi, channels, levels) {
    const bc = dxgi === 80 || dxgi === 83;
    let w = width, h = height;
    for (const data of levels) {
        const expected = bc ? Math.ceil(w/4)*Math.ceil(h/4)*channels*8 : w*h*channels;
        if (data.byteLength !== expected) throw new Error('DDS mip payload has the wrong size.');
        w = Math.max(1,w>>1); h = Math.max(1,h>>1);
    }
    const bytes = new Uint8Array(148 + levels.reduce((n,data)=>n+data.byteLength,0));
    const header = new DataView(bytes.buffer);
    const set = (i,v)=>header.setUint32(i*4,v,true);
    set(0,0x20534444); set(1,124); set(2,0x1007 | (bc?0x80000:8) | (levels.length>1?0x20000:0));
    set(3,height); set(4,width); set(5,bc?levels[0].length:width*channels); set(6,1); set(7,levels.length);
    set(19,32); set(20,4); set(21,0x30315844);
    set(27,0x1000 | (levels.length>1?0x400008:0));
    set(32,dxgi); set(33,3); set(35,1);
    let offset = 148;
    for (const data of levels) { bytes.set(data,offset); offset += data.byteLength; }
    return bytes;
}

// Endpoint / selector least squares in each BC4 channel. The final decoder is
// refitted against these exact standard palette values, not the prepack plane.
export function packBC(values, w, h, channels, lo, hi) {
    const bx = Math.ceil(w/4), by = Math.ceil(h/4);
    const groups = channels > 2 ? [2,channels-2] : [channels];
    const payloads = groups.map(n=>new Uint8Array(bx*by*n*8));
    const decoded = new Float32Array(values.length);
    for (let y = 0; y < by; y++) for (let x = 0; x < bx; x++) for (let ch = 0; ch < channels; ch++) {
        const samples = new Float64Array(16), range = hi[ch]-lo[ch];
        for (let t = 0; t < 16; t++) {
            const px = Math.min(w-1,x*4+(t&3)), py = Math.min(h-1,y*4+(t>>2));
            samples[t] = Math.max(0,Math.min(255,(values[(py*w+px)*channels+ch]-lo[ch])/range*255));
        }
        let a = Math.round(Math.max(...samples)), b = Math.round(Math.min(...samples));
        let best = null;
        for (let iteration = 0; iteration < 5; iteration++) {
            if (a <= b) { if (b < 255) a = b+1; else b = a-1; }
            const palette = [a,b,...Array.from({length:6},(_,i)=>((6-i)*a+(i+1)*b)/7)];
            const indices = new Uint8Array(16);
            let error=0, aa=0, ab=0, bb=0, ay=0, byValue=0;
            for (let t = 0; t < 16; t++) {
                let pick=0;
                for (let k=1;k<8;k++) if (Math.abs(samples[t]-palette[k]) < Math.abs(samples[t]-palette[pick])) pick=k;
                indices[t]=pick; error+=(samples[t]-palette[pick])**2;
                const u=pick===0?1:pick===1?0:(8-pick)/7, v=1-u;
                aa+=u*u; ab+=u*v; bb+=v*v; ay+=u*samples[t]; byValue+=v*samples[t];
            }
            if (!best || error < best.error) best={a,b,indices,palette,error};
            const determinant=aa*bb-ab*ab;
            if (determinant<1e-9) break;
            a=Math.max(0,Math.min(255,Math.round((ay*bb-byValue*ab)/determinant)));
            b=Math.max(0,Math.min(255,Math.round((byValue*aa-ay*ab)/determinant)));
        }
        const group=ch<2?0:1, gc=groups[group], local=ch%2, bytes=payloads[group];
        const offset=((y*bx+x)*gc+local)*8;
        bytes[offset]=best.a; bytes[offset+1]=best.b;
        for(let t=0;t<16;t++) {
            const bit=t*3, byte=offset+2+(bit>>3), shift=bit&7;
            bytes[byte] |= best.indices[t]<<shift;
            if (shift>5) bytes[byte+1] |= best.indices[t]>>(8-shift);
            const px=x*4+(t&3), py=y*4+(t>>2);
            if(px<w && py<h) decoded[(py*w+px)*channels+ch]=lo[ch]+best.palette[best.indices[t]]/255*range;
        }
    }
    return {payloads,decoded};
}

export function ranges(planes, channels) {
    const lo=Array(channels).fill(Infinity),hi=Array(channels).fill(-Infinity);
    for(const data of planes) for(let i=0;i<data.length;i++) {
        if (!Number.isFinite(data[i])) throw new Error('The fit produced a non-finite latent value.');
        const c=i%channels; lo[c]=Math.min(lo[c],data[i]);hi[c]=Math.max(hi[c],data[i]);
    }
    for(let c=0;c<channels;c++) if(hi[c]-lo[c]<1e-6) hi[c]=lo[c]+1e-6;
    return {lo,hi};
}

export function quantize(values, channels, lo, hi) {
    const stored=channels===3?4:channels, bytes=new Uint8Array(values.length/channels*stored), decoded=new Float32Array(values.length);
    for(let i=0;i<values.length/channels;i++) for(let c=0;c<channels;c++) {
        const k=Math.max(0,Math.min(255,Math.round((values[i*channels+c]-lo[c])/(hi[c]-lo[c])*255)));
        bytes[i*stored+c]=k; decoded[i*channels+c]=lo[c]+k/255*(hi[c]-lo[c]);
    }
    return {bytes,decoded,stored};
}

export function makeDescriptor({name,width,height,inputs,c0,c1,planes,range0,range1,weights,bc}) {
    const nin=c0+c1+c0*c1, mm=nin+1, nout=inputs.length*3;
    const texture=(level,channels,range)=>({level,width:planes[0][level?'w1':'w'],height:planes[0][level?'h1':'h'],
        channels_used:channels,bits_per_channel:Array(channels).fill(8),mip_count:planes.length,
        mip_sizes:planes.map(p=>[p[level?'w1':'w'],p[level?'h1':'h']]),
        dequantise:{kind:'range',lo:range.lo,hi:range.hi,levels:255}});
    const first=texture(0,c0,range0),second=texture(1,c1,range1);
    const entry=(file,channels,compressed)=>({file,channels_stored:channels,dxgi_format_id:compressed?(channels===1?80:83):channels===1?61:channels===2?49:28});
    if(bc && c0>2) first.files=[entry(`${name}_lat0a.dds`,2,true),entry(`${name}_lat0b.dds`,c0-2,true)];
    else Object.assign(first,entry(`${name}_lat0.dds`,bc?c0:c0===3?4:c0,bc));
    Object.assign(second,entry(`${name}_lat1.dds`,c1===3?4:c1,false));
    for(const t of [first,second]) for(const e of t.files||[t]) e.dxgi_format=dxgiName(e.dxgi_format_id);
    if(bc) first.bc_palette='standard';
    const matrix=[],bias=[];
    for(let o=0;o<nout;o++) { matrix.push(...weights.slice(o*mm,o*mm+nin)); bias.push(weights[o*mm+nin]); }
    return {format:'nntc-dds-1',encoder:{name:'nntc-cuda-webshader',version:1,optimizer:'four-color coordinate descent; f32 GPU normal equations; f64 host solve'},
        source:{width,height,inputs:inputs.map(file=>({file,filter:'box',srgb:false,edge:'clamp',normal_map:false,weight:1,rgb_weights:[1,1,1]}))},
        decode:{width:planes[0].w,height:planes[0].h,textures_out:inputs.length},block:4,lod_bias_level1:2,
        textures:[first,second],decoder:{type:'bilinear',terms:'a b sc',C0:c0,C1:c1,nin,nout,hidden:[],output:'identity',
            layers:[{rows:nout,cols:nin,weights:matrix,bias}]}};
}
