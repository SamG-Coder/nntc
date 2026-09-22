// SPDX-License-Identifier: Apache-2.0
import {GpuRuntime} from '../../vendor/cuda-webshader/src/runtime/runtime.js';
import {NormalEquations} from '../../vendor/cuda-webshader/src/runtime/least-squares.js';
import {prepare,validateInputs} from './prepare.js';
import {ranges,quantize,packBC,makeDescriptor,writeDDS} from './format.js';
import {buildAsset} from '../descriptor.js';

const TILE=4096;
const tick=()=>new Promise(resolve=>setTimeout(resolve,0));
const abort=signal=>{if(signal?.aborted) throw new DOMException('Encoding cancelled.','AbortError');};

export class BrowserEncoder {
    static async create({runtime}={}) {
        const own=!runtime;
        runtime ||= await GpuRuntime.create({useAdapterBufferLimits:true});
        try {
            const response=await fetch(new URL('./kernels.cu',import.meta.url));
            if(!response.ok) throw new Error(`Cannot load encoder kernels: HTTP ${response.status}`);
            const source=await response.text(),kernels={};
            for(const entry of ['make_rows','update_latent','decode']) kernels[entry]=await runtime.kernel(source,{entry,workgroupSize:[64,1,1]});
            return new BrowserEncoder(runtime,kernels,own);
        } catch(error) {if(own) runtime.dispose();throw error;}
    }
    constructor(runtime,kernels,own) {this.runtime=runtime;this.kernels=kernels;this.own=own;this.busy=false;}
    async dispose() {
        if(this.busy) throw new Error('Wait for the encode or cancellation before disposing.');
        await this.runtime.idle();if(this.own) this.runtime.dispose();
    }
    async encode(images,{name='material',c0=2,c1=4,iterations=6,sites=5,mips=true,bc=true,signal,onProgress=()=>{}}={}) {
        if(this.busy) throw new Error('An encode is already running.');
        const {width,height}=validateInputs(images,{c0,c1,iterations,sites});
        if(typeof name!=='string'||!name.length||!/^[\w.-]+$/.test(name)||name==='.'||name==='..') throw new Error('Output name must contain letters, digits, dots, underscores or hyphens.');
        const rt=this.runtime,limits=rt.device.limits,w=Math.ceil(width/4)*4,h=Math.ceil(height/4)*4,nout=images.length*3;
        const largest=w*h*Math.max(nout,c0)*4;
        if(largest>Math.min(limits.maxStorageBufferBindingSize,limits.maxBufferSize)||Math.ceil(w*h/64)>limits.maxComputeWorkgroupsPerDimension) throw new Error('Images exceed this adapter’s buffer or dispatch limits; reduce their dimensions.');
        // Includes source chain, both latent ping-pong buffers, host preparation,
        // readback and reconstruction; refuse before allocating those arrays.
        if(largest*5+w*h*(c0+c1)*16>1024**3) throw new Error('This browser encoder limits working memory to 1 GiB; reduce image dimensions.');
        this.busy=true;
        const resources=new Set(),alloc=data=>{const b=rt.createBuffer(data);resources.add(b);return b;};
        const report=async message=>{abort(signal);onProgress(message);await tick();abort(signal);};
        const launch=(entry,buffers,scalars,count)=>rt.batch().dispatch(this.kernels[entry].bind(buffers,scalars),[Math.ceil(count/64),1,1]).submit();
        try {
            await report('Preparing source mip chain and shared PCA basis…');
            const model=prepare(images,{c0,c1,iterations,sites,mips}),planes=model.planes,mm=c0+c1+c0*c1+1;
            const decoder=alloc(mm*nout*4),rows=alloc(TILE*(mm+nout)*4);
            for(const p of planes) {
                p.src=alloc(p.source);p.l0=alloc(p.values0);p.l1=alloc(p.values1);
                p.scratch0=alloc(p.values0.byteLength);p.scratch1=alloc(p.values1.byteLength);
                p.scalars={w:p.w,h:p.h,w1:p.w1,h1:p.h1,c0,c1,nout};
            }
            let weights;
            const fit=async()=>{
                const equations=await NormalEquations.create(rt,mm,nout);
                try {
                    for(const p of planes) for(let start=0;start<p.w*p.h*sites;start+=TILE) {
                        abort(signal);
                        const count=Math.min(TILE,p.w*p.h*sites-start);
                        // Equal per-pixel weight across stored planes, matching
                        // the default pixel-proportional contribution of mips.
                        launch('make_rows',{source:p.src,latent0:p.l0,latent1:p.l1,rows},{...p.scalars,sites,start,count,weight:1},count);
                        equations.append(rows,count);
                        if(start%(TILE*16)===0) {await rt.idle();await tick();}
                    }
                    weights=await equations.solve();rt.write(decoder,weights);
                } finally {await rt.idle();equations.dispose();}
            };
            await report('Fitting the initial decoder…');await fit();
            for(let iteration=0;iteration<iterations;iteration++) {
                await report(`Optimizing latents: pass ${iteration+1} of ${iterations}…`);
                for(const p of planes) for(const level of [0,1]) for(let color=0;color<4;color++) {
                    abort(signal);
                    const key=level?'l1':'l0',scratch=level?'scratch1':'scratch0';
                    launch('update_latent',{source:p.src,latent0:p.l0,latent1:p.l1,decoder,result:p[scratch]},
                        {...p.scalars,sites,level,color},level?p.w1*p.h1:p.w*p.h);
                    [p[key],p[scratch]]=[p[scratch],p[key]];
                    await rt.idle();await tick();
                }
                await fit();
            }
            await report('Quantizing the latent mip chains and packing DDS blocks…');
            for(const p of planes) {p.values0=await rt.read(p.l0);p.values1=await rt.read(p.l1);}
            const range0=ranges(planes.map(p=>p.values0),c0),range1=ranges(planes.map(p=>p.values1),c1);
            for(const p of planes) {
                abort(signal);
                if(bc) {const packed=packBC(p.values0,p.w,p.h,c0,range0.lo,range0.hi);p.payload0=packed.payloads;p.values0=packed.decoded;}
                else {const packed=quantize(p.values0,c0,range0.lo,range0.hi);p.payload0=[packed.bytes];p.values0=packed.decoded;}
                const packed=quantize(p.values1,c1,range1.lo,range1.hi);p.payload1=packed.bytes;p.values1=packed.decoded;
                rt.write(p.l0,p.values0);rt.write(p.l1,p.values1);await tick();
            }
            await report('Refitting the decoder against the stored, quantized representation…');await fit();
            const descriptor=makeDescriptor({name,width,height,inputs:images.map((image,i)=>image.name||`input${i}.png`),c0,c1,planes,range0,range1,weights,bc});
            const files=new Map();
            for(let level=0;level<2;level++) {
                const texture=descriptor.textures[level];
                (texture.files||[texture]).forEach((entry,index)=>files.set(entry.file,writeDDS(texture.width,texture.height,entry.dxgi_format_id,entry.channels_stored,
                    planes.map(p=>level?p.payload1:p.payload0[index]))));
            }
            const descriptorName=`${name}_nntc.json`;
            // Independent, existing viewer parser checks each DDS header and model.
            buildAsset(descriptorName,descriptor,files,true);
            files.set(descriptorName,new TextEncoder().encode(JSON.stringify(descriptor,null,2)+'\n'));
            await report('Measuring the exported representation…');
            const base=planes[0],output=alloc(base.source.byteLength);
            launch('decode',{latent0:base.l0,latent1:base.l1,decoder,output},base.scalars,base.w*base.h);
            const reconstructed=await rt.read(output),psnr=[];
            if(reconstructed.some(v=>!Number.isFinite(v))) throw new Error('Non-finite reconstruction; the encode was not exported.');
            for(let t=0;t<images.length;t++) {
                let error=0;
                for(let y=0;y<height;y++) for(let x=0;x<width;x++) for(let c=0;c<3;c++) {const i=(y*base.w+x)*nout+t*3+c;error+=(reconstructed[i]-base.source[i])**2;}
                psnr.push(error===0?Infinity:-10*Math.log10(error/(width*height*3)));
            }
            abort(signal);
            return {files,descriptor,descriptorName,psnr,reconstructed,width,height,paddedWidth:base.w,paddedHeight:base.h,nout};
        } finally {
            try {await rt.idle();} finally {for(const resource of resources) rt.destroyBuffer(resource);this.busy=false;}
        }
    }
}
