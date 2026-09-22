import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {compile} from '../vendor/cuda-webshader/src/compiler/compiler.js';
import {packBC,quantize,writeDDS,makeDescriptor,ranges} from '../webgpu/encoder/format.js';
import {prepare,validateInputs} from '../webgpu/encoder/prepare.js';
import {ddsRead} from '../webgpu/dds.js';
import {unpackImage} from '../webgpu/bcdec.js';
import {buildAsset} from '../webgpu/descriptor.js';
import {zipFiles} from '../webgpu/encoder/zip.js';

test('all browser CUDA entries compile through cuda-webshader',async()=>{
 const source=await readFile(new URL('../webgpu/encoder/kernels.cu',import.meta.url),'utf8');
 for(const entry of ['make_rows','update_latent','decode']) assert.match(compile(source,{entry,workgroupSize:[64,1,1]}).wgsl,/@compute/);
});
test('BC packing round-trips partial blocks, constant channels and split BC5/BC4 planes',()=>{
 for(let channels=1;channels<=4;channels++) {
  const w=7,h=5,values=Float32Array.from({length:w*h*channels},(_,i)=>i%channels===0?0.25:((i*17)%256)/255);
  const {payloads,decoded}=packBC(values,w,h,channels,Array(channels).fill(0),Array(channels).fill(1));
  let firstChannel=0;
  for(const payload of payloads) {
   const count=Math.min(2,channels-firstChannel),bytes=writeDDS(w,h,count===1?80:83,count,[payload]),dds=ddsRead('test.dds',bytes),unpacked=unpackImage(dds)[0];
   assert.equal(new DataView(bytes.buffer).getUint32(24,true),1,'upstream Python decoder requires DDS depth=1');
   for(let i=0;i<w*h;i++)for(let c=0;c<count;c++)assert.ok(Math.abs(unpacked[i*count+c]/255-decoded[i*channels+firstChannel+c])<=0.501/255);
   firstChannel+=count;
  }
 }
});
test('every C0/C1 layout produces descriptors accepted by the existing viewer',()=>{
 for(let c0=1;c0<=4;c0++)for(let c1=1;c1<=4;c1++)for(const bc of [false,true]) {
  const planes=[{w:16,h:16,w1:4,h1:4},{w:8,h:8,w1:2,h1:2}],r0={lo:Array(c0).fill(-1),hi:Array(c0).fill(1)},r1={lo:Array(c1).fill(-1),hi:Array(c1).fill(1)};
  const d=makeDescriptor({name:'test',width:16,height:16,inputs:['a.png','b.png'],c0,c1,planes,range0:r0,range1:r1,weights:new Float32Array((c0+c1+c0*c1+1)*6),bc}),files=new Map();
  for(let level=0;level<2;level++)for(const e of d.textures[level].files||[d.textures[level]]) {
   const compressed=e.dxgi_format_id===80||e.dxgi_format_id===83,texture=d.textures[level];
   files.set(e.file,writeDDS(texture.width,texture.height,e.dxgi_format_id,e.channels_stored,texture.mip_sizes.map(([w,h])=>new Uint8Array(compressed?Math.ceil(w/4)*Math.ceil(h/4)*e.channels_stored*8:w*h*e.channels_stored))));
  }
  assert.equal(buildAsset('test_nntc.json',d,files,true).texturesOut,2);
 }
});
test('preparation clamps padding and generates matching DDS mip dimensions',()=>{
 const width=35,height=19,data=new Uint8Array(width*height*4).fill(127);
 const result=prepare([{width,height,data}],{c0:3,c1:4});
 assert.equal(result.planes[0].w,36);assert.equal(result.planes[0].h,20);
 for(const [i,p] of result.planes.entries()) {
  assert.ok(p.values0.every(Number.isFinite));assert.ok(p.values1.every(Number.isFinite));
  assert.equal(p.w1,Math.max(1,result.planes[0].w1>>i));assert.equal(p.h1,Math.max(1,result.planes[0].h1>>i));
  assert.ok(p.source.every(v=>Math.abs(v-127/255)<1e-6));
 }
});
test('reject malformed inputs, nonfinite latents and incorrect DDS payloads',()=>{
 assert.throws(()=>validateInputs([]),/one to six/);
 const image={width:1,height:1,data:new Uint8Array(4)};
 assert.throws(()=>validateInputs([image],{c0:5}),/1..4/);
 assert.throws(()=>validateInputs([image],{iterations:-1}),/Iterations/);
 assert.throws(()=>validateInputs([image,{...image,width:2}]),/same dimensions/);
 assert.throws(()=>ranges([new Float32Array([NaN])],1),/non-finite/);
 assert.throws(()=>writeDDS(4,4,80,1,[new Uint8Array(7)]),/payload/);
 const r=ranges([new Float32Array([0,0])],1);assert.ok(r.hi[0]>r.lo[0]);
 assert.ok(quantize(new Float32Array([0,0]),1,r.lo,r.hi).decoded.every(Number.isFinite));
});
test('ZIP contains standard headers, correct CRC32 and intact bytes',async()=>{
 const data=new TextEncoder().encode('123456789'),zip=new Uint8Array(await zipFiles(new Map([['test.txt',data]])).arrayBuffer()),v=new DataView(zip.buffer);
 assert.equal(v.getUint32(0,true),0x04034b50);assert.equal(v.getUint32(14,true),0xcbf43926);
 assert.deepEqual(zip.slice(38,47),data);assert.equal(v.getUint32(47,true),0x02014b50);
 assert.equal(v.getUint32(zip.length-22,true),0x06054b50);assert.equal(v.getUint16(zip.length-12,true),1);
});
