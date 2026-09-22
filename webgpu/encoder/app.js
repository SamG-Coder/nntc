// SPDX-License-Identifier: Apache-2.0
import {BrowserEncoder} from './encoder.js';
import {zipFiles} from './zip.js';
const $=id=>document.getElementById(id);
let images=[],result=null,controller=null,encoder=null,elapsed=0,previewURLs=[];
const status=(message,error=false)=>{$('status').textContent=message;document.body.classList.toggle('error',error);};
async function loadImages(files) {
    if(controller) return;
    if(files.length<1||files.length>6) throw new Error('Choose one to six images.');
    const next=[];
    for(const file of files) {
        const bitmap=await createImageBitmap(file,{colorSpaceConversion:'none',premultiplyAlpha:'none'});
        try {
            if(bitmap.width>16384||bitmap.height>16384||bitmap.width*bitmap.height*files.length>32*1024*1024) throw new Error('This material is too large for the browser encoder; reduce the image dimensions.');
            const canvas=new OffscreenCanvas(bitmap.width,bitmap.height),ctx=canvas.getContext('2d',{willReadFrequently:true});
            ctx.drawImage(bitmap,0,0);const data=ctx.getImageData(0,0,canvas.width,canvas.height).data;
            next.push({name:file.name,width:canvas.width,height:canvas.height,data,file});
        } finally {bitmap.close();}
    }
    if(next.some(image=>image.width!==next[0].width||image.height!==next[0].height)) throw new Error('All material images must have the same dimensions.');
    images=next;for(const url of previewURLs) URL.revokeObjectURL(url);previewURLs=[];$('inputs').replaceChildren();
    for(const image of images) {const thumbnail=document.createElement('img');thumbnail.src=URL.createObjectURL(image.file);previewURLs.push(thumbnail.src);thumbnail.alt=image.name;thumbnail.title=image.name;$('inputs').append(thumbnail);}
    $('encode').disabled=false;
    $('name').value=images[0].name.replace(/\.[^.]+$/,'').replace(/[^\w.-]/g,'_')||'material';
    result=null;$('result').hidden=true;$('empty').hidden=false;$('texture').hidden=true;$('viewer-section').hidden=true;$('viewer').removeAttribute('src');
    const alpha=images.some(image=>image.data.some((value,i)=>i%4===3&&value!==255));
    status(`${images.length} map${images.length===1?'':'s'} · ${images[0].width} × ${images[0].height}. ${alpha?'Alpha will be ignored.':'Ready to encode.'}`);
}
$('images').addEventListener('change',()=>loadImages([...$('images').files]).catch(error=>status(error.message,true)));
for(const kind of ['dragover','dragleave','drop']) $('dropzone').addEventListener(kind,event=>{
    event.preventDefault();$('dropzone').classList.toggle('drag',kind==='dragover');
    if(kind==='drop') loadImages([...event.dataTransfer.files]).catch(error=>status(error.message,true));
});
$('example').addEventListener('click',async()=>{
    try {status('Loading the bundled 512 × 512 material…');const response=await fetch('../examples/m1.png');if(!response.ok) throw new Error('Could not load the example image.');await loadImages([new File([await response.blob()],'m1.png')]);}
    catch(error){status(error.message,true);}
});
function showComparison() {
    const texture=Number($('texture').value),image=images[texture];
    const source=$('source');source.width=image.width;source.height=image.height;source.getContext('2d').putImageData(new ImageData(image.data,image.width,image.height),0,0);
    const target=$('reconstruction');target.width=result.width;target.height=result.height;
    const pixels=new Uint8ClampedArray(result.width*result.height*4);
    for(let y=0;y<result.height;y++) for(let x=0;x<result.width;x++) {const offset=(y*result.width+x)*4;for(let c=0;c<3;c++) pixels[offset+c]=result.reconstructed[(y*result.paddedWidth+x)*result.nout+texture*3+c]*255;pixels[offset+3]=255;}
    target.getContext('2d').putImageData(new ImageData(pixels,result.width,result.height),0,0);
    const bytes=[...result.files.values()].reduce((n,data)=>n+data.length,0),metric=(value,label)=>{const block=document.createElement('div'),strong=document.createElement('strong'),span=document.createElement('span');strong.textContent=value;span.textContent=label;block.append(strong,span);return block;};
    $('metrics').replaceChildren(metric(`${result.psnr[texture].toFixed(1)} dB`,'Reconstruction PSNR'),metric(`${(bytes/1024).toFixed(1)} KB`,'DDS + descriptor'),metric(`${elapsed.toFixed(1)} s`,'Encoding time'));
}
$('texture').addEventListener('change',showComparison);
$('form').addEventListener('submit',async event=>{
    event.preventDefault();if(controller||!images.length) return;
    controller=new AbortController();$('settings').disabled=true;$('encode').disabled=true;$('cancel').hidden=false;document.body.classList.add('busy');
    const start=performance.now();
    try {
        status('Initializing CUDA WebShader and validating GPU kernels…');
        encoder ||= await BrowserEncoder.create();
        const next=await encoder.encode(images,{name:$('name').value,c0:Number($('c0').value),c1:Number($('c1').value),iterations:Number($('iterations').value),
            mips:$('mips').checked,sites:$('filtered').checked?5:1,bc:$('storage').value==='bc',signal:controller.signal,onProgress:status});
        result=next;elapsed=(performance.now()-start)/1000;
        $('texture').replaceChildren(...images.map((image,i)=>new Option(image.name,String(i))));$('texture').hidden=false;
        $('empty').hidden=true;$('result').hidden=false;showComparison();status('Encoding complete. The exported descriptor and DDS files passed the viewer’s format checks.');
        $('viewer-section').hidden=true;$('viewer').removeAttribute('src');
    } catch(error) {status(error.message,error.name!=='AbortError');}
    finally {controller=null;$('settings').disabled=false;$('encode').disabled=false;$('cancel').hidden=true;document.body.classList.remove('busy');}
});
$('cancel').addEventListener('click',()=>{controller?.abort();status('Cancelling after the current GPU dispatch…');});
$('download').addEventListener('click',()=>{
    if(!result)return;
    const url=URL.createObjectURL(zipFiles(result.files)),link=document.createElement('a');link.href=url;link.download=result.descriptorName.replace('_nntc.json','.zip');link.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
});
$('view').addEventListener('click',()=>{$('viewer-section').hidden=false;$('viewer').src='index.html?encoder=1';$('viewer-section').scrollIntoView({behavior:'smooth'});});
$('close-viewer').addEventListener('click',()=>{$('viewer-section').hidden=true;$('viewer').removeAttribute('src');});
window.addEventListener('message',event=>{
    if(event.origin!==location.origin||event.source!==$('viewer').contentWindow||event.data?.type!=='nntc-viewer-ready'||!result) return;
    $('viewer').contentWindow.postMessage({type:'nntc-encoded-files',files:[...result.files].map(([name,bytes])=>new File([bytes],name))},location.origin);
});
if(!navigator.gpu) {status('WebGPU is unavailable. Open localhost or HTTPS in a WebGPU-enabled browser.',true);$('settings').disabled=true;}
