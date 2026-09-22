import {chromium} from 'playwright';
import {createServer} from '../scripts/serve-web.mjs';
import {mkdir,writeFile} from 'node:fs/promises';
const server=createServer();await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;
try {
    browser=await chromium.launch({...(process.env.NNTC_BROWSER?{channel:process.env.NNTC_BROWSER}:process.platform==='win32'?{channel:'msedge'}:{}),headless:true,args:['--enable-unsafe-webgpu',...(process.env.NNTC_SOFTWARE_GPU==='1'?['--use-angle=swiftshader','--enable-unsafe-swiftshader']:[])]});
    const page=await browser.newPage();
    page.on('console',message=>{if(message.type()==='error') console.error(message.text());});
    page.on('pageerror',error=>console.error(error));
    await page.goto(`http://127.0.0.1:${server.address().port}/tests/web-encoder-gpu.html`);
    await page.waitForFunction(()=>window.runTest);
    const results=[];
    for(const options of [{repeat:true},{c0:4,c1:3,width:16,height:16},{c0:1,c1:1,width:9,height:7,bc:false},
        {textures:6,c0:4,c1:4,width:20,height:12},{width:1,height:1,constant:true},{width:36,height:20,c0:3,c1:2},
        {sites:1,mips:false,bc:false,c0:3,c1:3}]) {
        const result=await page.evaluate(options=>window.runTest(options),options);results.push({options,...result});
        console.log(JSON.stringify({options,psnr:result.psnr,maxDecodeDifference:result.difference,files:result.files.length}));
        if(!result.psnr.every(v=>v>20)) throw new Error('Ramp quality below 20 dB.');
    }
    await mkdir('out/web-tests',{recursive:true});
    await writeFile('out/web-tests/gpu.json',JSON.stringify(results,null,2));
} finally {await browser?.close();await new Promise(resolve=>server.close(resolve));}
