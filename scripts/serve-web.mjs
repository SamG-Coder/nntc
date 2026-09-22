import http from 'node:http';
import {readFile,stat} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));
const mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.css':'text/css','.cu':'text/plain','.wgsl':'text/plain','.png':'image/png','.jpg':'image/jpeg','.dds':'application/octet-stream'};
export function createServer() {
    return http.createServer(async(req,res)=>{
        try {
            const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
            if(pathname==='/') {res.writeHead(302,{Location:'/webgpu/encode.html'}).end();return;}
            const relative=pathname==='/'?'webgpu/encode.html':pathname.replace(/^\/+/,''),file=path.resolve(root,relative);
            if(!file.startsWith(root)||relative.split(/[\\/]/).some(p=>p.startsWith('.'))) {res.writeHead(403).end();return;}
            const target=(await stat(file)).isDirectory()?path.join(file,'index.html'):file;
            const bytes=await readFile(target);
            res.writeHead(200,{'Content-Type':mime[path.extname(target)]||'application/octet-stream','Cache-Control':'no-store'}).end(bytes);
        } catch {res.writeHead(404).end('Not found');}
    });
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
    const port=Number(process.env.PORT||5174);
    createServer().listen(port,'127.0.0.1',()=>console.log(`NNTC encoder: http://127.0.0.1:${port}/`));
}
