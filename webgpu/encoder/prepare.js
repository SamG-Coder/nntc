// SPDX-License-Identifier: Apache-2.0
export function validateInputs(images,{c0=2,c1=4,iterations=6,sites=5}={}) {
    if(!Array.isArray(images)||images.length<1||images.length>6) throw new Error('Select one to six RGB images.');
    if(![c0,c1].every(c=>Number.isInteger(c)&&c>=1&&c<=4)) throw new Error('Latent channels must be 1..4.');
    if(!Number.isInteger(iterations)||iterations<0||iterations>100) throw new Error('Iterations must be 0..100.');
    if(![1,5].includes(sites)) throw new Error('Use one center site or five filtered sites.');
    const {width,height}=images[0];
    if(![width,height].every(v=>Number.isInteger(v)&&v>0&&v<=16384)) throw new Error('Image dimensions must be 1..16384.');
    for(const image of images) {
        if(image.width!==width||image.height!==height) throw new Error('All material images must have the same dimensions.');
        if(!(image.data instanceof Uint8Array || image.data instanceof Uint8ClampedArray)||image.data.length!==width*height*4) throw new Error('Expected RGBA8 image data.');
    }
    return {width,height};
}

// A shared PCA basis across the mip chain, with deterministic Jacobi rotations.
function basis(data,channels,count) {
    const mean=new Float64Array(channels),cov=new Float64Array(channels*channels),vectors=new Float64Array(channels*channels),n=data.length/channels;
    for(let i=0;i<data.length;i++) mean[i%channels]+=data[i]/n;
    for(let i=0;i<n;i++) for(let a=0;a<channels;a++) for(let b=0;b<channels;b++) cov[a*channels+b]+=(data[i*channels+a]-mean[a])*(data[i*channels+b]-mean[b])/n;
    for(let a=0;a<channels;a++) vectors[a*channels+a]=1;
    for(let pass=0;pass<channels*channels*30;pass++) {
        let p=0,q=0,largest=0;
        for(let a=0;a<channels;a++) for(let b=a+1;b<channels;b++) if(Math.abs(cov[a*channels+b])>largest) {p=a;q=b;largest=Math.abs(cov[a*channels+b]);}
        if(largest<1e-12) break;
        const angle=0.5*Math.atan2(2*cov[p*channels+q],cov[q*channels+q]-cov[p*channels+p]),c=Math.cos(angle),s=Math.sin(angle);
        for(let a=0;a<channels;a++) {const x=cov[a*channels+p],y=cov[a*channels+q];cov[a*channels+p]=c*x-s*y;cov[a*channels+q]=s*x+c*y;}
        for(let a=0;a<channels;a++) {const x=cov[p*channels+a],y=cov[q*channels+a];cov[p*channels+a]=c*x-s*y;cov[q*channels+a]=s*x+c*y;}
        for(let a=0;a<channels;a++) {const x=vectors[a*channels+p],y=vectors[a*channels+q];vectors[a*channels+p]=c*x-s*y;vectors[a*channels+q]=s*x+c*y;}
    }
    const order=Array.from({length:channels},(_,i)=>i).sort((a,b)=>cov[b*channels+b]-cov[a*channels+a]);
    return {mean,axes:Array.from({length:count},(_,i)=>Array.from({length:channels},(_,j)=>vectors[j*channels+order[i%channels]]))};
}

function resize(data,w,h,nw,nh,channels) {
    const result=new Float32Array(nw*nh*channels);
    // Area averaging including partial source cells for non-power-of-two mips.
    for(let y=0;y<nh;y++) for(let x=0;x<nw;x++) {
        const left=x*w/nw,right=(x+1)*w/nw,top=y*h/nh,bottom=(y+1)*h/nh,area=(right-left)*(bottom-top);
        for(let sy=Math.floor(top);sy<Math.ceil(bottom);sy++) for(let sx=Math.floor(left);sx<Math.ceil(right);sx++) {
            const weight=(Math.min(right,sx+1)-Math.max(left,sx))*(Math.min(bottom,sy+1)-Math.max(top,sy))/area;
            for(let c=0;c<channels;c++) result[(y*nw+x)*channels+c]+=data[(sy*w+sx)*channels+c]*weight;
        }
    }
    return result;
}

function detailResidual(plane,channels) {
    const {w,h,w1,h1,source,coarse}=plane,result=new Float32Array(source.length);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++) {
        const gx=(x+0.5)*w1/w-0.5,gy=(y+0.5)*h1/h-0.5,ix=Math.floor(gx),iy=Math.floor(gy),fx=gx-ix,fy=gy-iy;
        for(let c=0;c<channels;c++) {
            let low=0;
            for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++) low+=coarse[(Math.max(0,Math.min(h1-1,iy+dy))*w1+Math.max(0,Math.min(w1-1,ix+dx)))*channels+c]*(dx?fx:1-fx)*(dy?fy:1-fy);
            result[(y*w+x)*channels+c]=source[(y*w+x)*channels+c]-low;
        }
    }
    return result;
}

export function prepare(images,{c0=2,c1=4,mips=true,...options}={}) {
    const {width,height}=validateInputs(images,{c0,c1,...options}),w=Math.ceil(width/4)*4,h=Math.ceil(height/4)*4,nout=images.length*3;
    const source=new Float32Array(w*h*nout);
    for(let y=0;y<h;y++) for(let x=0;x<w;x++) for(let t=0;t<images.length;t++) for(let c=0;c<3;c++) source[(y*w+x)*nout+t*3+c]=images[t].data[(Math.min(y,height-1)*width+Math.min(x,width-1))*4+c]/255;
    const planes=[];
    let pw=w,ph=h,src=source;
    while(true) {
        const w1=Math.max(1,Math.floor(pw/4)),h1=Math.max(1,Math.floor(ph/4));
        planes.push({w:pw,h:ph,w1,h1,source:src,coarse:resize(src,pw,ph,w1,h1,nout)});
        if(!mips||pw<16||ph<16) break;
        const nw=pw>>1,nh=ph>>1;
        // DDS halves each latent independently. Stop before rounding would make
        // floor(full/4) disagree with the DDS half-size of the previous latent.
        if(Math.floor(nw/4)!==(w1>>1)||Math.floor(nh/4)!==(h1>>1)) break;
        src=resize(src,pw,ph,nw,nh,nout);pw=nw;ph=nh;
    }
    const pca1=basis(planes[0].coarse,nout,c1),baseResidual=detailResidual(planes[0],nout),pca0=basis(baseResidual,nout,c0);
    for(const p of planes) {
        p.values0=new Float32Array(p.w*p.h*c0);p.values1=new Float32Array(p.w1*p.h1*c1);
        for(const [data,target,channels,pca] of [[p===planes[0]?baseResidual:detailResidual(p,nout),p.values0,c0,pca0],[p.coarse,p.values1,c1,pca1]]) for(let i=0;i<data.length/nout;i++) for(let c=0;c<channels;c++) {
            let value=0;for(let o=0;o<nout;o++) value+=(data[i*nout+o]-pca.mean[o])*pca.axes[c][o];
            target[i*channels+c]=value;
        }
        delete p.coarse;
    }
    return {width,height,nout,planes};
}
