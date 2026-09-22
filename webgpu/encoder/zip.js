// SPDX-License-Identifier: Apache-2.0
// Uncompressed ZIP (DDS is already block compressed); one explicit download.
export function zipFiles(files) {
    const table=Uint32Array.from({length:256},(_,n)=>{for(let k=0;k<8;k++) n=(n&1)?0xedb88320^(n>>>1):n>>>1;return n>>>0;});
    const local=[],central=[];let offset=0,centralSize=0;
    for(const [name,data] of files) {
        const encoded=new TextEncoder().encode(name);let crc=0xffffffff;
        for(const byte of data) crc=table[(crc^byte)&255]^(crc>>>8);
        crc=(crc^0xffffffff)>>>0;
        const header=new Uint8Array(30+encoded.length),v=new DataView(header.buffer);
        v.setUint32(0,0x04034b50,true);v.setUint16(4,20,true);v.setUint16(6,0x800,true);v.setUint16(12,33,true);
        v.setUint32(14,crc,true);v.setUint32(18,data.length,true);v.setUint32(22,data.length,true);v.setUint16(26,encoded.length,true);header.set(encoded,30);
        local.push(header,data);
        const entry=new Uint8Array(46+encoded.length),e=new DataView(entry.buffer);
        e.setUint32(0,0x02014b50,true);e.setUint16(4,20,true);e.setUint16(6,20,true);e.setUint16(8,0x800,true);e.setUint16(14,33,true);
        e.setUint32(16,crc,true);e.setUint32(20,data.length,true);e.setUint32(24,data.length,true);e.setUint16(28,encoded.length,true);e.setUint32(42,offset,true);entry.set(encoded,46);
        central.push(entry);centralSize+=entry.length;offset+=header.length+data.length;
    }
    const end=new Uint8Array(22),v=new DataView(end.buffer);
    v.setUint32(0,0x06054b50,true);v.setUint16(8,files.size,true);v.setUint16(10,files.size,true);v.setUint32(12,centralSize,true);v.setUint32(16,offset,true);
    return new Blob([...local,...central,end],{type:'application/zip'});
}
