// SPDX-License-Identifier: Apache-2.0
// Browser port of NNTC's bilinear features and fixed filter-aware site objective.
// Derived from Richard Geldreich Jr.'s src/sample.cuh (Copyright 2026).
// Host orchestration lives in encoder.js; cuda-webshader compiles these kernels.
__device__ float sample(const float* image, int w, int h, int channels, int ch, float x, float y) {
    int ix = (int)floorf(x), iy = (int)floorf(y);
    float fx = x - (float)ix, fy = y - (float)iy;
    int x0 = max(0, min(w - 1, ix)), x1 = max(0, min(w - 1, ix + 1));
    int y0 = max(0, min(h - 1, iy)), y1 = max(0, min(h - 1, iy + 1));
    float a = image[(y0 * w + x0) * channels + ch];
    float b = image[(y0 * w + x1) * channels + ch];
    float c = image[(y1 * w + x0) * channels + ch];
    float d = image[(y1 * w + x1) * channels + ch];
    return (a + (b - a) * fx) * (1.0f - fy) + (c + (d - c) * fx) * fy;
}
__device__ float site_x(int j) {
    if (j == 1) return -0.375f;
    if (j == 2) return -0.125f;
    if (j == 3) return 0.125f;
    if (j == 4) return 0.375f;
    return 0.0f;
}
__device__ float site_y(int j) {
    if (j == 1) return -0.125f;
    if (j == 2) return 0.375f;
    if (j == 3) return -0.375f;
    if (j == 4) return 0.125f;
    return 0.0f;
}
__device__ float influence(float coordinate, int texel, int extent) {
    int low = (int)floorf(coordinate);
    float f = coordinate - (float)low;
    float weight = 0.0f;
    if (max(0, min(extent - 1, low)) == texel) weight += 1.0f - f;
    if (max(0, min(extent - 1, low + 1)) == texel) weight += f;
    return weight;
}
__global__ void make_rows(const float* source, const float* latent0, const float* latent1, float* rows,
    int w, int h, int w1, int h1, int c0, int c1, int nout, int sites, int start, int count, float weight) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    int si = start + id, pixel = si / sites, j = si % sites;
    float x = (float)(pixel % w) + site_x(j), y = (float)(pixel / w) + site_y(j);
    float s[4], c[4];
    for (int i = 0; i < c0; i++) s[i] = sample(latent0,w,h,c0,i,x,y);
    for (int i = 0; i < c1; i++) c[i] = sample(latent1,w1,h1,c1,i,(x+0.5f)*(float)w1/(float)w-0.5f,(y+0.5f)*(float)h1/(float)h-0.5f);
    int mm = c0 + c1 + c0*c1 + 1, at = id*(mm+nout), k = 0;
    for (int i = 0; i < c1; i++) rows[at+k++] = c[i]*weight;
    for (int i = 0; i < c0; i++) rows[at+k++] = s[i]*weight;
    for (int i = 0; i < c0; i++) for (int t = 0; t < c1; t++) rows[at+k++] = s[i]*c[t]*weight;
    rows[at+k++] = weight;
    for (int o = 0; o < nout; o++) rows[at+k++] = sample(source,w,h,nout,o,x,y)*weight;
}

// Four-color block coordinate descent. With the opposite latent fixed, each
// texel's channel vector has a small convex quadratic. Gather all affected sites
// so neighboring work items never scatter/atomically accumulate floating values.
// Read and write distinct buffers: a dispatch sees a coherent previous plane.
__global__ void update_latent(const float* source, const float* latent0, const float* latent1,
    const float* decoder, float* result, int w, int h, int w1, int h1,
    int c0, int c1, int nout, int sites, int level, int color) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tw = level == 0 ? w : w1, th = level == 0 ? h : h1, channels = level == 0 ? c0 : c1;
    if (id >= tw*th) return;
    int tx = id % tw, ty = id / tw;
    for (int a = 0; a < channels; a++) result[id*channels+a] = level == 0 ? latent0[id*c0+a] : latent1[id*c1+a];
    if ((tx % 2) + 2*(ty % 2) != color) return;
    int mm = c0 + c1 + c0*c1 + 1;
    float matrix[16], rhs[4];
    for (int a = 0; a < 16; a++) matrix[a] = 0.0f;
    for (int a = 0; a < 4; a++) rhs[a] = 0.0f;
    float scaleX = (float)w/(float)tw, scaleY = (float)h/(float)th;
    int left = max(0,(int)floorf(((float)tx-0.5f)*scaleX-1.0f));
    int right = min(w-1,(int)ceilf(((float)tx+1.5f)*scaleX+1.0f));
    int top = max(0,(int)floorf(((float)ty-0.5f)*scaleY-1.0f));
    int bottom = min(h-1,(int)ceilf(((float)ty+1.5f)*scaleY+1.0f));
    for (int py = top; py <= bottom; py++) for (int px = left; px <= right; px++) for (int j = 0; j < sites; j++) {
        float x = (float)px+site_x(j), y = (float)py+site_y(j);
        float gx = (x+0.5f)/scaleX-0.5f, gy = (y+0.5f)/scaleY-0.5f;
        float q = influence(gx,tx,tw)*influence(gy,ty,th);
        if (q <= 0.0f) continue;
        float s[4], c[4], phi[25];
        for (int a = 0; a < c0; a++) s[a] = sample(latent0,w,h,c0,a,x,y);
        for (int a = 0; a < c1; a++) c[a] = sample(latent1,w1,h1,c1,a,(x+0.5f)*(float)w1/(float)w-0.5f,(y+0.5f)*(float)h1/(float)h-0.5f);
        int k = 0;
        for (int a = 0; a < c1; a++) phi[k++] = c[a];
        for (int a = 0; a < c0; a++) phi[k++] = s[a];
        for (int a = 0; a < c0; a++) for (int b = 0; b < c1; b++) phi[k++] = s[a]*c[b];
        phi[k] = 1.0f;
        for (int o = 0; o < nout; o++) {
            float predicted = 0.0f;
            for (int a = 0; a < mm; a++) predicted += decoder[o*mm+a]*phi[a];
            float residual = sample(source,w,h,nout,o,x,y)-predicted;
            float slope[4];
            for (int a = 0; a < channels; a++) {
                float derivative = decoder[o*mm+(level == 0 ? c1+a : a)];
                if (level == 0) {
                    for (int b = 0; b < c1; b++) derivative += decoder[o*mm+c1+c0+a*c1+b]*c[b];
                } else {
                    for (int b = 0; b < c0; b++) derivative += decoder[o*mm+c1+c0+b*c1+a]*s[b];
                }
                slope[a] = q*derivative;
            }
            for (int a = 0; a < channels; a++) {
                rhs[a] += slope[a]*residual;
                for (int b = 0; b < channels; b++) matrix[a*4+b] += slope[a]*slope[b];
            }
        }
    }
    float trace = 0.0f;
    for (int a = 0; a < channels; a++) trace += matrix[a*4+a];
    // Positive ridge makes the symmetric system definite even for constant input.
    for (int a = 0; a < channels; a++) matrix[a*4+a] += max(1.0e-8f,trace*1.0e-5f);
    for (int a = 0; a < channels; a++) {
        float diagonal = matrix[a*4+a];
        for (int b = a; b < channels; b++) matrix[a*4+b] /= diagonal;
        rhs[a] /= diagonal;
        for (int row = 0; row < channels; row++) if (row != a) {
            float factor = matrix[row*4+a];
            for (int b = a; b < channels; b++) matrix[row*4+b] -= factor*matrix[a*4+b];
            rhs[row] -= factor*rhs[a];
        }
    }
    // Bounded trust step retains numerical stability in nearly rank-deficient fits.
    float largest = 0.5f;
    for (int a = 0; a < channels; a++) largest = fmaxf(largest,fabsf(rhs[a]));
    for (int a = 0; a < channels; a++) result[id*channels+a] += rhs[a]*(0.5f/largest);
}

__global__ void decode(const float* latent0, const float* latent1, const float* decoder, float* output,
    int w, int h, int w1, int h1, int c0, int c1, int nout) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id >= w*h) return;
    float x = (float)(id%w), y = (float)(id/w), s[4], c[4], phi[25];
    for (int i = 0; i < c0; i++) s[i] = latent0[id*c0+i];
    for (int i = 0; i < c1; i++) c[i] = sample(latent1,w1,h1,c1,i,(x+0.5f)*(float)w1/(float)w-0.5f,(y+0.5f)*(float)h1/(float)h-0.5f);
    int k = 0;
    for (int i = 0; i < c1; i++) phi[k++] = c[i];
    for (int i = 0; i < c0; i++) phi[k++] = s[i];
    for (int i = 0; i < c0; i++) for (int j = 0; j < c1; j++) phi[k++] = s[i]*c[j];
    phi[k++] = 1.0f;
    for (int o = 0; o < nout; o++) {
        float value = 0.0f;
        for (int i = 0; i < k; i++) value += decoder[o*k+i]*phi[i];
        output[id*nout+o] = value;
    }
}
