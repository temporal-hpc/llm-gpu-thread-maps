// Block-level benchmark of the 2D Sierpinski carpet on a single GPU.
//   1. The map (index in base 8 -> block offsets in base 3) at block level.
//   2. Two bounding-box baselines: without filter and with discard per block.
//   3. Validation by counting: every kernel must write exactly 8^L cells.
// Same protocol as gpu_benchmark.cu: 5 warm-up launches, then at least 3 s
// of launches; time and energy are per kernel launch.
//
// Build: nvcc -O3 -std=c++14 -o carpet_benchmark carpet_benchmark.cu -lnvidia-ml
#include <stdio.h>
#include <cuda_runtime.h>
#include <nvml.h>

#define CK(c){cudaError_t e=c;if(e!=cudaSuccess){printf("CUDA %s L%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}
#define NK(c){nvmlReturn_t e=c;if(e!=NVML_SUCCESS){printf("NVML %s L%d\n",nvmlErrorString(e),__LINE__);exit(1);}}

#define TILE 27            // 3^3: the carpet is base 3, a block of 16 does not align

// A cell belongs to the carpet if no base-3 digit has x_i==1 and y_i==1.
__device__ __forceinline__ bool in_carpet(int x,int y,int L){
    for(int i=0;i<L;i++){ if((x%3)==1 && (y%3)==1) return false; x/=3; y/=3; }
    return true;
}
// A block holds cells of the carpet if its block coordinate is in the carpet
// of level L-3. Exact because TILE is a power of 3.
__device__ __forceinline__ bool block_in_carpet(int bx,int by,int BL){
    return in_carpet(bx,by,BL);
}

// Map: lambda in base 8 -> offsets in base 3.
__constant__ int OX[8]={0,1,2,0,2,0,1,2};
__constant__ int OY[8]={0,0,0,1,1,2,2,2};
__device__ __forceinline__ void carpet_map(unsigned int lam,int&bx,int&by,int BL){
    bx=0; by=0; int p=1;
    for(int i=0;i<BL;i++){ int d=lam&7u; lam>>=3; bx+=OX[d]*p; by+=OY[d]*p; p*=3; }
}

__global__ void k_bb_plain(int* img,int side,int L,unsigned long long* cnt){
    int gx=blockIdx.x*TILE+threadIdx.x, gy=blockIdx.y*TILE+threadIdx.y;
    if(gx>=side||gy>=side) return;
    if(in_carpet(gx,gy,L)){ img[(size_t)gy*side+gx]=1; if(cnt) atomicAdd(cnt,1ULL); }
}
__global__ void k_bb_discard(int* img,int side,int L,int BL,unsigned long long* cnt){
    if(!block_in_carpet(blockIdx.x,blockIdx.y,BL)) return;
    int gx=blockIdx.x*TILE+threadIdx.x, gy=blockIdx.y*TILE+threadIdx.y;
    if(gx>=side||gy>=side) return;
    if(in_carpet(gx,gy,L)){ img[(size_t)gy*side+gx]=1; if(cnt) atomicAdd(cnt,1ULL); }
}
__global__ void k_map(int* img,int side,int L,int BL,unsigned int nb,unsigned long long* cnt){
    unsigned int n=blockIdx.x; if(n>=nb) return;
    int bx,by; carpet_map(n,bx,by,BL);
    int gx=bx*TILE+threadIdx.x, gy=by*TILE+threadIdx.y;
    if(gx>=side||gy>=side) return;
    if(in_carpet(gx,gy,L)){ img[(size_t)gy*side+gx]=1; if(cnt) atomicAdd(cnt,1ULL); }
}

nvmlDevice_t dev;
static double now_mJ(){ unsigned long long e=0; nvmlDeviceGetTotalEnergyConsumption(dev,&e); return (double)e; }

struct Res{ double ms, J, W; };
template<class F> Res timeit(F launch){
    CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for(int i=0;i<5;i++) launch();
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float w=0; cudaEventElapsedTime(&w,a,b); w/=5;
    int iters=(int)(3000.0f/w); if(iters<10) iters=10;
    double e0=now_mJ();
    cudaEventRecord(a);
    for(int i=0;i<iters;i++) launch();
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    double e1=now_mJ();
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    Res r; r.ms=ms/iters; r.W=((e1-e0)/1000.0)/(ms/1000.0); r.J=r.W*r.ms/1000.0;   // J per launch
    return r;
}

int main(){
    NK(nvmlInit());
    // NVML and CUDA can number the GPUs differently: take the one CUDA uses
    int cd=0; char pci[64]; CK(cudaGetDevice(&cd)); CK(cudaDeviceGetPCIBusId(pci,sizeof(pci),cd));
    NK(nvmlDeviceGetHandleByPciBusId(pci,&dev)); printf("measuring GPU %s\n",pci);
    char nm[96]; nvmlDeviceGetName(dev,nm,sizeof(nm));
    printf("GPU: %s | TILE=%d (%d threads per block) | >=3 s per kernel\n\n",nm,TILE,TILE*TILE);

    unsigned long long* d_cnt; CK(cudaMalloc(&d_cnt,sizeof(unsigned long long)));

    for(int L=9; L<=10; L++){
        int BL=L-3;                                  // levels at block level
        int side=1; for(int i=0;i<L;i++) side*=3;    // 3^L cells per side
        unsigned int nb=1; for(int i=0;i<BL;i++) nb*=8;   // blocks of the carpet
        int T=(side+TILE-1)/TILE;                    // blocks of the bounding box
        unsigned long long expect=1; for(int i=0;i<L;i++) expect*=8ULL;
        size_t mem=(size_t)side*side*sizeof(int);
        printf("=== level L=%d | side=%d cells | %.2f GB | BB=%llu blocks | map=%u blocks | ratio=%.3fx\n",
               L,side,mem/(1024.0*1024*1024),(unsigned long long)T*T,nb,(double)T*T/nb);

        int* img; if(cudaMalloc(&img,mem)!=cudaSuccess){ printf("  not enough GPU memory, skipped\n\n"); continue; }
        CK(cudaMemset(img,0,mem));

        dim3 blk(TILE,TILE), gbb(T,T);
        unsigned long long c;

        // --- validation by counting, a single launch of each kernel
        CK(cudaMemset(d_cnt,0,sizeof(c))); k_map<<<nb,blk>>>(img,side,L,BL,nb,d_cnt);
        CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&c,d_cnt,sizeof(c),cudaMemcpyDeviceToHost));
        printf("  validation map : writes %llu cells, expected 8^%d = %llu  -> %s\n",
               c,L,expect, c==expect?"OK":"WRONG");
        CK(cudaMemset(d_cnt,0,sizeof(c))); k_bb_plain<<<gbb,blk>>>(img,side,L,d_cnt);
        CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&c,d_cnt,sizeof(c),cudaMemcpyDeviceToHost));
        printf("  validation BB  : writes %llu cells  -> %s\n", c, c==expect?"OK":"WRONG");

        Res r1=timeit([&]{ k_bb_plain  <<<gbb,blk>>>(img,side,L,nullptr); });
        Res r2=timeit([&]{ k_bb_discard<<<gbb,blk>>>(img,side,L,BL,nullptr); });
        Res r3=timeit([&]{ k_map       <<<nb ,blk>>>(img,side,L,BL,nb,nullptr); });

        printf("  %-26s %9.3f ms  %8.3f J  %7.1f W\n","BB without filter",r1.ms,r1.J,r1.W);
        printf("  %-26s %9.3f ms  %8.3f J  %7.1f W\n","BB with discard",r2.ms,r2.J,r2.W);
        printf("  %-26s %9.3f ms  %8.3f J  %7.1f W\n","GPU map",r3.ms,r3.J,r3.W);
        printf("  speedup  vs BB without filter = %.3fx   | vs BB with discard = %.3fx\n",
               r1.ms/r3.ms, r2.ms/r3.ms);
        printf("  greenup  vs BB without filter = %.3fx   | vs BB with discard = %.3fx\n\n",
               r1.J /r3.J , r2.J /r3.J );
        cudaFree(img);
    }

    cudaFree(d_cnt); nvmlShutdown();
    printf("\n=== end\n");
    return 0;
}
