// Block-level benchmark of the three fractal domains on a single GPU.
//
// Each fractal is measured on a complete level: the level whose number of
// elements is the closest to the N = 5e8 of the dense domains, among those
// whose bounding box fits in the memory of an A100 40 GB with one byte per
// cell.
//
//   2D Sierpinski gasket    3^17 elements, side 2^17, blocks of 16 x 16
//   2D Sierpinski carpet    8^10 elements, side 3^10, blocks of 27 x 27
//   3D Sierpinski pyramid   4^11 elements, side 2^11, blocks of 8 x 8 x 8
//
// The block side is a power of the base of the fractal, so that a block is
// either inside the fractal of the block level or completely outside. The
// bounding box launches the full enclosing grid and discards the blocks
// outside; the map launches only the blocks of the fractal. Before timing,
// every kernel is checked to write exactly base^level cells.
//
// Protocol of gpu_benchmark.cu: 5 warm-up launches, then at least 3 s of
// launches and never fewer than 10. Time and energy are per kernel launch.
//
// Build: nvcc -O3 -std=c++14 -o fractal_benchmark fractal_benchmark.cu -lnvidia-ml
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nvml.h>

#define CK(c){cudaError_t e=c;if(e!=cudaSuccess){printf("CUDA %s L%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}
#define NK(c){nvmlReturn_t e=c;if(e!=NVML_SUCCESS){printf("NVML %s L%d\n",nvmlErrorString(e),__LINE__);exit(1);}}
typedef unsigned char cell_t;
typedef unsigned long long u64;

// ---------------------------------------------------------------- gasket
// (x, y) belongs to the gasket iff (x & ~y) == 0.
#define GT 16
__device__ __forceinline__ void g_write(cell_t* a,int side,int bx,int by,u64* cnt){
    int gx=bx*GT+threadIdx.x, gy=by*GT+threadIdx.y;
    if((gx & ~gy)==0){ a[(size_t)gy*side+gx]=1; if(cnt) atomicAdd(cnt,1ULL); }
}
__global__ void g_bb(cell_t* a,int side,u64* cnt){
    int bx=blockIdx.x, by=blockIdx.y; if(bx & ~by) return;
    g_write(a,side,bx,by,cnt);
}
// Reference: one base-3 digit of the index per level.
__global__ void g_map(cell_t* a,int side,unsigned int nb,u64* cnt){
    unsigned int n=blockIdx.x; if(n>=nb) return;
    int bx=0, by=0, s=1;
    while(n){ int d=n%3; if(d>=1) by|=s; if(d==2) bx|=s; n/=3; s<<=1; }
    g_write(a,side,bx,by,cnt);
}
// OSS:120b: walks the rows, row y holds 2^popcount(y) elements, and places
// the bits of the offset on the set bits of y.
__global__ void g_model(cell_t* a,int side,unsigned int nb,u64* cnt){
    unsigned int n=blockIdx.x; if(n>=nb) return;
    unsigned int cum=0; int y=0, x=0;
    while(true){
        unsigned int c=1u<<__popc(y);
        if(cum+c>n){
            unsigned int off=n-cum; int i=0;
            for(int t=y,pos=0;t;t>>=1,pos++) if(t&1){ if((off>>i)&1) x|=1<<pos; i++; }
            break;
        }
        cum+=c; y++;
    }
    g_write(a,side,x,y,cnt);
}

// ---------------------------------------------------------------- carpet
// (x, y) belongs to the carpet iff no base-3 digit has x_i == 1 and y_i == 1.
#define CT 27
__device__ __forceinline__ bool in_carpet(int x,int y,int L){
    for(int i=0;i<L;i++){ if((x%3)==1 && (y%3)==1) return false; x/=3; y/=3; }
    return true;
}
__device__ __forceinline__ void c_write(cell_t* a,int side,int bx,int by,u64* cnt){
    // the block is in the carpet: only the 3 digits inside the block remain
    if(in_carpet(threadIdx.x,threadIdx.y,3)){
        int gx=bx*CT+threadIdx.x, gy=by*CT+threadIdx.y;
        a[(size_t)gy*side+gx]=1; if(cnt) atomicAdd(cnt,1ULL);
    }
}
__global__ void c_bb(cell_t* a,int side,int BL,u64* cnt){
    int bx=blockIdx.x, by=blockIdx.y; if(!in_carpet(bx,by,BL)) return;
    c_write(a,side,bx,by,cnt);
}
// Reference and OSS:120b: one base-8 digit of the index per level.
__constant__ int OX[8]={0,1,2,0,2,0,1,2};
__constant__ int OY[8]={0,0,0,1,1,2,2,2};
__global__ void c_map(cell_t* a,int side,unsigned int nb,u64* cnt){
    unsigned int n=blockIdx.x; if(n>=nb) return;
    int bx=0, by=0, p=1;
    while(n){ int d=n&7u; bx+=OX[d]*p; by+=OY[d]*p; n>>=3; p*=3; }
    c_write(a,side,bx,by,cnt);
}

// --------------------------------------------------------------- pyramid
// (x, y, z) belongs to the pyramid iff x, y and z share no set bit.
#define PT 8
__device__ __forceinline__ void p_write(cell_t* a,int side,int bx,int by,int bz,u64* cnt){
    int gx=bx*PT+threadIdx.x, gy=by*PT+threadIdx.y, gz=bz*PT+threadIdx.z;
    if((gx&gy)==0 && (gx&gz)==0 && (gy&gz)==0){
        a[((size_t)gz*side+gy)*side+gx]=1; if(cnt) atomicAdd(cnt,1ULL);
    }
}
__global__ void p_bb(cell_t* a,int side,u64* cnt){
    int bx=blockIdx.x, by=blockIdx.y, bz=blockIdx.z;
    if((bx&by)!=0 || (bx&bz)!=0 || (by&bz)!=0) return;
    p_write(a,side,bx,by,bz,cnt);
}
// This work and OSS:120b: one base-4 digit of the index per level.
__global__ void p_map(cell_t* a,int side,unsigned int nb,u64* cnt){
    unsigned int n=blockIdx.x; if(n>=nb) return;
    int bx=0, by=0, bz=0, s=1;
    while(n){ int d=n&3u; if(d==1) bx|=s; else if(d==2) by|=s; else if(d==3) bz|=s; n>>=2; s<<=1; }
    p_write(a,side,bx,by,bz,cnt);
}

// ------------------------------------------------------------ measurement
nvmlDevice_t dev;
static double now_mJ(){ unsigned long long mj=0; NK(nvmlDeviceGetTotalEnergyConsumption(dev,&mj)); return (double)mj; }
struct Res{ double ms, J, W; int iters; };
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
    Res r; r.iters=iters; r.ms=ms/iters; r.W=((e1-e0)/1000.0)/(ms/1000.0); r.J=r.W*r.ms/1000.0;
    return r;
}
u64* d_cnt;
template<class F> u64 count(F launch){
    u64 c=0; CK(cudaMemset(d_cnt,0,sizeof(u64))); launch();
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&c,d_cnt,sizeof(u64),cudaMemcpyDeviceToHost)); return c;
}
static u64 ipow(u64 b,int e){ u64 r=1; while(e--) r*=b; return r; }
static void header(const char* name,const char* lvl,u64 n,double gb){
    printf("\n>>> %s | %s = %llu elements | array %.2f GB\n",name,lvl,n,gb);
    printf("%-26s | %12s | %12s | %12s | %9s | %10s | %s\n","Row","Time (ms)","Blocks","Discarded","Power (W)","Energy (J)","Launches");
}
static void row(const char* name,Res r,u64 blocks,u64 disc){
    printf("%-26s | %12.4f | %12llu | %12llu | %9.1f | %10.4f | %d\n",name,r.ms,blocks,disc,r.W,r.J,r.iters);
}
static void check(const char* name,u64 c,u64 expect){
    printf("  check %-20s writes %llu cells, expected %llu -> %s\n",name,c,expect,c==expect?"OK":"WRONG");
    if(c!=expect) exit(2);
}

int main(){
    NK(nvmlInit());
    int cd=0; char pci[64]; CK(cudaGetDevice(&cd)); CK(cudaDeviceGetPCIBusId(pci,sizeof(pci),cd));
    NK(nvmlDeviceGetHandleByPciBusId(pci,&dev));
    char nm[96]; nvmlDeviceGetName(dev,nm,sizeof(nm));
    printf("=== Fractal block-level benchmark | %s (%s) | >= 3 s per row\n",nm,pci);
    CK(cudaMalloc(&d_cnt,sizeof(u64)));
    const double GB=1024.0*1024*1024;

    {   // ---- 2D Sierpinski gasket, 3^17
        const int L=17, BL=L-4; int side=1<<L, T=1<<BL;
        unsigned int nb=(unsigned int)ipow(3,BL); u64 expect=ipow(3,L), bb=(u64)T*T;
        size_t mem=(size_t)side*side; cell_t* a; CK(cudaMalloc(&a,mem)); CK(cudaMemset(a,0,mem));
        dim3 blk(GT,GT), g(T,T);
        header("2D Sierpinski gasket","3^17",expect,mem/GB);
        check("bounding box",count([&]{ g_bb   <<<g ,blk>>>(a,side,d_cnt); }),expect);
        check("reference map",count([&]{ g_map  <<<nb,blk>>>(a,side,nb,d_cnt); }),expect);
        check("OSS:120b map",count([&]{ g_model<<<nb,blk>>>(a,side,nb,d_cnt); }),expect);
        row("Bounding Box (baseline)",timeit([&]{ g_bb   <<<g ,blk>>>(a,side,nullptr); }),bb,bb-nb);
        row("Paper (Navarro 2017)",   timeit([&]{ g_map  <<<nb,blk>>>(a,side,nb,nullptr); }),nb,0);
        row("OSS:120b (100 Samples)", timeit([&]{ g_model<<<nb,blk>>>(a,side,nb,nullptr); }),nb,0);
        cudaFree(a);
    }
    {   // ---- 2D Sierpinski carpet, 8^10
        const int L=10, BL=L-3; int side=(int)ipow(3,L), T=(int)ipow(3,BL);
        unsigned int nb=(unsigned int)ipow(8,BL); u64 expect=ipow(8,L), bb=(u64)T*T;
        size_t mem=(size_t)side*side; cell_t* a; CK(cudaMalloc(&a,mem)); CK(cudaMemset(a,0,mem));
        dim3 blk(CT,CT), g(T,T);
        header("2D Sierpinski carpet","8^10",expect,mem/GB);
        check("bounding box",count([&]{ c_bb <<<g ,blk>>>(a,side,BL,d_cnt); }),expect);
        check("map",         count([&]{ c_map<<<nb,blk>>>(a,side,nb,d_cnt); }),expect);
        row("Bounding Box (baseline)",timeit([&]{ c_bb <<<g ,blk>>>(a,side,BL,nullptr); }),bb,bb-nb);
        row("Paper (reference)",      timeit([&]{ c_map<<<nb,blk>>>(a,side,nb,nullptr); }),nb,0);
        row("OSS:120b (100 Samples)", timeit([&]{ c_map<<<nb,blk>>>(a,side,nb,nullptr); }),nb,0);
        cudaFree(a);
    }
    {   // ---- 3D Sierpinski pyramid, 4^11
        const int L=11, BL=L-3; int side=1<<L, T=1<<BL;
        unsigned int nb=(unsigned int)ipow(4,BL); u64 expect=ipow(4,L), bb=(u64)T*T*T;
        size_t mem=(size_t)side*side*side; cell_t* a; CK(cudaMalloc(&a,mem)); CK(cudaMemset(a,0,mem));
        dim3 blk(PT,PT,PT), g(T,T,T);
        header("3D Sierpinski pyramid","4^11",expect,mem/GB);
        check("bounding box",count([&]{ p_bb <<<g ,blk>>>(a,side,d_cnt); }),expect);
        check("map",         count([&]{ p_map<<<nb,blk>>>(a,side,nb,d_cnt); }),expect);
        row("Bounding Box (baseline)",timeit([&]{ p_bb <<<g ,blk>>>(a,side,nullptr); }),bb,bb-nb);
        row("This work (closed form)",timeit([&]{ p_map<<<nb,blk>>>(a,side,nb,nullptr); }),nb,0);
        row("OSS:120b (100 Samples)", timeit([&]{ p_map<<<nb,blk>>>(a,side,nb,nullptr); }),nb,0);
        cudaFree(a);
    }
    cudaFree(d_cnt); nvmlShutdown();
    printf("\n=== end\n");
    return 0;
}
