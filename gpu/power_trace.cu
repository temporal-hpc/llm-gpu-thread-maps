// Power trace of the bounding box against the efficient map on the 2D
// triangular domain. Phases: idle -> bounding box -> idle -> map -> idle,
// with NVML sampled every 20 ms. Writes power_trace.csv (ms, watts, phase).
#include <stdio.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <thread>
#include <atomic>
#include <chrono>

#define NUM_POINTS 500000000
#define TILE_DIM   16
// Both kernel phases do THE SAME WORK (same number of launches), so the
// width of each phase in the trace is proportional to the time of one launch.
#define PHASE_LAUNCHES 2000   // launches per phase (about 5.9 s of bounding box)
#define IDLE_MS    2000

#define CK(c){cudaError_t e=c;if(e!=cudaSuccess){printf("CUDA %s L%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}
#define NK(c){nvmlReturn_t e=c;if(e!=NVML_SUCCESS){printf("NVML %s L%d\n",nvmlErrorString(e),__LINE__);exit(1);}}

__global__ void k_bb(int* img,int SIDE){
    int bx=blockIdx.x, by=blockIdx.y; if(bx>by) return;
    int gx=bx*TILE_DIM+threadIdx.x, gy=by*TILE_DIM+threadIdx.y;
    if(gx>=SIDE||gy>=SIDE) return;
    if(gx<=gy){ size_t i=(size_t)gy*SIDE+gx; img[i]=1; }
}
__global__ void k_ana(int* img,int SIDE,int nb){
    int bid=blockIdx.x; if(bid>=nb) return;
    float v=sqrtf(8.0f*(float)bid+1.0f);
    int by=(int)((v-1.0f)*0.5f);
    int bx=bid-((by*(by+1))>>1);
    int gx=bx*TILE_DIM+threadIdx.x, gy=by*TILE_DIM+threadIdx.y;
    if(gx>=SIDE||gy>=SIDE) return;
    if(gx<=gy){ size_t i=(size_t)gy*SIDE+gx; img[i]=1; }
}

nvmlDevice_t dev;
std::atomic<bool> run(true);
std::atomic<int>  phase(0);   // 0 idle, 1 bounding box, 2 map
FILE* out;

void sampler(){
    auto t0=std::chrono::steady_clock::now();
    while(run){
        unsigned int mw=0; nvmlDeviceGetPowerUsage(dev,&mw);
        double ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-t0).count();
        fprintf(out,"%.1f,%.2f,%d\n", ms, mw/1000.0, phase.load());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
void holdN(int n,int ph,void(*launch)()){
    phase=ph;
    for(int i=0;i<n;i++) launch();
    CK(cudaDeviceSynchronize());
}

void hold(int ms,int ph,void(*launch)()){
    phase=ph;
    auto t0=std::chrono::steady_clock::now();
    while(std::chrono::duration<double,std::milli>(
          std::chrono::steady_clock::now()-t0).count()<ms){
        if(launch) launch();
    }
    CK(cudaDeviceSynchronize());
}

int   SIDE, NB; int* d_img;
void runBB(){ int T=(SIDE+TILE_DIM-1)/TILE_DIM; dim3 g(T,T),b(TILE_DIM,TILE_DIM);
              k_bb<<<g,b>>>(d_img,SIDE); cudaDeviceSynchronize(); }
void runAN(){ dim3 b(TILE_DIM,TILE_DIM);
              k_ana<<<NB,b>>>(d_img,SIDE,NB); cudaDeviceSynchronize(); }

int main(){
    NK(nvmlInit());
    // Resolve through its PCI bus id the GPU that CUDA was given. Under SLURM
    // the NVML numbering is the physical one of the node and may differ from
    // the CUDA numbering, so a fixed index could read the GPU of another job.
    int cudaDev=0; CK(cudaGetDevice(&cudaDev));
    char pci[64]; CK(cudaDeviceGetPCIBusId(pci,sizeof(pci),cudaDev));
    NK(nvmlDeviceGetHandleByPciBusId(pci,&dev));
    printf("measuring GPU %s\n", pci);
    SIDE=(int)sqrt(2.0*NUM_POINTS)+128; NB=(NUM_POINTS+255)/256;
    CK(cudaMalloc(&d_img,(size_t)SIDE*SIDE*sizeof(int)));
    CK(cudaMemset(d_img,0,(size_t)SIDE*SIDE*sizeof(int)));
    out=fopen("power_trace.csv","w"); fprintf(out,"ms,watts,phase\n");
    std::thread th(sampler);
    hold(IDLE_MS,0,nullptr);
    holdN(PHASE_LAUNCHES,1,runBB);
    hold(IDLE_MS,0,nullptr);
    holdN(PHASE_LAUNCHES,2,runAN);
    hold(IDLE_MS,0,nullptr);
    run=false; th.join(); fclose(out);
    printf("done: power_trace.csv  SIDE=%d NB=%d\n",SIDE,NB);
    nvmlShutdown(); cudaFree(d_img); return 0;
}
