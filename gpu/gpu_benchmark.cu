#include <stdio.h>
#include <cuda_runtime.h>
#include <nvml.h> 
#include <math.h>
#include <stdarg.h>
#include <thread>
#include <atomic>
#include <chrono>

// =================================================================================
// Block-level benchmark of the thread maps: time, power and energy per kernel
// launch against the bounding-box baseline. N = 5e8 mapped points.
// Every row launches its kernel repeatedly for at least TARGET_TIME_MS (never
// fewer than 10 launches) while NVML integrates the energy drawn by GPU 0.
// =================================================================================
#define NUM_POINTS  500000000 
#define TILE_DIM    16        
#define TARGET_TIME_MS 3000.0f // minimum measured time per row

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

#define CHECK_NVML(call) { \
    nvmlReturn_t err = call; \
    if (err != NVML_SUCCESS) { \
        printf("NVML Error: %s (line %d)\n", nvmlErrorString(err), __LINE__); \
        exit(1); \
    } \
}

nvmlDevice_t nvml_device;

// =================================================================================
// KERNELS. The model rows are hand translations to CUDA of the Python
// functions emitted by the models (see responses/).
// =================================================================================

// --- 2D ---
__global__ void k_2d_BB(int* img, int SIDE) {
    int bx = blockIdx.x; int by = blockIdx.y; if (bx > by) return; 
    int gx = bx * TILE_DIM + threadIdx.x; int gy = by * TILE_DIM + threadIdx.y;
    if (gx >= SIDE || gy >= SIDE) return;
    if (gx <= gy) { size_t idx = (size_t)gy * SIDE + gx; img[idx] = 1; }
}
__global__ void k_2d_Analytical(int* img, int SIDE, int n_blocks) {
    int bid = blockIdx.x; if (bid >= n_blocks) return;
    float v = sqrtf(8.0f * (float)bid + 1.0f);
    int by = (int)((v - 1.0f) * 0.5f);
    int bx = bid - ((by * (by + 1)) >> 1);
    int gx = bx * TILE_DIM + threadIdx.x; int gy = by * TILE_DIM + threadIdx.y;
    if (gx >= SIDE || gy >= SIDE) return;
    if (gx <= gy) { size_t idx = (size_t)gy * SIDE + gx; img[idx] = 1; }
}
__global__ void k_2d_Robust(int* img, int SIDE, int n_blocks) {
    int bid = blockIdx.x; if (bid >= n_blocks) return;
    double temp = 8.0 * (double)bid + 1.0;
    int by = (int)((sqrt(temp) - 1.0) * 0.5);
    while ((long long)(by+1)*(by+2)/2 <= bid) by++;
    int bx = bid - (by * (by + 1)) / 2;
    int gx = bx * TILE_DIM + threadIdx.x; int gy = by * TILE_DIM + threadIdx.y;
    if (gx >= SIDE || gy >= SIDE) return;
    if (gx <= gy) { size_t idx = (size_t)gy * SIDE + gx; img[idx] = 1; }
}
__global__ void k_2d_Approx(int* img, int SIDE, int n_blocks) {
    int bid = blockIdx.x; if (bid >= n_blocks) return;
    int by = (int)sqrtf(2.0f * (float)bid);
    long long tri = (long long)by * (by + 1) / 2;
    if (tri > bid) by -= 1; 
    int bx = bid - (by * (by + 1)) / 2;
    int gx = bx * TILE_DIM + threadIdx.x; int gy = by * TILE_DIM + threadIdx.y;
    if (gx >= SIDE || gy >= SIDE) return;
    if (gx <= gy) { size_t idx = (size_t)gy * SIDE + gx; img[idx] = 1; }
}
__global__ void k_2d_BinSearch(int* img, int SIDE, int n_blocks) {
    int bid = blockIdx.x; if (bid >= n_blocks) return;
    long long n = bid;
    long long L=0, R=SIDE; 
    while (L < R) {
        long long mid = (L+R)/2;
        if (mid*(mid+1)/2 > n) R=mid; else L=mid+1;
    }
    int by = (int)L-1;
    int bx = (int)(n - (long long)by*(by+1)/2);
    int gx = bx * TILE_DIM + threadIdx.x; int gy = by * TILE_DIM + threadIdx.y;
    if (gx >= SIDE || gy >= SIDE) return;
    if (gx <= gy) { size_t idx = (size_t)gy * SIDE + gx; img[idx] = 1; }
}

// --- 3D ---
__global__ void k_3d_BB(int* vol, int SIDE) {
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;
    if (!(bx <= by && by <= bz)) return;
    int gx=bx*8+threadIdx.x; int gy=by*8+threadIdx.y; int gz=bz*4+threadIdx.z;
    if (gx >= SIDE || gy >= SIDE || gz >= SIDE) return;
    if (gx<=gy && gy<=gz) { size_t idx = ((size_t)gz * SIDE + gy) * SIDE + gx; vol[idx] = 1; }
}
__global__ void k_3d_Analytical(int* vol, int SIDE, int n_blocks) {
    int k = blockIdx.x; if (k >= n_blocks) return;
    float fz = cbrtf(6.0f * (float)k); int bz = (int)fz;
    long long tet = (long long)bz*(bz+1)*(bz+2)/6;
    if(tet>k){ bz--; tet=(long long)bz*(bz+1)*(bz+2)/6; }
    int rem = k - (int)tet;
    float fy = sqrtf(2.0f * (float)rem); int by = (int)fy;
    long long tri = (long long)by*(by+1)/2;
    if(tri>rem){ by--; tri=(long long)by*(by+1)/2; }
    int bx = rem - (int)tri;
    int gx=bx*8+threadIdx.x; int gy=by*8+threadIdx.y; int gz=bz*4+threadIdx.z;
    if (gx >= SIDE || gy >= SIDE || gz >= SIDE) return;
    if (gx<=gy && gy<=gz) { size_t idx = ((size_t)gz * SIDE + gy) * SIDE + gx; vol[idx] = 1; }
}
__global__ void k_3d_Robust(int* vol, int SIDE, int n_blocks) {
    int k = blockIdx.x; if (k >= n_blocks) return;
    int bz = (int)cbrtf(6.0f * (float)k);
    while ((long long)bz*(bz+1)*(bz+2)/6 > k) bz--;
    while ((long long)(bz+1)*(bz+2)*(bz+3)/6 <= k) bz++;
    long long tet = (long long)bz*(bz+1)*(bz+2)/6;
    int rem = k - (int)tet;
    int by = (int)sqrtf(2.0f * (float)rem);
    while ((long long)by*(by+1)/2 > rem) by--;
    int bx = rem - (by*(by+1)/2);
    int gx=bx*8+threadIdx.x; int gy=by*8+threadIdx.y; int gz=bz*4+threadIdx.z;
    if (gx >= SIDE || gy >= SIDE || gz >= SIDE) return;
    if (gx<=gy && gy<=gz) { size_t idx = ((size_t)gz * SIDE + gy) * SIDE + gx; vol[idx] = 1; }
}
__global__ void k_3d_BinSearch(int* vol, int SIDE, int n_blocks) {
    int k = blockIdx.x; if (k >= n_blocks) return;
    long long L=0, R=SIDE; 
    while(L<=R){ long long m=(L+R)/2; if(m*(m+1)*(m+2)/6 <= k) L=m+1; else R=m-1; }
    int bz = (int)R;
    int rem = k - bz*(bz+1)*(bz+2)/6;
    L=0; R=SIDE;
    while(L<=R){ long long m=(L+R)/2; if(m*(m+1)/2 <= rem) L=m+1; else R=m-1; }
    int by = (int)R;
    int bx = rem - by*(by+1)/2;
    int gx=bx*8+threadIdx.x; int gy=by*8+threadIdx.y; int gz=bz*4+threadIdx.z;
    if (gx >= SIDE || gy >= SIDE || gz >= SIDE) return;
    if (gx<=gy && gy<=gz) { size_t idx = ((size_t)gz * SIDE + gy) * SIDE + gx; vol[idx] = 1; }
}
__global__ void k_3d_Hybrid(int* vol, int SIDE, int n_blocks) {
    int k = blockIdx.x; if (k >= n_blocks) return;
    long long L=0, R=SIDE; 
    while(L<=R){ long long m=(L+R)/2; if(m*(m+1)*(m+2)/6 <= k) L=m+1; else R=m-1; }
    int bz = (int)R;
    int rem = k - bz*(bz+1)*(bz+2)/6;
    int by = 0;
    while ((long long)(by+1)*(by+2)/2 <= rem) by++;
    int bx = rem - by*(by+1)/2;
    int gx=bx*8+threadIdx.x; int gy=by*8+threadIdx.y; int gz=bz*4+threadIdx.z;
    if (gx >= SIDE || gy >= SIDE || gz >= SIDE) return;
    if (gx<=gy && gy<=gz) { size_t idx = ((size_t)gz * SIDE + gy) * SIDE + gx; vol[idx] = 1; }
}
__global__ void k_3d_Linear(int* vol, int SIDE, int n_blocks) {
    int k = blockIdx.x; if (k >= n_blocks) return;
    int bz = 0;
    while(1) {
        long long sz = (long long)(bz+1)*(bz+2)/2;
        if (k < sz) break;
        k -= sz;
        bz++;
    }
    int by = 0;
    while ((long long)(by+1) <= k) { k -= (by+1); by++; }
    int bx = k;
    int gx=bx*8+threadIdx.x; int gy=by*8+threadIdx.y; int gz=bz*4+threadIdx.z;
    if (gx >= SIDE || gy >= SIDE || gz >= SIDE) return;
    if (gx<=gy && gy<=gz) { size_t idx = ((size_t)gz * SIDE + gy) * SIDE + gx; vol[idx] = 1; }
}

// --- SIERPINSKI ---
__global__ void k_sierp2d_BB(int* img, int SIDE) {
    int bx = blockIdx.x; int by = blockIdx.y; if ((bx & by) != 0) return; 
    int gx = bx * 16 + threadIdx.x; int gy = by * 16 + threadIdx.y;
    if (gx >= SIDE || gy >= SIDE) return;
    if ((gx & gy) == 0) { size_t idx = (size_t)gy * SIDE + gx; img[idx] = 1; }
}
__global__ void k_sierp2d_Bit(int* img, int SIDE, int n_blocks) {
    int n = blockIdx.x; if (n >= n_blocks) return;
    int bx=0; int by=0; int s=1;
    while(n>0) { if(n&1) bx|=s; else by|=s; n>>=1; s<<=1; }
    int gx = bx * 16 + threadIdx.x; int gy = by * 16 + threadIdx.y;
    if (gx >= SIDE || gy >= SIDE) return;
    if ((gx & gy) == 0) { size_t idx = (size_t)gy * SIDE + gx; img[idx] = 1; }
}
__global__ void k_sierp3d_BB(int* vol, int SIDE) {
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;
    if ((bx&by)!=0 || (bx&bz)!=0 || (by&bz)!=0) return;
    int gx=bx*8+threadIdx.x; int gy=by*8+threadIdx.y; int gz=bz*4+threadIdx.z;
    if (gx >= SIDE || gy >= SIDE || gz >= SIDE) return;
    if ((gx&gy)==0 && (gx&gz)==0 && (gy&gz)==0) { size_t idx = ((size_t)gz * SIDE + gy) * SIDE + gx; vol[idx] = 1; }
}
__global__ void k_sierp3d_Bit(int* vol, int SIDE, int n_blocks) {
    int n = blockIdx.x; if (n >= n_blocks) return;
    int bx=0; int by=0; int bz=0; int s=1;
    while(n>0) { int d=n&3; if(d==1) bx|=s; else if(d==2) by|=s; else if(d==3) bz|=s; n>>=2; s<<=1; }
    int gx=bx*8+threadIdx.x; int gy=by*8+threadIdx.y; int gz=bz*4+threadIdx.z;
    if (gx >= SIDE || gy >= SIDE || gz >= SIDE) return;
    if ((gx&gy)==0 && (gx&gz)==0 && (gy&gz)==0) { size_t idx = ((size_t)gz * SIDE + gy) * SIDE + gx; vol[idx] = 1; }
}

// =================================================================================
// MONITORING
// =================================================================================

void setup_nvml() {
    CHECK_NVML(nvmlInit());
    // Resolve through its PCI bus id the GPU that CUDA was given. Under SLURM
    // the NVML numbering is the physical one of the node and may differ from
    // the CUDA numbering, so a fixed index could read the GPU of another job.
    int cuda_dev = 0; char pci[64];
    CHECK_CUDA(cudaGetDevice(&cuda_dev));
    CHECK_CUDA(cudaDeviceGetPCIBusId(pci, sizeof(pci), cuda_dev));
    CHECK_NVML(nvmlDeviceGetHandleByPciBusId(pci, &nvml_device));
    printf("measuring GPU %s\n", pci);
}

unsigned long long get_energy() {
    unsigned long long energy;
    CHECK_NVML(nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy));
    return energy;
}

// Runs in a separate thread and samples instantaneous power for the peak
void monitor_peak_power(std::atomic<bool>& keep_running, unsigned int& peak_power_mw) {
    unsigned int power_mw = 0;
    peak_power_mw = 0;
    while (keep_running) {
        nvmlReturn_t res = nvmlDeviceGetPowerUsage(nvml_device, &power_mw);
        if (res == NVML_SUCCESS && power_mw > peak_power_mw) {
            peak_power_mw = power_mw;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void dual_print(FILE* fp, const char* fmt, ...) {
    va_list args1, args2;
    va_start(args1, fmt); va_copy(args2, args1);
    vprintf(fmt, args1);
    if(fp) vfprintf(fp, fmt, args2);
    va_end(args1); va_end(args2);
}

void print_header(FILE* fp) {
    dual_print(fp, "%-30s | %10s | %9s | %9s | %10s | %8s | %-18s | %s\n",
            "Model (Samples)", "Time", "Avg Power", "Peak Pwr", "Energy", "Launches", "Algorithm Logic", "Complexity");
    dual_print(fp, "-------------------------------------------------------------------------------------------------------------------------------------\n");
}

void print_row(FILE* fp, const char* model, double avg_ms, unsigned long long startE, unsigned long long endE, 
               unsigned int peak_mw, const char* logic, const char* complexity, int iterations_run) {
    
    double total_energy_mj = (double)(endE - startE);
    double total_time_s = (avg_ms * iterations_run) / 1000.0;
    
    double avg_power_w = 0.0;
    if(total_time_s > 0) avg_power_w = (total_energy_mj / 1000.0) / total_time_s;

    double peak_power_w = peak_mw / 1000.0;

    // Energy of ONE kernel launch: energy drawn during the measurement divided
    // by the number of launches (equal to avg. power x avg. time).
    double energy_per_launch_j = (total_energy_mj / 1000.0) / iterations_run;

    dual_print(fp, "%-30s | %7.3f ms | %7.1f W | %7.1f W | %8.3f J | %8d | %-18s | %s\n",
            model, avg_ms, avg_power_w, peak_power_w, energy_per_launch_j, iterations_run, logic, complexity);
}

// =================================================================================
// MAIN
// =================================================================================

int main() {
    const char* filename = "gpu_benchmark_results.txt";
    FILE* fp = fopen(filename, "w");
    if(!fp) { printf("Error file\n"); return 1; }

    setup_nvml();

    dual_print(fp, "=== Block-level thread map benchmark ===\n");
    dual_print(fp, "Target Points: %d\n", NUM_POINTS);
    dual_print(fp, "Minimum measured time per row: %.1f s\n", TARGET_TIME_MS/1000.0);

    int SIDE_2D = (int)sqrt(2.0 * NUM_POINTS) + 128; 
    size_t MEM_2D = (size_t)SIDE_2D * SIDE_2D * sizeof(int);
    int SIDE_3D = (int)cbrtf(6.0 * NUM_POINTS) + 64;
    size_t MEM_3D = (size_t)SIDE_3D * SIDE_3D * SIDE_3D * sizeof(int);

    dual_print(fp, "VRAM: ~%.2f GB\n\n", (MEM_2D + MEM_3D)/(1024.0*1024.0*1024.0));

    int *d_grid2d, *d_grid3d;
    CHECK_CUDA(cudaMalloc(&d_grid2d, MEM_2D));
    CHECK_CUDA(cudaMalloc(&d_grid3d, MEM_3D));
    CHECK_CUDA(cudaMemset(d_grid2d, 0, MEM_2D));
    CHECK_CUDA(cudaMemset(d_grid3d, 0, MEM_3D));

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms_warmup = 0;
    unsigned long long e_start, e_end;
    int N_BLOCKS = (NUM_POINTS + 255) / 256; 
    
    // --- measurement of one row ---
    #define RUN_FULL(name, logic, complex, ...) \
        { \
            CHECK_CUDA(cudaDeviceSynchronize()); \
            /* 1. warm-up and time estimate */    \
            cudaEventRecord(start); \
            for(int i=0; i<5; i++) { __VA_ARGS__; } \
            cudaEventRecord(stop); \
            cudaEventSynchronize(stop); \
            cudaEventElapsedTime(&ms_warmup, start, stop); \
            double avg_ms = ms_warmup / 5.0; \
            int adaptive_iters = (int)(TARGET_TIME_MS / avg_ms); \
            if (adaptive_iters < 10) adaptive_iters = 10; \
            \
            /* 2. peak power monitor */         \
            std::atomic<bool> keep_monitoring(true); \
            unsigned int peak_mw = 0; \
            std::thread monitor_thread(monitor_peak_power, std::ref(keep_monitoring), std::ref(peak_mw)); \
            \
            /* 3. measured launches */ \
            CHECK_CUDA(cudaDeviceSynchronize()); \
            e_start = get_energy(); \
            cudaEventRecord(start); \
            for(int i=0; i<adaptive_iters; i++) { __VA_ARGS__; } \
            cudaEventRecord(stop); \
            cudaEventSynchronize(stop); \
            e_end = get_energy(); \
            \
            /* 4. stop monitor */ \
            keep_monitoring = false; \
            monitor_thread.join(); \
            \
            /* 5. report */  \
            float total_ms = 0; \
            cudaEventElapsedTime(&total_ms, start, stop); \
            print_row(fp, name, total_ms/adaptive_iters, e_start, e_end, peak_mw, logic, complex, adaptive_iters); \
        }

    // ===========================================================================
    // DOMAIN 1: 2D TRIANGULAR MAPPING
    // ===========================================================================
    dual_print(fp, "\n>>> DOMAIN: 2D TRIANGULAR MAPPING\n");
    print_header(fp);

    int TSIDE_2D = (SIDE_2D + TILE_DIM - 1) / TILE_DIM;
    dim3 grid(TSIDE_2D, TSIDE_2D); dim3 blk(TILE_DIM, TILE_DIM);
    
    RUN_FULL("Bounding Box (baseline)", "If(x<=y)", "O(1)", k_2d_BB<<<grid,blk>>>(d_grid2d, SIDE_2D));
    RUN_FULL("Paper (Navarro 2018)", "Analytical", "O(1)", k_2d_Analytical<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    RUN_FULL("R1:70b (20 Samples)", "Analytical", "O(1)", k_2d_Analytical<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    RUN_FULL("R1:70b (50 Samples)", "Analytical", "O(1)", k_2d_Analytical<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    RUN_FULL("OSS:120b (20 Samples)", "Analytical", "O(1)", k_2d_Analytical<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    RUN_FULL("OSS:120b (50 Samples)", "Analytical", "O(1)", k_2d_Analytical<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    RUN_FULL("OSS:120b (100 Samples)", "Analytical", "O(1)", k_2d_Analytical<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    RUN_FULL("Lla3.3:70b (20 Samples)", "Analytical", "O(1)", k_2d_Analytical<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    RUN_FULL("Qw3:235b (20 Samples)", "Analytical", "O(1)", k_2d_Analytical<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    RUN_FULL("Qw3:32b (20 Samples)", "Analytical", "O(1)", k_2d_Analytical<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    RUN_FULL("Qw3:32b (100 Samples)", "Analytical", "O(1)", k_2d_Analytical<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    
    RUN_FULL("R1:70b (100 Samples)", "Sqrt + Loop", "O(1)", k_2d_Robust<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    
    RUN_FULL("OSS:20b (50 Samples)", "Approx + If", "O(1)", k_2d_Approx<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    RUN_FULL("OSS:20b (100 Samples)", "Approx + If", "O(1)", k_2d_Approx<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));
    
    RUN_FULL("Qw3:32b (50 Samples)", "Binary Search", "O(log N)", k_2d_BinSearch<<<N_BLOCKS,blk>>>(d_grid2d, SIDE_2D, N_BLOCKS));

    // ===========================================================================
    // DOMAIN 2: 3D PYRAMID MAPPING
    // ===========================================================================
    dual_print(fp, "\n>>> DOMAIN: 3D PYRAMID MAPPING\n");
    print_header(fp);

    int TSIDE_3D_XY = (SIDE_3D + 7) / 8;
    int TSIDE_3D_Z  = (SIDE_3D + 3) / 4;
    dim3 grid3d(TSIDE_3D_XY, TSIDE_3D_XY, TSIDE_3D_Z); dim3 blk3d(8, 8, 4);
    
    RUN_FULL("Bounding Box (baseline)", "If(x<=y<=z)", "O(1)", k_3d_BB<<<grid3d,blk3d>>>(d_grid3d, SIDE_3D));
    RUN_FULL("Paper (Navarro 2016)", "Analytical", "O(1)", k_3d_Analytical<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));
    RUN_FULL("OSS:120b (100 Samples)", "Analytical", "O(1)", k_3d_Analytical<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));

    RUN_FULL("R1:70b (50 Samples)", "Cbrt + Loop", "O(1)", k_3d_Robust<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));
    RUN_FULL("Qw3:32b (20 Samples)", "Cbrt + Loop", "O(1)", k_3d_Robust<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));
    RUN_FULL("Qw3:32b (50 Samples)", "Cbrt + Loop", "O(1)", k_3d_Robust<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));
    RUN_FULL("Qw3:32b (100 Samples)", "Cbrt + Loop", "O(1)", k_3d_Robust<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));

    RUN_FULL("OSS:20b (100 Samples)", "BinSearch Z&Y", "O(log N)", k_3d_BinSearch<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));
    RUN_FULL("Qw3:235b (20 Samples)", "BinSearch Z&Y", "O(log N)", k_3d_BinSearch<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));

    RUN_FULL("OSS:120b (50 Samples)", "BinSearch+Lin", "O(N^1/3)", k_3d_Hybrid<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));

    // The linear-search kernels are slow: the adaptive count falls to about 25 launches
    RUN_FULL("OSS:120b (20 Samples)", "Linear Search", "O(N^1/3)", k_3d_Linear<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));
    RUN_FULL("OSS:20b (50 Samples)", "Linear Search", "O(N^1/3)", k_3d_Linear<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));

    // ===========================================================================
    // DOMAIN 3: 2D SIERPINSKI
    // ===========================================================================
    dual_print(fp, "\n>>> DOMAIN: 2D SIERPINSKI FRACTAL\n");
    print_header(fp);

    int S2D = (int)pow(N_BLOCKS, 0.631) + 100; if(S2D > SIDE_2D) S2D = SIDE_2D;
    dim3 grid2(S2D, S2D);
    
    RUN_FULL("Bounding Box (baseline)", "If(Fractal)", "O(1)", k_sierp2d_BB<<<grid2,blk>>>(d_grid2d, S2D));
    RUN_FULL("Paper (Navarro 2017)", "Bitwise", "O(log N)", k_sierp2d_Bit<<<N_BLOCKS,blk>>>(d_grid2d, S2D, N_BLOCKS));
    RUN_FULL("OSS:120b (100 Samples)", "Bitwise", "O(log N)", k_sierp2d_Bit<<<N_BLOCKS,blk>>>(d_grid2d, S2D, N_BLOCKS));

    // ===========================================================================
    // DOMAIN 4: 3D SIERPINSKI
    // ===========================================================================
    dual_print(fp, "\n>>> DOMAIN: 3D SIERPINSKI FRACTAL\n");
    print_header(fp);

    // S3D_blocks: side of the bounding box that encloses the fractal.
    // SIDE_3D: stride of the array (bounded by VRAM).
    // The BB launches grid(S3D_blocks/8, S3D_blocks/8, S3D_blocks/4) and writes
    // only inside [0, SIDE_3D), so the full cost of discarding blocks is paid.
    int S3D_blocks = (int)pow(N_BLOCKS, 0.5) * 2;
    int TSIDE_S3D_XY = (S3D_blocks + 7) / 8;
    int TSIDE_S3D_Z  = (S3D_blocks + 3) / 4;
    dim3 grid3(TSIDE_S3D_XY, TSIDE_S3D_XY, TSIDE_S3D_Z);

    RUN_FULL("Bounding Box (baseline)", "If(Fractal)", "O(1)", k_sierp3d_BB<<<grid3,blk3d>>>(d_grid3d, SIDE_3D));
    RUN_FULL("This work (closed form)", "Bitwise", "O(log N)", k_sierp3d_Bit<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));
    RUN_FULL("OSS:120b (100 Samples)", "Bitwise", "O(log N)", k_sierp3d_Bit<<<N_BLOCKS,blk3d>>>(d_grid3d, SIDE_3D, N_BLOCKS));

    cudaFree(d_grid2d); cudaFree(d_grid3d);
    CHECK_NVML(nvmlShutdown());
    fclose(fp);
    printf("\n=== done, results in gpu_benchmark_results.txt ===\n");
    return 0;
}