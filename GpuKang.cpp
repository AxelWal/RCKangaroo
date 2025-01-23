// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

#include "GpuKang.h"

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
void CallGpuKernelGen(TKparams Kparams);
void CallGpuKernelABC(TKparams Kparams);
void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
extern bool gGenMode; //tames generation mode

int RCGpuKang::CalcKangCnt()
{
    // Query device properties
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, CudaIndex);
    
    // Calculate optimal block and group sizes based on GPU architecture
    if (!IsOldGpu) {
        // Use warp-aligned block size for better memory coalescing
        int maxWarpsPerSM = props.maxThreadsPerMultiProcessor / props.warpSize;
        int optimalBlockSize = props.warpSize * 8;  // Use 8 warps per block
        
        // Ensure block size doesn't exceed device limits
        Kparams.BlockSize = std::min(optimalBlockSize, props.maxThreadsPerBlock);
        
        // Calculate group count to maximize occupancy
        int blocksPerSM = props.maxThreadsPerMultiProcessor / Kparams.BlockSize;
        Kparams.GroupCnt = (props.multiProcessorCount * blocksPerSM * 2) / 3; // Use 2/3 of max blocks
        
        // Ensure we don't exceed reasonable limits
        if (Kparams.GroupCnt > 32) {
            Kparams.GroupCnt = 32;
        }
    } else {
        // Keep original values for old GPUs
        Kparams.BlockSize = 512;
        Kparams.GroupCnt = 64;
    }
    
    Kparams.BlockCnt = mpCnt;
    return Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
}

//executes in main thread
bool RCGpuKang::Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
    PntToSolve = _PntToSolve;
    Range = _Range;
    DP = _DP;
    EcJumps1 = _EcJumps1;
    EcJumps2 = _EcJumps2;
    EcJumps3 = _EcJumps3;
    StopFlag = false;
    Failed = false;
    u64 total_mem = 0;
    memset(dbg, 0, sizeof(dbg));
    memset(SpeedStats, 0, sizeof(SpeedStats));
    cur_stats_ind = 0;

    cudaError_t err;
    err = cudaSetDevice(CudaIndex);
    if (err != cudaSuccess)
        return false;

    // Set device flags for optimal performance
    err = cudaSetDeviceFlags(
        cudaDeviceScheduleAuto |      // Let driver choose best scheduling
        cudaDeviceMapHost |           // Enable zero-copy memory
        cudaDeviceLmemResizeToMax     // Maximize local memory
    );
    if (err != cudaSuccess) {
        printf("GPU %d, Failed to set device flags: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    // Query device properties for optimization
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, CudaIndex);
    
    // Optimize thread and block configuration
    if (!IsOldGpu) {
        // Use warp-aligned block size
        int maxWarpsPerSM = props.maxThreadsPerMultiProcessor / props.warpSize;
        int optimalBlockSize = props.warpSize * 8;
        
        Kparams.BlockSize = std::min(optimalBlockSize, props.maxThreadsPerBlock);
        
        // Calculate group count for optimal occupancy
        int blocksPerSM = props.maxThreadsPerMultiProcessor / Kparams.BlockSize;
        Kparams.GroupCnt = (props.multiProcessorCount * blocksPerSM * 2) / 3;
        
        if (Kparams.GroupCnt > 32) {
            Kparams.GroupCnt = 32;
        }
    } else {
        Kparams.BlockSize = 512;
        Kparams.GroupCnt = 64;
    }
    
    Kparams.BlockCnt = mpCnt;
    KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
    Kparams.KangCnt = KangCnt;
    Kparams.DP = DP;
    
    // Optimize memory alignment for coalesced access
    int warpSize = props.warpSize;
    const int memoryAlignment = warpSize * 4;  // Align to 128 bytes for coalesced access
    
    // Ensure kernel sizes are warp-aligned for better memory access
    Kparams.KernelA_LDS_Size = ((64 * JMP_CNT + 16 * Kparams.BlockSize + memoryAlignment - 1) / memoryAlignment) * memoryAlignment;
    Kparams.KernelB_LDS_Size = ((64 * JMP_CNT + memoryAlignment - 1) / memoryAlignment) * memoryAlignment;
    Kparams.KernelC_LDS_Size = ((96 * JMP_CNT + memoryAlignment - 1) / memoryAlignment) * memoryAlignment;
    
    Kparams.IsGenMode = gGenMode;

    // Allocate GPU memory with optimal alignment
    u64 size;
    if (!IsOldGpu)
    {
        // Calculate optimal L2 cache size based on working set and warp size
        size_t workingSetPerWarp = Kparams.BlockSize * sizeof(float) * 4;
        int warpsPerBlock = Kparams.BlockSize / warpSize;
        size_t totalWorkingSet = workingSetPerWarp * warpsPerBlock;
        
        // Align to cache line size
        int L2size = ((Kparams.KangCnt * (3 * 32) + 127) & ~127);
        total_mem += L2size;
        
        err = cudaMalloc((void**)&Kparams.L2, L2size);
        if (err != cudaSuccess)
        {
            printf("GPU %d, Allocate L2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
            return false;
        }
        
        // Configure L2 cache for optimal throughput
        int l2CacheSize;
        cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, CudaIndex);
        
        // Use device's full L2 cache capacity if available
        size = std::min(totalWorkingSet, (size_t)l2CacheSize);
        err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
        
        // Configure stream for optimal cache usage
        cudaStreamAttrValue stream_attribute;
        stream_attribute.accessPolicyWindow.base_ptr = Kparams.L2;
        stream_attribute.accessPolicyWindow.num_bytes = size;
        stream_attribute.accessPolicyWindow.hitRatio = 0.6;  // Lower hit ratio for better throughput
        stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        
        err = cudaStreamSetAttribute(NULL, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
        if (err != cudaSuccess)
        {
            printf("GPU %d, cudaStreamSetAttribute failed: %s\n", CudaIndex, cudaGetErrorString(err));
            return false;
        }
    }
    size = MAX_DP_CNT * GPU_DP_SIZE + 16;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.DPs_out, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = KangCnt * 96;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.Kangs, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate pKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += JMP_CNT * 96;
    err = cudaMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += JMP_CNT * 96;
    err = cudaMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += JMP_CNT * 96;
    err = cudaMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate Jumps3 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = 2 * (u64)KangCnt * STEP_CNT;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.JumpsList, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.DPTable, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate DPTable memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = mpCnt * Kparams.BlockSize * sizeof(u64);
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.L1S2, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate L1S2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = (u64)KangCnt * MD_LEN * (2 * 32);
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.LastPnts, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = (u64)KangCnt * MD_LEN * sizeof(u64);
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.LoopTable, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += 1024;
    err = cudaMalloc((void**)&Kparams.dbg_buf, 1024);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate dbg_buf memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = sizeof(u32) * KangCnt + 8;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.LoopedKangs, size);
    if (err != cudaSuccess)
    {
        printf("GPU %d Allocate LoopedKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    DPs_out = (u32*)malloc(MAX_DP_CNT * GPU_DP_SIZE);

//jmp1
    u64* buf = (u64*)malloc(JMP_CNT * 96);
    for (int i = 0; i < JMP_CNT; i++)
    {
        memcpy(buf + i * 12, EcJumps1[i].p.x.data, 32);
        memcpy(buf + i * 12 + 4, EcJumps1[i].p.y.data, 32);
        memcpy(buf + i * 12 + 8, EcJumps1[i].dist.data, 32);
    }
    err = cudaMemcpy(Kparams.Jumps1, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("GPU %d, cudaMemcpy Jumps1 failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    free(buf);
//jmp2
    buf = (u64*)malloc(JMP_CNT * 96);
    u64* jmp2_table = (u64*)malloc(JMP_CNT * 64);
    for (int i = 0; i < JMP_CNT; i++)
    {
        memcpy(buf + i * 12, EcJumps2[i].p.x.data, 32);
        memcpy(jmp2_table + i * 8, EcJumps2[i].p.x.data, 32);
        memcpy(buf + i * 12 + 4, EcJumps2[i].p.y.data, 32);
        memcpy(jmp2_table + i * 8 + 4, EcJumps2[i].p.y.data, 32);
        memcpy(buf + i * 12 + 8, EcJumps2[i].dist.data, 32);
    }
    err = cudaMemcpy(Kparams.Jumps2, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("GPU %d, cudaMemcpy Jumps2 failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    free(buf);

    err = cuSetGpuParams(Kparams, jmp2_table);
    if (err != cudaSuccess)
    {
        free(jmp2_table);
        printf("GPU %d, cuSetGpuParams failed: %s!\r\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    free(jmp2_table);
//jmp3
    buf = (u64*)malloc(JMP_CNT * 96);
    for (int i = 0; i < JMP_CNT; i++)
    {
        memcpy(buf + i * 12, EcJumps3[i].p.x.data, 32);
        memcpy(buf + i * 12 + 4, EcJumps3[i].p.y.data, 32);
        memcpy(buf + i * 12 + 8, EcJumps3[i].dist.data, 32);
    }
    err = cudaMemcpy(Kparams.Jumps3, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("GPU %d, cudaMemcpy Jumps3 failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    free(buf);

    printf("GPU %d: allocated %llu MB, %d kangaroos. OldGpuMode: %s\r\n", CudaIndex, total_mem / (1024 * 1024), KangCnt, IsOldGpu ? "Yes" : "No");
    return true;
}

bool RCGpuKang::Init(int _CudaIndex)
{
    CudaIndex = _CudaIndex;
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, CudaIndex);
    
    // Force new GPU mode for RTX 30 series
    if (props.major >= 8 && props.minor >= 6) {
        IsOldGpu = false;  // RTX 3070 has compute capability 8.6
    } else {
        IsOldGpu = true;
    }
    
    return true;
}

void RCGpuKang::Release()
{
    free(RndPnts);
    free(DPs_out);
    cudaFree(Kparams.LoopedKangs);
    cudaFree(Kparams.dbg_buf);
    cudaFree(Kparams.LoopTable);
    cudaFree(Kparams.LastPnts);
    cudaFree(Kparams.L1S2);
    cudaFree(Kparams.DPTable);
    cudaFree(Kparams.JumpsList);
    cudaFree(Kparams.Jumps3);
    cudaFree(Kparams.Jumps2);
    cudaFree(Kparams.Jumps1);
    cudaFree(Kparams.Kangs);
    cudaFree(Kparams.DPs_out);
    if (!IsOldGpu)
        cudaFree(Kparams.L2);
}

void RCGpuKang::Stop()
{
    StopFlag = true;
}

void RCGpuKang::GenerateRndDistances()
{
    for (int i = 0; i < KangCnt; i++)
    {
        EcInt d;
        if (i < KangCnt / 3)
            d.RndBits(Range - 4); //TAME kangs
        else
        {
            d.RndBits(Range - 1);
            d.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
        }
        memcpy(RndPnts[i].priv, d.data, 24);
    }
}

bool RCGpuKang::Start()
{
    if (Failed)
        return false;

    cudaError_t err;
    err = cudaSetDevice(CudaIndex);
    if (err != cudaSuccess)
        return false;

    HalfRange.Set(1);
    HalfRange.ShiftLeft(Range - 1);
    PntHalfRange = ec.MultiplyG(HalfRange);
    NegPntHalfRange = PntHalfRange;
    NegPntHalfRange.y.NegModP();

    PntA = ec.AddPoints(PntToSolve, NegPntHalfRange);
    PntB = PntA;
    PntB.y.NegModP();

    RndPnts = (TPointPriv*)malloc(KangCnt * 96);
    GenerateRndDistances();
/* 
    //we can calc start points on CPU
    for (int i = 0; i < KangCnt; i++)
    {
        EcInt d;
        memcpy(d.data, RndPnts[i].priv, 24);
        d.data[3] = 0;
        d.data[4] = 0;
        EcPoint p = ec.MultiplyG(d);
        memcpy(RndPnts[i].x, p.x.data, 32);
        memcpy(RndPnts[i].y, p.y.data, 32);
    }
    for (int i = KangCnt / 3; i < 2 * KangCnt / 3; i++)
    {
        EcPoint p;
        p.LoadFromBuffer64((u8*)RndPnts[i].x);
        p = ec.AddPoints(p, PntA);
        p.SaveToBuffer64((u8*)RndPnts[i].x);
    }
    for (int i = 2 * KangCnt / 3; i < KangCnt; i++)
    {
        EcPoint p;
        p.LoadFromBuffer64((u8*)RndPnts[i].x);
        p = ec.AddPoints(p, PntB);
        p.SaveToBuffer64((u8*)RndPnts[i].x);
    }
    //copy to gpu
    err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
/**/
    //but it's faster to calc then on GPU
    u8 buf_PntA[64], buf_PntB[64];
    PntA.SaveToBuffer64(buf_PntA);
    PntB.SaveToBuffer64(buf_PntB);
    for (int i = 0; i < KangCnt; i++)
    {
        if (i < KangCnt / 3)
            memset(RndPnts[i].x, 0, 64);
        else
            if (i < 2 * KangCnt / 3)
                memcpy(RndPnts[i].x, buf_PntA, 64);
            else
                memcpy(RndPnts[i].x, buf_PntB, 64);
    }
    //copy to gpu
    err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    CallGpuKernelGen(Kparams);

    err = cudaMemset(Kparams.L1S2, 0, mpCnt * Kparams.BlockSize * 8);
    if (err != cudaSuccess)
        return false;
    cudaMemset(Kparams.dbg_buf, 0, 1024);
    cudaMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
    return true;
}

#ifdef DEBUG_MODE
int RCGpuKang::Dbg_CheckKangs()
{
    int kang_size = mpCnt * Kparams.BlockSize * Kparams.GroupCnt * 96;
    u64* kangs = (u64*)malloc(kang_size);
    cudaError_t err = cudaMemcpy(kangs, Kparams.Kangs, kang_size, cudaMemcpyDeviceToHost);
    int res = 0;
    for (int i = 0; i < KangCnt; i++)
    {
        EcPoint Pnt, p;
        Pnt.LoadFromBuffer64((u8*)&kangs[i * 12 + 0]);
        EcInt dist;
        dist.Set(0);
        memcpy(dist.data, &kangs[i * 12 + 8], 24);
        bool neg = false;
        if (dist.data[2] >> 63)
        {
            neg = true;
            memset(((u8*)dist.data) + 24, 0xFF, 16);
            dist.Neg();
        }
        p = ec.MultiplyG_Fast(dist);
        if (neg)
            p.y.NegModP();
        if (i < KangCnt / 3)
            p = p;
        else
            if (i < 2 * KangCnt / 3)
                p = ec.AddPoints(PntA, p);
            else
                p = ec.AddPoints(PntB, p);
        if (!p.IsEqual(Pnt))
            res++;
    }
    free(kangs);
    return res;
}
#endif

extern u32 gTotalErrors;

//executes in separate thread
void RCGpuKang::Execute()
{
    cudaSetDevice(CudaIndex);

    if (!Start())
    {
        gTotalErrors++;
        return;
    }
#ifdef DEBUG_MODE
    u64 iter = 1;
#endif
    cudaError_t err;	
    while (!StopFlag)
    {
        u64 t1 = GetTickCount64();
        cudaMemset(Kparams.DPs_out, 0, 4);
        cudaMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32));
        cudaMemset(Kparams.LoopedKangs, 0, 8);
        CallGpuKernelABC(Kparams);
        int cnt;
        err = cudaMemcpy(&cnt, Kparams.DPs_out, 4, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            printf("GPU %d, CallGpuKernel failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
            gTotalErrors++;
            break;
        }
        
        if (cnt >= MAX_DP_CNT)
        {
            cnt = MAX_DP_CNT;
            printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", CudaIndex);
        }
        u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

        if (cnt)
        {
            err = cudaMemcpy(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                gTotalErrors++;
                break;
            }
            AddPointsToList(DPs_out, cnt, (u64)KangCnt * STEP_CNT);
        }

        //dbg
        cudaMemcpy(dbg, Kparams.dbg_buf, 1024, cudaMemcpyDeviceToHost);

        u32 lcnt;
        cudaMemcpy(&lcnt, Kparams.LoopedKangs, 4, cudaMemcpyDeviceToHost);
        //printf("GPU %d, Looped: %d\r\n", CudaIndex, lcnt);

        u64 t2 = GetTickCount64();
        u64 tm = t2 - t1;
        if (!tm)
            tm = 1;
        int cur_speed = (int)(pnt_cnt / (tm * 1000));
        //printf("GPU %d kernel time %d ms, speed %d MH\r\n", CudaIndex, (int)tm, cur_speed);

        SpeedStats[cur_stats_ind] = cur_speed;
        cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
        if ((iter % 300) == 0)
        {
            int corr_cnt = Dbg_CheckKangs();
            if (corr_cnt)
            {
                printf("DBG: GPU %d, KANGS CORRUPTED: %d\r\n", CudaIndex, corr_cnt);
                gTotalErrors++;
            }
            else
                printf("DBG: GPU %d, ALL KANGS OK!\r\n", CudaIndex);
        }
        iter++;
#endif
    }

    Release();
}

int RCGpuKang::GetStatsSpeed()
{
    int res = SpeedStats[0];
    for (int i = 1; i < STATS_WND_SIZE; i++)
        res += SpeedStats[i];
    return res / STATS_WND_SIZE;
}