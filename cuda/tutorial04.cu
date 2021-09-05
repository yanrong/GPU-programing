#include <stdio.h>

/*Each thread writes to one block of 256 elements of global memory and contends for wirte access*/
__global__ void myhistory256kernel_01(
    const unsigned char const * d_hist_data,
    unsigned int * const d_bin_data)
{
    /*Work out our thread id*/
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;
    /*Fetch the data value*/
    const unsigned char value = d_hist_data[tid];
    atomicAdd(&(d_bin_data[value]), 1);
}

/*Each read is 4 bytes, not one, 32 x 4 = 128 byte reads*/
__global__ void myhistory256kernel_02(
    const unsigned char const * d_hist_data,
    unsigned int * const d_bin_data)
{
    /*Work out our thread id*/
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;
    /*Fetch the data value*/
    const unsigned int value_32 = d_hist_data[tid];
    atomicAdd(&(d_bin_data[((value_32 & 0x000000FF))]),1);
    atomicAdd(&(d_bin_data[((value_32 & 0x0000FF00) >> 8)]),1);
    atomicAdd(&(d_bin_data[((value_32 & 0x00FF0000) >> 16)]),1);
    atomicAdd(&(d_bin_data[((value_32 & 0xFF000000) >> 24)]),1);
}

__shared__ unsigned int d_bin_data_share[256];
/*Each read is 4 bytes, not one, 32 x 4 = 128 byte reads*/
__global__ void myhistory256kernel_03(
    const unsigned char const * d_hist_data,
    unsigned int * const d_bin_data)
{
    /*Work out our thread id*/
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;
    /*Clear shared memory*/
    d_bin_data_share[threadIdx.x] = 0;
    /*Fetch the data value as 32 bit*/
    const unsigned int value_32 = d_hist_data[tid];
    /*Wait for all threads to update shared memory*/
    __syncthreads();
    atomicAdd(&(d_bin_data_shared[((value_32 & 0x000000FF))]),1);
    atomicAdd(&(d_bin_data_shared[((value_32 & 0x0000FF00) >> 8)]),1);
    atomicAdd(&(d_bin_data_shared[((value_32 & 0x00FF0000) >> 16)]),1);
    atomicAdd(&(d_bin_data_shared[((value_32 & 0xFF000000) >> 24)]),1);
    /*Wait for all threads to update shared memory*/
    __syncthreads();
    /*The write the accumulated data back to global memory in blocks, not scattered*/
    atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_share[threadIdx.x]);
}

/*Each read is 4 bytes, not one, 32 x 4 = 128 byte reads*/
/*Accumulate int o shared memory N times*/
__global__ void myhistory256kernel_07(
    const unsigned char const * d_hist_data,
    unsigned int * const d_bin_data,
    unsigned int N)
{
    /*Work out our thread id*/
    const unsigned int idx = (blockIdx.x * (blockDim.x * N)) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int tid = idx + idy * (blockDim.x * N) * gridDim.x;
    /*Clear shared memory*/
    d_bin_data_share[threadIdx.x] = 0;

    /*Wait for all threads to update shared memory*/
    __syncthreads();
    for (int i = 0, tid_offset = 0; i < N; i++, tid_offset += 256)
    {
        const unsigned int value_32 = d_hist_data[tid + tid_offset];
        atomicAdd(&(d_bin_data_shared[((value_32 & 0x000000FF))]),1);
        atomicAdd(&(d_bin_data_shared[((value_32 & 0x0000FF00) >> 8)]),1);
        atomicAdd(&(d_bin_data_shared[((value_32 & 0x00FF0000) >> 16)]),1);
        atomicAdd(&(d_bin_data_shared[((value_32 & 0xFF000000) >> 24)]),1);
    }
    /*Wait for all threads to update shared memory*/
    __syncthreads();
    /*The write the accumulated data back to global memory in blocks, not scattered*/
    atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_share[threadIdx.x]);
}