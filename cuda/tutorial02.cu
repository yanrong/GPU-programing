#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 128
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))
unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];
unsigned int cpu_warp[ARRAY_SIZE];
unsigned int cpu_calc_thread[ARRAY_SIZE];

__global__ void what_is_my_id(unsigned int *const block, unsigned int * const thread,
	unsigned int *const warp, unsigned int * const calc_thread)
{
	/*Thread is the block index * block size + thread offset into the block */
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
	/*Calculate warp using build in variable warpSize*/
	warp[thread_idx] = threadIdx.x / warpSize;
	calc_thread[thread_idx] = thread_idx;
}

int main(int argc, char *argv[])
{
	/*Totoal thread count = 2 * 64 = 128*/
	const unsigned int num_blocks = 2;
	const unsigned int num_thread = 64;
	char ch;
	/*Declare pointer for GPU base params*/
	unsigned int *gpu_block;
	unsigned int *gpu_thread;
	unsigned int *gpu_warp;
	unsigned int *gpu_calc_thread;

	/*Decalre loop counter for user later*/
	unsigned int i;

	cudaMallocManaged((void**)&gpu_block, ARRAY_SIZE_IN_BYTES);
	cudaMallocManaged((void**)&gpu_thread, ARRAY_SIZE_IN_BYTES);
	cudaMallocManaged((void**)&gpu_warp, ARRAY_SIZE_IN_BYTES);
	cudaMallocManaged((void**)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);

	what_is_my_id<<<num_blocks, num_thread>>>(gpu_block, gpu_thread, gpu_warp, gpu_calc_thread);
	cudaDeviceSynchronize();

	/*Copy back the gpu results to the GPU*/
	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_warp, gpu_warp, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	/*Free the arrays on the GPU as now we're done with them */

	cudaFree(gpu_block);
	cudaFree(gpu_thread);
	cudaFree(gpu_warp);
	cudaFree(gpu_calc_thread);

	/*Iterate through the arrays and print*/
	for(i = 0; i < ARRAY_SIZE; i++)
	{
		printf("Calculated Thread %3u - Block: %2u - Warp %2u - Thread %3u\n",
		cpu_calc_thread[i],  cpu_block[i], cpu_warp[i], cpu_thread[i]);
	}
	ch = getchar();
	return 0;
}