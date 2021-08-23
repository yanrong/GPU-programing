#include <stdio.h>
#define ARRAY_SIZE_X 32
#define ARRAY_SIZE_Y 16
#define ARRAY_SIZE_IN_BYTES ((ARRAY_SIZE_X)*(ARRAY_SIZE_Y)*(sizeof(unsigned int)))

/*Declare statically six arrays of ARRAY_SIZE each*/
unsigned int cpu_block_x[ARRAY_SIZE_Y][ARRAY_SIZE_X]
unsigned int cpu_block_y[ARRAY_SIZE_Y][ARRAY_SIZE_X]
unsigned int cpu_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X]
unsigned int cpu_warp[ARRAY_SIZE_Y][ARRAY_SIZE_X]
unsigned int cpu_calc_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X]
unsigned int cpu_xthread[ARRAY_SIZE_Y][ARRAY_SIZE_X]
unsigned int cpu_ythread[ARRAY_SIZE_Y][ARRAY_SIZE_X]
unsigned int cpu_grid_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X]
unsigned int cpu_block_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_grid_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];

__global__ void what_is_my_id_2d_A(unsigned int *const block_x,
	unsigned int *const block_y,
	unsigned int *const thread,
	unsigned int *const calc_thread,
	unsigned int *const x_thread,
	unsigned int *const y_thread,
	unsigned int *const grid_dimx,
	unsigned int *const block_dimx,
	unsigned int *const grid_dimy,
	unsigned int *const block_dimy,)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;
	
	block_x[thread_idx] = 
}