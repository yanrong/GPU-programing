#include <stdio.h>

__global__ void vectorAdd(int *a, int *b, int *c, int n)
{
    //int i = threadIdx.x;
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(void)
{
    int *a, *b, *c;
    int SIZE = 1024;
    //alloc the memory
    cudaMallocManaged(&a, SIZE * sizeof(int));
    cudaMallocManaged(&b, SIZE * sizeof(int));
    cudaMallocManaged(&c, SIZE * sizeof(int));
    //fill the array
    for(int i = 0; i < SIZE; i++)
    {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
    }
    //call the kernel function
    vectorAdd<<<2, SIZE/2>>>(a, b, c, SIZE);
    cudaDeviceSynchronize();

    for(int i = 0; i < SIZE; i++)
    {
        printf("c[%d] = %d\n", i, c[i]);
    }
    //release function
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
