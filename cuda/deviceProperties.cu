#include <stdio.h>

int main(int argc, char *argv[])
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; ++i)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		printf("Device Numver %d\n", i);
		printf( "--- General Information for device %d ---\n", i);
		printf( "Name:%s\n", prop.name);
		printf( "Compute capability:%d.%d\n", prop.major, prop.minor);
		printf( "Clock rate:%d\n", prop.clockRate);
		printf( "Device copy overlap:");

		if (prop.deviceOverlap){
			printf("Enabled\n");
		}else{
			printf("Disabled\n");
		}
		printf("Kernel execition timeout :");
		if(prop.kernelExecTimeoutEnabled){
			printf("Enabled\n");
		}else{
			printf("Disabled\n");
		}
		printf("--- Memory Information for device %d ---\n", i);
		printf("Totoal Mem Block %ld(M) (%ld bytes)\n", prop.totalGlobalMem/1048576, prop.totalGlobalMem);
		printf("Total constant Mem:%ld\n", prop.totalConstMem);
		printf("Max mem pitch:%ld\n", prop.memPitch);
		printf("Texture Alignment:%ld\n", prop.textureAlignment);

		printf("--- MP Information for device %d ---\n", i);
		printf("Multiprocessor count:%d\n",prop.multiProcessorCount);
		printf("Shared mem per mp:%ld\n", prop.sharedMemPerBlock);
		printf("Mem clock rate (KHz) %d\n", prop.memoryClockRate);
		printf("Mem bandwidth(Gb/s) %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		printf("Registers per mp:%d\n", prop.regsPerBlock);
		printf("Threads in warp:%d\n", prop.warpSize);
		printf("Max threads per block:%d\n",prop.maxThreadsPerBlock);
		printf("Max thread dimensions:(%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions:(%d, %d, %d)\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	}
}