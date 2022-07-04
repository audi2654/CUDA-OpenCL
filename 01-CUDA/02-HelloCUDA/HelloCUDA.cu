//26/03/2022
//HPP CUDA Program to show execution of CUDA Kernel on GPU by calling from Host & simple Vector Addtion on GPU

//cmd : nvcc HelloCUDA.c or .cu -o HelloCUDA

#include <stdio.h>

//cuda headers
#include <cuda.h>

//global variables
const int iNumberOfArrayElements = 5;

//cpu array by ptr method
float* hostInput1 = NULL;
float* hostInput2 = NULL;
float* hostOutput = NULL;

//gpu device cuda arrays by ptr method
float* deviceInput1 = NULL;
float* deviceInput2 = NULL;
float* deviceOutput = NULL;

//CUDA kernel

//below func will run on GPU device but will b called from Host CPU Device

//__global is a CUDA user defined func specifier, total 3 ,__device__, __host__
//__global__ means call from Host CPU or GPU & execute on GPU Device
//__device__ means call from GPU & execute on GPU
//__host__ means call from CPU & execute on CPU

__global__ void vecAddGPU(float* in1, float* in2, float* out, int len)
{
	//code
	//we create a certain no. of thread using below equation (similar to formula of 2D array to 1D array conversion) 
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//blockIndex on x-axis * blockDimension in x-axis + threadIndex on x-axis, similar for y-axis & z-axis

	//we can't create or have an exact/accurate no. of desired threads, if we ask for 200 threads
	//GPU may create 209, 300, 275 as per its calculations

	if (i < len)
	{
		out[i] = in1[i] + in2[i];
	}
}

//entry point function
int main(void)
{
	//func signature
	void cleanup(void);

	//variable decl.
	int size = iNumberOfArrayElements * sizeof(float);
	cudaError_t result = cudaSuccess;

	//code
	//memory allocation for arrays on host
	hostInput1 = (float*)malloc(size);
	if (hostInput1 == NULL)
	{
		printf("Host Memory Allocation failed for hostInput1 array\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostInput2 = (float*)malloc(size);
	if (hostInput2 == NULL)
	{
		printf("Host Memory Allocation failed for hostInput2 array\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostOutput = (float*)malloc(size);
	if (hostOutput == NULL)
	{
		printf("Host Memory Allocation failed for hostOutput array\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	//filling values into host arrays
	hostInput1[0] = 101.0;
	hostInput1[1] = 102.0;
	hostInput1[2] = 103.0;
	hostInput1[3] = 104.0;
	hostInput1[4] = 105.0;

	hostInput2[0] = 201.0;
	hostInput2[1] = 202.0;
	hostInput2[2] = 203.0;
	hostInput2[3] = 204.0;
	hostInput2[4] = 205.0;

	//memory allocation for arrays on GPU device
	result = cudaMalloc((void**)&deviceInput1, size);
	if (result != cudaSuccess)
	{
		printf("Device Memory Allocation failed for deviceInput1 array\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMalloc((void**)&deviceInput2, size);
	if (result != cudaSuccess)
	{
		printf("Device Memory Allocation failed for deviceInput2 array\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMalloc((void**)&deviceOutput, size);
	if (result != cudaSuccess)
	{
		printf("Device Memory Allocation failed for deviceOutput array\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	//copy data from host arrays into device arrays
	result = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		printf("Host to Device Data Copy failed for deviceInput1 array\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		printf("Host to Device Data Copy failed for deviceInput2 array\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	//now call CUDA kernel to perform calculations

	//2 steps to call a CUDA kernel
	//STEP-1 : configure the CUDA kernel threads by configuring the grid & block dimensions for threads
	//i.e give how many threads of which dimensions to create
	
	dim3 dimGrid = dim3(iNumberOfArrayElements, 1, 1);		//desired no. of Blocks to have in a single Grid
	//above statement specifies in grid dimGrid dim3(x-dimension, y-dimension, z-dimension) we want 
	//x = iNumberOfArrayElements i.e 5 Blocks in y & z axis/dimension

	dim3 dimBlock = dim3(1, 1, 1);							//desired no. of Threads to have in a single Block
	//above statement specifies in block dimBlock dim3(x-dimension, y-dimension, z-dimension) we want 
	//x = 1 i.e 1 Thread in y & z axis/dimension

	//with above 2 statements we create 1 Grid with 1, 1 dimension in y, z axis having 5 Blocks
	//each Block with 1, 1 dimension in y, z axis having 1 Thread, 
	//i.e total 5 Threads each one in 5 Blocks in one Grid

	//STEP-2 : Actually call kernel by using CUDA kernel launch syntax
	//CUDA kernel for vector addition
	vecAddGPU <<< dimGrid, dimBlock >>> (deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);	 //here CUDA Runtime comes into picture

	//copy result data from device array to host array
	result = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess)
	{
		printf("Device to Host Data Copy failed for hostOutput array\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	//printinf vector addition result on Host for visualization
	for (int i = 0; i < iNumberOfArrayElements; i++)
	{
		printf("%f + %f = %f\n", hostInput1[i], hostInput2[i], hostOutput[i]);
	}

	//cleanup
	cleanup();

	return(0);
}

void cleanup(void)
{
	//code
	//safe release - free in reverse order of vars. initialized
	if (deviceOutput)
	{
		cudaFree(deviceOutput);
		deviceOutput = NULL;
	}

	if (deviceInput2)
	{
		cudaFree(deviceInput2);
		deviceInput2 = NULL;
	}

	if (deviceInput1)
	{
		cudaFree(deviceInput1);
		deviceInput1 = NULL;
	}

	if (hostOutput)
	{
		cudaFree(hostOutput);
		hostOutput = NULL;
	}

	if (hostInput2)
	{
		cudaFree(hostInput2);
		hostInput2 = NULL;
	}

	if (hostInput1)
	{
		cudaFree(hostInput1);
		hostInput1 = NULL;
	}
}
