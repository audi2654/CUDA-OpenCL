//26/03/2022
//HPP CUDA Program to show Vector Addition on GPU

//cmd : nvcc -o VecAdd --gpu-architecture=sm_50 VecAdd.cu
//	OR: nvcc -o VecAdd -arch=native VecAdd.cu
//	OR: nvcc -o VecAdd -arch=sm_50 VecAdd.cu

//header files
//standard headers
#include <stdio.h>

//cuda headers
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include\cuda_runtime.h"
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include\device_launch_parameters.h"
#include <cuda.h>
#include "helper_timer.h"

//macros
#define BLOCK_WIDTH 32

//global variables
const int iNumberOfArrayElements = 11444777;

//cpu array by ptr method
float* hostInput1 = NULL;
float* hostInput2 = NULL;
float* hostOutput = NULL;
float* gold = NULL;			//CUDA OpenCL calls this as gold because this array has most accurate results

//gpu device cuda arrays by ptr method
float* deviceInput1 = NULL;
float* deviceInput2 = NULL;
float* deviceOutput = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;

//CUDA kernel function
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

int main(int argc, char* argv[], char** envp)
{
	//func signature
	void fillFloatArrayWithRandomNumbers(float*, int);
	void vecAddCPU(const float*, const float*, float*, int);
	void cleanup(void);

	//variable decl.
	int size = iNumberOfArrayElements * sizeof(float);
	cudaError_t result = cudaSuccess;

	//code
	//memory allocation for arrays/matrices on host
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

	gold = (float*)malloc(size);
	if (gold == NULL)
	{
		printf("Host Memory Allocation failed for gold matrix\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	//filling values into host arrays
	fillFloatArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements);
	fillFloatArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements);

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
	//CUDA kernel configuration
	dim3 dimGrid = dim3((int)ceil((float)iNumberOfArrayElements / 256.0f), 1, 1);
	dim3 dimBlock = dim3(256, 1, 1);

	//create timer to benchmark results
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	//now call the kernel - vector addition on device GPU
	vecAddGPU << < dimGrid, dimBlock >> > (deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);	 //here CUDA Runtime comes into picture

	//stop the timer
	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;

	//copy result data from device array to host array
	result = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess)
	{
		printf("Device to Host Data Copy failed for hostOutput array\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	//vector addition on host CPU
	vecAddCPU(hostInput1, hostInput2, gold, iNumberOfArrayElements);

	//comparison for result accuracy
	const float epsilon = 0.000001f;
	int breakValue = -1;
	bool bAccuracy = true;

	for (int i = 0; i < iNumberOfArrayElements; i++)
	{
		float val1 = gold[i];
		float val2 = hostOutput[i];

		if (fabs(val1 - val2) > epsilon)			//fabs is float absolute
		{
			bAccuracy = false;
			breakValue = i;
			break;
		}
	}

	char str[128];
	if (bAccuracy == false)
	{
		sprintf(str, "Result Comparison of CPU & GPU Vector Addition is not within \
			Accuracy of 0.000001 at array index %d", breakValue);
	}
	else
	{
		sprintf(str, "Result Comparison of CPU & GPU Vector Addition is within Accuracy of 0.000001");
	}

	//output
	printf("\n");
	printf("Array1 begins from 0th index %.6f to %dth index %.6f\n", hostInput1[0], iNumberOfArrayElements - 1, hostInput1[iNumberOfArrayElements - 1]);
	printf("Array2 begins from 0th index %.6f to %dth index %.6f\n", hostInput2[0], iNumberOfArrayElements - 1, hostInput2[iNumberOfArrayElements - 1]);
	printf("CUDA Kernel Grid Dimension = %d, %d, %d & Block Dimension = %d, %d, %d\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
	printf("Output Array begins from 0th index %.6f to %dth index %.6f\n", hostOutput[0], iNumberOfArrayElements - 1, hostOutput[iNumberOfArrayElements - 1]);

	printf("\n");
	printf("Time taken for Vector Addition on CPU = %.6f\n", timeOnCPU);
	printf("Time taken for Vector Addition on GPU = %.6f\n", timeOnGPU);
	printf("%s\n", str);

	//cleanup
	cleanup();

	return(0);
}

//user defined function definitions
void fillFloatArrayWithRandomNumbers(float* arr, int len)
{
	const float fscale = 1.0f / (float)RAND_MAX;

	//code
	for (int i = 0; i < len; i++)
	{
		arr[i] = fscale * rand();
	}
}

void vecAddCPU(const float* arr1, const float* arr2, float* out, int len)
{
	//code
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	for (int i = 0; i < iNumberOfArrayElements; i++)
	{
		out[i] = arr1[i] + arr2[i];
	}

	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;
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

	if (gold)
	{
		free(gold);
		gold = NULL;
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
