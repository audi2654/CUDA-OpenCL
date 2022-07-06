//26/03/2022
//HPP CUDA Program to show Matrix Multiplication on GPU

//cmd : nvcc -o MatMul --gpu-architecture=sm_50 MatMul.cu
//	OR: nvcc -o MatMul -arch=native MatMul.cu
//	OR: nvcc -o MatMul -arch=sm_50 MatMul.cu

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
int* hostA = NULL;
int* hostB = NULL;
int* hostC = NULL;
int* gold = NULL;			//CUDA OpenCL calls this as gold because this array has most accurate results

int* deviceA = NULL;
int* deviceB = NULL;
int* deviceC = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;

//CUDA kernel function
__global__ void matMulGPU(int* A, int* B, int* C, int numARows, int numAColumns, int numBColumns, int numCColumns)
{
	//var decl.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	//code
	if ((row < numARows) && (column < numBColumns))
	{
		int value = 0;

		for (int k = 0; k < numAColumns; k++)
		{
			int a = A[row * numAColumns + k];			//using k here as column
			int b = B[k * numBColumns + column];		//using k here as row
			value += a * b;
		}
		C[row * numCColumns + column] = value;
	}
}

int main(int argc, char* argv[], char** envp)
{
	//func decl.
	void InitA(int* data, int, int);
	void InitB(int* data, int, int);
	void matMulCPU(int*, int*, int*, int, int, int, int);
	void cleanup(void);

	//var decl.
	int numARows = BLOCK_WIDTH;
	int numAColumns = BLOCK_WIDTH;
	int numBRows = BLOCK_WIDTH;
	int numBColumns = BLOCK_WIDTH;
	int numCRows = numARows;
	int numCColumns = numBColumns;

	int numGoldRows = numARows;
	int numGoldColumns = numBColumns;

	int sizeA = numARows * numAColumns * sizeof(int);
	int sizeB = numBRows * numBColumns * sizeof(int);
	int sizeC = numCRows * numCColumns * sizeof(int);
	int sizeGold = numGoldRows * numGoldColumns * sizeof(int);

	cudaError_t result = cudaSuccess;

	//code
	//memory allocation for arrays/matrices on host
	hostA = (int*)malloc(sizeA);
	if (hostA == NULL)
	{
		printf("Host Memory Allocation failed for hostA matrix\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostB = (int*)malloc(sizeB);
	if (hostB == NULL)
	{
		printf("Host Memory Allocation failed for hostB matrix\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostC = (int*)malloc(sizeC);
	if (hostC == NULL)
	{
		printf("Host Memory Allocation failed for hostC matrix\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	gold = (int*)malloc(sizeGold);
	if (gold == NULL)
	{
		printf("Host Memory Allocation failed for gold matrix\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	//printing matrix dimensions & sizes
	printf("Dimensions of matrix hostA : %d x %d\n", numARows, numAColumns);
	printf("Dimensions of matrix hostB : %d x %d\n", numBRows, numBColumns);
	printf("Dimensions of matrix hostC : %d x %d\n", numCRows, numCColumns);
	printf("Dimensions of matrix gold  : %d x %d\n", numGoldRows, numGoldColumns);

	printf("\n");
	printf("Size of matrix hostA       : %d\n", sizeA);
	printf("Size of matrix hostB       : %d\n", sizeB);
	printf("Size of matrix hostC       : %d\n", sizeC);
	printf("Size of matrix gold        : %d\n", sizeGold);

	//fill source matrices from host
	InitA(hostA, numARows, numAColumns);
	InitB(hostB, numBRows, numBColumns);

	//memory allocation for arrays/matrices on GPU device
	result = cudaMalloc((void**)&deviceA, sizeA);
	if (result != cudaSuccess)
	{
		printf("Device Memory Allocation failed for deviceA matrix\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMalloc((void**)&deviceB, sizeB);
	if (result != cudaSuccess)
	{
		printf("Device Memory Allocation failed for deviceB matrix\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMalloc((void**)&deviceC, sizeC);
	if (result != cudaSuccess)
	{
		printf("Device Memory Allocation failed for deviceC matrix\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	//copy data from host arrays/matrices into device arrays/matrices
	result = cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		printf("Host to Device Data Copy failed for deviceA matrix\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		printf("Host to Device Data Copy failed for deviceB matrix\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	//now call CUDA kernel to perform calculations
	//CUDA kernel configuration
	dim3 dimGrid = dim3(ceil((int)numBColumns / (int)BLOCK_WIDTH), ceil((int)numARows/(int)BLOCK_WIDTH), 1);
	dim3 dimBLock = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);

	//CUDA kernel for matrix multiplication
	//create timer to benchmark results
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	//now call the kernel - matrix multiplication on device GPU
	matMulGPU <<<dimGrid, dimBLock >>> (deviceA, deviceB, deviceC, numARows, numAColumns, numBColumns, numCColumns);

	//stop the timer
	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;

	//copy result data from device matrix to host matrix
	result = cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess)
	{
		printf("Device to Host Data Copy failed for hostC matrix\n\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	//matrix multiplication on host CPU
	matMulCPU(hostA, hostB, gold, numARows, numAColumns, numBColumns, numCColumns);

	//comparison for result accuracy
	int breakValue = -1;
	bool bAccuracy = true;
	for (int i = 0; i < numCRows * numCColumns; i++)
	{
		int val1 = gold[i];
		int val2 = hostC[i];

		if (val1 != val2)
		{
			bAccuracy = false;
			breakValue = i;
			break;
		}
	}

	char str[128];
	if (bAccuracy == false)
	{
		sprintf(str, "Result Comparison of CPU & GPU Matrix Multiplication is not Accurate at array index %d", breakValue);
	}
	else
	{
		sprintf(str, "Result Comparison of CPU & GPU Matrix Multiplication is Accurate");
	}

	printf("\n");
	printf("Time taken for Matrix Multiplication on CPU = %.6f\n", timeOnCPU);
	printf("Time taken for Matrix Multiplication on GPU = %.6f\n", timeOnGPU);
	printf("%s\n", str);

	//cleanup
	cleanup();
	
	return(0);
}

//user defined function definitions
void InitA(int* data, int row, int col)
{
	int num = 1;

	//code
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			*(data + i * col + j) = num;
			num++;
		}
	}
}

void InitB(int* data, int row, int col)
{
	int num = BLOCK_WIDTH;

	//code
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			*(data + i * col + j) = num;
			num--;
		}
	}
}

void matMulCPU(int* A, int* B, int* C, int numARows, int numAColumns, int numBColumns, int numCColumns)
{
	//code
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	for (int i = 0; i < numARows; ++i)
	{
		for (int j = 0; j < numBColumns; ++j)
		{
			int value = 0;
			
			for (int k = 0; k < numAColumns; ++k)
			{
				int a = A[i * numAColumns + k];				//using k here as column
				int b = B[k * numBColumns + j];				//using k here as row
				value += a * b;
			}
			C[i * numCColumns + j] = value;
		}
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
	if (deviceC)
	{
		cudaFree(deviceC);
		deviceC = NULL;
	}

	if (deviceB)
	{
		cudaFree(deviceC);
		deviceC = NULL;
	}

	if (deviceA)
	{
		cudaFree(deviceC);
		deviceC = NULL;
	}

	if (gold)
	{
		free(gold);
		gold = NULL;
	}

	if (hostC)
	{
		free(hostC);
		hostC = NULL;
	}

	if (hostB)
	{
		free(hostB);
		hostB = NULL;
	}

	if (hostA)
	{
		free(hostA);
		hostA = NULL;
	}
}
