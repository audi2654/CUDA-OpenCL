//26/03/2022
//HPP OpenCL Program to show Matrix Multiplication on GPU

//cmd : cl.exe /c /EHsc /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include" MatMul.cpp
//      link.exe MatMul.obj /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\lib\x64" opencl.lib

//headers
//standard headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>			//for fabs()

//OpenCL headers
#include <CL\opencl.h>                                                                              //single compulsory file
#include <C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include\CL\cl.h>                 //optional - added on for intellisense
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include\CL\cl_platform.h"        //optional - added on for intellisense
#include "helper_timer.h"

//macros
#define BLOCK_WIDTH 64

//global variables
cl_platform_id oclPlatformID;
cl_device_id oclDeviceID;

cl_context oclContext;
cl_command_queue oclCommandQueue;

cl_program oclProgram;
cl_kernel oclKernel;

int* hostA = NULL;
int* hostB = NULL;
int* hostC = NULL;
int* gold = NULL;			//CUDA OpenCL calls this as gold because this array has most accurate results

cl_mem deviceA = NULL;
cl_mem deviceB = NULL;
cl_mem deviceC = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;

//OpenCL kernel function
const char* oclSourceCode =
		"__kernel void matMulGPU(__global int *A, __global int *B, __global int *C, int numARows, int numAColumns, int numBColumns, int numCColumns)" \
		"{" \
			"int row = get_global_id(0);" \
			"int column = get_global_id(1);" \
			"if((row < numARows) && (column < numBColumns))" \
			"{" \
				"int value = 0;" \
				"for(int k = 0; k < numAColumns; k++)" \
					"{" \
						"int a = A[row * numAColumns + k];" \
						"int b = B[k * numBColumns + column];" \
						"value += a * b;" \
					"}" \
				"C[row * numCColumns + column] = value;" \
			"}" \
		"}";
		
//entry point function
int main(int argc, char* argv[])
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

	int numCRows = BLOCK_WIDTH;
	int numCColumns = numBColumns;

	int numGoldRows = numARows;
	int numGoldColumns = numBColumns;

	int sizeA = numARows * numAColumns * sizeof(int);
	int sizeB = numBRows * numBColumns * sizeof(int);
	int sizeC = numCRows * numCColumns * sizeof(int);
	int sizeGold = numGoldRows * numGoldColumns * sizeof(int);

	cl_int result;

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

	//get OpenCL supporting platform's ID
	result = clGetPlatformIDs(1, &oclPlatformID, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clGetPlatformIDs() failed : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	printf("\nALL GOOD till line number : %d\n", __LINE__);

	//get OpenCL supporting CPU device's ID
	result = clGetDeviceIDs(oclPlatformID, CL_DEVICE_TYPE_GPU, 1, &oclDeviceID, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clGetDeviceIDs() failed : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//create OpenCL compute context
	oclContext = clCreateContext(NULL, 1, &oclDeviceID, NULL, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateContext() failed : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//create command queue
	oclCommandQueue = clCreateCommandQueue(oclContext, oclDeviceID, 0, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateCommandQueue() failed : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//create OpenCL program from .cl
	oclProgram = clCreateProgramWithSource(oclContext, 1, (const char**)&oclSourceCode, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateProgramWithSource() failed : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//build OpenCL program
	result = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(oclProgram, oclDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("Program build log : %s\n", buffer);
		printf("clBuildProgram() failed : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//create OpenCL kernel by passing kernel function name that we used in .cl file
	oclKernel = clCreateKernel(oclProgram, "matMulGPU", &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateKernel() failed : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//memory allocation for arrays/matrices on GPU device
	deviceA = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeA, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateBuffer() failed for deviceA matrix : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	deviceB = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeB, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateBuffer() failed for deviceB matrix : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	deviceC = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeC, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateBuffer() failed for deviceC Output matrix : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//set 0 based 0th argument i.e deviceA
	result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void*)&deviceA);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArg() failed for 1st argument : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//set 0 based 1st argument i.e deviceB
	result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void*)&deviceB);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArg() failed for 2nd argument : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//set 0 based 2nd argument i.e deviceC
	result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void*)&deviceC);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArg() failed for 3rd argument : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//set 0 based 3rd argument i.e numARows
	result = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void*)&numARows);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArg() failed for 4th argument : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//set 0 based 4th argument i.e numARows
	result = clSetKernelArg(oclKernel, 4, sizeof(cl_int), (void*)&numAColumns);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArg() failed for 5th argument : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//set 0 based 5th argument i.e numBColumns
	result = clSetKernelArg(oclKernel, 5, sizeof(cl_int), (void*)&numBColumns);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArg() failed for 6th argument : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//set 0 based 6th argument i.e numCColumns
	result = clSetKernelArg(oclKernel, 6, sizeof(cl_int), (void*)&numCColumns);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArg() failed for 7th argument : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//write above 'input' device buffer to device memory
	result = clEnqueueWriteBuffer(oclCommandQueue, deviceA, CL_FALSE, 0, sizeA, hostA, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clEnqueueWriteBuffer() failed for 1st Input Device Buffer : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clEnqueueWriteBuffer(oclCommandQueue, deviceB, CL_FALSE, 0, sizeB, hostB, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clEnqueueWriteBuffer() failed for 2nd Input Device Buffer : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//kernel configuration
	size_t globalWorkSize[2];		//1D 2 element array operation
	globalWorkSize[0] = BLOCK_WIDTH;
	globalWorkSize[1] = BLOCK_WIDTH;

	//create timer to benchmark results
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	result = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clEnqueueNDRangeKernel() failed : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//finish OpenCL command queue
	clFinish(oclCommandQueue);

	//stop the timer
	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;

	//read back results from GPU device (i.e from deviceC) into CPU variable (hostC)
	result = clEnqueueReadBuffer(oclCommandQueue, deviceC, CL_TRUE, 0, sizeC, hostC, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clEnqueueReadBuffer() failed : %d\n", result);
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
		clReleaseMemObject(deviceC);
		deviceC = NULL;
	}

	if (deviceB)
	{
		clReleaseMemObject(deviceB);
		deviceB = NULL;
	}

	if (deviceA)
	{
		clReleaseMemObject(deviceA);
		deviceA = NULL;
	}

	if (oclKernel)
	{
		clReleaseKernel(oclKernel);
		oclKernel = NULL;
	}

	if (oclProgram)
	{
		clReleaseProgram(oclProgram);
		oclProgram = NULL;
	}

	if (oclCommandQueue)
	{
		clReleaseCommandQueue(oclCommandQueue);
		oclCommandQueue = NULL;
	}

	if (oclContext)
	{
		clReleaseContext(oclContext);
		oclContext = NULL;
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