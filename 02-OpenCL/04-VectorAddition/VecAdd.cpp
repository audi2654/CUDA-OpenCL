//26/03/2022
//HPP OpenCL Program to show Vector Addition on GPU

//cmd : cl.exe /c /EHsc /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include" VecAdd.cpp
//      link.exe VecAdd.obj /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\lib\x64" opencl.lib

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

//global variables
const int iNumberOfArrayElements = 11444777;

cl_platform_id oclPlatformID;
cl_device_id oclDeviceID;

cl_context oclContext;
cl_command_queue oclCommandQueue;

cl_program oclProgram;
cl_kernel oclKernel;

//cpu arrays by ptr method
float* hostInput1 = NULL;
float* hostInput2 = NULL;
float* hostOutput = NULL;
float* gold = NULL;			//CUDA OpenCL calls this as gold because this array has most accurate results

//for gpu device arrays
cl_mem deviceInput1 = NULL;
cl_mem deviceInput2 = NULL;
cl_mem deviceOutput = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;

//OpenCL kernel function
const char* oclSourceCode =
"__kernel void vecAddGPU(__global float *in1, __global float *in2, __global float *out, int len)" \
"{" \
"int i = get_global_id(0);" \
"if(i < len)" \
"{" \
"out[i] = in1[i] + in2[i];" \
"}" \
"}";
		
//entry point function
int main(int argc, char* argv[])
{
	//func prototype
	void fillFloatArrayWithRandomNumbers(float*, int);
	size_t roundGlobalSizeToNearestMultipleOfLocalSize(int, unsigned int);
	void vecAddCPU(const float*, const float*, float*, int);
	void cleanup(void);

	//var decl.
	int size = iNumberOfArrayElements * sizeof(float);
	cl_int result;

	//code
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
	oclKernel = clCreateKernel(oclProgram, "vecAddGPU", &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateKernel() failed : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//memory allocation for arrays on GPU device
	deviceInput1 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateBuffer() failed for deviceInput1 array : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	deviceInput2 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateBuffer() failed for deviceInput2 array : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	deviceOutput = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
	if (result != CL_SUCCESS)
	{
		printf("clCreateBuffer() failed for deviceOutput array : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//set 0 based 0th argument of kernel i.e deviceInput1
	result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void*)&deviceInput1);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArg() failed for 1st argument : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//set 0 based 1st argument of kernel i.e deviceInput2
	result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void*)&deviceInput2);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArg() failed for 2nd argument : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//set 0 based 2nd argument of kernel i.e deviceOutput
	result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void*)&deviceOutput);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArg() failed for 3rd argument : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//set 0 based 3rd argument of kernel i.e len
	result = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void*)&iNumberOfArrayElements);
	if (result != CL_SUCCESS)
	{
		printf("clSetKernelArg() failed for 4th argument : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//write above 'input' device buffer to device memory
	result = clEnqueueWriteBuffer(oclCommandQueue, deviceInput1, CL_FALSE, 0, size, hostInput1, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clEnqueueWriteBuffer() failed for 1st Input Device Buffer : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clEnqueueWriteBuffer(oclCommandQueue, deviceInput2, CL_FALSE, 0, size, hostInput2, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clEnqueueWriteBuffer() failed for 2nd Input Device Buffer : %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}

	//kernel configuration
	size_t localWorkSize = 256;
	size_t globalWorkSize;
	
	globalWorkSize = roundGlobalSizeToNearestMultipleOfLocalSize(localWorkSize, iNumberOfArrayElements);

	//create timer to benchmark results
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	result = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
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

	//read back results from GPU device (i.e from deviceOutput) into CPU variable (hostOutput)
	result = clEnqueueReadBuffer(oclCommandQueue, deviceOutput, CL_TRUE, 0, size, hostOutput, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		printf("clEnqueueReadBuffer() failed : %d\n", result);
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
	printf("OpenCL Kernel Global Work Size = %zu & Local Work Size = %zu\n", globalWorkSize, localWorkSize);
	printf("Output Array begins from 0th index %.6f to %dth index %.6f\n", hostOutput[0], iNumberOfArrayElements - 1, hostOutput[iNumberOfArrayElements - 1]);

	printf("\n");
	printf("Time taken for Matrix Multiplication on CPU = %.6f\n", timeOnCPU);
	printf("Time taken for Matrix Multiplication on GPU = %.6f\n", timeOnGPU);
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

size_t roundGlobalSizeToNearestMultipleOfLocalSize(int local_size, unsigned int global_size)
{
	//code
	unsigned int r = global_size % local_size;
	if (r == 0)
	{
		return(global_size);
	}
	else
	{
		return(global_size + local_size - r);
	}
}

void cleanup(void)
{
	if (deviceOutput)
	{
		clReleaseMemObject(deviceOutput);
		deviceOutput = NULL;
	}

	if (deviceInput2)
	{
		clReleaseMemObject(deviceInput2);
		deviceInput2 = NULL;
	}

	if (deviceInput1)
	{
		clReleaseMemObject(deviceInput1);
		deviceInput1 = NULL;
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

	if (hostOutput)
	{
		free(hostOutput);
		hostOutput = NULL;
	}

	if (hostInput2)
	{
		free(hostInput2);
		hostInput2 = NULL;
	}

	if (hostInput1)
	{
		free(hostInput1);
		hostInput1 = NULL;
	}
}
