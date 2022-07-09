//26/03/2022
//HPP OpenCL Program to show Matrix Multiplication on GPU

//cmd : cl.exe /c /EHsc /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include" MatMul.c
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
		""