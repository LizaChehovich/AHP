#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <Windows.h>
#include <malloc.h>
#include <time.h>
#include <iostream>
#include <cuda_profiler_api.h>
using namespace std;

#define constant 0
#define Size 14000
#define count 1
const int BlockDim = 16;

#if constant 
__constant__ float d_input[Size*Size];

__global__ void transponse(float* output)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if ((x < Size) && (y < Size))
	{
		output[y + x * Size] = d_input[y * Size + x];
	}
}

__global__ void transponseShared(float* output)
{
	__shared__ int smem[BlockDim][BlockDim + 1];
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if ((x < Size) && (y < Size))
	{
		smem[threadIdx.y][threadIdx.x] = d_input[y * Size + x];
	}
	__syncthreads();
	y = blockDim.y * blockIdx.y + threadIdx.x;
	x = blockDim.x * blockIdx.x + threadIdx.y;
	if ((x < Size) && (y < Size))
	{
		output[x * Size + y] = smem[threadIdx.x][threadIdx.y];
	}
}

#endif

__global__ void transponse(const float* input, float* output)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if ((x < Size) && (y < Size))
	{
		output[y + x * Size] = input[y * Size + x];
	}
}

__global__ void transponseShared(const float* input, float* output)
{
	__shared__ int smem[BlockDim][BlockDim + 1];
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if ((x < Size) && (y < Size))
	{
		smem[threadIdx.y][threadIdx.x] = input[y * Size + x];
	}
	__syncthreads();
	y = blockDim.y * blockIdx.y + threadIdx.x;
	x = blockDim.x * blockIdx.x + threadIdx.y;
	if ((x < Size) && (y < Size))
	{
		output[x * Size + y] = smem[threadIdx.x][threadIdx.y];
	}
}

__global__ void transponse(const float* input, float* output, const int pitch)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if ((x < Size) && (y < Size))
	{
		output[y + x * Size] = input[y * pitch + x];
	}
}

__global__ void transponseShared(const float* input, float* output, const int pitch)
{
	__shared__ int smem[BlockDim][BlockDim + 1];
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if ((x < Size) && (y < Size))
	{
		smem[threadIdx.y][threadIdx.x] = input[y * pitch + x];
	}
	__syncthreads();
	y = blockDim.y * blockIdx.y + threadIdx.x;
	x = blockDim.x * blockIdx.x + threadIdx.y;
	if ((x < Size) && (y < Size))
	{
		output[x * Size + y] = smem[threadIdx.x][threadIdx.y];
	}
}

double transposeCPU(const float* input, float* output);

double transponseCUDA(const float* input, float* output, bool fast = false);

double transponseCUDAGlobal(const float* input, float* output, bool fast = false);

double transponseCUDAConst(const float* input, float* output, bool fast = false);

float* getRandomMatrix();

void showMatrix(float* matrix);

bool equals(float* mA, float* mB);

int main()
{
	cout << "Block Size:" << BlockDim << endl;
	cudaProfilerStart();
	cout << "Size:" << Size << "*" << Size << endl;
	float *matrix, *transponseMatrix, *cudaMatrix, *fastCudaMatrix, *gcudaMatrix, *gfastCudaMatrix;
	matrix = getRandomMatrix();
	transponseMatrix = (float*)malloc(Size*Size * sizeof(float));
	cudaMatrix = (float*)malloc(Size*Size * sizeof(float));
	fastCudaMatrix = (float*)malloc(Size*Size * sizeof(float));
	gcudaMatrix = (float*)malloc(Size*Size * sizeof(float));
	gfastCudaMatrix = (float*)malloc(Size*Size * sizeof(float));
	double cpu = 0.0, cuda = 0.0, fcuda = 0.0, gcuda = 0.0, gfastcuda;
	for (int i = 0; i < count; i++)
	{
		cpu += transposeCPU(matrix, transponseMatrix);
		cuda += transponseCUDA(matrix, cudaMatrix, false);
		fcuda += transponseCUDA(matrix, fastCudaMatrix, true);
		gcuda += transponseCUDAGlobal(matrix, gcudaMatrix, false);
		gfastcuda += transponseCUDAGlobal(matrix, fastCudaMatrix, true);
	}
	cout << "CPU:" << cpu/count << endl;
	cout << "CUDA:" << cuda/count << endl;
	cout << "Fast CUDA:" << fcuda/count << endl;
	cout << "CUDA global:" << gcuda/count << endl;
	cout << "Fast CUDA global:" <<  gfastcuda/count << endl;
#if constant
	cudaMatrix = (float*)malloc(Size*Size * sizeof(float));
	cout << "CUDA const:" << transponseCUDAConst(matrix, cudaMatrix, false) << endl;
	free(cudaMatrix);
	fastCudaMatrix = (float*)malloc(Size*Size * sizeof(float));
	cout << "Fast CUDA const:" << transponseCUDAConst(matrix, fastCudaMatrix, true) << endl;
	free(fastCudaMatrix);
#endif
	free(matrix);
	free(transponseMatrix);
	free(cudaMatrix);
	free(fastCudaMatrix);
	free(gcudaMatrix);
	free(gfastCudaMatrix);
	cudaProfilerStop();
	return 0;
}

double transposeCPU(const float* input, float* output)
{
	int start = clock();
	for (int i = 0; i < Size; i++)
	{
		for (int j = 0; j < Size; j++)
		{
			output[i + Size * j] = input[i * Size + j];
		}
	}
	return clock() - start;
}

double transponseCUDA(const float* input, float* output, bool fast)
{
	float* dev_input, *dev_output;
	float time = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaMalloc((void**)&dev_input, Size * Size * sizeof(float));
	cudaMalloc((void**)&dev_output, Size * Size * sizeof(float));
	cudaMemcpy(dev_input, input, Size * Size * sizeof(float), cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(BlockDim, BlockDim);
	dim3 numBlocks((Size + BlockDim - 1) / BlockDim, (Size + BlockDim - 1) / BlockDim);
	cudaEventRecord(start, 0);
	if (fast)
	{
		transponseShared << <numBlocks, threadsPerBlock >> > (dev_input, dev_output);
	}
	else
	{
		transponse << <numBlocks, threadsPerBlock >> > (dev_input, dev_output);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaMemcpy(output, dev_output, Size * Size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_input);
	cudaFree(dev_output);
	return time;
}

double transponseCUDAGlobal(const float * input, float * output, bool fast)
{
	float* dev_input, *dev_output;
	float time = 0;
	cudaEvent_t start, stop;
	int pitch = ((Size * sizeof(float) + 127) / 128) * 128 / sizeof(float);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaMalloc((void**)&dev_input, pitch * Size * sizeof(float));
	cudaMalloc((void**)&dev_output, pitch * Size * sizeof(float));
	for (int i = 0; i < Size; i++)
	{
		cudaMemcpy(&dev_input[i*pitch], &input[i*Size], Size * sizeof(float), cudaMemcpyHostToDevice);
	}
	dim3 threadsPerBlock(BlockDim, BlockDim);
	dim3 numBlocks((Size + BlockDim - 1) / BlockDim, (Size + BlockDim - 1) / BlockDim);
	cudaEventRecord(start, 0);
	if (fast)
	{
		transponseShared << <numBlocks, threadsPerBlock >> > (dev_input, dev_output, pitch);
	}
	else
	{
		transponse << <numBlocks, threadsPerBlock >> > (dev_input, dev_output, pitch);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaMemcpy(output, dev_output, Size * Size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_input);
	cudaFree(dev_output);
	return time;
}

#if constant
double transponseCUDAConst(const float * input, float * output, bool fast)
{
	float* dev_output;
	float time = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaMalloc((void**)&dev_output, Size * Size * sizeof(float));
	cudaMemcpyToSymbol(d_input, input, Size * Size * sizeof(float), 0, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(BlockDim, BlockDim);
	dim3 numBlocks((Size + BlockDim - 1) / BlockDim, (Size + BlockDim - 1) / BlockDim);
	cudaEventRecord(start, 0);
	if (fast)
	{
		transponseShared << <numBlocks, threadsPerBlock >> > (dev_output);
	}
	else
	{
		transponse << <numBlocks, threadsPerBlock >> > (dev_output);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaMemcpy(output, dev_output, Size * Size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_output);
	return time;
}
#endif

float * getRandomMatrix()
{
	float* matrix = (float*)malloc(Size*Size * sizeof(float));
	if (matrix != nullptr)
	{
		srand(time(0));
		for (int i = 0; i < Size*Size; i++)
		{
			matrix[i] = rand() % 100;
		}
	}
	return matrix;
}

void showMatrix(float* matrix)
{
	for (int i = 0; i < Size*Size; i++)
	{
		cout << matrix[i] << " ";
		if (i % Size == Size - 1)
			cout << endl;
	}
	cout << endl;
}

bool equals(float * mA, float * mB)
{
	for (int i = 0; i < Size*Size; i++)
	{
		if (mA[i] != mB[i])
			return false;
	}
	return true;
}
