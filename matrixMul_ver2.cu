#define WIN32
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "genmatrix.h"

#include <cuda_runtime.h>

 __global__ void matrixMulMultiBlock(float *C, float *A, float *B, int width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float C_local = 0;

	if(row < width && col < width)
	{
		for(int k = 0; k < width; k++)
			C_local += A[row*width+k] * B[k*width+col];
	
		C[row*width+col] = C_local;
	}
}

static float totalTime = 0.0f;

int performMultiBlockTest(dim3 block_size, int width)
{
	cudaError_t error;

	float *A = (float*)malloc(width*width*sizeof(float));
	float *B = (float*)malloc(width*width*sizeof(float));

	generateTestMatrix(A, width);
	generateTestMatrix(B, width);
	
	float *C = (float*)malloc(width*width*sizeof(float));
	memset(C, 0, width*width*sizeof(float));

	float *A_d, *B_d, *C_d;

	error = cudaMalloc((void**)&A_d, width*width*sizeof(float));

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not allocate memory on the device for matrix A: %s (line: %d)\n", cudaGetErrorString(error), __LINE__);
		return -1;
	}

	error = cudaMalloc((void**)&B_d, width*width*sizeof(float));

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not allocate memory on the device for matrix B: %s (line: %d)\n", cudaGetErrorString(error), __LINE__);
		return -1;
	}

	error = cudaMalloc((void**)&C_d, width*width*sizeof(float));

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not allocate memory on the device for matrix C: %s (line: %d)\n", cudaGetErrorString(error), __LINE__);
		return -1;
	}

	error = cudaMemcpy(A_d, A, width*width*sizeof(float), cudaMemcpyHostToDevice);

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not copy data from host to device: %s (line: %d)\n", cudaGetErrorString(error), __LINE__);
		return -1;
	}

	error = cudaMemcpy(B_d, B, width*width*sizeof(float), cudaMemcpyHostToDevice);

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not copy data from host to device: %s (line: %d)\n", cudaGetErrorString(error), __LINE__);
		return -1;
	}

	cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event: %s (line: %d)\n", cudaGetErrorString(error), __LINE__);
        return -1;
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event: %s (line: %d)\n", cudaGetErrorString(error), __LINE__);
        return -1;
    }

	error = cudaEventRecord(start, NULL);

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not record start event: %s (line: %d)\n", cudaGetErrorString(error), __LINE__);
		return -1;
	}

	int grid_side = (int)ceil((float)width/(float)block_size.x);
	for(int current_test = 0; current_test < TEST_COUNT; current_test++)
	{
		matrixMulMultiBlock<<<dim3(grid_side, grid_side),block_size>>>(C_d, A_d, B_d, width);

		cudaDeviceSynchronize();
	}

	error = cudaEventRecord(stop, NULL);

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not record stop event: %s (line: %d)\n", cudaGetErrorString(error), __LINE__);
		return -1;
	}

	error = cudaEventSynchronize(stop);

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not synchronize with stop event: %s (line: %d)\n", cudaGetErrorString(error), __LINE__);
        return -1;
	}
	
	totalTime = 0.0f;
	error = cudaEventElapsedTime(&totalTime, start, stop);

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not calculate elapsed time: %s (line: %d)\n", cudaGetErrorString(error), __LINE__);
        return -1;
	}

	float msecPerMatrixMul = totalTime / (float)TEST_COUNT;
    double flopsPerMatrixMul = 2.0 * (double)width * (double)width * (double)width;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

	printf("%dx%d\t%dx%d\t%dx%d\t%.3f\t%.2f\n", width, width, block_size.x, block_size.y, grid_side, grid_side, msecPerMatrixMul, gigaFlops);

	error = cudaMemcpy(C, C_d, width*width*sizeof(float), cudaMemcpyDeviceToHost);
	
	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not copy data from device to host: %s (line: %d)\n", cudaGetErrorString(error), __LINE__);
		return -1;
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(C_d);
	cudaFree(B_d);
	cudaFree(A_d);

	free(C);
	free(B);
	free(A);

	return 0;
}

void performMultiBlockTests(void)
{
	srand((unsigned int)time(NULL));

	dim3 blockSizes[] = { dim3(8,8), dim3(16,16), dim3(22,22), dim3(32,32)};
	int matrixSizes[] = { 32, 64, 128 };

	for(int i = 0; i < sizeof(matrixSizes)/sizeof(int); i++)
	{
		for(int j = 0; j < sizeof(blockSizes)/sizeof(dim3); j++)
		{
			performMultiBlockTest(blockSizes[j], matrixSizes[i]);
		}
	}

	cudaDeviceReset();
}