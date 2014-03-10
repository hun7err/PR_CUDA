#define WIN32
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

int performMultiBlockTest(dim3 block_size, int width)
{
	cudaError_t error;

	float A[] = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f,13.0f,14.0f,15.0f,16.0f};
	float B[] = {17.0f,18.0f,19.0f,20.0f,21.0f,22.0f,23.0f,24.0f,25.0f,26.0f,27.0f,28.0f,29.0f,30.0f,31.0f,32.0f};

	float *C = (float*)malloc(16*sizeof(float));
	memset(C, 0.0f, 16*sizeof(float));

	float *A_d, *B_d, *C_d;

	error = cudaMalloc((void**)&A_d, sizeof(A));

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not allocate memory on the device for matrix A (line: %d)\n", __LINE__);
		return -1;
	}

	error = cudaMalloc((void**)&B_d, sizeof(B));

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not allocate memory on the device for matrix B (line: %d)\n", __LINE__);
		return -1;
	}

	error = cudaMalloc((void**)&C_d, 16*sizeof(float));

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not allocate memory on the device for matrix C (line: %d)\n", __LINE__);
		return -1;
	}

	error = cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice);

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not copy data from host to device (line: %d)\n", __LINE__);
		return -1;
	}

	error = cudaMemcpy(B_d, B, sizeof(B), cudaMemcpyHostToDevice);

	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not copy data from host to device (line: %d)\n", __LINE__);
		return -1;
	}

	matrixMulMultiBlock<<<dim3(ceil((float)width/(float)block_size.x), ceil((float)width/(float)block_size.y)),block_size>>>(C_d, A_d, B_d, 4);

	error = cudaMemcpy(C, C_d, 16*sizeof(float), cudaMemcpyDeviceToHost);
	
	if(error != cudaSuccess)
	{
		fprintf(stderr, "Could not copy data from device to host (line: %d)\n", __LINE__);
		return -1;
	}

	cudaFree(C_d);
	cudaFree(B_d);
	cudaFree(A_d);

	free(C);
 }

void performMultiBlockTests(void)
{
	performMultiBlockTest(dim3(3,3), 4);
	/*performMultiBlockTest(dim3(8,8), 10);
	performMultiBlockTest(dim3(16,16), 10);
	performMultiBlockTest(dim3(22,22), 10);
	performMultiBlockTest(dim3(32,32), 10);*/
}