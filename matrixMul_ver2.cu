#define WIN32
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

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

void performTest(void)
{
	float A[] = {1.0f,2.0f,3.0f,4.0f};
	float B[] = {5.0f,6.0f,7.0f,8.0f};

	float *C = (float*)malloc(4*sizeof(float));
	memset(C, 0.0f, 4*sizeof(float));

	float *A_d, *B_d, *C_d;

	cudaMalloc((void**)&A_d, sizeof(A));
	cudaMalloc((void**)&B_d, sizeof(B));
	cudaMalloc((void**)&C_d, 4*sizeof(float));

	cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, sizeof(B), cudaMemcpyHostToDevice);

	matrixMulMultiBlock<<<1,dim3(2,2)>>>(C_d, A_d, B_d, 2);

	cudaMemcpy(C, C_d, 4*sizeof(float), cudaMemcpyDeviceToHost);

	printf("[[%f,%f],\n[%f,%f]]\n", C[0],C[1],C[2],C[3]);

	cudaFree(C_d);
	cudaFree(B_d);
	cudaFree(A_d);

	free(C);
}