#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

__global__ void matrixMulSingleBlock(float *C, float *A, float *B, int width)
{
	int blocksPerDim = (int)ceil((float)width / (float)blockDim.x);
	int row;
	int col;

	for(int i = 0; i < blocksPerDim; i++)
	{
		for(int j = 0; j < blocksPerDim; j++)
		{
			float C_local = 0;

			for(int k = 0; k < width; k++)
			{
				row = threadIdx.y + i * blockDim.y;
				col = threadIdx.x + j * blockDim.x;

				C_local += A[row * width + k] * B[k * width + col];
			}
			
			C[row * width + col] = C_local;
		}
		
		/*float C_local = 0;
		for(int k = 0; k < width; k++)
		{
			C_local += A[row*width+k] * B[k*width+col];
		}
		C[row*width+col] = C_local;*/
	}
}

void performSingleBlockTest() {
	float A[] = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f,13.0f,14.0f,15.0f,16.0f};
	float B[] = {17.0f,18.0f,19.0f,20.0f,21.0f,22.0f,23.0f,24.0f,25.0f,26.0f,27.0f,28.0f,29.0f,30.0f,31.0f,32.0f};

	int width = 4;
	
	float *C = (float*)malloc(width*width*sizeof(float));
	memset(C, 0.0f, width*width*sizeof(float));

	float *A_d, *B_d, *C_d;
	cudaMalloc((void**)&A_d, width*width*sizeof(float));
	cudaMalloc((void**)&B_d, width*width*sizeof(float));
	cudaMalloc((void**)&C_d, width*width*sizeof(float));

	cudaMemcpy(A_d, A, width*width*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, width*width*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(C_d, 0, width*width*sizeof(float));

	matrixMulSingleBlock<<<1, dim3(2,2)>>>(C_d, A_d, B_d, width);

	cudaMemcpy(C, C_d, width*width*sizeof(float), cudaMemcpyDeviceToHost);

	printf("[\n");
	for(int i = 0; i < 4; i++)
	{
		printf("\t[");
		for(int j = 0; j < 4; j++)
		{
			printf("%f,", C[i*4+j]);
		}
		printf("],\n");
	}
	printf("]\n");

	cudaFree(C_d);
	cudaFree(B_d);
	cudaFree(A_d);

	free(C);
}