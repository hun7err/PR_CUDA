#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cuda_runtime.h>

template <int BLOCK_SIZE> __global__ void matrixMulSharedMemPrefetch(float *C, float *A, float *B, int width) // dodaæ sprawdzanie < width
{
	int a_start = width * BLOCK_SIZE * blockIdx.y, a_offset,
		b_start = BLOCK_SIZE * blockIdx.x, b_offset;

	__shared__ float A_shared[BLOCK_SIZE*BLOCK_SIZE];
	__shared__ float B_shared[BLOCK_SIZE*BLOCK_SIZE];

	float C_local = 0.0f;

	float a_prefetched = A[a_start + threadIdx.y * width + threadIdx.x],
			b_prefetched = B[b_start + threadIdx.y * width + threadIdx.x];

	for(int index = 0; index < gridDim.x;) // równie dobrze mog³oby byæ gridDim.y bo s¹ równe
	{
		++index;

		a_offset = index * BLOCK_SIZE;
		b_offset = index * BLOCK_SIZE * width;

		A_shared[threadIdx.y * blockDim.x + threadIdx.x] = a_prefetched;
		B_shared[threadIdx.y * blockDim.x + threadIdx.x] = b_prefetched;
		
		__syncthreads();

		if(index < gridDim.x)
		{
			a_prefetched = A[a_start + a_offset + threadIdx.y * width + threadIdx.x];
			b_prefetched = B[b_start + b_offset + threadIdx.y * width + threadIdx.x];
		}

		for(int k = 0; k < BLOCK_SIZE; k++)
		{
			C_local += A_shared[threadIdx.y * BLOCK_SIZE + k] * B_shared[k * BLOCK_SIZE + threadIdx.x];
		}

		__syncthreads(); // bariera synchronizacyjna, czekamy a¿ wszystkie w¹tki w bloku oblicz¹ wynik cz¹stkowy

		if(index * BLOCK_SIZE >= width)
			break;
	}
	
	int c_start = blockIdx.y * width * BLOCK_SIZE,
		c_offset = blockIdx.x * BLOCK_SIZE;
	C[c_start + c_offset + width * threadIdx.y + threadIdx.x] = C_local;
}

void performImprovedSharedMemTest(void)
{
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

	matrixMulSharedMemPrefetch<2><<<dim3(2,2), dim3(2,2)>>>(C_d, A_d, B_d, width);
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
	printf("]\n\n");

	cudaFree(C_d);
	cudaFree(B_d);
	cudaFree(A_d);

	free(C);
}
