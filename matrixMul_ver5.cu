#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

template <int BLOCK_SIZE, int threadElemsPerDim> __global__ void matrixMulSharedMemPrefetchMultipleElements(float *C, float *A, float *B, int width) // sprawdziæ czemu to nie dzia³a
{
	int a_start = width * BLOCK_SIZE * threadElemsPerDim * blockIdx.y, a_offset, // ok
		b_start = BLOCK_SIZE * threadElemsPerDim * blockIdx.x, b_offset;		// 

	__shared__ float A_shared[BLOCK_SIZE*threadElemsPerDim*BLOCK_SIZE*threadElemsPerDim];
	__shared__ float B_shared[BLOCK_SIZE*threadElemsPerDim*BLOCK_SIZE*threadElemsPerDim];

	//float C_local = 0.0f;
	float C_local[threadElemsPerDim*threadElemsPerDim];

	float a_prefetched[threadElemsPerDim*threadElemsPerDim],
			b_prefetched[threadElemsPerDim*threadElemsPerDim];

	int row, col;

	for(row = 0; row < threadElemsPerDim; row++)
	{
		for(col = 0; col < threadElemsPerDim; col++)
		{
			a_prefetched[row*threadElemsPerDim+col] = A[a_start + (threadIdx.y + row) * width + threadIdx.x + col];
			b_prefetched[row*threadElemsPerDim+col] = B[b_start + (threadIdx.y + row) * width + threadIdx.x + col];
			C_local[row*threadElemsPerDim+col] = 0.0f;
		}
	}


	for(int index = 0; index < gridDim.x;) // równie dobrze mog³oby byæ gridDim.y bo s¹ równe
	{
		++index;

		a_offset = index * BLOCK_SIZE * threadElemsPerDim;
		b_offset = index * BLOCK_SIZE * threadElemsPerDim * width;

		for(row = 0; row < threadElemsPerDim; row++)
		{
			for(col = 0; col < threadElemsPerDim; col++)
			{
				A_shared[(threadIdx.y + row) * blockDim.x * threadElemsPerDim + threadIdx.x + col] = a_prefetched[row*threadElemsPerDim+col];
				B_shared[(threadIdx.y + row) * blockDim.x * threadElemsPerDim + threadIdx.x + col] = b_prefetched[row*threadElemsPerDim+col];
				
			}
		}

		__syncthreads(); // bariera synchronizacyjna, czekamy a¿ wszystkie w¹tki w bloku wype³ni¹ pamiêæ wspó³dzielon¹

		for(row = 0; row < threadElemsPerDim; row++)
		{
			for(col = 0; col < threadElemsPerDim; col++)
			{
				if(a_start + a_offset + (threadIdx.y + row) * width + threadIdx.x + col < width)
				{
					a_prefetched[row*threadElemsPerDim+col] = A[a_start + a_offset + (threadIdx.y + row) * width + threadIdx.x + col];
					b_prefetched[row*threadElemsPerDim+col] = B[b_start + b_offset + (threadIdx.y + row) * width + threadIdx.x + col];
				}
			
				for(int k = 0; k < BLOCK_SIZE*threadElemsPerDim; k++)
				{
					C_local[row*threadElemsPerDim+col] += A_shared[(threadIdx.y + row) * BLOCK_SIZE * threadElemsPerDim + k] * B_shared[k * BLOCK_SIZE * threadElemsPerDim + threadIdx.x + col];
				}
			}
		}

		__syncthreads(); // bariera synchronizacyjna, czekamy a¿ wszystkie w¹tki w bloku oblicz¹ wynik cz¹stkowy

		if(index * BLOCK_SIZE * threadElemsPerDim >= width)
			break;
	}

	int c_start = blockIdx.y * width * BLOCK_SIZE * threadElemsPerDim,
		c_offset = blockIdx.x * BLOCK_SIZE * threadElemsPerDim;
	for(row = 0; row < threadElemsPerDim; row++)
	{
		for(col = 0; col < threadElemsPerDim; col++)
		{
			C[c_start + c_offset + width * (threadIdx.y + row) + threadIdx.x + col] = C_local[row*threadElemsPerDim+col];
		}
	}
}

void performImprovedSharedMemMultipleElemsTest(void)
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

	const unsigned int threadElemsPerDim = 2;
	const unsigned int blockSideOrig = 2;
	const unsigned int blockSide = (int)ceil((float)blockSideOrig/(float)threadElemsPerDim);

	//matrixMulSharedMemPrefetchMultipleElements<blockSide,threadElemsPerDim> <<< dim3(2,2), dim3(blockSide, blockSide) >>>(C_d, A_d, B_d, width); // niby na tej linii jest jakiœ b³¹d ._.
	printf("grid size: %dx%d\n", (int)ceil((float)width/(float)blockSideOrig), (int)ceil((float)width/(float)blockSideOrig));
	printf("elements per thread: a %dx%d submatrix\n", threadElemsPerDim, threadElemsPerDim);
	printf("new block size: %dx%d (original %dx%d)\n", blockSide, blockSide, blockSideOrig, blockSideOrig);
	matrixMulSharedMemPrefetchMultipleElements<1,2><<<dim3((int)ceil((float)width/(float)blockSideOrig),(int)ceil((float)width/(float)blockSideOrig)),dim3(blockSide,blockSide)>>>(C_d, A_d, B_d, width); // niby na tej linii jest jakiœ b³¹d ._.
	
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
