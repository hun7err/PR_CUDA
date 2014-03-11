#include "genmatrix.h"
#include <time.h>
#include <stdlib.h>

void generateTestMatrix(float *matPtr, int width)
{
	srand((unsigned int)time(NULL));

	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < width; j++)
		{
			matPtr[i*width+j] = (float)rand() / (float)RAND_MAX;
		}
	}
}