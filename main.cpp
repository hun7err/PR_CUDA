#include <stdio.h>
#include <stdlib.h>

extern void performMultiBlockTests(void);
extern void performSingleBlockTests(void);
//
extern void performSharedMemTest(void);

int main(int argc, char ** argv)
{
	/*
	// wariant 1.
	printf("=== Single block tests ===\n");
	performSingleBlockTests();
	printf("\n");
	// wariant 2.
	printf("=== Multi block Tests ===\n");
	performMultiBlockTests();
	printf("\n");
	*/

	system("PAUSE");
	return 0;
}