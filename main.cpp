#include <stdio.h>
#include <stdlib.h>

extern void performMultiBlockTests(void);
extern void performSingleBlockTests(void);
//
extern void performSharedMemTest(void);
extern void performImprovedSharedMemTest(void);
extern void performImprovedSharedMemMultipleElemsTest(void);

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
	/*printf("Basic SharedMem\n");
	performSharedMemTest();
	printf("Improved SharedMem\n");
	performImprovedSharedMemTest();*/
	performImprovedSharedMemMultipleElemsTest();

	system("PAUSE");
	return 0;
}