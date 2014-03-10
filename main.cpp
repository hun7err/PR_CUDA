#include <stdio.h>
#include <stdlib.h>

extern void performMultiBlockTests(void);

int main(int argc, char ** argv)
{
	performMultiBlockTests();

	system("PAUSE");
	return 0;
}