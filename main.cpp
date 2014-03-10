#include <stdio.h>
#include <stdlib.h>

extern void performTest(void);

int main(int argc, char ** argv)
{
	performTest();

	system("PAUSE");
	return 0;
}