#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <fcntl.h>

extern void performSingleBlockTests(void);
extern void performMultiBlockTests(void);
extern void performSharedMemTests(void);
extern void performImprovedSharedMemTests(void);
extern void performImprovedSharedMemMultipleElemsTests(void);

int main(int argc, char ** argv)
{
	if(argc < 2)
	{
		printf("Uzycie: %s numer_testu\n", argv[0]);
		printf("numer_testu: liczba calkowita z przedzialu domknietego [1,5]\n");
		printf("1 - jeden blok watkow\n"
				"2 - grid wieloblokowy\n"
				"3 - wykorzystanie pamieci wspoldzielonej\n"
				"4 - wersja 3. + zrownoleglenie pobierania danych do pamieci wspoldz. i obliczen\n"
				"5 - wersja 4. + wiecej pracy per watek\n");

		return 1;
	}

	int test = atoi(argv[1]);
	
	switch(test)
	{
		case 1:
			printf("Macierz\tBlok\tCzas\tGFLOPS\n");
			performSingleBlockTests();
		break;
		case 2:
			printf("Macierz\tBlok\tGrid\tCzas\tGFLOPS\n");
			performMultiBlockTests();
		break;
		case 3:
			printf("Macierz\tBlok\tGrid\tCzas\tGFLOPS\n");
			performSharedMemTests();
		break;
		case 4:
			printf("Macierz\tBlok\tGrid\tCzas\tGFLOPS\n");
			performImprovedSharedMemTests();
		break;
		case 5:
			printf("Macierz\tOryginalny blok\tNowy blok\tGrid\tCzas\tGFLOPS\n"); // EpW = elementy per w¹tek, or. blok = oryginalny blok
			performImprovedSharedMemMultipleElemsTests();
		break;
	}
	//przywroc normalne stdout

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
	printf("Improved SharedMem\n");
	performImprovedSharedMemTest();*/
	//performImprovedSharedMemMultipleElemsTest();

	//system("PAUSE");
	return 0;
}