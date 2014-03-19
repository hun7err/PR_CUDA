#include <stdio.h>
#include <stdlib.h>

extern void performSingleBlockTests(void);
extern void performMultiBlockTests(void);
extern void performSharedMemTests(void);

extern void performImprovedSharedMemTest(void);
extern void performImprovedSharedMemMultipleElemsTest(void);

int main(int argc, char ** argv)
{
	if(argc != 2)
	{
		printf("Uzycie: %s numer_testu [nazwa_pliku]\n", argv[0]);
		printf("numer_testu: liczba calkowita z przedzialu domknietego [1,5]\n");
		printf("1 - jeden blok watkow\n"
				"2 - grid wieloblokowy\n"
				"3 - wykorzystanie pamieci wspoldzielonej\n"
				"4 - wersja 3. + zrownoleglenie pobierania danych do pamieci wspoldz. i obliczen\n"
				"5 - wersja 4. + wiecej pracy per watek\n");
		printf("nazwa_pliku: nazwa pliku, w ktorym zapisane zostana wyniki\n");

		//return 1;
	}

	int test = 3; //atoi(argv[1]);
	
	if(argc > 2)
	{
		// otworz plik i przekieruj tam stdout
	}
	
	
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
			printf("Macierz\tBlokGrid\t\tCzas\tGFLOPS\n");
		break;
		case 5:
			printf("Macierz\tOr. blok\tEpW\tGrid\tCzas\tGFLOPS\n"); // EpW = elementy per w¹tek, or. blok = oryginalny blok
		break;
	}

	// przywroc normalne stdout

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

	system("PAUSE");
	return 0;
}