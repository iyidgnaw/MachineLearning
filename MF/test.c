#include <stdio.h>
#include <time.h>

int main()
{
	int matrix1[100][80];
	int matrix2[80][100];
	time_t start, end;
	for (int i = 0;i<100;i++)
	{
		for (int j =0; j<80;j++)
		{
			matrix1[i][j] = i;
			matrix2[j][i] = j;
		}
	}

	time(&start);
	for(int p=0;p<500;p++)
	{
		int result[100][100];
		for(int i =0; i<100; i++)
		{
			for(int j =0; j <100; j++)
			{
				for(int k =0;k<80;k++)
				{
					result[i][j] += matrix1[i][k]*matrix2[k][j];
				}
			}
		}
	}
	time(&end);
	printf("The difference is: %f seconds", difftime(end, start));
	return 0;

}


