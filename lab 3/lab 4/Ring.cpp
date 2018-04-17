#include "stdafx.h"
#include "Ring.h"

Ring::Ring(int n)
{
	N = n;
	CountElementsInLine = BlockSize / N / sizeof(int);
	array = new int[Offset*N];
	for (int i = 0; i < CountElementsInLine; i++)
	{
		for (int j = 0; j < N - 1; j++)
		{
			array[i + j * Offset] = (j + 1)*Offset + i;
		}
		array[i + (N - 1) * Offset] = i + 1;
	}
	array[(N - 1) * Offset + CountElementsInLine - 1] = 0;
}

Ring::~Ring()
{
	delete[] array;
}

int Ring::getTimeOfRead(int count)
{
	long long start, end;
	int countElement = CountElementsInLine * N;
	int element = 0;
	start = __rdtsc();
	for (int i = 0; i < countElement*count; i++)
	{
		element = array[element];
	}
	end = __rdtsc();
	return (end - start)/(count*countElement);
}
