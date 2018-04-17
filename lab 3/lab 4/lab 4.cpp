#include "stdafx.h"
#include "Ring.h"
#include <iostream>

void draw(int, int);
void draw(int*, int);
int getMaxTime(int*, int);

int main()
{
	int N = 20, count = 100;
	int* t = new int[N];
	for (int i = 1; i < N + 1; i++)
	{
		Ring ring = Ring(i);
		//std::cout << ring.getTimeOfRead(count) << std::endl;
		draw(ring.getTimeOfRead(count), i);
		t[i - 1] = ring.getTimeOfRead(count);
	}
	//draw(t, N);
    return 0;
}

void draw(int count, int number)
{
	std::cout << number << ' ';
	if (number < 10)
	{
		std::cout << ' ';
	}
	for (int i = 0; i < count; i++)
	{
		std::cout << (char)254;
	}
	std::cout<<" " <<count<< std::endl;
}

void draw(int * timeArray, int N)
{
	int M = getMaxTime(timeArray, N);
	for (int i = M; i > 0; i--)
	{
		std::cout << i << ' ';
		if (i < 10)
		{
			std::cout << ' ';
		}
		for (int j = 0; j < N; j++)
		{
			std::cout << (timeArray[j] < i ? ' ' : (char)222);
		}
		std::cout << std::endl;
	}
}

int getMaxTime(int * timeArray, int N)
{
	int max = timeArray[0];
	for (int i = 1; i < N; i++)
	{
		if (timeArray[i] > max)
		{
			max = timeArray[i];
		}
	}
	return max;
}
