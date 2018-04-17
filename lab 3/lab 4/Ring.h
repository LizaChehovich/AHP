#pragma once
#include <Windows.h>

class Ring
{
	static const int BlockSize = 256 * 1024;
	static const int Offset = 1024 * 1024;
	int N;
	int CountElementsInLine;
	int* array;
public:
	Ring(int N);
	~Ring();
	int getTimeOfRead(int count);
};

