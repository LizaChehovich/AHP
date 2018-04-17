#pragma once
#include "MatrixOfMatrix.h"
#include <Windows.h>
#include <immintrin.h>
#include <stdbool.h>
#include <inttypes.h>

template <typename T>
class MatrixController
{
	int bigSize, lowSize;
	LARGE_INTEGER start, finish, freq;
public:
	MatrixController(int big, int low);
	~MatrixController();

	float multiply(MatrixOfMatrix<T> mA, MatrixOfMatrix<T> mB, MatrixOfMatrix<T> mR);
	float SSE(MatrixOfMatrix<T> mA, MatrixOfMatrix<T> mB, MatrixOfMatrix<T> mR);
};

template<typename T>
inline MatrixController<T>::MatrixController(int big, int low)
{
	bigSize = big;
	lowSize = low;
}

template<typename T>
inline MatrixController<T>::~MatrixController()
{
}

template<typename T>
inline float MatrixController<T>::multiply(MatrixOfMatrix<T> mA, MatrixOfMatrix<T> mB, MatrixOfMatrix<T> mR)
{
	/*T**** a = mA.getMatrix();
	T**** b = mB.getMatrix();
	T**** r = mR.getMatrix();*/
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	T* r, *b, a;
	for (int i = 0; i < bigSize; i++)
	{
		for (int j = 0; j < bigSize; j++)
		{
			for (int k = 0; k < lowSize; k++)
			{
				r = mR(i, j, k);
				for (int m = 0; m < lowSize; m++)
				{
					b = mB(i, j, m);
					a = mA(i, j, k, m);
//#pragma loop(no_vector)
					for (int l = 0; l < 16; l++)
					{
						r[l] += b[l] * a;
					}
				}
			}
		}
	}
	QueryPerformanceCounter(&finish);
	return (finish.QuadPart - start.QuadPart) / (float)freq.QuadPart;
}

template<typename T>
inline float MatrixController<T>::SSE(MatrixOfMatrix<T> mA, MatrixOfMatrix<T> mB, MatrixOfMatrix<T> mR)
{
	float* res, *matrixB;
	__m128 a, b, r;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	for (int i = 0; i < bigSize; i++)
	{
		for (int j = 0; j < bigSize; j++)
		{
			for (int k = 0; k < lowSize; k++)
			{
				res = mR(i, j, k);
				for (int m = 0; m < lowSize; m++)
				{
					matrixB = mB(i, j, m);
					a = _mm_set1_ps(mA(i, j, k, m));
					for (int l = 0; l < lowSize; l += 4)
					{
						r = _mm_load_ps(res + l);
						b = _mm_load_ps(matrixB + l);
						_mm_store_ps(res + l, _mm_add_ps(r, _mm_mul_ps(a, b)));
					}
				}
			}
		}
	}
	QueryPerformanceCounter(&finish);
	return (finish.QuadPart - start.QuadPart) / (float)freq.QuadPart;
}
