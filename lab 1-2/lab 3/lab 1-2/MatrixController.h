#pragma once
#include "MatrixOfMatrix.h"
#include <Windows.h>
#include <immintrin.h>
#include <stdbool.h>
#include <inttypes.h>
#include <omp.h> 

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
	float OpenMP(MatrixOfMatrix<T> mA, MatrixOfMatrix<T> mB, MatrixOfMatrix<T> mR);
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
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	for (int i = 0; i < bigSize; i++)
	{
		for (int j = 0; j < bigSize; j++)
		{
			for (int k = 0; k < lowSize; k++)
			{
				for (int m = 0; m < lowSize; m++)
				{
					for (int l = 0; l < lowSize; l++)
					{
						mR(i, j, k, m) += mA(i, j, k, l)*mB(i, j, l, m);
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
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	for (int i = 0; i < bigSize; i++)
	{
		for (int j = 0; j < bigSize; j++)
		{
			for (int k = 0; k < lowSize; k++)
			{
				float* res = mR(i, j, k);
				for (int m = 0; m < lowSize; m++)
				{
					float* matrixB = mB(i, j, m);
					__m128 a = _mm_set1_ps(mA(i, j, k, m));
					for (int l = 0; l < lowSize; l += 4)
					{
						__m128 r = _mm_load_ps(res + l);
						__m128 b = _mm_load_ps(matrixB + l);
						_mm_add_ps(r, _mm_mul_ps(a, b));
						_mm_store_ps(res + l, r);
					}

				}
			}
		}
	}
	QueryPerformanceCounter(&finish);
	return (finish.QuadPart - start.QuadPart) / (float)freq.QuadPart;
}

template<typename T>
inline float MatrixController<T>::OpenMP(MatrixOfMatrix<T> mA, MatrixOfMatrix<T> mB, MatrixOfMatrix<T> mR)
{
	/*T**** a = mA.getMatrix();
	T**** b = mB.getMatrix();
	T**** r = mR.getMatrix();*/
	omp_set_dynamic(0);
	omp_set_num_threads(8);
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);

#pragma omp parallel for
	for (int i = 0; i < bigSize; i++)
	{
		for (int j = 0; j < bigSize; j++)
		{
			for (int k = 0; k < lowSize; k++)
			{
				for (int m = 0; m < lowSize; m++)
				{
					for (int l = 0; l < lowSize; l++)
					{
						mR(i, j, k, m) += mA(i, j, k, l)*mB(i, j, l, m);
						//r[i][j][k][m] += a[i][j][k][l] * b[i][j][l][m];
					}
				}
			}
		}
	}
	QueryPerformanceCounter(&finish);
	return (finish.QuadPart - start.QuadPart) / (float)freq.QuadPart;
}
