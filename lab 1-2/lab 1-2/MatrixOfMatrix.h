#pragma once
#include <malloc.h>
#include <time.h>
#include <stdlib.h>

template <typename T>
class MatrixOfMatrix
{
private:
	T**** matrix;
	int alignment = 16;
	int bigSize, lowSize;
public:
	MatrixOfMatrix(int bigSize, int lowSize);
	//MatrixOfMatrix(const MatrixOfMatrix &m);
	~MatrixOfMatrix();

	int getBigSize();
	int getLowSize();
	T**** getMatrix();
	bool equals(MatrixOfMatrix<T> m);

	void initializeMatrix();

	T& operator()(int i, int j, int k, int m);
	T* operator()(int i, int j, int k);
};

template<typename T>
MatrixOfMatrix<T>::MatrixOfMatrix(int _bigSize, int _lowSize)
{
	bigSize = _bigSize;
	lowSize = _lowSize;
	matrix = (T****)_aligned_malloc(bigSize * sizeof(T***), alignment);
	for (int i = 0; i < bigSize; i++)
	{
		matrix[i] = (T***)_aligned_malloc(bigSize * sizeof(T**), alignment);
		for (int j = 0; j < bigSize; j++)
		{
			matrix[i][j] = (T**)_aligned_malloc(lowSize * sizeof(T*), alignment);
			for (int k = 0; k < lowSize; k++)
			{
				matrix[i][j][k] = (T*)_aligned_malloc(lowSize * sizeof(T), alignment);
				for (int m = 0; m < lowSize; m++)
				{
					matrix[i][j][k][m] = 0.0;
				}
			}
		}
	}
}

//template<typename T>
//inline MatrixOfMatrix<T>::MatrixOfMatrix(const MatrixOfMatrix & _matrix)
//{
//	bigSize = _matrix.bigSize;
//	lowSize = _matrix.lowSize;
//	matrix = (T****)_aligned_malloc(bigSize * sizeof(T***), alignment);
//	for (int i = 0; i < bigSize; i++)
//	{
//		matrix[i] = (T***)_aligned_malloc(bigSize * sizeof(T**), alignment);
//		for (int j = 0; j < bigSize; j++)
//		{
//			matrix[i][j] = (T**)_aligned_malloc(lowSize * sizeof(T*), alignment);
//			for (int k = 0; k < lowSize; k++)
//			{
//				matrix[i][j][k] = (T*)_aligned_malloc(lowSize * sizeof(T), alignment);
//				for (int m = 0; m < lowSize; m++)
//				{
//					matrix[i][j][k][m] = _matrix.matrix[i][j][k][m];
//				}
//			}
//		}
//	}
//}

template<typename T>
MatrixOfMatrix<T>::~MatrixOfMatrix()
{
	/*for (int i = 0; i < bigSize; i++)
	{
		for (int j = 0; j < bigSize; j++)
		{
			for (int k = 0; k < lowSize; k++)
			{
				_aligned_free(matrix[i][j][k]);
			}
			_aligned_free(matrix[i][j]);
		}
		_aligned_free(matrix[i]);
	}
	_aligned_free(matrix);*/
}

template<typename T>
inline int MatrixOfMatrix<T>::getBigSize()
{
	return bigSize;
}

template<typename T>
inline int MatrixOfMatrix<T>::getLowSize()
{
	return lowSize;
}

template<typename T>
inline T **** MatrixOfMatrix<T>::getMatrix()
{
	return matrix;
}

template<typename T>
inline bool MatrixOfMatrix<T>::equals(MatrixOfMatrix<T> ma)
{
	for (int i = 0; i < bigSize; i++)
	{
		for (int j = 0; j < bigSize; j++)
		{
			for (int k = 0; k < lowSize; k++)
			{
				for (int m = 0; m < lowSize; m++)
				{
					if (ma(i, j, k, m) != matrix[i][j][k][m])
						return false;
				}
			}
		}
	}
	return true;
}

template<typename T>
inline void MatrixOfMatrix<T>::initializeMatrix()
{
	srand(time(0));
	for (int i = 0; i < bigSize; i++)
	{
		for (int j = 0; j < bigSize; j++)
		{
			for (int k = 0; k < lowSize; k++)
			{
				for (int m = 0; m < lowSize; m++)
				{
					matrix[i][j][k][m] = (float)(rand() % 10) / 10.0;
				}
			}
		}
	}
}

template<typename T>
inline T& MatrixOfMatrix<T>::operator()(int i, int j, int k, int m)
{
	return matrix[i][j][k][m];
}

template<typename T>
inline T* MatrixOfMatrix<T>::operator()(int i, int j, int k)
{
	return matrix[i][j][k];
}

