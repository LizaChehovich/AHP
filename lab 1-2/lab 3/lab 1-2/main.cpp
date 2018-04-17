#include "stdafx.h"
#include "MatrixOfMatrix.h"
#include "MatrixController.h"
#include <iostream>

const int B = 256;
const int L = 16;

int main()
{
	MatrixOfMatrix<float> mA = MatrixOfMatrix<float>(B, L);
	MatrixOfMatrix<float> mB = MatrixOfMatrix<float>(B, L);
	MatrixOfMatrix<float> mR = MatrixOfMatrix<float>(B, L);
	MatrixOfMatrix<float> mOpenMP = MatrixOfMatrix<float>(B, L);
	mA.initializeMatrix();
	mB.initializeMatrix();
	MatrixController<float> c = MatrixController<float>(B, L);
	std::cout << c.multiply(mA, mB, mR) << "\n"; 
	std::cout << c.OpenMP(mA, mB, mOpenMP) << "\n";
    return 0;
}

