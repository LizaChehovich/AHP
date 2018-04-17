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
	MatrixOfMatrix<float> mV = MatrixOfMatrix<float>(B, L);
	MatrixOfMatrix<float> mSSE = MatrixOfMatrix<float>(B, L);
	mA.initializeMatrix();
	mB.initializeMatrix();
	MatrixController<float> c = MatrixController<float>(B, L);
	std::cout << c.multiply(mA, mB, mV) << "\n";
	//std::cout << *mV(0,0,1) << "\n";
	std::cout << c.SSE(mA, mB, mSSE) << "\n";
	//std::cout << *mSSE(0,0,1) << "\n";
	//std::cout << mV.equals(mSSE);
	system("pause");
    return 0;
}

