#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//������
float filter[3][3] = { { -1,-1,-1 },{ -1,9,-1 },{ -1,-1,-1 } };
//������ ��� ����������
__constant__ float d_filter[3][3];

//����������� �������� �������
#define MIN 0
//������������ �������� �������
#define MAX 255

#define CountStreamPerDevice 3

//������ ����� �� ��� X
#define Xdim 32
//������ ����� �� ��� Y
#define Ydim 16