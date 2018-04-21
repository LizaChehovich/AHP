#pragma once
#include <stdint.h>
#include <Windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//����������� �������� �������
#define MIN 0
//������������ �������� �������
#define MAX 255

//������ ����� �� ��� X
#define Xdim 32
//������ ����� �� ��� Y
#define Ydim 16

//���������� ������� ��� ������� 
#define CountStream 2

//������
float filter[3][3] = { { -1,-1,-1 },{ -1,9,-1 },{ -1,-1,-1 } };
//������ ��� ����������
__constant__ float d_filter[3][3];
