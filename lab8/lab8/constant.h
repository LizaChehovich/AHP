#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//фильтр
float filter[3][3] = { { -1,-1,-1 },{ -1,9,-1 },{ -1,-1,-1 } };
//фильтр для устройства
__constant__ float d_filter[3][3];

//минимальное значение пикселя
#define MIN 0
//максимальное значение пикселя
#define MAX 255

#define CountStreamPerDevice 3

//размер блока по оси X
#define Xdim 32
//размер блока по оси Y
#define Ydim 16