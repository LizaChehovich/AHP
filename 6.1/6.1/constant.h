#pragma once
#include <stdint.h>
#include <Windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//минимальное значение пикселя
#define MIN 0
//максимальное значение пикселя
#define MAX 255

//размер блока по оси X
#define Xdim 32
//размер блока по оси Y
#define Ydim 16

//количество стримов для расчёта 
#define CountStream 2

//фильтр
float filter[3][3] = { { -1,-1,-1 },{ -1,9,-1 },{ -1,-1,-1 } };
//фильтр для устройства
__constant__ float d_filter[3][3];
