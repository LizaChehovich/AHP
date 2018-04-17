#pragma once
#include <stdint.h>
#include <Windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//pixel range
#define MIN 0
#define MAX 255

//block size
#define Xdim 32
#define Ydim 16

#define CountStream 4

//mini matrix
float filter[3][3] = { { -1,-1,-1 },{ -1,9,-1 },{ -1,-1,-1 } };
__constant__ float d_filter[3][3];
