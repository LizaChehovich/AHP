//#include "cuda_functions_gray.h"
//
//__device__ void gray_center_kernel(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
//{
//	const int count_row = Xdim + 2;
//	const int count_column = Ydim + 2;
//	const int pitch_in_int = pitch / sizeof(int);
//	const int absX = blockIdx.x * blockDim.x + threadIdx.x;
//	const int absY = blockIdx.y * blockDim.y + threadIdx.y;
//
//	const int absXinByte = absX * sizeof(int);
//
//	const int Width = width + 2 * sizeof(int);
//	const int Height = heigth + 2;
//
//	__shared__ int buf[count_column][count_row];
//
//	//load data in shared memory
//
//	buf[threadIdx.y][threadIdx.x] = input[absY*pitch_in_int + absX];
//
//	if (threadIdx.y < 2)
//		buf[threadIdx.y + blockDim.y][threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX];
//
//	if (threadIdx.x < 2)
//		buf[threadIdx.y][count_row - 2 + threadIdx.x] = input[absY*pitch_in_int + absX + blockDim.x];
//
//	if (threadIdx.y < 2  && threadIdx.x < 2)
//		buf[threadIdx.y + blockDim.y][count_row - 2 + threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x];
//
//	__syncthreads();
//
//	int val = 0;
//	float sum
//	int bytePosY = threadIdx.y + 1;
//	int bytePosX = (threadIdx.x + 1) * 4;
//
//	for (int k = 0; k < 4; k++)
//	{
//		sum = 0;
//
//		for (int i = -1; i < 2; i++)
//		{
//			for (int j = -1; j < 2; j++)
//			{
//				sum += ((byte*)buf[bytePosY + i])[bytePosX + j] * d_filter[i + 1][j + 1];
//			}
//		}
//		if (sum < 0) sum = 0;
//		if (sum > 255) sum = 255;
//
//		((byte*)&val)[k] = sum;
//		bytePosX++;
//	}
//
//	result[absY*res_pitch / sizeof(int) + absX] = val;
//}
//
//__device__ void gray_frame_kernel(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
//{
//	const int count_row = Xdim + 2;
//	const int count_column = Ydim + 2;
//	const int pitch_in_int = pitch / sizeof(int);
//	const int absX = blockIdx.x * blockDim.x + threadIdx.x;
//	const int absY = blockIdx.y * blockDim.y + threadIdx.y;
//
//	const int absXinByte = absX * sizeof(int);
//
//	const int Width = width + 2 * sizeof(int);
//	const int Height = heigth + 2;
//
//	__shared__ int buf[count_column][count_row];
//
//	//load data in shared memory
//
//	if (absY < Height && absXinByte < Width)
//		buf[threadIdx.y][threadIdx.x] = input[absY*pitch_in_int + absX];
//
//	if (threadIdx.y < 2 && ((absY + blockDim.y) < Height) && absXinByte < Width)
//		buf[threadIdx.y + blockDim.y][threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX];
//
//	if (absY < Height && threadIdx.x < 2 && ((absX + blockDim.x) * sizeof(int)) < Width)
//		buf[threadIdx.y][count_row - 2 + threadIdx.x] = input[absY*pitch_in_int + absX + blockDim.x];
//
//	if (threadIdx.y < 2 && (absY + blockDim.y) < Height && threadIdx.x < 2 && ((absX + blockDim.x) * sizeof(int)) < Width)
//		buf[threadIdx.y + blockDim.y][count_row - 2 + threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x];
//
//	__syncthreads();
//
//	int val = 0;
//	float sum
//	int bytePosY = threadIdx.y + 1;
//	int bytePosX = (threadIdx.x + 1) * 4;
//
//	for (int k = 0; k < 4; k++)
//	{
//		if (absY >= Height && absXinByte >= Width)
//			break;
//		sum = 0;
//
//		for (int i = -1; i < 2; i++)
//		{
//			for (int j = -1; j < 2; j++)
//			{
//				sum += ((byte*)buf[bytePosY + i])[bytePosX + j] * d_filter[i + 1][j + 1];
//			}
//		}
//		if (sum < 0) sum = 0;
//		if (sum > 255) sum = 255;
//
//		((byte*)&val)[k] = sum;
//		bytePosX++;
//	}
//
//	if (absY < heigth && absXinByte < width)
//	{
//		result[absY*res_pitch / sizeof(int) + absX] = val;
//	}
//}
//
//__global__ void cuda_gray_processing(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
//{
//	if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1)
//		gray_frame_kernel(input, result, width, heigth, pitch, res_pitch);
//	else
//		gray_center_kernel(input, result, width, heigth, pitch, res_pitch);
//}
//
//__global__ void cuda_gray_processing(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch, const int frame)
//{
//	if (frame)
//		if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1)
//			gray_frame_kernel(input, result, width, heigth, pitch, res_pitch);
//		else
//			gray_center_kernel(input, result, width, heigth, pitch, res_pitch);
//	else
//		gray_center_kernel(input, result, width, heigth, pitch, res_pitch);
//}