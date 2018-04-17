//#include "cuda_functions_color.h"
//
//__device__ void color_center_kernel(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
//{
//	const int channels = 3;
//	const int count_row = (Xdim*channels + 2);
//	const int count_column = Ydim + 2;
//	const int pitch_in_int = pitch / sizeof(int);
//	const int absX = (blockIdx.x * blockDim.x)*channels + threadIdx.x;
//	const int absY = blockIdx.y * blockDim.y + threadIdx.y;
//
//	const int absXinByte = absX * sizeof(int);
//	const int blockDimXinByte = blockDim.x * sizeof(int);
//
//	const int Width = (width * channels + 2 * sizeof(int));
//	const int Height = heigth + 2;
//
//	__shared__ int buf[count_column][count_row];
//
//	//load data in shared memory
//
//	for (int i = 0; i < channels; i++) {
//		buf[threadIdx.y][threadIdx.x + blockDim.x*i] = input[absY*pitch_in_int + absX + blockDim.x*i];
//
//		buf[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x*i] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x*i];
//	}
//
//	if (threadIdx.x < 2)
//		buf[threadIdx.y][count_row - 2 + threadIdx.x] = input[absY*pitch_in_int + absX + blockDim.x*channels];
//
//	if (threadIdx.y < 2 && threadIdx.x < 2)
//		buf[threadIdx.y + blockDim.y][count_row - 2 + threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x*channels];
//
//	__syncthreads();
//
//	int val;
//	float sum
//
//	for (int c = 0; c < channels; c++)
//	{
//
//		val = 0;
//
//		int bytePosY = threadIdx.y + 1;
//		int bytePosX = (threadIdx.x + 1 + c*blockDim.x) * sizeof(int);
//
//		for (int k = 0; k < 4; k++)
//		{
//			sum = 0;
//
//			for (int i = -1; i < 2; i++)
//			{
//				for (int j = -1; j < 2; j++)
//				{
//					sum += ((byte*)buf[bytePosY + i])[bytePosX + j*channels] * d_filter[i + 1][j + 1];
//				}
//			}
//			if (sum < 0) sum = 0;
//			if (sum > 255) sum = 255;
//
//			((byte*)&val)[k] = sum;
//			bytePosX++;
//		}
//
//		result[absY*res_pitch / sizeof(int) + absX + c*blockDim.x] = val;
//	}
//}
//
//__device__ void color_frame_kernel(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
//{
//	const int channels = 3;
//	const int count_row = (Xdim*channels + 2);
//	const int count_column = Ydim + 2;
//	const int pitch_in_int = pitch / sizeof(int);
//	const int absX = (blockIdx.x * blockDim.x)*channels + threadIdx.x;
//	const int absY = blockIdx.y * blockDim.y + threadIdx.y;
//
//	const int absXinByte = absX * sizeof(int);
//	const int blockDimXinByte = blockDim.x * sizeof(int);
//
//	const int Width = (width * channels + 2 * sizeof(int));
//	const int Height = heigth + 2;
//
//	__shared__ int buf[count_column][count_row];
//
//	//load data in shared memory
//
//	for (int i = 0; i < channels; i++) {
//		if (absY < Height && absXinByte + blockDimXinByte*i < Width)
//			buf[threadIdx.y][threadIdx.x + blockDim.x*i] = input[absY*pitch_in_int + absX + blockDim.x*i];
//
//		if (threadIdx.y < 2 && ((absY + blockDim.y) < Height) && absXinByte + blockDimXinByte*i < Width)
//			buf[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x*i] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x*i];
//	}
//
//	if (absY < Height && threadIdx.x < 2 && ((absX + blockDim.x*channels) * sizeof(int)) < Width)
//		buf[threadIdx.y][count_row - 2 + threadIdx.x] = input[absY*pitch_in_int + absX + blockDim.x*channels];
//
//	if (threadIdx.y < 2 && (absY + blockDim.y) < Height && threadIdx.x < 2 && ((absX + blockDim.x*channels) * sizeof(int)) < Width)
//		buf[threadIdx.y + blockDim.y][count_row - 2 + threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x*channels];
//
//	__syncthreads();
//
//	int val;
//	float sum
//
//	for (int c = 0; c < channels; c++)
//	{
//
//		val = 0;
//
//		int bytePosY = threadIdx.y + 1;
//		int bytePosX = (threadIdx.x + 1 + c*blockDim.x) * sizeof(int);
//
//		for (int k = 0; k < 4; k++)
//		{
//			if (absY >= Height && absXinByte + c*blockDim.x * sizeof(int) >= Width)
//				break;
//			sum = 0;
//
//			for (int i = -1; i < 2; i++)
//			{
//				for (int j = -1; j < 2; j++)
//				{
//					sum += ((byte*)buf[bytePosY + i])[bytePosX + j*channels] * d_filter[i + 1][j + 1];
//				}
//			}
//			if (sum < 0) sum = 0;
//			if (sum > 255) sum = 255;
//
//			((byte*)&val)[k] = sum;
//			bytePosX++;
//		}
//
//		if (absY < heigth && absXinByte + c * blockDimXinByte < width * channels)
//		{
//			result[absY*res_pitch / sizeof(int) + absX + c*blockDim.x] = val;
//		}
//	}
//}
//
//__global__ void cuda_color_processing(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
//{
//	if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1)
//		color_frame_kernel(input, result, width, heigth, pitch, res_pitch);
//	else
//		color_center_kernel(input, result, width, heigth, pitch, res_pitch);
//}
//
//__global__ void cuda_color_processing(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch, const int frame)
//{
//	if (frame)
//		if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1)
//			color_frame_kernel(input, result, width, heigth, pitch, res_pitch);
//		else
//			color_center_kernel(input, result, width, heigth, pitch, res_pitch);
//	else
//		color_center_kernel(input, result, width, heigth, pitch, res_pitch);
//}