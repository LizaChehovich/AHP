//#include "cpu_functions.h"
//


//double cpu_convert_image(uint8_t * input, uint8_t * result, unsigned int width, unsigned int height, unsigned int channels)
//{
//	float val;
//	LARGE_INTEGER start, finish, freq;
//	QueryPerformanceFrequency(&freq);
//	QueryPerformanceCounter(&start);
//	for (int x = 0; x < width * channels; x++)
//	{
//		for (int y = 0; y < height; y++)
//		{
//			val = 0;
//			for (int i = -1; i < 2; i++)
//			{
//				for (int j = -1; j < 2; j++)
//				{
//					if (x + j * channels < 0 || x + j * channels >= width*channels || y + i < 0 || y + i >= height)
//						continue;
//					val += input[(y + i)*width * channels + (x + j * channels)] * filter[i + 1][j + 1];
//				}
//			}
//			result[y * width * channels + x] = val<MIN ? MIN : val>MAX ? MAX : val;
//		}
//	}
//	QueryPerformanceCounter(&finish);
//	return (finish.QuadPart - start.QuadPart) * 1000 / (double)freq.QuadPart;
//}