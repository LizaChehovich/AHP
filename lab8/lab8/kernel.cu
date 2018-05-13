#include <iostream>
#include "image_helper.h"
#include <malloc.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "mpi.h"

typedef uint8_t byte;
using namespace std;

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

int main_main();
double gpu_stream_convert_image(byte* input, byte* result, unsigned int width, unsigned int height, unsigned int channels);

bool error(cudaError_t val)
{
	if (!val)
		return false;
	cout << "Cuda Error " << cudaGetLastError() << endl;
	cudaDeviceReset();
	return true;
}

__device__ void color_center_kernel(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
{
	const int channels = 3;
	const int count_row = (Xdim*channels + 2);
	const int count_column = Ydim + 2;
	const int pitch_in_int = pitch / sizeof(int);
	const int absX = (blockIdx.x * blockDim.x)*channels + threadIdx.x;
	const int absY = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int buf[count_column][count_row];
	__shared__ int res[Ydim][Xdim];

	//загрузка данны в разделяемую память
	//каждая нить загружает 3 раза по 1 int - транзакции
	for (int i = 0; i < channels; i++) {
		buf[threadIdx.y][threadIdx.x + blockDim.x*i] = input[absY*pitch_in_int + absX + blockDim.x*i];

		if (threadIdx.y < 2)
			buf[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x*i] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x*i];
	}

	//2 строки
	if (threadIdx.x < 2)
		buf[threadIdx.y][count_row - 2 + threadIdx.x] = input[absY*pitch_in_int + absX + blockDim.x*channels];

	//квадрат 2*2
	if (threadIdx.y < 2 && threadIdx.x < 2)
		buf[threadIdx.y + blockDim.y][count_row - 2 + threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x*channels];
	
	__syncthreads();

	float sum;
	int bytePosY = threadIdx.y + 1;

	//3 раза обрабатываем по 4 байта
	for (int c = 0; c < channels; c++)
	{
		res[threadIdx.y][threadIdx.x] = 0;

		for (int k = 0; k < 4; k++)
		{
			const int bytePosX = (threadIdx.x + 1 + c*blockDim.x) * sizeof(int) + k;
			sum = 0;

			for (int i = -1; i < 2; i++)
			{
				for (int j = -1; j < 2; j++)
				{
					sum += ((byte*)buf[bytePosY + i])[bytePosX + j * channels] * d_filter[i + 1][j + 1];
				}
			}
			sum = round(sum);
			if (sum < 0) sum = 0;
			if (sum > 255) sum = 255;

			((byte*)&(res[threadIdx.y][threadIdx.x]))[k] = sum;
		}

		result[absY*res_pitch / sizeof(int) + absX + c*blockDim.x] = res[threadIdx.y][threadIdx.x];
	}
}

//ядро обработки крайних правых и нижних блоков цветного изображения
__device__ void color_frame_kernel(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
{
	const int channels = 3;
	const int count_row = (Xdim*channels + 2);
	const int count_column = Ydim + 2;
	const int pitch_in_int = pitch / sizeof(int);
	const int absX = (blockIdx.x * blockDim.x)*channels + threadIdx.x;
	const int absY = blockIdx.y * blockDim.y + threadIdx.y;

	const int absXinByte = absX * sizeof(int);
	const int blockDimXinByte = blockDim.x * sizeof(int);

	const int Width = (width * channels + 2 * sizeof(int));
	const int Height = heigth + 2;

	__shared__ int buf[count_column][count_row];
	__shared__ int res[Ydim][Xdim];

	//load data in shared memory

	for (int i = 0; i < channels; i++) {
		if (absY < Height && absXinByte + blockDimXinByte*i < Width)
			buf[threadIdx.y][threadIdx.x + blockDim.x*i] = input[absY*pitch_in_int + absX + blockDim.x*i];

		if (threadIdx.y < 2 && ((absY + blockDim.y) < Height) && absXinByte + blockDimXinByte*i < Width)
			buf[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x*i] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x*i];
	}

	if (absY < Height && threadIdx.x < 2 && ((absX + blockDim.x*channels) * sizeof(int)) < Width)
		buf[threadIdx.y][count_row - 2 + threadIdx.x] = input[absY*pitch_in_int + absX + blockDim.x*channels];

	if (threadIdx.y < 2 && (absY + blockDim.y) < Height && threadIdx.x < 2 && ((absX + blockDim.x*channels) * sizeof(int)) < Width)
		buf[threadIdx.y + blockDim.y][count_row - 2 + threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x*channels];

	__syncthreads();
	
	float sum;
	const int bytePosY = threadIdx.y + 1;

	for (int c = 0; c < channels; c++)
	{

		res[threadIdx.y][threadIdx.x] = 0;

		for (int k = 0; k < 4; k++)
		{
			const int bytePosX = (threadIdx.x + 1 + c*blockDim.x) * sizeof(int) + k;
			if (absY >= Height || absXinByte + c*blockDim.x * sizeof(int) + k >= width* channels)
				break;
			sum = 0;

			for (int i = -1; i < 2; i++)
			{
				for (int j = -1; j < 2; j++)
				{
					sum += ((byte*)buf[bytePosY + i])[bytePosX + j * channels] * d_filter[i + 1][j + 1];
				}
			}
			sum = round(sum);
			if (sum < 0) sum = 0;
			if (sum > 255) sum = 255;

			((byte*)&(res[threadIdx.y][threadIdx.x]))[k] = sum;
		}

		if (absY < heigth && absXinByte + c * blockDimXinByte < width * channels)
		{
			result[absY*res_pitch / sizeof(int) + absX + c*blockDim.x] = res[threadIdx.y][threadIdx.x];
		}
	}
}

//функция разделения блоков на центральные и граничные для цветного изображения
__global__ void cuda_color_processing(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
{
	if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1)
		color_frame_kernel(input, result, width, heigth, pitch, res_pitch);
	else
		color_center_kernel(input, result, width, heigth, pitch, res_pitch);
}


int main(int argc, char* argv[])
{
	//MPI_Init(&argc, &argv);

	const int res = main_main();

	//MPI_Finalize();

	return res;
}

int main_main()
{
	char input[] = "input.ppm";
	char output[] = "output.ppm";

	unsigned int width, height, channels;
	byte* in_image = nullptr;

	if (!load_ppm(input, &in_image, &width, &height, &channels))
	{
		cout << "Error in loading file" << endl;
		return 1;
	}

	if (channels != 3)
	{
		cout << "Error in count channels" << endl;
		free(in_image);
		return 1;
	}

	byte* out_image = (byte*)malloc(width * height * sizeof(uint8_t) * channels);
	if (!out_image)
	{
		cout << "Error in memory allocation" << endl;
		free(in_image);
		return 1;
	}

	gpu_stream_convert_image(in_image, out_image, width, height, channels);

	if (!save_ppm(output, out_image, width, height, channels))
	{
		cout << "Error in save file" << endl;
	}

	free(in_image);
	free(out_image);
	return 0;
}

double gpu_stream_convert_image(byte* input, byte* result, unsigned int width, unsigned int height, unsigned int channels)
{
	cudaMemcpyToSymbol(d_filter, filter, 9 * sizeof(float), 0, cudaMemcpyHostToDevice);

	float time = 0;

	int countDevice;
	cudaGetDeviceCount(&countDevice);
	const int countStream = CountStreamPerDevice * countDevice;

	//высота части изображения, обрабатываемой одним потоком
	int stream_height = ceil(((float)height) / Ydim / countStream) * Ydim;
	int Width = width * channels;

	size_t input_pitch;
	size_t res_pitch;

	cudaError_t err = cudaSuccess;

	cudaStream_t* stream = new cudaStream_t[countStream];

	byte** dev_input = new byte*[countStream];
	byte** dev_output = new byte*[countStream];

	//регистрируем входную и выходные матрицы как пиннед-память
	err = cudaHostRegister(input, Width*height, cudaHostRegisterPortable);
	err = cudaHostRegister(result, Width*height, cudaHostRegisterPortable);

	int height_count;

	for (int i = 0; i < countStream; i++)
	{
		err = cudaStreamCreate(&stream[i]);
	}

	int offset = 0;

		for (int i = 0; i < countStream; i++)
		{
			if (i%CountStreamPerDevice == 0)
				cudaSetDevice(i / CountStreamPerDevice);
			//количество строк изображения, обрабатываемых данным стримом
			height_count = (height > stream_height*(i + 1)) ? stream_height : height - stream_height*i;
			err = cudaMallocPitch(&dev_input[i], &input_pitch, (Width + 2 * sizeof(int)), height_count + 2);
			if (error(err))		return 0.0;
			//err = cudaMemset2DAsync(dev_input[i], input_pitch, 0, (Width + 2 * sizeof(int)), height_count + 2, stream[i]);
			err = cudaMallocPitch(&dev_output[i], &res_pitch, Width, height_count);
			if (error(err))		return 0.0;
			//копируем часть изображения во входной массив для устройства с отступами
			err = cudaMemcpy2DAsync(dev_input[i] + input_pitch + sizeof(int), input_pitch, input + stream_height*i*Width, Width, Width * sizeof(byte), height_count, cudaMemcpyHostToDevice, stream[i]);
			if (error(err))		return 0.0;
			//вычисляем какую строку надо скопировать в верхний отступ и копируем
			offset = i > 0 ? (stream_height*i - 1) * Width : 0;
			err = cudaMemcpy2DAsync(dev_input[i] + sizeof(int), input_pitch, input + offset, Width, Width * sizeof(byte), 1, cudaMemcpyHostToDevice, stream[i]);
			if (error(err))		return 0.0;
			//вычисляем какую строку скопировать в нижний отступ и копируем
			offset = i == countStream - 1 ? Width*(height - 1) : stream_height*(i + 1) * Width;
			err = cudaMemcpy2DAsync(dev_input[i] + input_pitch * (height_count + 1) + sizeof(int), input_pitch, input + offset, Width, Width * sizeof(byte), 1, cudaMemcpyHostToDevice, stream[i]);
			if (error(err))		return 0.0;
		}

		for (int i = 0; i < countStream; i++)
		{
			if (i%CountStreamPerDevice == 0)
				cudaSetDevice(i / CountStreamPerDevice);
			height_count = (height > stream_height*(i + 1)) ? stream_height : height - stream_height*i;
			dim3 threadsPerBlock(Xdim, Ydim);
			dim3 numBlocks(ceil(((float)width) / Xdim / 4), ceil((float)height_count / Ydim));
			cuda_color_processing << <numBlocks, threadsPerBlock, 0, stream[i] >> > (reinterpret_cast<int*>(dev_input[i]), reinterpret_cast<int*>(dev_output[i]), width, height_count, input_pitch, res_pitch);
		}

	for (int i = 0; i < countStream; i++)
	{
		if (i%CountStreamPerDevice == 0)
			cudaSetDevice(i / CountStreamPerDevice);
		height_count = (height > stream_height*(i + 1)) ? stream_height : height - stream_height*i;
		err = cudaMemcpy2DAsync(result + stream_height*i*Width, Width, dev_output[i], res_pitch, Width * sizeof(byte), height_count, cudaMemcpyDeviceToHost, stream[i]);
	}

	for (int i = 0; i < countStream; i++)
	{
		cudaFree(dev_input[i]);
		cudaFree(dev_output[i]);
	}

	err = cudaHostUnregister(input);
	if (error(err))		return 0.0;

	err = cudaHostUnregister(result);
	if (error(err))		return 0.0;
	return time;
}
