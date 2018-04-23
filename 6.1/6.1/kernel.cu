#include "helper.h"
#include "image_helper.h"
#include "constant.h"

#include <cmath>
#include "cuda_profiler_api.h"

using namespace std;

//запуск фильтрации изображения на устройстве
double gpu_convert_image(uint8_t* input, uint8_t* result, unsigned int width, unsigned int height, unsigned int channels);

//запуск фильтрации изображения на устройстве в нескольких стримах
double gpu_stream_convert_image(uint8_t* input, uint8_t* result, unsigned int width, unsigned int height, unsigned int channels);

//фильтрация изображения на хосте
double cpu_convert_image(uint8_t* input, uint8_t* result, unsigned int width, unsigned int height, unsigned int chnnels);

//фильтрация цветного изображения при помощи разложения на матрицы отдельных цветов и обработки на устройстве
double cpu_and_gpu_convert_image(uint8_t* input, uint8_t* result, unsigned int width, unsigned int height, unsigned int chnnels);

//ядро обработки центральных блоков чёрно-белого изображения
__device__ void gray_center_kernel(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
{
	const int count_row = Xdim + 2;
	const int count_column = Ydim + 2;
	const int pitch_in_int = pitch / sizeof(int);
	const int absX = blockIdx.x * blockDim.x + threadIdx.x;
	const int absY = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int buf[count_column][count_row];

	//загрузка данных в разделяемую память
	//основная часть - транзакции
	buf[threadIdx.y][threadIdx.x] = input[absY*pitch_in_int + absX];
	//две последнии строки - транзакции
	if (threadIdx.y < 2)
		buf[threadIdx.y + blockDim.y][threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX];
	//два столбца 
	if (threadIdx.x < 2)
		buf[threadIdx.y][count_row - 2 + threadIdx.x] = input[absY*pitch_in_int + absX + blockDim.x];
	//квадрат 2*2 в правом нижнем углу
	if (threadIdx.y < 2  && threadIdx.x < 2)
		buf[threadIdx.y + blockDim.y][count_row - 2 + threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x];

	//синхронизация потоков блока
	__syncthreads();

	//заполнение граничных левых бит в блоках, обрабатывающих левую границу
	//конфликта обращения к разделяемой памяти не должно быть, т.к. каждая новая строка смещается относительно предыдущей на 2 банка
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		buf[threadIdx.y][threadIdx.x] = buf[threadIdx.y][threadIdx.x + 1] << 24;
		if (threadIdx.y < 2)
			buf[threadIdx.y + blockDim.y][threadIdx.x] = buf[threadIdx.y + blockDim.y][threadIdx.x + 1] << 24;
	}

	__syncthreads();

	int val = 0;
	float sum;
	int bytePosY = threadIdx.y + 1;
	int bytePosX = (threadIdx.x + 1) * 4;

	for (int k = 0; k < 4; k++)
	{
		sum = 0;

		//я не совсем понимаю, где здесь конфликт обращения к разделяемой памяти,
		//т.е. он, вроде как, есть (профайлер показывает эффективность обращения 33%), но я не могу понять из-за чего
		for (int i = -1; i < 2; i++)
		{
			for (int j = -1; j < 2; j++)
			{
				sum += ((byte*)buf[bytePosY + i])[bytePosX + j] *d_filter[i + 1][j + 1];
			}
		}
		sum = round(sum);
		if (sum < 0) sum = 0;
		if (sum > 255) sum = 255;

		((byte*)&val)[k] = sum;
		bytePosX++;
	}

	//записываем в память результаты работы всего варпа - транцакция
	result[absY*res_pitch / sizeof(int) + absX] = val;
}

//ядро обработки крайних правых и нижних блоков чёрно-белого изображения
__device__ void gray_frame_kernel(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
{
	const int count_row = Xdim + 2;
	const int count_column = Ydim + 2;
	const int pitch_in_int = pitch / sizeof(int);
	const int absX = blockIdx.x * blockDim.x + threadIdx.x;
	const int absY = blockIdx.y * blockDim.y + threadIdx.y;

	const int absXinByte = absX * sizeof(int);

	const int Width = width + 2 * sizeof(int);
	const int Height = heigth + 2;
	//количество int, занятых изображением
	const int image_width = blockDim.x - (gridDim.x*blockDim.x * sizeof(int) - width) / sizeof(int);

	__shared__ int buf[count_column][count_row];

	//load data in shared memory

	if (absY < Height && absXinByte < Width)
		buf[threadIdx.y][threadIdx.x] = input[absY*pitch_in_int + absX];

	if (threadIdx.y < 2 && ((absY + blockDim.y) < Height) && absXinByte < Width)
		buf[threadIdx.y + blockDim.y][threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX];

	if (absY < Height && threadIdx.x < 2 && ((absX + blockDim.x) * sizeof(int)) < Width)
		buf[threadIdx.y][count_row - 2 + threadIdx.x] = input[absY*pitch_in_int + absX + blockDim.x];

	if (threadIdx.y < 2 && (absY + blockDim.y) < Height && threadIdx.x < 2 && ((absX + blockDim.x) * sizeof(int)) < Width)
		buf[threadIdx.y + blockDim.y][count_row - 2 + threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x];

	__syncthreads();

	//дублируем последний байт изображения в соседний int
	if (blockIdx.x == gridDim.x-1 && threadIdx.x == 0) {
		buf[threadIdx.y][image_width +1] = buf[threadIdx.y][image_width]>>24;
		if (threadIdx.y < 2)
			buf[threadIdx.y + blockDim.y][image_width +1] = buf[threadIdx.y + blockDim.y][image_width]>>24;
	}

	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		buf[threadIdx.y][threadIdx.x] = buf[threadIdx.y][threadIdx.x + 1] << 24;
		if (threadIdx.y<2)
			buf[threadIdx.y + blockDim.y][threadIdx.x] = buf[threadIdx.y + blockDim.y][threadIdx.x + 1] << 24;
	}

	__syncthreads();

	int val = 0;
	float sum;
	int bytePosY = threadIdx.y + 1;
	int bytePosX = (threadIdx.x + 1) * 4;

	for (int k = 0; k < 4; k++)
	{
		if (absY >= Height && absXinByte >= Width)
			break;
		sum = 0;

		for (int i = -1; i < 2; i++)
		{
			for (int j = -1; j < 2; j++)
			{
				sum += ((byte*)buf[bytePosY + i])[bytePosX + j] *d_filter[i + 1][j + 1];
			}
		}
		sum = round(sum);
		if (sum < 0) sum = 0;
		if (sum > 255) sum = 255;

		((byte*)&val)[k] = sum;
		bytePosX++;
	}

	if (absY < heigth && absXinByte < width)
	{
		result[absY*res_pitch / sizeof(int) + absX] = val;
	}
}

//функция разделения блоков на центральные и граничные для чёрно-белого изображения
__global__ void cuda_gray_processing(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
{
	if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1)
		gray_frame_kernel(input, result, width, heigth, pitch, res_pitch);
	else
		gray_center_kernel(input, result, width, heigth, pitch, res_pitch);
}

//ядро обработки центральных блоков цветного изображения
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

	//заполняем границы левых крайних блоков
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		buf[threadIdx.y][threadIdx.x] = buf[threadIdx.y][threadIdx.x + 1] << 8;
		if (threadIdx.y < 2)
			buf[threadIdx.y + blockDim.y][threadIdx.x] = buf[threadIdx.y + blockDim.y][threadIdx.x + 1] << 8;
	}

	__syncthreads();

	float sum;
	//int val;
	int bytePosY = threadIdx.y + 1;

	//3 раза обрабатываем по 4 байта
	for (int c = 0; c < channels; c++)
	{
		//val = 0;
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
			//((byte*)&val)[k] = sum;
		}

		result[absY*res_pitch / sizeof(int) + absX + c*blockDim.x] = res[threadIdx.y][threadIdx.x];
		//result[absY*res_pitch / sizeof(int) + absX + c*blockDim.x] = val;
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

	//здесь нет смысла переносить значения крайнего правого пикселя в соседнюю ячейку, т.к. в зависимости от размера изображения 
	//будет варьироваться положения бит итого пикселя в int. Проще два раза учитывать сам пиксель с разными коэффициентами домножения

	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		buf[threadIdx.y][threadIdx.x] = buf[threadIdx.y][threadIdx.x + 1] << 8;
		if (threadIdx.y < 2)
			buf[threadIdx.y + blockDim.y][threadIdx.x] = buf[threadIdx.y + blockDim.y][threadIdx.x + 1] << 8;
	}

	__syncthreads();

	float sum;
	//int val;
	const int bytePosY = threadIdx.y + 1;
	int offset;

	for (int c = 0; c < channels; c++)
	{

		res[threadIdx.y][threadIdx.x] = 0;
		//val = 0;
		

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
					offset = j*channels;
					//Собстрвенно, если нам нужен сосед крайнего правого - берём крайний правый
					if (absXinByte + c*blockDim.x * sizeof(int) + k >= (width - 1)*channels&& offset > 0)
						offset = 0;
					sum += ((byte*)buf[bytePosY + i])[bytePosX + offset] * d_filter[i + 1][j + 1];
				}
			}
			sum = round(sum);
			if (sum < 0) sum = 0;
			if (sum > 255) sum = 255;

			((byte*)&(res[threadIdx.y][threadIdx.x]))[k] = sum;
			//((byte*)&val)[k] = sum;
		}

		if (absY < heigth && absXinByte + c * blockDimXinByte < width * channels)
		{
			//result[absY*res_pitch / sizeof(int) + absX + c*blockDim.x] = val;
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

int main()
{
	cudaProfilerStart();
	return menu();
	cudaProfilerStop();
}

double gpu_convert_image(uint8_t* input, uint8_t* result, unsigned int width, unsigned int height, unsigned int channels)
{
	//копируем фильтр в константную память
	cudaMemcpyToSymbol(d_filter, filter, 9 * sizeof(float), 0, cudaMemcpyHostToDevice);

	uint8_t* dev_input;
	uint8_t* dev_output;
	float time = 0;
	unsigned int Width = width*channels;

	size_t input_pitch;
	size_t res_pitch;

	cudaError_t err = cudaSuccess;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);


	dim3 threadsPerBlock(Xdim, Ydim);
	dim3 numBlocks(ceil(((float)width ) / Xdim / 4), ceil(((float)height) / Ydim));

	cudaEventRecord(start);

	err = cudaMallocPitch(&dev_input, &input_pitch, (Width + 2 * sizeof(int)), height + 2);
	if (error(err))		return 0.0;
	//err = cudaMemset2D(dev_input, input_pitch, 0, (Width + 2 * sizeof(int)), height + 2);
	//if (error(err))		return 0.0;

	//копируем матрицу с отступами
	err = cudaMemcpy2D(dev_input + input_pitch + sizeof(int), input_pitch, input, Width, Width * sizeof(uint8_t), height, cudaMemcpyHostToDevice);
	if (error(err))		return 0.0;
	//копируем в верхнюю строку расширения первую строку изображения
	err = cudaMemcpy2D(dev_input + sizeof(int), input_pitch, input, Width, Width * sizeof(uint8_t), 1, cudaMemcpyHostToDevice);
	if (error(err))		return 0.0;
	//копируем в нижнюю строку расширения последнюю строку изображения
	err = cudaMemcpy2D(dev_input + input_pitch*(height+1) + sizeof(int), input_pitch, input + Width*(height-1), Width, Width * sizeof(uint8_t), 1, cudaMemcpyHostToDevice);
	if (error(err))		return 0.0;

	err = cudaMallocPitch(&dev_output, &res_pitch, Width, height);
	if (error(err))		return 0.0;

	if (channels == 3)
		cuda_color_processing << <numBlocks, threadsPerBlock >> > (reinterpret_cast<int*>(dev_input), reinterpret_cast<int*>(dev_output), width, height, input_pitch, res_pitch);
	else
		cuda_gray_processing << <numBlocks, threadsPerBlock >> > (reinterpret_cast<int*>(dev_input), reinterpret_cast<int*>(dev_output), width, height, input_pitch, res_pitch);


	err = cudaMemcpy2D(result, Width, dev_output, res_pitch, Width* sizeof(uint8_t), height, cudaMemcpyDeviceToHost);
	if (error(err))		return 0.0;

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);

	cudaFree(dev_input);
	cudaFree(dev_output);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	return time;
}

double gpu_stream_convert_image(uint8_t* input, uint8_t* result, unsigned int width, unsigned int height, unsigned int channels)
{
	cudaMemcpyToSymbol(d_filter, filter, 9 * sizeof(float), 0, cudaMemcpyHostToDevice);

	float time = 0;

	//высота части изображения, обрабатываемой одним потоком
	int stream_height = ceil(((float)height) / Ydim / CountStream)*Ydim;
	int Width = width*channels;
	
	size_t input_pitch;
	size_t res_pitch;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaError_t err = cudaSuccess;

	cudaStream_t stream[CountStream];

	uint8_t* dev_input[CountStream];
	uint8_t* dev_output[CountStream];

	//регистрируем входную и выходные матрицы как пиннед-память
	err = cudaHostRegister(input, Width*height, cudaHostRegisterPortable);
	if (error(err))		return 0.0;

	err = cudaHostRegister(result, Width*height, cudaHostRegisterPortable);
	if (error(err))		return 0.0;

	int height_count;

	for (int i = 0; i < CountStream; i++)
	{
		err = cudaStreamCreate(&stream[i]);
		if (error(err))		return 0.0;
	}

	int offset = 0;

	cudaEventRecord(start);

	for (int i = 0; i < CountStream; i++)
	{
		//количество строк изображения, обрабатываемых данным стримом
		height_count = (height > stream_height*(i + 1)) ? stream_height : height - stream_height*i;
		err = cudaMallocPitch(&dev_input[i], &input_pitch, (Width + 2 * sizeof(int)), height_count + 2);
		if (error(err))		return 0.0;
		//err = cudaMemset2DAsync(dev_input[i], input_pitch, 0, (Width + 2 * sizeof(int)), height_count + 2, stream[i]);
		err = cudaMallocPitch(&dev_output[i], &res_pitch, Width, height_count);
		if (error(err))		return 0.0;
		//копируем часть изображения во входной массив для устройства с отступами
		err = cudaMemcpy2DAsync(dev_input[i] + input_pitch + sizeof(int), input_pitch, input + stream_height*i*Width, Width, Width * sizeof(uint8_t), height_count, cudaMemcpyHostToDevice, stream[i]);
		if (error(err))		return 0.0;
		//вычисляем какую строку надо скопировать в верхний отступ и копируем
		offset = i > 0 ? (stream_height*i - 1) * Width : 0;
		err = cudaMemcpy2DAsync(dev_input[i] + sizeof(int), input_pitch, input + offset, Width, Width * sizeof(uint8_t), 1, cudaMemcpyHostToDevice, stream[i]);
		if (error(err))		return 0.0;
		//вычисляем какую строку скопировать в нижний отступ и копируем
		offset = i == CountStream - 1 ? Width*(height - 1) : stream_height*(i + 1) * Width;
		err = cudaMemcpy2DAsync(dev_input[i] + input_pitch * (height_count + 1) + sizeof(int), input_pitch, input + offset, Width, Width * sizeof(uint8_t), 1, cudaMemcpyHostToDevice, stream[i]);
		if (error(err))		return 0.0;
	}

	for (int i = 0; i < CountStream; i++)
	{
		height_count = (height > stream_height*(i + 1)) ? stream_height : height - stream_height*i;

		dim3 threadsPerBlock(Xdim, Ydim);
		dim3 numBlocks(ceil(((float)width) / Xdim / 4), ceil((float)height_count / Ydim));

		if (channels == 3)
			cuda_color_processing << <numBlocks, threadsPerBlock,0,stream[i] >> > (reinterpret_cast<int*>(dev_input[i]), reinterpret_cast<int*>(dev_output[i]), width, height_count, input_pitch, res_pitch);
		else
			cuda_gray_processing << <numBlocks, threadsPerBlock,0,stream[i] >> > (reinterpret_cast<int*>(dev_input[i]), reinterpret_cast<int*>(dev_output[i]), width, height_count, input_pitch, res_pitch);
	}

	for (int i = 0; i < CountStream; i++)
	{
		height_count = (height > stream_height*(i + 1)) ? stream_height : height - stream_height*i;

		//err = cudaStreamSynchronize(stream[i]);

		err = cudaMemcpy2DAsync(result + stream_height*i*Width, Width, dev_output[i], res_pitch, Width * sizeof(uint8_t), height_count, cudaMemcpyDeviceToHost, stream[i]);
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);

	for (int i = 0; i < CountStream; i++)
	{
		cudaFree(dev_input[i]);
		cudaFree(dev_output[i]);
	}
	
	err = cudaHostUnregister(input);
	if (error(err))		return 0.0;

	err = cudaHostUnregister(result);
	if (error(err))		return 0.0;

	cudaEventDestroy(start);
	cudaEventDestroy(end);
	return time;
}

double cpu_convert_image(uint8_t * input, uint8_t * result, unsigned int width, unsigned int height, unsigned int channels)
{
	float val;
	int xCoord, yCoord;
	LARGE_INTEGER start, finish, freq;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	for (int x = 0; x < width * channels; x++)
	{
		for (int y = 0; y < height; y++)
		{
			val = 0;
			for (int i = -1; i < 2; i++)
			{
				yCoord = y + i;
				yCoord += yCoord < 0 ? 1 : yCoord >= height ? -1 : 0;
				for (int j = -1; j < 2; j++)
				{
					xCoord = x + j*channels;
					xCoord += xCoord < 0 ? channels : xCoord >= width*channels ? channels*(-1) : 0;

					val += input[yCoord*width * channels + xCoord] * filter[i + 1][j + 1];
				}
			}
			val = round(val);
			result[y * width * channels + x] = val<MIN ? MIN : val>MAX ? MAX : val;
		}
	}
	QueryPerformanceCounter(&finish);
	return (finish.QuadPart - start.QuadPart) * 1000 / (double)freq.QuadPart;
}

double cpu_and_gpu_convert_image(uint8_t * input, uint8_t * result, unsigned int width, unsigned int height, unsigned int channels)
{
	if (channels == 1)
		return gpu_convert_image(input, result, width, height, channels);

	cudaMemcpyToSymbol(d_filter, filter, 9 * sizeof(float), 0, cudaMemcpyHostToDevice);

	float cuda_time = 0.0;
	int data_size = width*height;

	cudaError_t err = cudaSuccess;

	size_t input_pitch;
	size_t res_pitch;

	uint8_t* color[3];
	uint8_t* r_color[3];

	uint8_t* d_color[3];
	uint8_t* d_r_color[3];

	cudaStream_t stream[3];

	for (int i = 0; i < channels; i++) {
		cudaStreamCreate(&stream[i]);
		if (error(err))		return 0.0;
		cudaMallocHost((void**)&color[i], data_size);
		if (error(err))		return 0.0;
		cudaMallocHost((void**)&r_color[i], data_size);
		if (error(err))		return 0.0;
		cudaMallocPitch(&d_color[i], &input_pitch, width + 2 * sizeof(int), height + 2);
		if (error(err))		return 0.0;
		cudaMallocPitch(&d_r_color[i], &res_pitch, width, height);
		if (error(err))		return 0.0;
	}

	dim3 threadsPerBlock(Xdim, Ydim);
	dim3 numBlocks(ceil(((float)width) / Xdim / 4), ceil(((float)height) / Ydim));
	
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	LARGE_INTEGER start, finish, freq;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);

	for (int x = 0; x < width; x++)
		for (int y = 0; y < height; y++)
		{
			color[0][y*width + x] = input[(y*width + x)*channels];
			color[1][y*width + x] = input[(y*width + x)*channels + 1];
			color[2][y*width + x] = input[(y*width + x)*channels + 2];
		}

	cudaEventRecord(begin);

	for (int i = 0; i < channels; i++)
	{
		cudaMemcpy2DAsync(d_color[i] + input_pitch + sizeof(int), input_pitch, color[i], width, width * sizeof(uint8_t), height, cudaMemcpyHostToDevice, stream[i]);
		if (error(err))		return 0.0;
		cudaMemcpy2DAsync(d_color[i] +  sizeof(int), input_pitch, color[i], width, width * sizeof(uint8_t), 1, cudaMemcpyHostToDevice, stream[i]);
		if (error(err))		return 0.0;
		cudaMemcpy2DAsync(d_color[i] + input_pitch*(height+1) + sizeof(int), input_pitch, color[i] + width* (height-1), width, width * sizeof(uint8_t), 1, cudaMemcpyHostToDevice, stream[i]);
		if (error(err))		return 0.0;
	}

	for (int i = 0; i < channels; i++)
	{
		cuda_gray_processing << <numBlocks, threadsPerBlock, 0, stream[i] >> >(reinterpret_cast<int*>(d_color[i]), reinterpret_cast<int*>(d_r_color[i]), width, height, input_pitch, res_pitch);
	}

	for(int i = 0; i <channels; i++)
	{
		cudaMemcpy2DAsync(r_color[i], width, d_r_color[i], res_pitch, width * sizeof(uint8_t), height, cudaMemcpyDeviceToHost, stream[i]);
		if (error(err))		return 0.0;
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&cuda_time, begin, end);

	for (int x = 0; x < width; x++)
		for (int y = 0; y < height; y++)
		{
			result[(y*width + x)*channels] = r_color[0][y*width + x];
			result[(y*width + x)*channels + 1] = r_color[1][y*width + x];
			result[(y*width + x)*channels + 2] = r_color[2][y*width + x];
		}

	QueryPerformanceCounter(&finish);

	cout << "Cuda kernel time " << cuda_time << " ms" << endl;

	for (int i = 0; i < channels; i++)
	{
		cudaFree(d_color[i]);
		cudaFree(d_r_color[i]);
		cudaFreeHost(color[i]);
		cudaFreeHost(r_color[i]);
		cudaStreamDestroy(stream[i]);
	}

	cudaEventDestroy(begin);
	cudaEventDestroy(end);

	return (finish.QuadPart - start.QuadPart) * 1000 / (double)freq.QuadPart;
}

int menu()
{
	unsigned int width, height, channels;
	int image_choise = 1;
	char* input_file = nullptr;
	char* cpu_result_file = nullptr;
	char* gpu_result_file = nullptr;

	while (image_choise) {

		choise_image_and_format(&input_file, &cpu_result_file, &gpu_result_file);

		uint8_t* input_image = nullptr;

		if (!load_ppm(input_file, &input_image, &width, &height, &channels))
		{
			cout << "Error in loading file" << endl;
			return 1;
		}

		if (channels != 1 && channels != 3)
		{
			cout << "Error in count channels" << endl;
			free(input_image);
			return 1;
		}

		cout << endl << "Width " << width << endl << "Height " << height << endl << "Channels " << channels << endl;
		
		if (image_processing_menu(input_file, cpu_result_file, gpu_result_file, input_image, width, height, channels))
			return 1;

		free(input_image);

		cout << "Convert new image? 1 - yes, 0 - no" << endl;
		image_choise = cin_int();
	}
	return 0;
}

int image_processing_menu(const char* input_file, const char* cpu_result_file, const char* gpu_result_file, 
						  uint8_t* input_image, const int width, const int height, const int channels)
{
	int choise = 1;

	while (choise)
	{
		change_filter();

		uint8_t* cpu_image = memory_alloc(width*height, channels);
		if (!cpu_image)
		{
			cout << "Error in memory allocation" << endl;
			free(input_image);
			return 1;
		}

		uint8_t* gpu_image = memory_alloc(width*height, channels);
		if (!gpu_image)
		{
			cout << "Error in memory allocation" << endl;
			free(input_image);
			free(cpu_image);
			return 1;
		}

		cout << endl << "CPU processing" << endl;
		cout << "CPU time " << cpu_convert_image(input_image, cpu_image, width, height, channels) << " ms" << endl;

		cout << endl << "Use cuda stream? 1-yes, 0-no" << endl;
		choise = cin_int();

		if (choise) {
			choise = 0;
			if (channels == 3)
			{
				cout << "Use cuda stream and cpu? 1-yes, 0-no" << endl;
				choise = cin_int();
			}
			cout << endl << "GPU processing" << endl;
			cout << "Time " << (choise == 1 ?
				cpu_and_gpu_convert_image(input_image, gpu_image, width, height, channels) :
				gpu_stream_convert_image(input_image, gpu_image, width, height, channels)) << " ms" << endl;
		}
		else {
			cout << endl << "GPU processing" << endl;
			cout << "GPU time " << gpu_convert_image(input_image, gpu_image, width, height, channels) << " ms" << endl;
		}

		int result = equals(cpu_image, gpu_image, width, height, channels);

		cout << ((result == -1) ? "Image is equals" : "Error in byte ");
		if (result != -1)
			cout << result << endl;
		else
			cout << endl;

		cout << endl << "Saving of results" << endl;

		if (!save_ppm(cpu_result_file, cpu_image, width, height, channels))
		{
			cout << "Error in save file" << endl;
			free(input_image);
			free(cpu_image);
			free(gpu_image);
			return 1;
		}

		if (!save_ppm(gpu_result_file, gpu_image, width, height, channels))
		{
			cout << "Error in save file" << endl;
			free(input_image);
			free(cpu_image);
			free(gpu_image);
			return 1;
		}
		free(cpu_image);
		free(gpu_image);

		cout << "Convert image again? 1 - yes, 0 - no" << endl;
		choise = cin_int();
	}
	return 0;
}

void change_filter()
{
	show_filter();
	int choise = 0;
	cout << "Change filter? 1-yes, 0-no" << endl;
	choise = cin_int();
	if (!choise)
		return;
	cout << "Select: 1 - enter yourself, 0 - select from specified" << endl;
	choise = cin_int();
	if (!choise)
		choise_filter();
	else
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
			{
				cout << "Enter the number" << endl;
				filter[i][j] = cin_float();
			}

	show_filter();
}

void show_filter()
{
	cout << endl << "Filter:" << endl << endl;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cout << filter[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void fill_filter(const float new_filter[3][3])
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			filter[i][j] = new_filter[i][j];
	}
}

bool error(cudaError_t val)
{
	if (!val)
		return false;
	cout << "Cuda Error " << cudaGetLastError() << endl;
	cudaDeviceReset();
	return true;
}