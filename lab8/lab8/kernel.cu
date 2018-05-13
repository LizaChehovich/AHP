#include "kernel.h"
#include "constant.h"

bool error(cudaError_t val);

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

	//�������� ����� � ����������� ������
	//������ ���� ��������� 3 ���� �� 1 int - ����������
	for (int i = 0; i < channels; i++) {
		buf[threadIdx.y][threadIdx.x + blockDim.x*i] = input[absY*pitch_in_int + absX + blockDim.x*i];

		if (threadIdx.y < 2)
			buf[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x*i] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x*i];
	}

	//2 ������
	if (threadIdx.x < 2)
		buf[threadIdx.y][count_row - 2 + threadIdx.x] = input[absY*pitch_in_int + absX + blockDim.x*channels];

	//������� 2*2
	if (threadIdx.y < 2 && threadIdx.x < 2)
		buf[threadIdx.y + blockDim.y][count_row - 2 + threadIdx.x] = input[(absY + blockDim.y)*pitch_in_int + absX + blockDim.x*channels];
	
	__syncthreads();

	float sum;
	int bytePosY = threadIdx.y + 1;

	//3 ���� ������������ �� 4 �����
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

//���� ��������� ������� ������ � ������ ������ �������� �����������
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

//������� ���������� ������ �� ����������� � ��������� ��� �������� �����������
__global__ void cuda_color_processing(const int* input, int* result, const int width, const int heigth, const int pitch, const int res_pitch)
{
	if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1)
		color_frame_kernel(input, result, width, heigth, pitch, res_pitch);
	else
		color_center_kernel(input, result, width, heigth, pitch, res_pitch);
}

double gpu_stream_convert_image(byte* input, byte* result, unsigned int width, unsigned int height, unsigned int channels, const int rank)
{
	cudaMemcpyToSymbol(d_filter, filter, 9 * sizeof(float), 0, cudaMemcpyHostToDevice);

	float time = 0;

	int countDevice;
	cudaGetDeviceCount(&countDevice);
	const int countStream = CountStreamPerDevice * countDevice;

	//������ ����� �����������, �������������� ����� �������
	int stream_height = ceil(((float)height) / Ydim / countStream) * Ydim;
	int Width = width * channels + 2 * sizeof(int);

	size_t input_pitch;
	size_t res_pitch;

	cudaError_t err = cudaSuccess;

	cudaStream_t* stream = new cudaStream_t[countStream];

	byte** dev_input = new byte*[countStream];
	byte** dev_output = new byte*[countStream];

	//������������ ������� � �������� ������� ��� ������-������
	err = cudaHostRegister(input, Width*height, cudaHostRegisterPortable);
	err = cudaHostRegister(result, width*height*channels, cudaHostRegisterPortable);

	int height_count;

	for (int i = 0; i < countStream; i++)
	{
		err = cudaStreamCreate(&stream[i]);
	}
	
	for (int i = 0; i < countStream; i++)
	{
		if (i%CountStreamPerDevice == 0)
			cudaSetDevice(i / CountStreamPerDevice);
		//���������� ����� �����������, �������������� ������ �������
		height_count = (height > stream_height*(i + 1)) ? stream_height : height - stream_height*i;
		err = cudaMallocPitch(&dev_input[i], &input_pitch, Width, height_count + 2);
		if (error(err))		return 0.0;
		err = cudaMallocPitch(&dev_output[i], &res_pitch, width*channels, height_count);
		if (error(err))		return 0.0;
		//�������� ����� ����������� �� ������� ������ ��� ����������
		err = cudaMemcpy2DAsync(dev_input[i] + input_pitch, input_pitch, input + stream_height*i*Width, Width, Width * sizeof(byte), height_count, cudaMemcpyHostToDevice, stream[i]);
		if (error(err))		return 0.0;
		//��������� ����� ������ ���� ����������� � ������� ������ � ��������
		if (i > 0)
		{
			err = cudaMemcpy2DAsync(dev_input[i], input_pitch, input + (stream_height*i - 1) * Width, Width, Width * sizeof(byte), 1, cudaMemcpyHostToDevice, stream[i]);
			if (error(err))		return 0.0;
		}
		else
		{
			err = cudaMemset2DAsync(dev_input[i], input_pitch, 0, Width, 1, stream[i]);
			if (error(err))		return 0.0;
		}
		//��������� ����� ������ ����������� � ������ ������ � ��������
		if (i != countStream - 1)
		{
			err = cudaMemcpy2DAsync(dev_input[i] + input_pitch * (height_count + 1), input_pitch, input + stream_height*(i + 1) * Width, Width, Width * sizeof(byte), 1, cudaMemcpyHostToDevice, stream[i]);
			if (error(err))		return 0.0;
		}
		else
		{
			err = cudaMemset2DAsync(dev_input[i] + input_pitch * (height_count + 1), input_pitch, 0, Width, 1, stream[i]);
			if (error(err))		return 0.0;
		}
		//cout <<"input: " << rank << " " << i << " " << width << " " << stream_height << " " << stream_height*i* Width << endl;
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
		err = cudaMemcpy2DAsync(result + stream_height*i*width*channels, width*channels, dev_output[i], res_pitch, width * channels * sizeof(byte), height_count, cudaMemcpyDeviceToHost, stream[i]);

		//cout << "output: " << rank << " " << i << " " << stream_height*i*width*channels << endl;
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
	return 0.0;
}

bool error(cudaError_t val)
{
	if (!val)
		return false;
	cout << "Cuda Error " << cudaGetLastError() << endl;
	cudaDeviceReset();
	return true;
}