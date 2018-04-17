//#include "host_cuda_functions.h"
//
//using namespace std;
//
//double gpu_convert_image(uint8_t* input, uint8_t* result, unsigned int width, unsigned int height, unsigned int channels)
//{
//	cudaMemcpyToSymbol(d_filter, filter, 9 * sizeof(float), 0, cudaMemcpyHostToDevice);
//
//	uint8_t* dev_input;
//	uint8_t* dev_output;
//	float time = 0;
//
//	size_t input_pitch;
//	size_t res_pitch;
//
//	cudaError_t err = cudaSuccess;
//
//	cudaEvent_t start, end;
//	cudaEventCreate(&start);
//	cudaEventCreate(&end);
//
//
//	dim3 threadsPerBlock(Xdim, Ydim);
//	dim3 numBlocks(ceil(((float)width + Xdim - 1) / Xdim / 4), ceil((height + Ydim - 1) / Ydim));
//
//	cudaEventRecord(start);
//
//	err = cudaMallocPitch(&dev_input, &input_pitch, (width * channels + 2 * sizeof(int)), height + 2);
//	if (error(err))		return 0.0;
//	err = cudaMemset2D(dev_input, input_pitch, 0, (width * channels + 2 * sizeof(int)), height + 2);
//	if (error(err))		return 0.0;
//	err = cudaMemcpy2D(dev_input + input_pitch + sizeof(int), input_pitch, input, width * channels, width * channels * sizeof(uint8_t), height, cudaMemcpyHostToDevice);
//	if (error(err))		return 0.0;
//	err = cudaMallocPitch(&dev_output, &res_pitch, width * channels, height);
//	if (error(err))		return 0.0;
//
//	if (channels == 3)
//		cuda_color_processing << <numBlocks, threadsPerBlock >> > (reinterpret_cast<int*>(dev_input), reinterpret_cast<int*>(dev_output), width, height, input_pitch, res_pitch);
//	else
//		cuda_gray_processing << <numBlocks, threadsPerBlock >> > (reinterpret_cast<int*>(dev_input), reinterpret_cast<int*>(dev_output), width, height, input_pitch, res_pitch);
//
//
//	err = cudaMemcpy2D(result, width * channels, dev_output, res_pitch, width * channels * sizeof(uint8_t), height, cudaMemcpyDeviceToHost);
//	if (error(err))		return 0.0;
//
//	cudaEventRecord(end);
//	cudaEventSynchronize(end);
//	cudaEventElapsedTime(&time, start, end);
//
//	cudaFree(dev_input);
//	cudaFree(dev_output);
//	cudaEventDestroy(start);
//	cudaEventDestroy(end);
//	return time;
//}
//
//double gpu_stream_convert_image(uint8_t* input, uint8_t* result, unsigned int width, unsigned int height, unsigned int channels)
//{
//	cudaMemcpyToSymbol(d_filter, filter, 9 * sizeof(float), 0, cudaMemcpyHostToDevice);
//
//	float time = 0;
//	int stream_height = ceil(((float)height) / Ydim / CountStream)*Ydim;
//	int Width = width*channels;
//
//	size_t input_pitch;
//	size_t res_pitch;
//
//	cudaEvent_t start, end;
//	cudaEventCreate(&start);
//	cudaEventCreate(&end);
//
//	cudaError_t err = cudaSuccess;
//
//	cudaStream_t stream[CountStream];
//
//	uint8_t* dev_input[CountStream];
//	uint8_t* dev_output[CountStream];
//
//	err = cudaHostRegister(input, Width*height, cudaHostRegisterPortable);
//	if (error(err))		return 0.0;
//
//	err = cudaHostRegister(result, Width*height, cudaHostRegisterPortable);
//	if (error(err))		return 0.0;
//
//	int height_count;
//
//	for (int i = 0; i< CountStream; i++)
//		err = cudaStreamCreate(&stream[i]);
//
//	cudaEventRecord(start);
//
//	for (int i = 0; i < CountStream; i++)
//	{
//		height_count = (height > stream_height*(i + 1)) ? stream_height : height - stream_height*i;
//		err = cudaMallocPitch(&dev_input[i], &input_pitch, (Width + 2 * sizeof(int)), height_count + 2);
//		err = cudaMemset2DAsync(dev_input[i], input_pitch, 0, (Width + 2 * sizeof(int)), height_count + 2, stream[i]);
//		err = cudaMallocPitch(&dev_output[i], &res_pitch, Width, height_count);
//		err = cudaMemcpy2DAsync(dev_input[i] + input_pitch + sizeof(int), input_pitch, input + stream_height*i*Width, Width, Width * sizeof(uint8_t), height_count, cudaMemcpyHostToDevice, stream[i]);
//		if (i > 0)
//		{
//			err = cudaMemcpy2DAsync(dev_input[i] + sizeof(int), input_pitch, input + stream_height*i * Width - Width, Width, Width * sizeof(uint8_t), 1, cudaMemcpyHostToDevice, stream[i]);
//		}
//		if (i < CountStream - 1)
//		{
//			err = cudaMemcpy2DAsync(dev_input[i] + input_pitch + sizeof(int) + height_count*input_pitch, input_pitch, input + stream_height*(i + 1) *Width, Width, Width * sizeof(uint8_t), 1, cudaMemcpyHostToDevice, stream[i]);
//		}
//	}
//
//	int frame;
//
//	for (int i = 0; i < CountStream; i++)
//	{
//		height_count = (height > stream_height*(i + 1)) ? stream_height : height - stream_height*i;
//
//		dim3 threadsPerBlock(Xdim, Ydim);
//		dim3 numBlocks(ceil(((float)width + Xdim - 1) / Xdim / 4), ceil((float)height_count / Ydim));
//
//		frame = i == CountStream - 1 ? 1 : 0;
//
//		if (channels == 3)
//			cuda_color_processing << <numBlocks, threadsPerBlock, 0, stream[i] >> > (reinterpret_cast<int*>(dev_input[i]), reinterpret_cast<int*>(dev_output[i]), width, height_count, input_pitch, res_pitch);
//		else
//			cuda_gray_processing << <numBlocks, threadsPerBlock, 0, stream[i] >> > (reinterpret_cast<int*>(dev_input[i]), reinterpret_cast<int*>(dev_output[i]), width, height_count, input_pitch, res_pitch, frame);
//	}
//
//	for (int i = 0; i < CountStream; i++)
//	{
//		height_count = (height > stream_height*(i + 1)) ? stream_height : height - stream_height*i;
//
//		//err = cudaStreamSynchronize(stream[i]);
//
//		err = cudaMemcpy2DAsync(result + stream_height*i*Width, Width, dev_output[i], res_pitch, Width * sizeof(uint8_t), height_count, cudaMemcpyDeviceToHost, stream[i]);
//	}
//
//	cudaEventRecord(end);
//	cudaEventSynchronize(end);
//	cudaEventElapsedTime(&time, start, end);
//
//	for (int i = 0; i < CountStream; i++)
//	{
//		cudaFree(dev_input[i]);
//		cudaFree(dev_output[i]);
//	}
//
//	err = cudaHostUnregister(input);
//	if (error(err))		return 0.0;
//
//	err = cudaHostUnregister(result);
//	if (error(err))		return 0.0;
//
//	cudaEventDestroy(start);
//	cudaEventDestroy(end);
//	return time;
//}
//
//double cpu_and_gpu_convert_image(uint8_t * input, uint8_t * result, unsigned int width, unsigned int height, unsigned int channels)
//{
//	if (channels == 1)
//		return gpu_convert_image(input, result, width, height, channels);
//
//	cudaMemcpyToSymbol(d_filter, filter, 9 * sizeof(float), 0, cudaMemcpyHostToDevice);
//
//	float cuda_time = 0.0;
//	int data_size = width*height;
//
//	cudaError_t err = cudaSuccess;
//	size_t input_pitch;
//	size_t res_pitch;
//
//	uint8_t* color[3];
//	uint8_t* r_color[3];
//
//	uint8_t* d_color[3];
//	uint8_t* d_r_color[3];
//
//	cudaStream_t stream[3];
//
//	for (int i = 0; i < channels; i++) {
//		cudaStreamCreate(&stream[i]);
//		cudaMallocHost((void**)&color[i], data_size);
//		cudaMallocHost((void**)&r_color[i], data_size);
//		cudaMallocPitch(&d_color[i], &input_pitch, width + 2 * sizeof(int), height + 2);
//		cudaMemset2DAsync(d_color[i], input_pitch, 0, width + 2 * sizeof(int), height + 2);
//		cudaMallocPitch(&d_r_color[i], &res_pitch, width, height);
//	}
//
//	dim3 threadsPerBlock(Xdim, Ydim);
//	dim3 numBlocks(ceil(((float)width + Xdim - 1) / Xdim / 4), ceil((height + Ydim - 1) / Ydim));
//
//	cudaEvent_t begin, end;
//	cudaEventCreate(&begin);
//	cudaEventCreate(&end);
//
//	LARGE_INTEGER start, finish, freq;
//	QueryPerformanceFrequency(&freq);
//	QueryPerformanceCounter(&start);
//
//	for (int x = 0; x < width; x++)
//		for (int y = 0; y < height; y++)
//		{
//			color[0][y*width + x] = input[(y*width + x)*channels];
//			color[1][y*width + x] = input[(y*width + x)*channels + 1];
//			color[2][y*width + x] = input[(y*width + x)*channels + 2];
//		}
//
//	cudaEventRecord(begin);
//
//	for (int i = 0; i < channels; i++)
//	{
//		cudaMemcpy2DAsync(d_color[i] + input_pitch + sizeof(int), input_pitch, color[i], width, width * sizeof(uint8_t), height, cudaMemcpyHostToDevice, stream[i]);
//	}
//
//	for (int i = 0; i < channels; i++)
//	{
//		cuda_gray_processing << <numBlocks, threadsPerBlock, 0, stream[i] >> >(reinterpret_cast<int*>(d_color[i]), reinterpret_cast<int*>(d_r_color[i]), width, height, input_pitch, res_pitch);
//	}
//
//	for (int i = 0; i <channels; i++)
//	{
//		cudaMemcpy2DAsync(r_color[i], width, d_r_color[i], res_pitch, width * sizeof(uint8_t), height, cudaMemcpyDeviceToHost, stream[i]);
//	}
//
//	cudaEventRecord(end);
//	cudaEventSynchronize(end);
//	cudaEventElapsedTime(&cuda_time, begin, end);
//
//	for (int x = 0; x < width; x++)
//		for (int y = 0; y < height; y++)
//		{
//			result[(y*width + x)*channels] = r_color[0][y*width + x];
//			result[(y*width + x)*channels + 1] = r_color[1][y*width + x];
//			result[(y*width + x)*channels + 2] = r_color[2][y*width + x];
//		}
//
//	QueryPerformanceCounter(&finish);
//
//	cout << "Cuda kernel time " << cuda_time << endl;
//
//	for (int i = 0; i < channels; i++)
//	{
//		cudaFree(d_color[i]);
//		cudaFree(d_r_color[i]);
//		cudaFreeHost(color[i]);
//		cudaFreeHost(r_color[i]);
//		cudaStreamDestroy(stream[i]);
//	}
//
//	cudaEventDestroy(begin);
//	cudaEventDestroy(end);
//
//	return (finish.QuadPart - start.QuadPart) * 1000 / (double)freq.QuadPart;
//}