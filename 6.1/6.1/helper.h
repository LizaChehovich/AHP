#pragma once
#include <stdlib.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "constant.h"

int menu();
int image_processing_menu(const char* input_file, const char* cpu_result_file, const char* gpu_result_file, 
						  uint8_t* input_image, const int width, const int height, const int channels);

uint8_t* memory_alloc(int size, int channels);

int equals(uint8_t* m1, uint8_t* m2, int width, int heigth, int channels);

bool error(cudaError_t val);

void change_filter();

void show_filter();

void choise_filter();

void fill_filter(float new_filter[3][3]);

void choise_image_and_format(char** input, char** cpu, char** gpu);

char* get_path(char* folder, char* file);

char* get_filename(char* file, char* format);
