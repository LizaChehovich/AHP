#pragma once
#include <stdlib.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//обща€ структура программы
int menu();

//обработка изображени€
int image_processing_menu(const char* input_file, const char* cpu_result_file, const char* gpu_result_file, 
						  uint8_t* input_image, const int width, const int height, const int channels);

//выделение пам€ти под изображение
uint8_t* memory_alloc(int size, int channels);

//сравнение матриц. ¬озвращает -1 при равенстве или номер первого несовпадающего байта
int equals(uint8_t* m1, uint8_t* m2, int width, int heigth, int channels);

//проверка ошибок cuda
bool error(cudaError_t val);

//изменение фильтра изобажени€
void change_filter();

//вывод фильтра
void show_filter();

//выбор нового фильтра
void choise_filter();

//перезапись значений в фильтре
void fill_filter(const float new_filter[3][3]);

//выбор изображени€ и его формата
void choise_image_and_format(char** input, char** cpu, char** gpu);

//формирование пути к файлу
char* get_filename(const char* folder, const char* file, const char* format);

//ввод целого числа с обработкой ошибок и проверкой выхода за границы
int cin_int(int min = 0, int max = 1);

//ввод дробного числа с обработкой ошибок
float cin_float();
