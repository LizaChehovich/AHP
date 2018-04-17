#include "helper.h"
#include <string.h>

using namespace std;

uint8_t * memory_alloc(int size, int channels)
{
	return (uint8_t*)malloc(size * sizeof(uint8_t)*channels);
}

int equals(uint8_t * m1, uint8_t * m2, int width, int heigth, int channels)
{
	for (int i = 0; i < width*heigth*channels; i++)
	{
		if (m1[i] != m2[i])	
			return i;
	}
	return -1;
}

void choise_filter()
{
	float val = 0.12;
	float filters[6][3][3] = {
		{ { 0.0625,0.125,0.0625 },{ 0.125,0.25,0.125 },{ 0.0625,0.125,0.0625 } },
		{ { val,val,val },{ val,val,val },{ val,val,val } },
		{ { 0,-1,0 },{ -1,5,-1 },{ 0,-1,0 } },
		{ { 0,1,0 },{ 1,-4,1 },{ 0,1,0 } },
		{ { -2,-1,0 },{ -1,1,1 },{ 0,1,2 } },
		{ { -1,-1,-1 },{ -1,9,-1 },{ -1,-1,-1 } }
	};
	cout << "Select filter:\n 1 - blur (Gausian),\n 2 - blur,\n 3 - contrast,\n 4 - edge selection,\n 5 - embossments,\n others - sharpening" << endl;
	int choise = 0;
	cin >> choise;
	if (choise <= 0 || choise > 5)
		choise = 6;
	fill_filter(filters[--choise]);
}

void choise_image_and_format(char** input, char** cpu, char** gpu)
{
	if (*input != nullptr)
		free(*input);
	if (*cpu != nullptr)
		free(*cpu);
	if (*gpu != nullptr)
		free(*gpu);

	cout << "Choise image:" << endl << " 1 - Cat (9216*6144)" << endl << " 2 - Field (5000*3189)" << endl << " 3 - Natur (10000*4558)" << endl << " others - Flover (500*500)" << endl;
	int image = 0;
	cin >> image;
	int format = 0;
	cout << "Choise image format: 1-pgm, others-ppm" << endl;
	cin >> format;

	switch (image)
	{
	case 1:
		*input = get_filename("Image\\Cat\\input.", (format == 1 ? "pgm" : "ppm"));
		*cpu = get_filename("Image\\Cat\\cpu.", (format == 1 ? "pgm" : "ppm"));
		*gpu = get_filename("Image\\Cat\\gpu.", (format == 1 ? "pgm" : "ppm"));
		break;
	case 2:
		*input = get_filename("Image\\Field\\input.", (format == 1 ? "pgm" : "ppm"));
		*cpu = get_filename("Image\\Field\\cpu.", (format == 1 ? "pgm" : "ppm"));
		*gpu = get_filename("Image\\Field\\gpu.", (format == 1 ? "pgm" : "ppm"));
		break;
	case 3:
		*input = get_filename("Image\\Natur\\input.", (format == 1 ? "pgm" : "ppm"));
		*cpu = get_filename("Image\\Natur\\cpu.", (format == 1 ? "pgm" : "ppm"));
		*gpu = get_filename("Image\\Natur\\gpu.", (format == 1 ? "pgm" : "ppm"));
		break;
	default:
		*input = get_filename("Image\\Flover\\input.", (format == 1 ? "pgm" : "ppm"));
		*cpu = get_filename("Image\\Flover\\cpu.", (format == 1 ? "pgm" : "ppm"));
		*gpu = get_filename("Image\\Flover\\gpu.", (format == 1 ? "pgm" : "ppm"));
		break;
	}
}

char* get_path(char* folder, char* file)
{
	char first_folder[] = "Image\\";
	int length_folder = strlen(folder);
	int file_length = strlen(file);
	int first_folder_length = strlen(first_folder);
	return first_folder;
}

char* get_filename(char* file, char* format)
{
	int length = strlen(file) + 4;
	char* filename = new char[length];
	for (int i = 0; i < length - 4; i++)
	{
		filename[i] = file[i];
	}
	for (int i = length - 4, j = 0; i < length - 1; i++, j++)
	{
		filename[i] = format[j];
	}
	filename[length - 1] = '\0';
	return filename;
}