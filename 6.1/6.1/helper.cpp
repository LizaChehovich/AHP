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
		{
			cout << (int)m1[i] << ", " << (int)m2[i] << endl;
			return i;
		}
	}
	return -1;
}

void choise_filter()
{
	float val = 0.11;
	const float filters[6][3][3] = {
		{ { 0.0625,0.125,0.0625 },{ 0.125,0.25,0.125 },{ 0.0625,0.125,0.0625 } },
		{ { val,val,val },{ val,val,val },{ val,val,val } },
		{ { 0,-1,0 },{ -1,5,-1 },{ 0,-1,0 } },
		{ { 0,1,0 },{ 1,-4,1 },{ 0,1,0 } },
		{ { -2,-1,0 },{ -1,1,1 },{ 0,1,2 } },
		{ { -1,-1,-1 },{ -1,9,-1 },{ -1,-1,-1 } }
	};
	cout << "Select filter:\n 1 - blur (Gausian),\n 2 - blur,\n 3 - contrast,\n 4 - edge selection,\n 5 - embossments,\n 6 - sharpening" << endl;
	int choise =  cin_int(1, 6);
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

	const char* folders[] = { "Cat\\","Field\\","Natur\\","Flover\\" };
	const char* files[] = { "input.", "cpu.", "gpu." };
	const char* formats[] = { "pgm","ppm" };

	cout << "Choise image:" << endl << " 1 - Cat (9216*6144)" << endl << " 2 - Field (5000*3189)" << endl << " 3 - Natur (10000*4558)" << endl << " 4 - Flover (500*500)" << endl;
	int image = cin_int(1, 4);
	image--;
	cout << "Choise image format: 1-pgm, 2-ppm" << endl;
	int format = cin_int(1, 2);
	format--;

	*input = get_filename(folders[image], files[0], formats[format]);
	*cpu = get_filename(folders[image], files[1], formats[format]);
	*gpu = get_filename(folders[image], files[2], formats[format]);
}

char* get_filename(const char* folder, const char* file, const char* format)
{
	string str = "Image\\" + string(folder) + string(file) + string(format);
	char* filename = (char*)malloc(str.length() + 1);
	for (int i = 0; i < str.length() + 1; i++)
	{
		filename[i] = str[i];
	}
	filename[str.length()] = '\0';
	return filename;
}

int cin_int(int min, int max)
{
	int val;
	while (true)
	{
		cin >> val;
		if (cin.fail())
		{
			if (cin.bad())
			{
				cout << "Error in cin " << endl;
				break;
			}
			cin.clear();
			while (cin.get() != '\n');
		}
		if (min == max)
			break;
		if (val<min || val>max)
			cout << "Error. Min = " << min << " max = " << max << endl;
		else
			break;
	}
	return val;
}

float cin_float()
{
	float val;
	while (true)
	{
		cin >> val;
		if (cin.fail())
		{
			if (cin.bad())
			{
				cout << "Error in cin " << endl;
				break;
			}
			cin.clear();
			while (cin.get() != '\n');
		}
		else
			break;
	}
	return val;
}
