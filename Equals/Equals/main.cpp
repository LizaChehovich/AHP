#include <stdint.h>
#include <iostream>
#include "image_helper.h"

using namespace std;

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

int main()
{
	char cpu[] = "cpu.ppm";
	char gpu[] = "output.ppm";

	uint8_t* cpu_image = nullptr, *gpu_image = nullptr;
	unsigned int width, height, channels;

	if (!load_ppm(cpu, &cpu_image, &width, &height, &channels))
	{
		cout << "Error in loading file" << endl;
		return 1;
	}

	if(!load_ppm(gpu, &gpu_image, &width, &height, &channels))
	{
		cout << "Error in loading file" << endl;
		return 1;
	}

	int result = equals(cpu_image, gpu_image, width, height, channels);

	cout << ((result == -1) ? "Image is equals" : "Error in byte ");
	if (result != -1)
		cout << result << endl;
	else
		cout << endl;

	return 0;
}