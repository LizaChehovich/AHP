#pragma once
#include <malloc.h>
#include <iostream>

using namespace std;

typedef uint8_t byte;

double gpu_stream_convert_image(byte* input, byte* result, unsigned int width, unsigned int height, unsigned int channels, int rank);
