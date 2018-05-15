#include "image_helper.h"

#include <fstream>
#include <iostream>
#include <assert.h>

using namespace std;

bool load_ppm(const char * file, unsigned char ** data, unsigned int * w, unsigned int * h, unsigned int * channels)
{
	FILE *fp = NULL;

	if (fopen_s(&fp, file, "rb")!=0)
	{
		cerr << "__LoadPPM() : Failed to open file: " << file << endl;
		return false;
	}

	char header[HeaderSize];

	if (fgets(header, HeaderSize, fp) == NULL)
	{
		cerr << "__LoadPPM() : reading PGM header returned NULL" << endl;
		return false;
	}

	if (strncmp(header, "P5", 2) == 0)
	{
		*channels = 1;
	}
	else if (strncmp(header, "P6", 2) == 0)
	{
		*channels = 3;
	}
	else
	{
		cerr << "__LoadPPM() : File is not a PPM or PGM image" << endl;
		*channels = 0;
		return false;
	}

	// parse header, read maxval, width and height
	unsigned int width = 0;
	unsigned int height = 0;
	unsigned int maxval = 0;
	unsigned int i = 0;

	while (i < 3)
	{
		if (fgets(header, HeaderSize, fp) == NULL)
		{
			cerr << "__LoadPPM() : reading PGM header returned NULL" << endl;
			return false;
		}

		if (header[0] == '#')
		{
			continue;
		}

		if (i == 0)
		{
			i += sscanf_s(header, "%u %u %u", &width, &height, &maxval);
		}
		else if (i == 1)
		{
			i += sscanf_s(header, "%u %u", &height, &maxval);
		}
		else if (i == 2)
		{
			i += sscanf_s(header, "%u", &maxval);
		}
	}

	// check if given handle for the data is initialized
	if (NULL != *data)
	{
		if (*w != width || *h != height)
		{
			cerr << "__LoadPPM() : Invalid image dimensions." << endl;
		}
	}
	else
	{
		*data = (unsigned char *)malloc(sizeof(unsigned char) * width * height **channels);
		*w = width;
		*h = height;
	}

	// read and close file
	if (fread(*data, sizeof(unsigned char), width * height **channels, fp) == 0)
	{
		cerr << "__LoadPPM() read data returned error." << endl;
	}

	fclose(fp);

	return true;
}

bool save_ppm(const char * file, unsigned char * data, unsigned int w, unsigned int h, unsigned int channels)
{
	assert(NULL != data);
	assert(w > 0);
	assert(h > 0);

	fstream fh(file, fstream::out | fstream::binary);

	if (fh.bad())
	{
		cerr << "__savePPM() : Opening file failed." << endl;
		return false;
	}

	if (channels == 1)
	{
		fh << "P5\n";
	}
	else if (channels == 3)
	{
		fh << "P6\n";
	}
	else
	{
		cerr << "__savePPM() : Invalid number of channels." << endl;
		return false;
	}

	fh << w << "\n" << h << "\n" << 0xff << endl;

	for (unsigned int i = 0; (i < (w*h*channels)) && fh.good(); ++i)
	{
		fh << data[i];
	}

	fh.flush();

	if (fh.bad())
	{
		cerr << "__savePPM() : Writing data failed." << endl;
		return false;
	}

	fh.close();

	return true;
}
