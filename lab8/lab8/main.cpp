#include <fstream>
#include "kernel.h"
#include <mpi.h>

unsigned width_full = 0;
unsigned height_full = 0;

int main_main();
bool load_part(const char* file, byte** data, unsigned int* w, unsigned int* h, unsigned int* channels);
bool load_header(const char* file, unsigned int* w, unsigned int* h, unsigned int* channels, unsigned* data_start_pos);
bool load_header_0(const char* file, unsigned int* w, unsigned int* h, unsigned int* channels, unsigned* data_start_pos);

bool save_part(const char* file, byte* data, unsigned int w, unsigned int h, unsigned int channels);
bool save_header(const char* file, unsigned w, unsigned h, unsigned channels, unsigned* data_start_pos);
bool save_header_0(const char* file, unsigned w, unsigned h, unsigned channels, unsigned* data_start_pos);

int get_rank();
int get_world_size();
MPI_Offset calc_main_block_size(MPI_Offset all_size);
MPI_Offset calc_current_block_size(MPI_Offset all_size);
MPI_Offset calc_start_pos(MPI_Offset all_size);


int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	const int res = main_main();

	MPI_Finalize();

	return res;
}

int main_main()
{
	char input[] = "input.ppm";
	char output[] = "output.ppm";

	unsigned int width, height, channels;
	byte* in_image = nullptr;

	if (!load_part(input, &in_image, &width, &height, &channels))
	{
		return 1;
	}

	byte* out_image = (byte*)malloc(width * height * sizeof(uint8_t) * channels);
	if (!out_image)
	{
		free(in_image);
		return 1;
	}

	gpu_stream_convert_image(in_image, out_image, width, height, channels, get_rank());

	save_part(output, out_image, width, height, channels);

	free(in_image);
	free(out_image);
	return 0;
}

bool load_part(const char * file, byte ** data, unsigned int * w, unsigned int * h, unsigned int * channels)
{
	unsigned w_full;
	unsigned h_full;
	unsigned data_start_pos;
	if (!load_header(file, &w_full, &h_full, channels, &data_start_pos))
	{
		return false;
	}

	width_full = w_full;
	height_full = h_full;

	MPI_File fh;
	const int err = MPI_File_open(MPI_COMM_WORLD, file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	if (err)
	{
		return false;
	}

	const MPI_Offset one_row_size = w_full * (*channels);
	const MPI_Offset current_block_w_size = calc_current_block_size(w_full)* (*channels);
	const MPI_Offset w_offset = calc_start_pos(w_full)*(*channels);
	const int rank = get_rank();

	*h = h_full;
	*w = static_cast<unsigned int>(current_block_w_size/(*channels));

	int w_add = 2 * sizeof(int);

	*data = static_cast<byte*>(calloc((current_block_w_size + w_add)* h_full, sizeof(byte)));
	unsigned char* ptr = *data;

	int offset = data_start_pos + w_offset - sizeof(int);
	if (rank == 0) {
		offset += sizeof(int);
		ptr += sizeof(int);
		w_add -= sizeof(int);
	}

	for (int i = 0; i < h_full; i++)
	{
		MPI_File_seek(fh, offset + i * one_row_size, MPI_SEEK_SET);
		MPI_File_read(fh, ptr, current_block_w_size + w_add, MPI_CHAR, MPI_STATUS_IGNORE);
		ptr += current_block_w_size + 2 * sizeof(int);
	}

	MPI_File_close(&fh);

	return true;
}

bool load_header(const char * file, unsigned int * w, unsigned int * h, unsigned int * channels, unsigned * data_start_pos)
{
	bool res = false;
	if (get_rank() == 0)
	{
		res = load_header_0(file, w, h, channels, data_start_pos);
	}

	int is_header_loaded_int = res ? 1 : 0;
	MPI_Bcast(&is_header_loaded_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (is_header_loaded_int != 1)
	{
		return false;
	}

	MPI_Bcast(w, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	MPI_Bcast(h, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	MPI_Bcast(channels, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	MPI_Bcast(data_start_pos, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	return true;
}

bool load_header_0(const char * file, unsigned int * w, unsigned int * h, unsigned int * channels, unsigned * data_start_pos)
{
	FILE *fp = nullptr;

	if (fopen_s(&fp, file, "rb") != 0)
	{
		return false;
	}

	// check header
	char header[64];

	if (fgets(header, 64, fp) == nullptr)
	{
		return false;
	}

	if (strncmp(header, "P6", 2) == 0)
	{
		*channels = 3;
	}
	else
	{
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
		if (fgets(header, 64, fp) == nullptr)
		{
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

	fpos_t file_pos;
	fgetpos(fp, &file_pos);

	*w = width;
	*h = height;

	*data_start_pos = static_cast<unsigned int>(file_pos);

	fclose(fp);

	return true;
}

bool save_part(const char * file, byte * data, unsigned int w, unsigned int h, unsigned int channels)
{
	unsigned data_start_pos;
	if (!save_header(file, width_full, height_full, channels, &data_start_pos))
	{
		return false;
	}

	MPI_File fh;
	const int err = MPI_File_open(MPI_COMM_WORLD, file, MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	if (err)
	{
		return false;
	}

	const MPI_Offset one_row_size = width_full * (channels);
	const MPI_Offset current_block_w_size = calc_current_block_size(width_full)* (channels);
	const MPI_Offset w_offset = calc_start_pos(width_full) * (channels);
	const int rank = get_rank();

	unsigned char* ptr = data;
	
	for (int i = 0; i < height_full; i++)
	{
		MPI_File_seek(fh, data_start_pos + w_offset + i * one_row_size, MPI_SEEK_SET);
		MPI_File_write(fh, ptr, current_block_w_size, MPI_CHAR, MPI_STATUS_IGNORE);
		ptr += current_block_w_size;
	}

	MPI_File_close(&fh);

	return true;
}

bool save_header(const char * file, unsigned w, unsigned h, unsigned channels, unsigned * data_start_pos)
{
	bool res = false;
	if (get_rank() == 0)
	{
		res = save_header_0(file, w, h, channels, data_start_pos);
	}

	int is_header_loaded_int = res ? 1 : 0;
	MPI_Bcast(&is_header_loaded_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (is_header_loaded_int != 1)
	{
		return false;
	}

	MPI_Bcast(data_start_pos, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	return true;
}

bool save_header_0(const char * file, unsigned w, unsigned h, unsigned channels, unsigned * data_start_pos)
{
	fstream fh(file, fstream::out | fstream::binary);

	if (fh.bad())
	{
		return false;
	}

	if (channels == 3)
	{
		fh << "P6\n";
	}
	else
	{
		return false;
	}

	fh << w << "\n" << h << "\n" << 0xff << endl;

	fh.flush();

	if (fh.bad())
	{
		return false;
	}

	*data_start_pos = static_cast<unsigned>(fh.tellp());

	fh.close();

	return true;
}

int get_rank()
{
	static int rank = -1;
	if (rank < 0)
	{
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	}
	return rank;
}

int get_world_size()
{
	static int size = -1;
	if (size < 0)
	{
		MPI_Comm_size(MPI_COMM_WORLD, &size);
	}
	return size;
}

MPI_Offset calc_main_block_size(MPI_Offset all_size)
{
	return ((all_size - 1) / get_world_size() + 1);
}

MPI_Offset calc_current_block_size(MPI_Offset all_size)
{
	const MPI_Offset one_block_size = calc_main_block_size(all_size);
	const MPI_Offset calculated_size = (all_size - one_block_size * get_rank());
	const MPI_Offset calculated_normilized_size = calculated_size > 0 ? calculated_size : 0;
	return calculated_normilized_size < one_block_size ? calculated_normilized_size : one_block_size;
}

MPI_Offset calc_start_pos(MPI_Offset all_size)
{
	return calc_main_block_size(all_size) * get_rank();
}