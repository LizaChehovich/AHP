#pragma once

const unsigned int HeaderSize = 64;

bool load_ppm(const char *file, unsigned char **data, unsigned int *w, unsigned int *h, unsigned int *channels);

bool save_ppm(const char *file, unsigned char *data, unsigned int w, unsigned int h, unsigned int channels);