#ifndef __TRT_INFER_H

#define __TRT_INFER_H

#define BATCH_SIZE 1

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <experimental/filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <chrono>
#include <string.h>

/*
This function allocates memory on host and provides a cpu pointer and
gpu pointer for use with cuda functions. Since the memory on jetson
is unified, the two pointers point to the same physical memory location.

Notes:
* This function is not x86 compatible, and works only  an embedded platform
*/

bool zero_copy_malloc(void** cpu_ptr, void** gpu_ptr, size_t size);

bool file_exists(std::string filename);
bool save_to_disk(nvinfer1::ICudaEngine* eg, std::string filename);
//bool read_engine_from_disk(nvinfer1::ICudaEngine* eg, std::string enginepath);
bool read_engine_from_disk(char* estream, size_t esize, const std::string enginepath);

cudaStream_t create_cuda_stream(bool nonblocking);
//struct iobinding 
class iobinding
{
public:
	nvinfer1::Dims dims;
	std::string name;
	float* cpu_ptr=nullptr;
	float* gpu_ptr=nullptr;
	uint32_t size=0;
	uint32_t binding;

	uint32_t get_size();
	void allocate_buffers();
	void destroy_buffers();
	void* tCPU;
	void* tGPU;

};

iobinding get_engine_bindings(nvinfer1::ICudaEngine* eg, const char* name, bool is_onnx);

#endif
