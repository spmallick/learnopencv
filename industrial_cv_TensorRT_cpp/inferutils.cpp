#include "inferutils.h"

#ifndef BATCH_SIZE
#define BATCH_SIZE 1 //prevent any possible errors from  BATCH_SIZE
#endif

bool zero_copy_malloc(void** cpu_ptr, void** gpu_ptr, size_t size)
{
	if (size==0) {return false;}

	cudaError_t alloc_err = cudaHostAlloc(cpu_ptr, size, cudaHostAllocMapped);

	if (alloc_err != cudaSuccess)
			return false;

	cudaError_t err= cudaHostGetDevicePointer(gpu_ptr, *cpu_ptr, 0);

	if (err != cudaSuccess)
			return false;

	memset(*cpu_ptr, 0, size);
	return true;

}

bool file_exists(std::string filename)
{
	FILE* ftr=NULL;
	ftr=fopen(filename.c_str(), "rb");
	if (!ftr)
	{
		return false;
	}
	else
	{
		return true;
	}


}

bool read_engine_from_disk(char* estream, size_t esize, const std::string enginepath)
{
	FILE* efile = NULL;
	efile = fopen(enginepath.c_str(), "rb");
	if (!efile)
	{
		//File not found error
		return false;
	}
	
	std::experimental::filesystem::path fp("/home/agx/agxnvme/trtapp/build/trt.engine");
	esize = std::experimental::filesystem::file_size(fp);
	
	if (esize==0)
	{
		//invalid engine file
		return false;
	}
	else
	{
		/*DO NOTHING*/
	}

	estream= (char*)malloc(esize);

	const size_t bytesread = fread(estream, 1, esize, efile);
	
	if (bytesread != esize)
	{
		//corrupt file error
		return false;
	}

	fclose(efile);
	return true;

}

bool save_to_disk(nvinfer1::ICudaEngine* eg, std::string filename)
{

nvinfer1::IHostMemory* serialized_engine = eg->serialize();

if (!serialized_engine)
{
return false;
}

const char* edata = (char *)serialized_engine->data();
const size_t esize= serialized_engine->size();

FILE* diskfile=NULL;
diskfile=fopen(filename.c_str(),"wb");

if( fwrite(edata, 1, esize, diskfile) != esize)
{
return false;
}

fclose(diskfile);

serialized_engine->destroy();
return true;
}

uint32_t iobinding::get_size()
{

uint32_t sz=BATCH_SIZE*sizeof(float);

for (int i=0; i<4; i++)
{
sz*=std::max(1,dims.d[i]);
}
return sz;
}

void iobinding::allocate_buffers()
{
if (!size)
	size=get_size();

if (!zero_copy_malloc(&tCPU,&tGPU, size))
	throw std::runtime_error("Cannot allocate buffers for binding");

cpu_ptr = (float*)tCPU;
gpu_ptr = (float*)tGPU;

}

void iobinding::destroy_buffers()
{
cudaFreeHost(tCPU);
}

nvinfer1::Dims validate_shift(const nvinfer1::Dims dims, bool shift=true)
{
/* FOr compatibility with onnx models in TensorRT 7.0 and above */
	nvinfer1::Dims out_dims=dims;
	if (shift)
	{
		out_dims.d[0]=std::max(1,dims.d[1]);
		out_dims.d[1]=std::max(1,dims.d[2]);
		out_dims.d[2]=std::max(1,dims.d[3]);
		out_dims.d[3]=1;
	}

	for (int n=out_dims.nbDims; n < nvinfer1::Dims::MAX_DIMS; n++)
		out_dims.d[n] = 1;

	return out_dims;
}

iobinding get_engine_bindings(nvinfer1::ICudaEngine* eg, const char* name, bool is_onnx=true)
{

iobinding io;
io.binding=eg->getBindingIndex(name);
if (io.binding<0)
{
	std::cout << "Could not find binding of name: " << name << std::endl;
	throw std::runtime_error("Binding not found error");
}

io.name=name;
io.dims=validate_shift(eg->getBindingDimensions(io.binding), is_onnx);
io.allocate_buffers();
return io;
}

cudaStream_t create_cuda_stream(bool nonblocking)
{
uint32_t flags = nonblocking?cudaStreamNonBlocking:cudaStreamDefault;

cudaStream_t stream = NULL;

cudaError_t err = cudaStreamCreateWithFlags(&stream, flags);

if (err != cudaSuccess)
	return NULL;

//SetStream(stream);
return stream;

}
