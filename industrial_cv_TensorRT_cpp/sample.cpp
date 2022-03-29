#include "inferutils.h"

#ifndef BATCH_SIZE
#define BATCH_SIZE 1
#endif

class Logger : public nvinfer1::ILogger
{
void log(Severity severity, const char* msg) noexcept
{
if (severity <= Severity::kINFO)
	std::cout << "\033[1;35m [TRT] " << msg << "\033[0m\n";
}
} gLogger;

int main(int argc, char* argv[])
{

if (argc<2)
{
	std::cout <<"Please specify onnx file name\n";
	return 1;
}

std::string onnxpath = argv[1];
std::string epath = "trt.engine";

nvinfer1::ICudaEngine* engine;
nvonnxparser::IParser* parser;
nvinfer1::IBuilderConfig* config;
nvinfer1::INetworkDefinition* network;
nvinfer1::IBuilder* builder;
nvinfer1::IRuntime* infer;
char* estream=NULL;
size_t esize=0;
//Logger* gLogger = new Logger();

//here
if (file_exists(epath))
{
	std::cout << "File already exists";
	if(!read_engine_from_disk(estream, esize, epath))
	{
		std::cout << "Could not read engine\n";
	}

	infer= nvinfer1::createInferRuntime(gLogger);
	std::cout << "Engine size: " << (int)esize << std::endl;
	engine=infer->deserializeCudaEngine(&estream, esize);

	//return 2;
}
else
{
	//engine = build_engine(gLogger, BATCH_SIZE, onnxpath, epath);
	builder=nvinfer1::createInferBuilder(gLogger);

	builder->setMaxBatchSize(BATCH_SIZE);

	network= builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

	parser= nvonnxparser::createParser(*network, gLogger);

	if (!(parser->parseFromFile(onnxpath.c_str(), static_cast<uint32_t>(nvinfer1::ILogger::Severity::kWARNING))))
	{
	//we use parseFromFile instead of parse, since it has better error logging
	for (int i=0; i< parser->getNbErrors(); i++)
	{
	std::cout << parser->getError(i)->desc() <<std::endl;
	}
	throw std::runtime_error("Could not parse onnx model from file");
	}

	config = builder->createBuilderConfig();

	config->setMaxWorkspaceSize(1<<25);
	config->setFlag(nvinfer1::BuilderFlag::kFP16);

	engine = builder->buildEngineWithConfig(*network, *config);

	if (!save_to_disk(engine, epath))
	{
	throw std::runtime_error("Failed to serialize engine and save it to disk");
	}
	else
	{
	std::cout << "Inference engine serialized and saved to " << epath << std::endl;
	}

}
//upto here
 
nvinfer1::IExecutionContext* ctx = engine->createExecutionContext();

auto net_inputs = get_engine_bindings(engine, "image", true);
auto segout = get_engine_bindings(engine, "probabilities", true);

void* buffers[2]={net_inputs.gpu_ptr, segout.gpu_ptr};

//net_inputs.print_binding();
//segout.print_binding();

auto t1=std::chrono::system_clock::now();

for(int i=0; i<1000; i++)
{
	ctx->execute(BATCH_SIZE, buffers);//, stream, NULL);
}

auto t2=std::chrono::system_clock::now();

auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

float fps= 1000/(duration*1e-3);

std::cout << "Network FPS = " << std::to_string(fps) << std::endl;

net_inputs.destroy_buffers();
segout.destroy_buffers();

/*
ctx->destroy();
parser->destroy();
network->destroy();
config->destroy();
builder->destroy();
*/

return 0;
}
