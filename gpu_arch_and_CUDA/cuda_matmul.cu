#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>

//#define VERIFY
//uncomment above to print difference between CPU and GPU calculations

__global__ void matmul_kernel(
	const float* M1, 
	const float* M2, 
	float* M3, 
	const int m, 
	const int n, 
	const int p
	)

	{
		/*
		CUDA kernel for matrix multiplication M3 = M1 * M2
		This function will be executed by every CUDA thread
		The instructions are the same, but each thread will work
		on a separate chunk of the data, as specified by the array indices.
		Note that the kernel definition is preceded by the __global__
		qualifier. Further, the kernel function returns nothing (void)
		Thus, we must modify the output matrix M3 within this function.
		The changes made to M3 (or M1 and M2) will all be visible outside
		the kernel to CPU and GPU memory after the kernel has executed.
		*/

		//Get the x and y indices of output entry for this thread
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int j = blockIdx.x * blockDim.x + threadIdx.x;
		
		/*
		Wait! what are blockDim,  blockIdx and threadIdx??
		These are structs provided by CUDA, which tells the thread
		how many blocks have been launched, what block number does 
		the current thread reside in and finally, what is the x and y 
		index of the current thread within the block.
		These variables allow each thread to choose which sub-section
		of the A, B and C matrices it should work on and we use them next.
		*/

		if ((i>=m)||(j>=p))
		{
			return;
			//this just means that dont process anything outside the
			//bounds of the output matrix size
		}

		float cout=0.0;
		//this is a local variable we have defined within the thread
		//so, this variable will reside in register memory as explained earlier
		
		for (int k=0; k<n; k++)
		{
			cout += M1[i*n + k]*M2[k*p + j];
			//loop through elements of one row of M1 and
			//one column of M2, multiply corresponding elements
			//and add them up. We are just doing standard matrix 
			//multiplication.
		}
		
		M3[i*p+j] = cout;
		//here we modify M3
	}

int main(int argc, char* argv[])
{
	/*
	In this demo, we will create matrices of size
	A:	M x N
	B:	N x P
	C:	M x P <-- for GPU
	D:	M x P <-- for CPU
	
	We will initialize A, B, C, D and perform matrix multiplications: 
	C = A*B (on GPU)
	D = A*B (on CPU)
	*/

	if (argc != 4)
	{
		printf("Matrix multiplication example for A[MxN] and B[NxP]\nUsage: cu_mm.out M N P\n");
		exit(1);		
	}

	int M=atoi(argv[1]); //2049;
	int N=atoi(argv[2]); //257;
	int P=atoi(argv[3]); //512;

	float *A, *B, *C, *D;

	/*
	Let's use unified memory
	cudaMallocManaged allows us to allocate memory 
	once and use it across both CPU and GPU.
	*/

	cudaMallocManaged(&A, M*N*sizeof(float));//input Mat1
	cudaMallocManaged(&B, N*P*sizeof(float));//input Mat2
	
	cudaMallocManaged(&C, M*P*sizeof(float));//output Mat for GPU

	cudaMallocManaged(&D, M*P*sizeof(float));//output Mat for CPU
	//we will do matmul in both CPU and GPU and compare the execution times

	for (int i=0; i<M*N; i++)
	{
		A[i]=sin((float)i/100); 
		//init with sine of index, just as an example
	}

	for (int i=0; i<N*P; i++)
	{
		B[i]=cos((float)i/100); 
		//init with sine of index, just as an example
	}

	//C and D can be left uninitialized

	float elapsed_time_gpu=0.0;
	double elapsed_time_cpu=0.0;
	cudaEvent_t gpu_start, gpu_stop;
	struct timespec cpu_start, cpu_stop;

	//BEGIN GPU MATMUL
	dim3 blocks_per_grid(ceil(M/32),ceil(P/32));
	dim3 threads_per_block(32, 32);

	/*
	We use CUDA events to accurately measure the time taken by matmul op
	Refer to page 16 of CUDA C++ Best Practices Guide:
	https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf
	*/
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);

	cudaEventRecord(gpu_start, 0);

	matmul_kernel<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, P);
	cudaEventRecord(gpu_stop, 0);

	cudaEventSynchronize(gpu_stop);
	//END GPU MATMUL

	timespec_get(&cpu_start, TIME_UTC);

	//BEGIN CPU MATMUL
	for (int i=0; i<M; i++)
	{
		for (int j=0; j< P; j++)
		{
			float cout=0.0;

			for(int k=0; k<N; k++)
			{
				cout+=A[i*N+k]*B[k*P+j];
			}

			D[i*P+j]=cout;
		}
	}
	//END CPU MATMUL

	timespec_get(&cpu_stop, TIME_UTC);
	
	//Measure elapsed times
	cudaEventElapsedTime(&elapsed_time_gpu, gpu_start, gpu_stop);
	elapsed_time_cpu = ((double)(cpu_stop.tv_sec - cpu_start.tv_sec)) * 1000000 + ((double)(cpu_stop.tv_nsec - cpu_start.tv_nsec)) / 1000;
	//tv_nsec is in nanoseconds

	/*
	Define VERIFY above to print diffs for the 
	first 100 entries
	you will get all values very close to zero
	*/
	#ifdef VERIFY
	for (int i=0; i<100; i++)
	{
		float diff=C[i]-D[i];
		printf("%f, ", diff);
	}
	printf("\n");
	#endif

	//convert microseconds to milliseconds
	printf("Elapsed time (CPU)= %f milliseconds\n", elapsed_time_cpu/1000);
	printf("Elapsed time (GPU)= %f milliseconds\n", elapsed_time_gpu);
	//cudaEventElapsedTime reports time in milliseconds
	
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(D);
}