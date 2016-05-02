#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define THREADS 512 

/**
 * Function on CPU code
 */
void SumOnCPU (int* sum, const int* datas, const int numElements)
{
	int i;
	*sum = 0;
	for (i=0; i<numElements; ++i)
		*sum += *(datas+i);
}

void UnrolledSumOnCPU (int* sum, const int* datas, const int numElements)
{
	int i;
	*sum = 0;
	for (i=0; i<numElements; i+=2)
		*sum += *(datas+i);
	for (i=1; i<numElements; i+=2)
		*sum += *(datas+i);
}

/**
 * CUDA Kernel Device code
 */

// global atomicAdd
__global__ void SumAtomic(int* sum, const int* datas, const int numElements)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numElements) return;
	atomicAdd(sum, datas[id]);
}

// shared atomicAdd
__global__ void SumReductionAtomic(int* sum, const int* datas, const int numElements)
{
	__shared__ int sh_sum;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numElements) return;
	
	if (threadIdx.x==0)
		sh_sum = 0;
	__syncthreads();

	atomicAdd(&sh_sum, datas[id]);
	__syncthreads();

	if (threadIdx.x == 0) atomicAdd(sum, sh_sum);
}

// binary sum reduction 
__global__ void SumReductionBinary(int* sum, const int* datas, const int numElements)
{
	__shared__ int sh_arr[THREADS]; // size: nThreads
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id >= numElements) {
		sh_arr[threadIdx.x] = 0;
	} else {
		sh_arr[threadIdx.x] = datas[id];
	}
	__syncthreads();

	int offset;
	for (offset=blockDim.x>>1; offset>0; offset>>=1) {
		if (threadIdx.x < offset)
			sh_arr[threadIdx.x] += sh_arr[threadIdx.x + offset];
		__syncthreads();
	}

	if (threadIdx.x == 0)
		atomicAdd(sum, sh_arr[0]);
}

/**
  * shuffle
  */

// shuffle sum reduce function with __shfl_down()
__device__ int reduceSumWarp(int val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}
// if all threads in the warp need the result, you have to using __shfl_xor()
__device__ int reduceInWarp(int val)
{
	for (int mask=warpSize/2; mask > 0; mask/=2) {
		val += __shfl_xor(val, mask);
	}
	return val;
}

__global__ void SumReductionShuffleAtom(int* sum, const int* datas, const int numElements)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int idInWarp = threadIdx.x % warpSize;

	if (id < numElements) {
		int val = datas[id];
//		val = reduceInWarp(val);
		val = reduceSumWarp(val);
		if (idInWarp == 0)
			atomicAdd (sum, val);
	}
}

__global__ void SumReductionShuffleShared(int* sum, const int* datas, const int numElements)
{
	__shared__ int s_sum;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int idInWarp = threadIdx.x % warpSize;
	if (threadIdx.x == 0)
		s_sum = 0;

	if (id < numElements) {
		int val = datas[id];
		val = reduceSumWarp(val);
		if (idInWarp == 0)
			atomicAdd(&s_sum, val);
		__syncthreads();
		if (threadIdx.x == 0)
			atomicAdd (sum, s_sum);
	}
}

__device__ int blockReduceSum(int val) {
	static __shared__ float shared[32];
	int lane = threadIdx.x % warpSize;	// thread index within the warp
	int wid  = threadIdx.x / warpSize;	// warp ID

	// warp reduction (only the threads with 0 index within the warp has warp reduction result)
	val = reduceSumWarp(val);
	if (lane == 0) shared[wid] = val;
	__syncthreads();

	// there will be at most 1024 threads within a block and at most 1024 blocks within a grid.
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid == 0) val = reduceSumWarp(val);

	return val;
}

__global__ void SumReductionShuffleQ(int* sum, const int* datas, const int numElements)
{
	int tSum = 0;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x)
		tSum += datas[i];
	tSum = blockReduceSum(tSum);
	if (threadIdx.x == 0)
		atomicAdd(sum, tSum);
}

float getduration (const struct timeval begin, const struct timeval finish)
{
	return ((finish.tv_sec-begin.tv_sec) * 1000.0
			+(double)(finish.tv_usec-begin.tv_usec)/1000.0);
}

/**
 * Host main routine
 */
int main(int argc, char **argv)
{
	if (argc < 3) {
		printf("ERROR: have to insert arguments\n");
		printf("usage: %s {# of elements} {sum reduction method}\n", argv[0]);
		printf(" - sum reduction methods are\n\
  * cpu : run sum reduction by for loop in cpu thread\n\
  * cpu_ur : loop unrolling with cpu\n\
  * g_atom : run sum reduction by global atomicAdd in gpu kernel\n\
  * s_atom : shared atomicAdd with gpu\n\
  * binary : binary sum reduction with gpu\n\
  * shfl_a : shuffle + global atomicAdd with gpu\n\
  * shfl_s : shuffle + shared atomicAdd with gpu\n\
  * shfl_q : quantitative reduction with shuffle\n");

		return -1;
	}

	struct timeval begin, finish;
	int numElements = atoi(argv[1]);
	int *datas, *d_datas;
	int sum, *d_sum;

	// init value
	datas = (int*) malloc(sizeof(int) * numElements);
	for (int i=0; i<numElements; ++i)
		datas[i] = 1;

	// Device Memory Allocate
	cudaMalloc((void **)&d_datas, sizeof(int) * numElements);
	cudaMalloc((void **)&d_sum, sizeof(int));

	// Copy the Host Value to Device
	cudaMemcpy(d_datas, datas, sizeof(int) * numElements, cudaMemcpyHostToDevice);
	cudaMemset(d_sum, 0, sizeof(int));

	printf("\nSummation %d elements\n", numElements);

	// set CUDA block/thread
	int threads = THREADS;
	int blocks = numElements % threads ? (numElements / threads) + 1 : numElements / threads;

	// run sum reduction & print result
	gettimeofday(&begin, NULL);
	if (!strcmp(argv[2], "cpu"))
		SumOnCPU(&sum, datas, numElements);
	else if (!strcmp(argv[2], "cpu_ur"))
		UnrolledSumOnCPU(&sum, datas, numElements);
	else if (!strcmp(argv[2], "g_atom")) {
		SumAtomic<<<blocks, threads>>>(d_sum, d_datas, numElements);
		cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
	} else if (!strcmp(argv[2], "s_atom")) {
		SumReductionAtomic<<<blocks, threads>>>(d_sum, d_datas, numElements);
		cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
	} else if (!strcmp(argv[2], "binary")) {
		SumReductionBinary<<<blocks, threads>>>(d_sum, d_datas, numElements);
		cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
	} else if (!strcmp(argv[2], "shfl_a")) {
		SumReductionShuffleAtom<<<blocks, threads>>>(d_sum, d_datas, numElements);
		cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
	} else if (!strcmp(argv[2], "shfl_s")) {
		SumReductionShuffleShared<<<blocks, threads>>>(d_sum, d_datas, numElements);
		cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
	} else if (!strcmp(argv[2], "shfl_q")) {
		SumReductionShuffleQ<<<1024, 1024>>>(d_sum, d_datas, numElements);
		cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
	} else {
		printf("ERROR: you used wrong method\n");
		return -1;
	}
	gettimeofday(&finish, NULL);
	printf("** Used %s method\n", argv[2]);
	printf("  sum result: %d\n", sum);
	printf("  Sum Duration: %.3f(ms)\n", getduration(begin, finish));

	// Free device global memory
	cudaFree(d_datas);
	cudaFree(d_sum);

	// Free host memory
	free(datas);

	// Reset the device and exit
	cudaDeviceReset();
	return 0;
}

