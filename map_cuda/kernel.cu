﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h";
#include <ctime>
#include <cstdio>;
#include <iostream>;

using namespace std;

__global__ void addTen(float* d, int count) {
	int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
	int threadPosInBlock = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
	int blockPosInGrid = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

	int tid = blockPosInGrid * threadsPerBlock + threadPosInBlock;
	if (tid < count) {
		d[tid] = d[tid] * 1000 + 1;
	}
}


int main() {
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32); // init generator
	curandSetPseudoRandomGeneratorSeed(gen, time(0)); // provide seed value

	const int count = 123456;
	const int size = count * sizeof(float);
	float* d;
	float h[count];
	cudaMalloc(&d, size);
	curandGenerateUniform(gen, d, count);
	
	dim3 block(8, 8, 8);
	dim3 grid(16, 16);

	addTen<<<grid, block>>>(d, count);
	cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
	cudaFree(d);

	for (int i = 0; i < count; ++i) {
		cout << h[i] << endl;
	}

	getchar();
	return 0;
}
