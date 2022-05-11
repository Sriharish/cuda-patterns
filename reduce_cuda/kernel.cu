
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void reduce(int* dA) {
	int threadId = threadIdx.x;
	
	for (int tc = blockDim.x, stepSize = 1; tc > 0; tc >>= 1, stepSize <<= 1) {
		if (threadId < tc) {
			int pA = threadId * stepSize * 2;
			int pB = pA + stepSize;
			dA[pA] += dA[pB];
		}
	}
}

__global__ void reduce_shared(int* dA) {
	extern __shared__ int dACopy[];
	int threadId = threadIdx.x;
	dACopy[threadId * 2 + 1] = dA[threadId * 2];

	for (int tc = blockDim.x, stepSize = 1; tc > 0; tc >>= 1, stepSize <<= 1) {
		if (threadId < tc) {
			int pA = threadId * stepSize * 2;
			int pB = pA + stepSize;
			dACopy[pA] += dACopy[pB];
		}
	}

	if (threadId == 0) {
		dA[0] = dACopy[0];
	}
}

int main() {
	const int count = 4;
	int computeSize = count * sizeof(int);
	int hA[] = {2, 2, 3, 4};
	int* dA;

	cudaMalloc(&dA, computeSize);
	cudaMemcpy(dA, hA, computeSize, cudaMemcpyHostToDevice);
	
	int numThreads = count / 2;
	reduce_shared<<<1, numThreads, computeSize>>>(dA);
	
	cudaMemcpy(hA, dA, computeSize, cudaMemcpyDeviceToHost);
	cudaFree(dA);

	for (int i = 0; i < count; ++i) {
		cout << hA[i] << endl;
	}

	getchar();
	return 0;
}