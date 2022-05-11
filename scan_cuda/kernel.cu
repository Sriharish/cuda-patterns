
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void sum(int* dA) {
	int threadId = threadIdx.x;


}

int main() {

	const int count = 512;
	int computeSize = count * sizeof(int);
	int hA[count];
	int* dA;

	for (int i = 0; i < count; i++) {
		hA[i] = i + 1;
	}

	cudaMalloc(&dA, computeSize);
	cudaMemcpy(dA, hA, computeSize, cudaMemcpyHostToDevice);

	sum<<<1, count - 1>>>(dA);

	cudaMemcpy(hA, dA, computeSize, cudaMemcpyDeviceToHost);
	cudaFree(dA);

	cout << hA[count - 1] << endl;
	getchar();
	return 0;
}