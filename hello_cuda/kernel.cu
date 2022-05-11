
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void addArrays(int* a, int* b, int* c) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main() {
	const int count = 5;
	int dataSize = count * sizeof(int);
	int hA[] = { 1, 2, 3, 4, 5 };
	int hB[] = { 10, 20, 30, 40, 50 };
	int hC[count];

	// Cuda Constants
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cudaDeviceProp prop;

	for (int i = 0; i < deviceCount; i++) {
		cudaGetDeviceProperties(&prop, i);
		cout << "Device " << i << ": " << prop.name << endl;
		cout << "Compute Power: " << prop.major << "." << prop.minor << endl;
		cout << "Max grid dimensions: (" << prop.maxGridSize[0] << " x " <<
			prop.maxGridSize[1] << " x " <<
			prop.maxGridSize[2] << ")" << endl;
		cout << "Max block dimensions: (" << prop.maxThreadsDim[0] << " x " <<
			prop.maxThreadsDim[1] << " x " <<
			prop.maxThreadsDim[2] << ")" << endl << endl;
	}

	getchar();

	int* dA, *dB, *dC;
	cudaMalloc(&dA, dataSize);
	cudaMalloc(&dB, dataSize);
	cudaMalloc(&dC, dataSize);
	
	cudaMemcpy(dA, hA, dataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, dataSize, cudaMemcpyHostToDevice);
	
	addArrays <<<1, count>>>(dA, dB, dC);

	cudaMemcpy(hC, dC, dataSize, cudaMemcpyDeviceToHost);
	
	/*for (int i = 0; i < count; ++i) {
		addArrays(a, b, c);
	}*/
	
	for (int i = 0; i < count; ++i) {
		printf("%d ", hC[i]);
	}

	return 0;
}

