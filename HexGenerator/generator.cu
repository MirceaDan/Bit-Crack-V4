#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "generator.cuh"

__constant__ const int __numbersPer4096MB = 1000000;
__constant__ const int __spaceOccupiedBy1MillionValues = 4096;//MB
__constant__ const uint32_t __UINT32_MAX = (uint32_t) - 1;

__global__ void generate(long long startingPoint, long long* set)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	set[idx] = startingPoint + idx;
	//printf("idx=%d, value=%lld\n", idx, (long long)set[idx]);
}

__global__ void add(uint32_t* startingPoint, uint32_t* set)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int carry = idx;
	//printf("Before addition idx=%d, v[0]=%d, v[1]=%d, v[2]=%d, v[3]=%d, v[4]=%d, v[5]=%d, v[6]=%d, v[7]=%d\n", idx, startingPoint[0], startingPoint[1], startingPoint[2], startingPoint[3], startingPoint[4], startingPoint[5], startingPoint[6], startingPoint[7]);
	for (int i = 0; i < 8; i++)
	{
		uint64_t newVal = startingPoint[i] + carry;
		set[idx*8 + i] = uint32_t(newVal % __UINT32_MAX);
		carry = newVal / __UINT32_MAX;
	}
	//printf("After addition idx=%d, v[0]=%d, v[1]=%d, v[2]=%d, v[3]=%d, v[4]=%d, v[5]=%d, v[6]=%d, v[7]=%d\n", idx, set[idx * 8], set[idx * 8 + 1], set[idx * 8 + 2], set[idx * 8 + 3], set[idx * 8 + 4], set[idx * 8 + 5], set[idx * 8 + 6], set[idx * 8 + 7]);
}

uint32_t* GenerateValues(uint32_t* startingPoint, int totalPoints, int runIndex)
{
	uint32_t* set = new uint32_t[totalPoints * 8];
	uint32_t* d_set;
	uint32_t* d_startingPoint;

	cudaMalloc(&d_set, 8 * totalPoints * sizeof(uint32_t));
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(err));
	}

	cudaMalloc(&d_startingPoint, 8 * sizeof(uint32_t));
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(err));
	}

	cudaMemcpy(d_startingPoint, startingPoint, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock = totalPoints > 128 ? dim3(128, 1) : dim3(totalPoints, 1);
	dim3 blocksPerGrid = dim3((ceil)(totalPoints / threadsPerBlock.x), 1);

	// generate random numbers
	add << <blocksPerGrid, threadsPerBlock >> > (d_startingPoint, d_set);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(err));
	}

	cudaDeviceSynchronize();

	cudaMemcpy(set, d_set, 8 * totalPoints * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	cudaFree(d_set);
	cudaFree(d_startingPoint);

	return set;
}

namespace Generator {
	int CountGPUs()
	{
		int devCount;
		cudaGetDeviceCount(&devCount);
		return devCount;
	}

	string* GetGPUNames()
	{
		int devCount;
		cudaGetDeviceCount(&devCount);

		string* gpuNames = new string[devCount];
		for (int i = 0; i < devCount; i++)
		{
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			gpuNames[i] = std::string(prop.name);
		}

		return gpuNames;
	}

	uint32_t* Generate(uint32_t* startingPoint, int totalPoints)
	{
		int devCount;
		cudaGetDeviceCount(&devCount);

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, devCount - 1);

		uint64_t totalMemory = deviceProp.totalGlobalMem;
		uint64_t totalMemoryS = totalMemory / ((uint64_t)1024 * 1024);

		long maximumPossibleNumbersAt1Run = __numbersPer4096MB * totalMemoryS / __spaceOccupiedBy1MillionValues;
		if (maximumPossibleNumbersAt1Run >= totalPoints / 64)
		{
			return GenerateValues(startingPoint, totalPoints, 1);
		}
		else
		{
			uint32_t* numbers = new uint32_t[8 * totalPoints];
			int runs = (ceil)((8 * totalPoints / 64) / maximumPossibleNumbersAt1Run) + 1;
			for (int i = 0; i < runs; i++)
			{
				uint32_t* set;
				if (i != runs - 1)
				{
					set = GenerateValues(startingPoint, maximumPossibleNumbersAt1Run, i + 1);
				}
				else
				{
					set = GenerateValues(startingPoint, 8 * totalPoints - i * maximumPossibleNumbersAt1Run, i + 1);
				}

				if (i == 0)
				{
					int idx = 0;
					for (int j = i * maximumPossibleNumbersAt1Run; j < i* maximumPossibleNumbersAt1Run + maximumPossibleNumbersAt1Run; j += 10)
					{
						numbers[j] = set[idx];
						numbers[j + 1] = set[idx + 1];
						numbers[j + 2] = set[idx + 2];
						numbers[j + 3] = set[idx + 3];
						numbers[j + 4] = set[idx + 4];
						numbers[j + 5] = set[idx + 5];
						numbers[j + 6] = set[idx + 6];
						numbers[j + 7] = set[idx + 7];
						numbers[j + 8] = set[idx + 8];
						numbers[j + 9] = set[idx + 9];
						idx += 10;
					}
				}
				else
				{
					int idx = 0;
					for (int j = i * maximumPossibleNumbersAt1Run; j < 8 * totalPoints; j++)
					{
						numbers[j] = set[idx];
						idx++;
					}
				}
			}

			return numbers;
		}
	}
}