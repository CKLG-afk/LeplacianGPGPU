
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>

using namespace std;

__global__ void laplacianKernel(unsigned char* input, unsigned char* output,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)
        return;

    int idx = y * width + x;

    int sum =
        -input[(y - 1) * width + x] +
        -input[y * width + (x - 1)] +
        4 * input[idx] +
        -input[y * width + (x + 1)] +
        -input[(y + 1) * width + x];

    sum = max(0, min(255, sum));

    output[idx] = (unsigned char)sum;
}

int main()
{
    int width, height, channels;

    unsigned char* h_input = stbi_load("input.jpg", &width, &height, &channels, 1);
    if (!h_input) {
        cout << "Blad wczytywania obrazu!\n";
        return -1;
    }
    
    cout << "Wczytano obraz: " << width << "x" << height << endl;

    size_t size = width * height * sizeof(unsigned char);

    unsigned char* h_output = new unsigned char[width * height];

    unsigned char* d_input, * d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    laplacianKernel << <gridSize, blockSize >> > (d_input, d_output, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "Kernel error: " << cudaGetErrorString(err) << endl;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        cout << "Sync error: " << cudaGetErrorString(err) << endl;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    stbi_write_png("output1.png", width, height, 1, h_output, width);

    for (int i = 0; i < width * height; i++) {
        h_output[i] = (h_output[i] > 10) ? 255 : 0;
    }

    stbi_write_png("output2.png", width, height, 1, h_output, width);

    cout << "Zapisano output1.png i output2.png\n";

    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_input);
    delete[] h_output;

    return 0;
}