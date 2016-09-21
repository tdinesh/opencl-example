#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define N 512

#define STRINGIFY(...) #__VA_ARGS__

const char* kernelSource = STRINGIFY(
__kernel void VecAdd(__global float* A,
                     __global float* B,
                     __global float* C)
{
    int i = get_global_id(0);

    C[i] = A[i] + B[i];
}
);

int main(void)
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    std::unique_ptr<cl::Context> context;
    std::unique_ptr<cl::CommandQueue> queue;
    std::unique_ptr<cl::Program> program;
    std::unique_ptr<cl::Kernel> kernel;
    std::unique_ptr<cl::Buffer> d_a, d_b, d_c;

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);

    std::generate(a.begin(), a.end(), rand);
    std::generate(b.begin(), b.end(), rand);

    try {
        cl::Platform::get(&platforms);
        if (!platforms.size())
            throw std::exception();

        cl::Platform& platform = platforms[0];

        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (!devices.size())
            throw std::exception();

        cl::Device& device = devices[0];

        context.reset(new cl::Context(device));
        queue.reset(new cl::CommandQueue(*context, device));

        program.reset(new cl::Program(
                *context,
                std::string(kernelSource),
                true));

        kernel.reset(new cl::Kernel(
                *program,
                "VecAdd"));

        d_a.reset(new cl::Buffer(
                *context,
                CL_MEM_READ_WRITE,
                N * sizeof(float)));

        d_b.reset(new cl::Buffer(
                *context,
                CL_MEM_READ_WRITE,
                N * sizeof(float)));

        d_c.reset(new cl::Buffer(
                *context,
                CL_MEM_READ_WRITE,
                N * sizeof(float)));

        queue->enqueueWriteBuffer(
                *d_a,
                true,
                0,
                N * sizeof(float),
                a.data());
        queue->enqueueWriteBuffer(
                *d_b,
                true,
                0, N * sizeof(float),
                b.data());

        kernel->setArg(0, *d_a);
        kernel->setArg(1, *d_b);
        kernel->setArg(2, *d_c);

        queue->enqueueNDRangeKernel(
                *kernel,
                cl::NullRange,
                {N},
                {1});

        queue->finish();

        queue->enqueueReadBuffer(
                *d_c,
                true,
                0,
                N * sizeof(float),
                c.data());

        for (unsigned int i = 0; i < c.size(); ++i) {
            if (c[i] != a[i] + b[i]) {
                std::cerr << c[i] << " != " << a[i] << " + " << b[i] << "\n";
                ::exit(-1);
            }

            std::cout << c[i] << " = " << a[i] << " + " << b[i] << "\n";
        }
    } catch (cl::Error& e) {
        std::cerr << "OpenCL C++ API error : " << e.what() << "\n";
        ::exit(-1);
    }
    return 0;
}
