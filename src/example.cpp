#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <boost/program_options.hpp>

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

int main(int argc, char **argv)
{
    std::srand(std::time(0));
    namespace po = boost::program_options;
    unsigned int number;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("version", "Show version number")
        ("number,n",
         po::value<unsigned int>(&number)->default_value(N),
         "number of elements to treat")
        ("show,s", "Show list of computed elements");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << '\n';
        return 0;
    }

    if (vm.count("version")) {
        std::cout << "Version 1.0.0\n";
        return 0;
    }

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    std::unique_ptr<cl::Context> context;
    std::unique_ptr<cl::CommandQueue> queue;
    std::unique_ptr<cl::Program> program;
    std::unique_ptr<cl::Kernel> kernel;
    std::unique_ptr<cl::Buffer> d_a, d_b, d_c;

    std::vector<float> a(number);
    std::vector<float> b(number);
    std::vector<float> c(number);

    std::generate(a.begin(), a.end(), rand);
    std::generate(b.begin(), b.end(), rand);

    try {

        cl::Platform::get(&platforms);
        if (!platforms.size())
            throw std::exception();

        cl::Platform plat;
        for (auto &p : platforms) {
            std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
            std::cout << "Platform: " << platver << std::endl;

            std::vector<cl::Device> all_devices;
            p.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
            if(all_devices.size()==0){
                std::cout<<" No devices found. Check OpenCL installation!\n";
                throw std::exception();
            }
            for (auto &p : all_devices) {
                std::string dev_name = p.getInfo<CL_DEVICE_NAME>();
                std::string vendor = p.getInfo<CL_DEVICE_VENDOR>();
                std::string version = p.getInfo<CL_DEVICE_VERSION>();
                std::cout << "Device Name: " <<dev_name << "\nVendor: " << vendor << "\nVersion: " << version << "\nType: " << p.getInfo<CL_DEVICE_TYPE>() << std::endl;
            }

            if (platver.find("OpenCL 2.") != std::string::npos) {
                plat = p;
            }
        }
        if (plat() == 0)  {
            std::cout << "No OpenCL 2.0 platform found." << std::endl;
        }

        cl::Platform& platform = platforms[0];

        //get default device of the default platform

        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (!devices.size())
            throw std::exception();

        cl::Device& device = devices[0];
        std::cout<< "Using device: "<<device.getInfo<CL_DEVICE_NAME>()<<"\n";

        //std::cout<< "Highest OpenCL Supported: "<<device.getInfo<CL_DEVICE_OPENCL_C_VERSION>()<<"\n";

        std::cout<< "Global Mem(KB): "<<device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/1000<<"\n";
        std::cout<< "Local Mem(KB): "<<device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()/1000<<"\n";
        std::cout<< "Max Const Buffer Size(KB): "<<device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>()/1000<<"\n";

        std::cout<< "Max Compute Units: "<<device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()<<"\n";

        std::cout<< "Max WorkGroup Size: "<<device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()<<"\n";
        std::cout<< "Max WorkItem  Dims: "<<device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>()<<"\n";

        auto sz = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
        for(const auto s : sz)
          std::cout<< "Max WorkItem  Sizes in each Dim: "<<s<<"\n";

        context.reset(new cl::Context(device));
        queue.reset(new cl::CommandQueue(*context, device));

        program.reset(new cl::Program(
                *context,
                std::string(kernelSource),
                true));

        kernel.reset(new cl::Kernel(
                *program,
                "VecAdd"));

        auto workGroupSize = kernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        std::cout << "Max WorkItems in WorkGroup " << workGroupSize << std::endl;

        auto localMemSize = kernel->getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device);
        std::cout << "Local Mem in WorkGroup " << localMemSize << std::endl;

        d_a.reset(new cl::Buffer(
                *context,
                CL_MEM_READ_WRITE,
                number * sizeof(float)));

        d_b.reset(new cl::Buffer(
                *context,
                CL_MEM_READ_WRITE,
                number * sizeof(float)));

        d_c.reset(new cl::Buffer(
                *context,
                CL_MEM_READ_WRITE,
                number * sizeof(float)));

        queue->enqueueWriteBuffer(
                *d_a,
                true,
                0,
                number * sizeof(float),
                a.data());
        queue->enqueueWriteBuffer(
                *d_b,
                true,
                0,
                number * sizeof(float),
                b.data());

        kernel->setArg(0, *d_a);
        kernel->setArg(1, *d_b);
        kernel->setArg(2, *d_c);

        queue->enqueueNDRangeKernel(
                *kernel,
                cl::NullRange,
                {number},
                {1});

        queue->enqueueReadBuffer(
                *d_c,
                true,
                0,
                number * sizeof(float),
                c.data());

        queue->finish();

        for (unsigned int i = 0; i < c.size(); ++i) {
            if (c[i] != a[i] + b[i]) {
                std::cerr << c[i] << " != " << a[i] << " + " << b[i] << '\n';
                ::exit(-1);
            }

            if (vm.count("show"))
                std::cout << c[i] << " = " << a[i] << " + " << b[i] << '\n';
        }
    } catch (cl::Error& e) {
        std::cerr << "OpenCL C++ API error : " << e.what() << "\n";
        ::exit(-1);
    }

    return 0;
}
