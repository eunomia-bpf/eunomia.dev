/**
 * OpenCL Basic Example - Vector Addition
 * 
 * This program demonstrates the fundamental concepts of OpenCL programming:
 * 1. Platform and device discovery
 * 2. Creating OpenCL context and command queue
 * 3. Allocating memory buffers on the GPU
 * 4. Loading and compiling OpenCL kernel source
 * 5. Setting kernel arguments and executing on GPU
 * 6. Reading results back to host
 * 7. Cleanup and resource management
 * 
 * This example shows the OpenCL equivalent of CUDA vector addition,
 * highlighting the differences and similarities between the two approaches.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// OpenCL kernel source code
const char* kernelSource = 
"__kernel void vectorAdd(__global const float* A,\n"
"                       __global const float* B,\n"
"                       __global float* C,\n"
"                       const int numElements) {\n"
"    int i = get_global_id(0);\n"
"    if (i < numElements) {\n"
"        C[i] = A[i] + B[i];\n"
"    }\n"
"}\n";

// Helper function to get error string
const char* getErrorString(cl_int error) {
    switch(error) {
        case CL_SUCCESS: return "Success";
        case CL_DEVICE_NOT_FOUND: return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES: return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling info not available";
        case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE: return "Build program failure";
        case CL_MAP_FAILURE: return "Map failure";
        case CL_INVALID_VALUE: return "Invalid value";
        case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
        case CL_INVALID_PLATFORM: return "Invalid platform";
        case CL_INVALID_DEVICE: return "Invalid device";
        case CL_INVALID_CONTEXT: return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
        case CL_INVALID_HOST_PTR: return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
        case CL_INVALID_SAMPLER: return "Invalid sampler";
        case CL_INVALID_BINARY: return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
        case CL_INVALID_PROGRAM: return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
        case CL_INVALID_KERNEL: return "Invalid kernel";
        case CL_INVALID_ARG_INDEX: return "Invalid argument index";
        case CL_INVALID_ARG_VALUE: return "Invalid argument value";
        case CL_INVALID_ARG_SIZE: return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS: return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
        case CL_INVALID_EVENT: return "Invalid event";
        case CL_INVALID_OPERATION: return "Invalid operation";
        case CL_INVALID_GL_OBJECT: return "Invalid GL object";
        case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL: return "Invalid mip level";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid global work size";
        case -1001: return "Platform not found (CL_PLATFORM_NOT_FOUND_KHR)";
        default: return "Unknown error";
    }
}

// Helper function to check OpenCL errors
void checkError(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        printf("Error during %s: %d (%s)\n", operation, error, getErrorString(error));
        exit(1);
    }
}

void printOpenCLInfo() {
    printf("\n=== OpenCL System Information ===\n");
    
    cl_uint num_platforms = 0;
    cl_int ret = clGetPlatformIDs(0, NULL, &num_platforms);
    
    if (ret != CL_SUCCESS || num_platforms == 0) {
        printf("No OpenCL platforms found!\n");
        printf("\nPossible solutions:\n");
        printf("1. Install OpenCL runtime for your hardware:\n");
        printf("   - NVIDIA GPU: Install CUDA toolkit\n");
        printf("   - AMD GPU: Install ROCm or AMD drivers\n");
        printf("   - Intel CPU/GPU: Install Intel OpenCL runtime\n");
        printf("2. For Ubuntu/Debian:\n");
        printf("   sudo apt-get install ocl-icd-opencl-dev\n");
        printf("   sudo apt-get install nvidia-opencl-dev (for NVIDIA)\n");
        printf("3. Check if your system has any OpenCL-capable devices\n");
        return;
    }
    
    printf("Found %d OpenCL platform(s):\n", num_platforms);
    
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    ret = clGetPlatformIDs(num_platforms, platforms, NULL);
    
    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_name[256];
        char platform_vendor[256];
        char platform_version[256];
        
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, NULL);
        
        printf("\nPlatform %d:\n", i);
        printf("  Name: %s\n", platform_name);
        printf("  Vendor: %s\n", platform_vendor);
        printf("  Version: %s\n", platform_version);
        
        // Get devices for this platform
        cl_uint num_devices = 0;
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        
        if (ret == CL_SUCCESS && num_devices > 0) {
            printf("  Devices: %d\n", num_devices);
            
            cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
            ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
            
            for (cl_uint j = 0; j < num_devices; j++) {
                char device_name[256];
                cl_device_type device_type;
                cl_ulong global_mem_size;
                
                clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
                clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
                clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
                
                printf("    Device %d: %s", j, device_name);
                if (device_type & CL_DEVICE_TYPE_GPU) printf(" (GPU)");
                else if (device_type & CL_DEVICE_TYPE_CPU) printf(" (CPU)");
                else printf(" (Other)");
                printf(" - %lu MB\n", global_mem_size / (1024 * 1024));
            }
            
            free(devices);
        } else {
            printf("  No devices found for this platform\n");
        }
    }
    
    free(platforms);
    printf("\n");
}

int main() {
    // Vector size and memory size
    const int numElements = 50000;
    const size_t dataSize = numElements * sizeof(float);
    
    printf("OpenCL Vector addition of %d elements\n", numElements);
    
    // Print OpenCL system information
    printOpenCLInfo();
    
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    
    // Get platform
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS || ret_num_platforms == 0) {
        printf("ERROR: No OpenCL platforms available!\n");
        printf("This system does not have OpenCL support.\n");
        printf("Please install OpenCL runtime for your hardware.\n");
        return 1;
    }
    
    // Get platform info
    char platform_name[1024];
    clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    printf("Using OpenCL platform: %s\n", platform_name);
    
    // Get device (prefer GPU, fallback to any device)
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    if (ret != CL_SUCCESS) {
        printf("No GPU found, trying any device type...\n");
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
        if (ret != CL_SUCCESS) {
            printf("ERROR: No OpenCL devices found on this platform!\n");
            return 1;
        }
    }
    
    if (ret_num_devices == 0) {
        printf("ERROR: No OpenCL devices available!\n");
        return 1;
    }
    
    // Allocate host memory
    float *h_A = (float*)malloc(dataSize);
    float *h_B = (float*)malloc(dataSize);
    float *h_C = (float*)malloc(dataSize);
    
    if (!h_A || !h_B || !h_C) {
        printf("Failed to allocate host memory\n");
        return 1;
    }
    
    // Initialize host arrays with random data
    for (int i = 0; i < numElements; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    // Get device info
    char device_name[1024];
    cl_device_type device_type;
    cl_ulong global_mem_size;
    cl_uint compute_units;
    size_t max_work_group_size;
    
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    
    printf("Using device: %s\n", device_name);
    printf("Device type: %s\n", (device_type & CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU");
    printf("Global memory: %lu MB\n", global_mem_size / (1024 * 1024));
    printf("Compute units: %u\n", compute_units);
    printf("Max work group size: %zu\n", max_work_group_size);
    
    // Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    checkError(ret, "creating context");
    
    // Create command queue (use newer API if available)
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {0};
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &ret);
#else
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
#endif
    checkError(ret, "creating command queue");
    
    // Create memory buffers on device
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &ret);
    checkError(ret, "creating buffer A");
    
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &ret);
    checkError(ret, "creating buffer B");
    
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, &ret);
    checkError(ret, "creating buffer C");
    
    // Copy data to device buffers
    ret = clEnqueueWriteBuffer(command_queue, d_A, CL_TRUE, 0, dataSize, h_A, 0, NULL, NULL);
    checkError(ret, "writing buffer A");
    
    ret = clEnqueueWriteBuffer(command_queue, d_B, CL_TRUE, 0, dataSize, h_B, 0, NULL, NULL);
    checkError(ret, "writing buffer B");
    
    // Create program from source
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &ret);
    checkError(ret, "creating program");
    
    // Build program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("Error building program: %s\n", getErrorString(ret));
        
        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build log:\n%s\n", log);
        free(log);
        return 1;
    }
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "vectorAdd", &ret);
    checkError(ret, "creating kernel");
    
    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_A);
    checkError(ret, "setting kernel arg 0");
    
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_B);
    checkError(ret, "setting kernel arg 1");
    
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_C);
    checkError(ret, "setting kernel arg 2");
    
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&numElements);
    checkError(ret, "setting kernel arg 3");
    
    // Execute kernel
    size_t globalWorkSize = numElements;
    size_t localWorkSize = 256; // Work group size
    
    // Adjust global work size to be multiple of local work size
    if (globalWorkSize % localWorkSize != 0) {
        globalWorkSize = ((globalWorkSize / localWorkSize) + 1) * localWorkSize;
    }
    
    printf("OpenCL kernel launch with global work size %zu and local work size %zu\n", 
           globalWorkSize, localWorkSize);
    
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    checkError(ret, "enqueueing kernel");
    
    // Wait for kernel to complete
    ret = clFinish(command_queue);
    checkError(ret, "waiting for kernel to finish");
    
    // Read result back to host
    ret = clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0, dataSize, h_C, 0, NULL, NULL);
    checkError(ret, "reading result buffer");
    
    // Verify the result
    printf("Verifying results...\n");
    for (int i = 0; i < numElements; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            printf("Result verification failed at element %d!\n", i);
            printf("Expected: %f, Got: %f\n", h_A[i] + h_B[i], h_C[i]);
            return 1;
        }
    }
    printf("Test PASSED\n");
    
    // Cleanup
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(d_A);
    ret = clReleaseMemObject(d_B);
    ret = clReleaseMemObject(d_C);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Done\n");
    return 0;
} 