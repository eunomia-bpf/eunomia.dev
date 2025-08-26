// Copyright 2021 NVIDIA Corporation. All rights reserved
//
// This sample demonstrates two ways to use the CUPTI Profiler API with concurrent kernels.
// By taking the ratio of runtimes for a consecutive series of kernels, compared
// to a series of concurrent kernels, one can difinitively demonstrate that concurrent
// kernels were running while metrics were gathered and the User Replay mechanism was in use.
//
// Example:
// 4 kernel launches, with 1x, 2x, 3x, and 4x amounts of work, each sized to one SM (one warp
// of threads, one thread block).
// When run synchronously, this comes to 10x amount of work.
// When run concurrently, the longest (4x) kernel should be the only measured time (it hides the others).
// Thus w/ 4 kernels, the concurrent : consecutive time ratio should be 4:10.
// On test hardware this does simplify to 3.998:10.  As the test is affected by memory layout, this may not
// hold for certain architectures where, for example, cache sizes may optimize certain kernel calls.
//
// After demonstrating concurrency using multpile streams, this then demonstrates using multiple devices.
// In this 3rd configuration, the same concurrent workload with streams is then duplicated and run
// on each device concurrently using streams.
// In this case, the wallclock time to launch, run, and join the threads should be roughly the same as the
// wallclock time to run the single device case.  If concurrency was not working, the wallcock time
// would be (num devices) times the single device concurrent case.
//
//  * If the multiple devices have different performance, the runtime may be significantly different between
//    devices, but this does not mean concurrent profiling is not happening.

// Standard STL headers
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <thread>
#include <vector>

// CUDA headers
#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

// CUPTI headers
#include <helper_cupti.h>
#include <cupti_range_profiler_util.h>

using namespace cupti::utils;

// Consolidate CUPTI Profiler options into one location
// This contains fields for multiple levels of Profiler API configuration, with only one Session and Config
// More complicated configurations can configure multiple sessions and multiple configs w/in a session:
// Session
//     Config
//     ...
//     Config
// ...
// Session
//     Config
//     ...
//     Config

// Global  structures and variables
typedef struct ProfilingConfig_st
{
    // Compute device number.
    int device;
    // Maximum number of Ranges that may be encountered in this Session. (Nested Ranges are multiplicative.)
    int maxNumRanges;
    // CUPTI_AutoRange or CUPTI_UserRange.
    CUpti_ProfilerRange rangeMode;
    // CUPTI_KernelReplay, CUPTI_UserReplay, or CUPTI_ApplicationReplay.
    CUpti_ProfilerReplayMode replayMode;
    // Profiling data images.
    std::vector<uint8_t> counterDataImage;
    // CUDA driver context, or NULL if default context has already been initialized.
    CUcontext context;
} ProfilingConfig;

// Per-device configuration, buffers, stream and device information, and device pointers.
typedef struct PerDeviceData_st
{
    int deviceID;
    // Each device (or each context) needs its own CUPTI profiling config.
    ProfilingConfig config;
    // MetricEvaluator for the device.
    MetricEvaluator *evaluator;
    // RangeInfos for the device.
    std::vector<MetricEvaluator::RangeInfo> rangeInfos;
    // Each device needs its own streams.
    std::vector<cudaStream_t> streams;
    // Device memory allocations.
    std::vector<double *> d_x;
    std::vector<double *> d_y;
} PerDeviceData;

bool explicitlyInitialized = false;

// Initialize kernel values
double a = 2.5;

// Normally you would want multiple warps, but to emphasize concurrency with streams and multiple devices.
// We run the kernels on a single warp.
int threadsPerBlock = 32;
int threadBlocks = 1;

// Configurable number of kernels. (streams, when running concurrently.)
int const numKernels = 4;
int const numStreams = numKernels;
std::vector<size_t> elements(numKernels);

// Each kernel call allocates and computes (call number) * (blockSize) elements.
// For 4 calls, this is 4k elements * 2 arrays * (1 + 2 + 3 + 4 stream mul) * 8B/elem =~ 640KB
int const blockSize = 4 * 1024;

// Macros
#define DAXPY_REPEAT 32768

// Kernels
// Loop over array of elements performing daxpy multiple times.
// To be launched with only one block (artificially increasing serial time to better demonstrate overlapping replay).
__global__
void DaxpyKernel(
    int elements,
    double a,
    double *x,
    double *y
)
{
    for (int i = threadIdx.x; i < elements; i += blockDim.x)
        // Artificially increase kernel runtime to emphasize concurrency.
        for (int j = 0; j < DAXPY_REPEAT; j++)
            y[i] = a * x[i] + y[i]; // daxpy
}

// Wrapper which will launch numKernel kernel calls on a single device.
// The device streams vector is used to control which stream each call is made on.
// If 'serial' is non-zero, the device streams are ignored and instead the default stream is used.
void ProfileKernels(
    PerDeviceData &deviceData,
    char const * const RangeName,
    bool serial,
    std::vector<std::string> metricNames
)
{
    // Switch to desired device
    RUNTIME_API_CALL(cudaSetDevice(deviceData.deviceID));
    DRIVER_API_CALL(cuCtxSetCurrent(deviceData.config.context));

    std::unique_ptr<RangeProfiler> rangeProfiler = std::make_unique<RangeProfiler>(
        deviceData.config.context,
        (size_t)deviceData.deviceID
    );

    CUPTI_API_CALL(rangeProfiler->EnableRangeProfiler());

    CUPTI_API_CALL(rangeProfiler->SetConfig(
        deviceData.config.rangeMode,
        deviceData.config.replayMode,
        metricNames,
        deviceData.config.counterDataImage,
        deviceData.config.maxNumRanges
    ));

    RangeProfiler::PassState passState = {};
    // Perform multiple passes if needed to provide all configured metrics.
    // Note that in this mode, kernel input data is not restored to initial values before each pass.
    do
    {
        CUPTI_API_CALL(rangeProfiler->StartRangeProfiler());
        CUPTI_API_CALL(rangeProfiler->PushRange(RangeName));

        for (unsigned int stream = 0; stream < deviceData.streams.size(); stream++)
        {
            cudaStream_t streamId = (serial ? 0 : deviceData.streams[stream]);
            DaxpyKernel <<< threadBlocks, threadsPerBlock, 0, streamId >>> (elements[stream], a, deviceData.d_x[stream], deviceData.d_y[stream]);
            RUNTIME_API_CALL(cudaGetLastError());

        }

        // After launching all work, synchronize all streams.
        if (serial == false)
        {
            for (unsigned int stream = 0; stream < deviceData.streams.size(); stream++)
            {
                RUNTIME_API_CALL(cudaStreamSynchronize(deviceData.streams[stream]));
            }
        }
        else
        {
            RUNTIME_API_CALL(cudaStreamSynchronize(0));
        }

        CUPTI_API_CALL(rangeProfiler->PopRange());
        CUPTI_API_CALL(rangeProfiler->StopRangeProfiler());
        passState = rangeProfiler->GetPassState();
    }
    while (!passState.isAllPassSubmitted);

    CUPTI_API_CALL(rangeProfiler->DecodeCounterData());
    std::vector<MetricEvaluator::RangeInfo> rangeInfos;
    deviceData.evaluator->evaluateAllRanges(deviceData.config.counterDataImage, metricNames, rangeInfos);
    CUPTI_API_CALL(rangeProfiler->DisableRangeProfiler());

    deviceData.rangeInfos.insert(deviceData.rangeInfos.end(), rangeInfos.begin(), rangeInfos.end());
}

int main(
    int argc,
    char *argv[]
)
{
    // These two metrics will demonstrate whether kernels within a Range were run serially or concurrently.
    std::vector<std::string> metricNames;
    metricNames.push_back("sm__cycles_active.sum");
    metricNames.push_back("sm__cycles_elapsed.max");
    // This metric shows that the same number of flops were executed on each run.
    metricNames.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum");

    int numDevices;
    RUNTIME_API_CALL(cudaGetDeviceCount(&numDevices));

    // Per-device information.
    std::vector<int> deviceIds;

    // Find all devices capable of running CUPTI Profiling.
    for (int i = 0; i < numDevices; i++)
    {
        // Get device compatibility.
        CUptiResult result = RangeProfiler::CheckDeviceSupport(i);
        if (result == CUPTI_SUCCESS) {
            deviceIds.push_back(i);
        }
    }

    numDevices = deviceIds.size();
    std::cout << "Found " << numDevices << " compatible devices" << std::endl;

    // Ensure we found at least one device.
    if (numDevices == 0)
    {
        std::cerr << "No devices detected compatible with CUPTI Profiling" << std::endl;
        exit(EXIT_WAIVED);
    }

    // Initialize kernel input to some known numbers.
    std::vector<double> h_x(blockSize * numKernels);
    std::vector<double> h_y(blockSize * numKernels);
    for (size_t i = 0; i < blockSize * numKernels; i++)
    {
        h_x[i] = 1.5 * i;
        h_y[i] = 2.0 * (i - 3000);
    }

    // Initialize a vector of 'default stream' values to demonstrate serialized kernels.
    std::vector<cudaStream_t> defaultStreams(numStreams);
    for (int stream = 0; stream < numStreams; stream++) {
        defaultStreams[stream] = 0;
    }

    // Scale per-kernel work by stream number.
    for (int stream = 0; stream < numStreams; stream++) {
        elements[stream] = blockSize * (stream + 1);
    }

    // For each device, configure profiling, set up buffers, copy kernel data.
    std::vector<PerDeviceData> deviceData(numDevices);

    for (int device = 0; device < numDevices; device++)
    {
        int deviceId = deviceIds[device];
        RUNTIME_API_CALL(cudaSetDevice(deviceId));
        std::cout << "Configuring device " << deviceId << std::endl;

        // Required CUPTI Profiling configuration & initialization.
        // Can be done ahead of time or immediately before startSession() call.
        // Initialization & configuration images can be generated separately, then passed to later calls.
        // For simplicity's sake, in this sample, a single config struct is created per device and passed to each CUPTI Profiler API call.
        // For more complex cases, each combination of CUPTI Profiler Session and Config requires additional initialization.
        ProfilingConfig config;
        // Device ID, used to get device name for metrics enumeration.
        config.device = deviceId;

        // Device 0 has max of 3 passes; other devices only run one pass in this sample code
        if (device == 0)
        {
            // Maximum number of ranges that may be profiled in the current Session
            config.maxNumRanges = 3;
        }
        else
        {
            // Maximum number of ranges that may be profiled in the current Session
            config.maxNumRanges = 1;
        }

        // CUPTI_AutoRange or CUPTI_UserRange.
        config.rangeMode = CUPTI_UserRange;
        // CUPTI_KernelReplay (only for CUPTI_AutoRange), CUPTI_UserReplay, or CUPTI_ApplicationReplay.
        config.replayMode = CUPTI_UserReplay;
        // Either set to a context, or may be NULL if a default context has been created.
        DRIVER_API_CALL(cuCtxCreate(&(config.context), (CUctxCreateParams*)0, 0, device));
        // Save this device config.
        deviceData[device].config = config;

        // Create a MetricEvaluator for the device.
        if (deviceData[device].evaluator == nullptr) {
            deviceData[device].evaluator = new MetricEvaluator(config.context);
        }

        // Per-stream initialization & memory allocation - copy from constant host array to each device array.
        deviceData[device].streams.resize(numStreams);
        deviceData[device].d_x.resize(numStreams);
        deviceData[device].d_y.resize(numStreams);
        for (int stream = 0; stream < numStreams; stream++)
        {
            RUNTIME_API_CALL(cudaStreamCreate(&(deviceData[device].streams[stream])));

            // Each kernel does (stream #) * blockSize work on doubles.
            size_t size = elements[stream] * sizeof(double);

            RUNTIME_API_CALL(cudaMalloc(&(deviceData[device].d_x[stream]), size));
            MEMORY_ALLOCATION_CALL(deviceData[device].d_x[stream]);
            RUNTIME_API_CALL(cudaMemcpy(deviceData[device].d_x[stream], h_x.data(), size, cudaMemcpyHostToDevice));

            RUNTIME_API_CALL(cudaMalloc(&(deviceData[device].d_y[stream]), size));
            MEMORY_ALLOCATION_CALL(deviceData[device].d_y[stream]);
            RUNTIME_API_CALL(cudaMemcpy(deviceData[device].d_y[stream], h_y.data(), size, cudaMemcpyHostToDevice));
        }
    }

    // First version - single device, kernel calls serialized on default stream.
    // Use wallclock time to measure performance.
    auto begin_time = std::chrono::high_resolution_clock::now();

    // Run on first device and use default streams, which run serially.
    ProfileKernels(deviceData[0], "single_device_serial", true, metricNames);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_serial_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time);
    int numBlocks = 0;
    for (int i = 1; i <= numKernels; i++) {
        numBlocks += i;
    }
    std::cout << "It took " << elapsed_serial_ms.count() << "ms on the host to profile " << numKernels << " kernels in serial." << std::endl;

    // Second version - same kernel calls as before on the same device, but now using separate streams for concurrency.
    // (Should be limited by the longest running kernel.)
    begin_time = ::std::chrono::high_resolution_clock::now();

    // Still only use first device, but this time use its allocated streams for parallelism.
    ProfileKernels(deviceData[0], "single_device_async", false, metricNames);

    end_time = ::std::chrono::high_resolution_clock::now();
    auto elapsed_single_device_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time);
    std::cout << "It took " << elapsed_single_device_ms.count() << "ms on the host to profile " << numKernels << " kernels on a single device on separate streams." << std::endl;
    std::cout << "--> If the separate stream wallclock time is less than the serial version, the streams were profiling concurrently." << std::endl;

    // Third version - same as the second case, but duplicates the concurrent work across devices to show cross-device concurrency.
    // This is done using devices so no serialization is needed between devices.
    // (Should have roughly the same wallclock time as second case if the devices have similar performance)

    if (numDevices == 1)
    {
        std::cout << "Only one compatible device found; skipping the multi-threaded test." << std::endl;
    }
    else
    {
        std::cout << "Running on " << numDevices << " devices, one thread per device." << std::endl;

        // Time creation of the same multiple streams. (on multiple devices, if possible.)
        std::vector<std::thread> threads;
        begin_time = std::chrono::high_resolution_clock::now();

        // Now launch parallel thread work, duplicated on one thread per device.
        for (int thread = 0; thread < numDevices; thread++) {
            threads.push_back(std::thread(ProfileKernels, std::ref(deviceData[thread]), "multi_device_async", false, metricNames));
        }

        // Wait for all threads to finish.
        for (auto &t: threads) {
            t.join();
        }

        // Record time used when launching on multiple devices.
        end_time = ::std::chrono::high_resolution_clock::now();
        auto elapsed_multiple_device_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time);
        std::cout << "It took " << elapsed_multiple_device_ms.count() << "ms on the host to profile the same " << numKernels << " kernels on each of the " << numDevices << " devices in parallel" << std::endl;
        std::cout << "--> Wallclock ratio of parallel device launch to single device launch is " << elapsed_multiple_device_ms.count() / static_cast<double>(elapsed_single_device_ms.count()) << std::endl;
        std::cout << "--> If the ratio is close to 1, that means there was little overhead to profile in parallel on multiple devices compared to profiling on a single device." << std::endl;
        std::cout << "--> If the devices have different performance, the ratio may not be close to one, and this should be limited by the slowest device." << std::endl;
    }

    // Free stream memory for each device.
    for (int i = 0; i < numDevices; i++)
    {
        for (int j = 0; j < numKernels; j++)
        {
            RUNTIME_API_CALL(cudaFree(deviceData[i].d_x[j]));
            RUNTIME_API_CALL(cudaFree(deviceData[i].d_y[j]));
        }
    }

    // Display metric values.
    std::cout << std::endl << "Metrics for device #0:" << std::endl;
    std::cout << "Look at the sm__cycles_elapsed.max values for each test." << std::endl;
    std::cout << "This value represents the time spent on device to run the kernels in each case, and should be longest for the serial range, and roughly equal for the single and multi device concurrent ranges." << std::endl;
    deviceData[0].evaluator->printMetricData(deviceData[0].rangeInfos);

    // Only display next device info if needed.
    if (numDevices > 1)
    {
        std::cout << std::endl << "Metrics for the remaining devices only display the multi device async case and should all be similar to the first device's values if the device has similar performance characteristics." << std::endl;
        std::cout << "If devices have different performance characteristics, the runtime cycles calculation may vary by device." << std::endl;
    }
    for (int i = 1; i < numDevices; i++)
    {
        std::cout << std::endl << "Metrics for device #" << i << ":" << std::endl;
        deviceData[i].evaluator->printMetricData(deviceData[i].rangeInfos);
    }

    exit(EXIT_SUCCESS);
}
