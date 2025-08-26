#pragma once

#include <algorithm>
#include <vector>
#include <string>
#include <memory>

#include <cupti_target.h>
#include <cupti_range_profiler.h>
#include <helper_cupti.h>
#include <cupti_profiler_host_util.h>

#define MAX_NUM_OF_RANGES 64
#define MAX_NUM_OF_NESTING_LEVELS 1

namespace cupti { namespace utils {

class RangeProfiler
{
public:
    struct PassState
    {
        size_t passIndex = 0;
        size_t targetNestingLevel = 0;
        bool isAllPassSubmitted = false;
    };

    RangeProfiler(
        CUcontext ctx,
        size_t deviceIndex = 0
    );

    ~RangeProfiler();

    CUptiResult EnableRangeProfiler();

    CUptiResult DisableRangeProfiler();

    CUptiResult StartRangeProfiler();

    CUptiResult StopRangeProfiler();

    CUptiResult PushRange(
        const char* rangeName
    );

    CUptiResult PopRange();

    CUptiResult SetConfig(
        CUpti_ProfilerRange range,
        CUpti_ProfilerReplayMode replayMode,
        std::vector<std::string>& metrics,
        std::vector<uint8_t>& counterDataImage,
        uint32_t numOfRanges
    );

    CUptiResult DecodeCounterData();

    CUptiResult CreateCounterDataImage(
        uint32_t numOfRanges,
        std::vector<std::string>& metrics,
        std::vector<uint8_t>& counterDataImage
    );

    CUptiResult InitializeCounterDataImage(
        std::vector<uint8_t>& counterDataImage
    );

    PassState GetPassState();

    static CUptiResult CheckDeviceSupport(
        CUdevice device
    );

private:
    CUcontext m_context = nullptr;
    size_t m_deviceIndex = 0;
    std::vector<const char*> m_metricNames = {};
    std::vector<uint8_t> m_configImage = {};
    CUpti_RangeProfiler_Object* m_rangeProfilerObjectPtr = nullptr;
    PassState m_passState = {};
};

using RangeProfilerPtr = std::shared_ptr<RangeProfiler>;

inline
RangeProfiler::RangeProfiler(
    CUcontext ctx,
    size_t deviceIndex
) : m_context(ctx),
    m_deviceIndex(deviceIndex)
{
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    (cuptiProfilerInitialize(&profilerInitializeParams));
}

inline
RangeProfiler::~RangeProfiler()
{
    CUpti_Profiler_DeInitialize_Params profilerDeinitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    (cuptiProfilerDeInitialize(&profilerDeinitializeParams));
}

inline
CUptiResult RangeProfiler::EnableRangeProfiler()
{
    CUpti_RangeProfiler_Enable_Params enableRangeProfiler {CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
    enableRangeProfiler.ctx = m_context;
    CUPTI_API_CALL(cuptiRangeProfilerEnable(&enableRangeProfiler));
    m_rangeProfilerObjectPtr = enableRangeProfiler.pRangeProfilerObject;
    return CUPTI_SUCCESS;
}

inline
CUptiResult RangeProfiler::DisableRangeProfiler()
{
    CUpti_RangeProfiler_Disable_Params disableRangeProfiler {CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
    disableRangeProfiler.pRangeProfilerObject = m_rangeProfilerObjectPtr;
    CUPTI_API_CALL(cuptiRangeProfilerDisable(&disableRangeProfiler));
    m_rangeProfilerObjectPtr = nullptr;
    m_passState = {};
    return CUPTI_SUCCESS;
}

inline
CUptiResult RangeProfiler::StartRangeProfiler()
{
    CUpti_RangeProfiler_Start_Params startRangeProfiler {CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
    startRangeProfiler.pRangeProfilerObject = m_rangeProfilerObjectPtr;
    CUPTI_API_CALL(cuptiRangeProfilerStart(&startRangeProfiler));
    return CUPTI_SUCCESS;
}

inline
CUptiResult RangeProfiler::StopRangeProfiler()
{
    CUpti_RangeProfiler_Stop_Params stopRangeProfiler {CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
    stopRangeProfiler.pRangeProfilerObject = m_rangeProfilerObjectPtr;
    CUPTI_API_CALL(cuptiRangeProfilerStop(&stopRangeProfiler));
    m_passState.passIndex = stopRangeProfiler.passIndex;
    m_passState.targetNestingLevel = stopRangeProfiler.targetNestingLevel;
    m_passState.isAllPassSubmitted = stopRangeProfiler.isAllPassSubmitted;
    return CUPTI_SUCCESS;
}

inline
CUptiResult RangeProfiler::PushRange(
    const char* rangeName
)
{
    CUpti_RangeProfiler_PushRange_Params pushRange {CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE};
    pushRange.pRangeProfilerObject = m_rangeProfilerObjectPtr;
    pushRange.pRangeName = rangeName;
    CUPTI_API_CALL(cuptiRangeProfilerPushRange(&pushRange));
    return CUPTI_SUCCESS;
}

inline
CUptiResult RangeProfiler::PopRange()
{
    CUpti_RangeProfiler_PopRange_Params popRange {CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE};
    popRange.pRangeProfilerObject = m_rangeProfilerObjectPtr;
    CUPTI_API_CALL(cuptiRangeProfilerPopRange(&popRange));
    return CUPTI_SUCCESS;
}

inline
CUptiResult RangeProfiler::SetConfig(
    CUpti_ProfilerRange range,
    CUpti_ProfilerReplayMode replayMode,
    std::vector<std::string>& metrics,
    std::vector<uint8_t>& counterDataImage,
    uint32_t numOfRanges
)
{
    // Create config image
    MetricScheduler metricScheduler(m_context, CUPTI_PROFILER_TYPE_RANGE_PROFILER);
    metricScheduler.createConfigImage(metrics, m_configImage);

    // Create counter data image (scratch space)
    CreateCounterDataImage(numOfRanges, metrics, counterDataImage);

    CUpti_RangeProfiler_SetConfig_Params setConfig {CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
    setConfig.pRangeProfilerObject = m_rangeProfilerObjectPtr;
    setConfig.configSize = m_configImage.size();
    setConfig.pConfig = m_configImage.data();
    setConfig.counterDataImageSize = counterDataImage.size();
    setConfig.pCounterDataImage = counterDataImage.data();
    setConfig.range = range;
    setConfig.replayMode = replayMode;
    setConfig.maxRangesPerPass = MAX_NUM_OF_RANGES;
    setConfig.numNestingLevels = MAX_NUM_OF_NESTING_LEVELS;
    setConfig.minNestingLevel = 1;
    setConfig.passIndex = m_passState.passIndex;
    setConfig.targetNestingLevel = m_passState.targetNestingLevel;
    CUPTI_API_CALL(cuptiRangeProfilerSetConfig(&setConfig));
    return CUPTI_SUCCESS;
}

inline
CUptiResult RangeProfiler::DecodeCounterData()
{
    CUpti_RangeProfiler_DecodeData_Params decodeData {CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
    decodeData.pRangeProfilerObject = m_rangeProfilerObjectPtr;
    CUPTI_API_CALL(cuptiRangeProfilerDecodeData(&decodeData));
    if (decodeData.numOfRangeDropped > 0) {
        std::cout << "numOfRangeDropped: " << decodeData.numOfRangeDropped << std::endl;
    }
    return CUPTI_SUCCESS;
}

inline
CUptiResult RangeProfiler::CreateCounterDataImage(
    uint32_t numOfRanges,
    std::vector<std::string>& metrics,
    std::vector<uint8_t>& counterDataImage
)
{
    std::vector<const char*> metricNames;
    std::transform(metrics.begin(), metrics.end(), std::back_inserter(metricNames), [](const std::string& metric) { return metric.c_str(); });

    CUpti_RangeProfiler_GetCounterDataSize_Params getCounterDataSizeParams {CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
    getCounterDataSizeParams.pRangeProfilerObject = m_rangeProfilerObjectPtr;
    getCounterDataSizeParams.pMetricNames = metricNames.data();
    getCounterDataSizeParams.numMetrics = metricNames.size();
    getCounterDataSizeParams.maxNumOfRanges = numOfRanges;
    getCounterDataSizeParams.maxNumRangeTreeNodes = numOfRanges;
    CUPTI_API_CALL(cuptiRangeProfilerGetCounterDataSize(&getCounterDataSizeParams));

    counterDataImage.resize(getCounterDataSizeParams.counterDataSize, 0);
    CUPTI_API_CALL(InitializeCounterDataImage(counterDataImage));

    return CUPTI_SUCCESS;
}

inline
CUptiResult RangeProfiler::InitializeCounterDataImage(
    std::vector<uint8_t>& counterDataImage
)
{
    CUpti_RangeProfiler_CounterDataImage_Initialize_Params initializeCounterDataImageParams {CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeCounterDataImageParams.pRangeProfilerObject = m_rangeProfilerObjectPtr;
    initializeCounterDataImageParams.pCounterData = counterDataImage.data();
    initializeCounterDataImageParams.counterDataSize = counterDataImage.size();
    CUPTI_API_CALL(cuptiRangeProfilerCounterDataImageInitialize(&initializeCounterDataImageParams));
    return CUPTI_SUCCESS;
}

inline
RangeProfiler::PassState RangeProfiler::GetPassState()
{
    return m_passState;
}

inline
CUptiResult RangeProfiler::CheckDeviceSupport(
    CUdevice device
)
{
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = device;
    params.api = CUPTI_PROFILER_RANGE_PROFILING;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        std::cerr << "Unable to profile on device " << (size_t)device << std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            std::cerr << "\tdevice architecture is not supported" << std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            std::cerr << "\tdevice sli configuration is not supported" << std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            std::cerr << "\tdevice vgpu configuration is not supported" << std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            std::cerr << "\tdevice vgpu configuration disabled profiling support" << std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            std::cerr << "\tdevice confidential compute configuration is not supported" << std::endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << std::endl;
        }

        if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            std::cerr << "\tWSL is not supported" << std::endl;
        }
        exit(EXIT_WAIVED);
    }
    return CUPTI_SUCCESS;
}


} } // namespace cupti::utils
