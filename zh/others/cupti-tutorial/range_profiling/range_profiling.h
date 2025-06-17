//
// Copyright 2024 NVIDIA Corporation. All rights reserved
//

// System headers
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <unordered_map>
#include <memory>

// CUPTI headers
#include "helper_cupti.h"
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cupti_profiler_host.h>
#include <cupti_range_profiler.h>

struct ProfilerRange
{
    size_t rangeIndex;
    std::string rangeName;
    std::unordered_map<std::string, double> metricValues;
};

struct RangeProfilerConfig
{
    size_t maxNumOfRanges = 0;
    size_t numOfNestingLevel = 0;
    size_t minNestingLevel = 0;
};

class CuptiProfilerHost
{
public:
    CuptiProfilerHost() = default;
    ~CuptiProfilerHost() = default;

    void SetUp(std::string chipName, std::vector<uint8_t>& counterAvailibilityImage);
    void TearDown();

    CUptiResult CreateConfigImage(
        std::vector<const char*> metricsList,
        std::vector<uint8_t>& configImage
    );

    CUptiResult EvaluateCounterData(
        size_t rangeIndex,
        std::vector<const char*> metricsList,
        std::vector<uint8_t>& counterDataImage
    );

    CUptiResult GetNumOfRanges(
        std::vector<uint8_t>& counterDataImage,
        size_t& numOfRanges
    );

    void PrintProfilerRanges();

private:
    CUptiResult Initialize(std::vector<uint8_t>& counterAvailibilityImage);
    CUptiResult Deinitialize();

    std::string m_chipName;
    std::vector<ProfilerRange> m_profilerRanges;
    CUpti_Profiler_Host_Object* m_pHostObject = nullptr;
};

using CuptiProfilerHostPtr = std::shared_ptr<CuptiProfilerHost>;


class RangeProfilerTarget
{
public:
    RangeProfilerTarget(CUcontext ctx, RangeProfilerConfig& config);
    ~RangeProfilerTarget();

    CUptiResult EnableRangeProfiler();
    CUptiResult DisableRangeProfiler();

    CUptiResult StartRangeProfiler();
    CUptiResult StopRangeProfiler();

    CUptiResult PushRange(const char* rangeName);
    CUptiResult PopRange();

    CUptiResult SetConfig(
        CUpti_ProfilerRange range,
        CUpti_ProfilerReplayMode replayMode,
        std::vector<uint8_t>& configImage,
        std::vector<uint8_t>& counterDataImage
    );

    CUptiResult DecodeCounterData();
    CUptiResult CreateCounterDataImage(std::vector<const char*>& metrics, std::vector<uint8_t>& counterDataImage);

    bool IsAllPassSubmitted() const { return bIsAllPassSubmitted;}
    static CUptiResult GetChipName(size_t deviceIndex, std::string& chipName);
    static CUptiResult GetCounterAvailabilityImage(CUcontext ctx, std::vector<uint8_t>& counterAvailabilityImage);

private:
    CUcontext m_context = nullptr;
    size_t isProfilingActive = 0;
    bool bIsAllPassSubmitted = false;

    std::vector<const char*> metricNames = {};
    std::vector<uint8_t> configImage = {};
    RangeProfilerConfig mConfig = {};
    CUpti_RangeProfiler_Object* rangeProfilerObject = nullptr;
    bool bIsCuptiInitialized = false;
};

using RangeProfilerTargetPtr = std::shared_ptr<RangeProfilerTarget>;


inline void CuptiProfilerHost::SetUp(std::string chipName, std::vector<uint8_t>& counterAvailibilityImage)
{
    m_chipName = chipName;
    CUPTI_API_CALL(Initialize(counterAvailibilityImage));
}

inline void CuptiProfilerHost::TearDown()
{
    CUPTI_API_CALL(Deinitialize());
}

inline CUptiResult CuptiProfilerHost::CreateConfigImage(
    std::vector<const char*> metricsList,
    std::vector<uint8_t>& configImage
)
{
    // Add metrics to config image
    {
        CUpti_Profiler_Host_ConfigAddMetrics_Params configAddMetricsParams {CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
        configAddMetricsParams.pHostObject = m_pHostObject;
        configAddMetricsParams.ppMetricNames = metricsList.data();
        configAddMetricsParams.numMetrics = metricsList.size();
        CUPTI_API_CALL(cuptiProfilerHostConfigAddMetrics(&configAddMetricsParams));
    }

    // Get Config image size and data
    {
        CUpti_Profiler_Host_GetConfigImageSize_Params getConfigImageSizeParams {CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
        getConfigImageSizeParams.pHostObject = m_pHostObject;
        CUPTI_API_CALL(cuptiProfilerHostGetConfigImageSize(&getConfigImageSizeParams));
        configImage.resize(getConfigImageSizeParams.configImageSize);

        CUpti_Profiler_Host_GetConfigImage_Params getConfigImageParams = {CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
        getConfigImageParams.pHostObject = m_pHostObject;
        getConfigImageParams.pConfigImage = configImage.data();
        getConfigImageParams.configImageSize = configImage.size();
        CUPTI_API_CALL(cuptiProfilerHostGetConfigImage(&getConfigImageParams));
    }

    // Get Num of Passes
    {
        CUpti_Profiler_Host_GetNumOfPasses_Params getNumOfPassesParam {CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
        getNumOfPassesParam.pConfigImage = configImage.data();
        getNumOfPassesParam.configImageSize = configImage.size();
        CUPTI_API_CALL(cuptiProfilerHostGetNumOfPasses(&getNumOfPassesParam));
        std::cout << "Num of Passes: " << getNumOfPassesParam.numOfPasses << "\n";
    }

    return CUPTI_SUCCESS;
}

inline CUptiResult CuptiProfilerHost::EvaluateCounterData(
    size_t rangeIndex,
    std::vector<const char*> metricsList,
    std::vector<uint8_t>& counterDataImage
)
{
    m_profilerRanges.push_back(ProfilerRange{});
    ProfilerRange& profilerRange = m_profilerRanges.back();

    CUpti_RangeProfiler_CounterData_GetRangeInfo_Params getRangeInfoParams = {CUpti_RangeProfiler_CounterData_GetRangeInfo_Params_STRUCT_SIZE};
    getRangeInfoParams.counterDataImageSize = counterDataImage.size();
    getRangeInfoParams.pCounterDataImage = counterDataImage.data();
    getRangeInfoParams.rangeIndex = rangeIndex;
    getRangeInfoParams.rangeDelimiter = "/";
    CUPTI_API_CALL(cuptiRangeProfilerCounterDataGetRangeInfo(&getRangeInfoParams));

    profilerRange.rangeIndex = rangeIndex;
    profilerRange.rangeName = getRangeInfoParams.rangeName;

    std::vector<double> metricValues(metricsList.size());
    CUpti_Profiler_Host_EvaluateToGpuValues_Params evalauateToGpuValuesParams {CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    evalauateToGpuValuesParams.pHostObject = m_pHostObject;
    evalauateToGpuValuesParams.pCounterDataImage = counterDataImage.data();
    evalauateToGpuValuesParams.counterDataImageSize = counterDataImage.size();
    evalauateToGpuValuesParams.ppMetricNames = metricsList.data();
    evalauateToGpuValuesParams.numMetrics = metricsList.size();
    evalauateToGpuValuesParams.rangeIndex = rangeIndex;
    evalauateToGpuValuesParams.pMetricValues = metricValues.data();
    CUPTI_API_CALL(cuptiProfilerHostEvaluateToGpuValues(&evalauateToGpuValuesParams));

    for (size_t i = 0; i < metricsList.size(); ++i) {
        profilerRange.metricValues[metricsList[i]] = metricValues[i];
    }

    return CUPTI_SUCCESS;
}

CUptiResult CuptiProfilerHost::GetNumOfRanges(
    std::vector<uint8_t>& counterDataImage,
    size_t& numOfRanges
)
{
    CUpti_RangeProfiler_GetCounterDataInfo_Params getCounterDataInfoParams = {CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE};
    getCounterDataInfoParams.pCounterDataImage = counterDataImage.data();
    getCounterDataInfoParams.counterDataImageSize = counterDataImage.size();
    CUPTI_API_CALL(cuptiRangeProfilerGetCounterDataInfo(&getCounterDataInfoParams));
    numOfRanges = getCounterDataInfoParams.numTotalRanges;
    return CUPTI_SUCCESS;
}

inline void CuptiProfilerHost::PrintProfilerRanges()
{
    std::cout << "Total num of Ranges: " << m_profilerRanges.size() << "\n\n";
    for(size_t rangeIndex = 0; rangeIndex < m_profilerRanges.size(); ++rangeIndex)
    {
        const auto& profilerRange = m_profilerRanges[rangeIndex];
        std::cout << "Range Name: " << profilerRange.rangeName << "\n";
        std::cout << "-----------------------------------------------------------------------------------\n";
        for (const auto& metric : profilerRange.metricValues)
        {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << std::setw(50) << std::left << metric.first;
            std::cout << std::setw(30) << std::right << metric.second << "\n";
        }
        std::cout << "-----------------------------------------------------------------------------------\n\n";
    }
}

inline CUptiResult CuptiProfilerHost::Initialize(std::vector<uint8_t>& counterAvailibilityImage)
{
    CUpti_Profiler_Host_Initialize_Params hostInitializeParams = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    hostInitializeParams.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    hostInitializeParams.pChipName = m_chipName.c_str();
    hostInitializeParams.pCounterAvailabilityImage = counterAvailibilityImage.data();
    CUPTI_API_CALL(cuptiProfilerHostInitialize(&hostInitializeParams));
    m_pHostObject = hostInitializeParams.pHostObject;
    return CUPTI_SUCCESS;
}

inline CUptiResult CuptiProfilerHost::Deinitialize()
{
    CUpti_Profiler_Host_Deinitialize_Params deinitializeParams = {CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
    deinitializeParams.pHostObject = m_pHostObject;
    CUPTI_API_CALL(cuptiProfilerHostDeinitialize(&deinitializeParams));
    m_pHostObject = nullptr;
    return CUPTI_SUCCESS;
}

inline RangeProfilerTarget::RangeProfilerTarget(CUcontext ctx, RangeProfilerConfig& config) :
    m_context(ctx),
    isProfilingActive(0),
    mConfig(config)
{
    m_context = ctx;
    bIsCuptiInitialized = false;
    bIsAllPassSubmitted = false;

    if (!bIsCuptiInitialized)
    {
        CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
        (cuptiProfilerInitialize(&profilerInitializeParams));
        bIsCuptiInitialized = true;
    }
}

inline RangeProfilerTarget::~RangeProfilerTarget()
{
    if (bIsCuptiInitialized)
    {
        bIsCuptiInitialized = false;
    }
}

inline CUptiResult RangeProfilerTarget::EnableRangeProfiler()
{
    CUpti_RangeProfiler_Enable_Params enableRangeProfiler {CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
    enableRangeProfiler.ctx = m_context;
    CUPTI_API_CALL(cuptiRangeProfilerEnable(&enableRangeProfiler));
    rangeProfilerObject = enableRangeProfiler.pRangeProfilerObject;
    return CUPTI_SUCCESS;
}

inline CUptiResult RangeProfilerTarget::CreateCounterDataImage(
    std::vector<const char*>& metrics,
    std::vector<uint8_t>& counterDataImage
)
{
    CUpti_RangeProfiler_GetCounterDataSize_Params getCounterDataSizeParams {CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
    getCounterDataSizeParams.pRangeProfilerObject = rangeProfilerObject;
    getCounterDataSizeParams.pMetricNames = metrics.data();
    getCounterDataSizeParams.numMetrics = metrics.size();
    getCounterDataSizeParams.maxNumOfRanges = mConfig.maxNumOfRanges;
    getCounterDataSizeParams.maxNumRangeTreeNodes = mConfig.maxNumOfRanges;
    CUPTI_API_CALL(cuptiRangeProfilerGetCounterDataSize(&getCounterDataSizeParams));

    counterDataImage.resize(getCounterDataSizeParams.counterDataSize, 0);
    CUpti_RangeProfiler_CounterDataImage_Initialize_Params initializeCounterDataImageParams {CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeCounterDataImageParams.pRangeProfilerObject = rangeProfilerObject;
    initializeCounterDataImageParams.pCounterData = counterDataImage.data();
    initializeCounterDataImageParams.counterDataSize = counterDataImage.size();
    CUPTI_API_CALL(cuptiRangeProfilerCounterDataImageInitialize(&initializeCounterDataImageParams));

    return CUPTI_SUCCESS;
}

inline CUptiResult RangeProfilerTarget::DisableRangeProfiler()
{
    CUpti_RangeProfiler_Disable_Params disableRangeProfiler {CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
    disableRangeProfiler.pRangeProfilerObject   = rangeProfilerObject;
    CUPTI_API_CALL(cuptiRangeProfilerDisable(&disableRangeProfiler));

    isProfilingActive = 0;
    rangeProfilerObject = nullptr;
    return CUPTI_SUCCESS;
}

inline CUptiResult RangeProfilerTarget::StartRangeProfiler()
{
    CUpti_RangeProfiler_Start_Params startRangeProfiler {CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
    startRangeProfiler.pRangeProfilerObject = rangeProfilerObject;
    CUPTI_API_CALL(cuptiRangeProfilerStart(&startRangeProfiler));
    isProfilingActive = 1;
    return CUPTI_SUCCESS;
}

inline CUptiResult RangeProfilerTarget::StopRangeProfiler()
{
    CUpti_RangeProfiler_Stop_Params stopRangeProfiler {CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
    stopRangeProfiler.pRangeProfilerObject      = rangeProfilerObject;
    CUPTI_API_CALL(cuptiRangeProfilerStop(&stopRangeProfiler));
    bIsAllPassSubmitted = stopRangeProfiler.isAllPassSubmitted;
    return CUPTI_SUCCESS;
}

inline CUptiResult RangeProfilerTarget::PushRange(const char* rangeName)
{
    CUpti_RangeProfiler_PushRange_Params pushRangeParams {CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE};
    pushRangeParams.pRangeProfilerObject = rangeProfilerObject;
    pushRangeParams.pRangeName = rangeName;
    CUPTI_API_CALL(cuptiRangeProfilerPushRange(&pushRangeParams));
    return CUPTI_SUCCESS;
}

inline CUptiResult RangeProfilerTarget::PopRange()
{
    CUpti_RangeProfiler_PopRange_Params popRangeParams {CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE};
    popRangeParams.pRangeProfilerObject = rangeProfilerObject;
    CUPTI_API_CALL(cuptiRangeProfilerPopRange(&popRangeParams));
    return CUPTI_SUCCESS;
}

inline CUptiResult RangeProfilerTarget::SetConfig(
    CUpti_ProfilerRange range,
    CUpti_ProfilerReplayMode replayMode,
    std::vector<uint8_t>& configImageBlob,
    std::vector<uint8_t>& counterDataImage
)
{
    configImage.resize(configImageBlob.size());
    std::copy(configImageBlob.begin(), configImageBlob.end(), configImage.begin());

    CUpti_RangeProfiler_SetConfig_Params setConfigParams {CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
    setConfigParams.pRangeProfilerObject = rangeProfilerObject;
    setConfigParams.pConfig = configImage.data();
    setConfigParams.configSize = configImage.size();
    setConfigParams.pCounterDataImage = counterDataImage.data();
    setConfigParams.counterDataImageSize = counterDataImage.size();
    setConfigParams.maxRangesPerPass = mConfig.maxNumOfRanges;
    setConfigParams.numNestingLevels = mConfig.numOfNestingLevel;
    setConfigParams.minNestingLevel = mConfig.minNestingLevel;
    setConfigParams.passIndex = 0;
    setConfigParams.targetNestingLevel = 1;
    setConfigParams.range = range;
    setConfigParams.replayMode = replayMode;
    CUPTI_API_CALL(cuptiRangeProfilerSetConfig(&setConfigParams));
    return CUPTI_SUCCESS;
}

inline CUptiResult RangeProfilerTarget::DecodeCounterData()
{
    CUpti_RangeProfiler_DecodeData_Params decodeDataParams {CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
    decodeDataParams.pRangeProfilerObject = rangeProfilerObject;
    CUPTI_API_CALL(cuptiRangeProfilerDecodeData(&decodeDataParams));
    return CUPTI_SUCCESS;
}

inline CUptiResult RangeProfilerTarget::GetChipName(size_t deviceIndex, std::string& chipName)
{
    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = deviceIndex;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    chipName = getChipNameParams.pChipName;
    return CUPTI_SUCCESS;
}

inline CUptiResult RangeProfilerTarget::GetCounterAvailabilityImage(CUcontext ctx, std::vector<uint8_t>& counterAvailabilityImage)
{
    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = { CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE };
    getCounterAvailabilityParams.ctx = ctx;
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    counterAvailabilityImage.clear();
    counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);
    getCounterAvailabilityParams.pCounterAvailabilityImage = counterAvailabilityImage.data();
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));
    return CUPTI_SUCCESS;
}



