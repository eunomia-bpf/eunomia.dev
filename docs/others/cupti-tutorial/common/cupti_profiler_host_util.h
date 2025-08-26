/**
 * Copyright 2025 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <cupti_profiler_host.h>
#include <cupti_target.h>
#include <cupti_range_profiler.h>
#include <helper_cupti.h>

namespace cupti { namespace utils {

class ProfilerHost
{
public:
    ProfilerHost() = default;
    ~ProfilerHost() = default;

    void setup(
        std::string chipName,
        std::vector<uint8_t>& counterAvailibilityImage,
        CUpti_ProfilerType profilerType
    )
    {
        if (m_pHostObject != nullptr)
        {
            std::cerr << "ProfilerHost already initialized" << std::endl;
            return;
        }

        m_chipName = chipName;
        m_counterAvailibilityImage = counterAvailibilityImage;
        m_profilerType = profilerType;

        CUpti_Profiler_Host_Initialize_Params hostInitializeParams = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
        hostInitializeParams.profilerType = profilerType;
        hostInitializeParams.pChipName = m_chipName.c_str();
        hostInitializeParams.pCounterAvailabilityImage = counterAvailibilityImage.data();
        CUptiResult cuptiStatus = cuptiProfilerHostInitialize(&hostInitializeParams);
        if (cuptiStatus != CUPTI_SUCCESS) {
            std::cerr << "ERROR!! Failed to initialize profiler host.\n";
            CUPTI_API_CALL(cuptiStatus);
        }
        m_pHostObject = hostInitializeParams.pHostObject;
    }

    void teardown()
    {
        if (m_pHostObject == nullptr)
        {
            std::cerr << "ProfilerHost not initialized" << std::endl;
            return;
        }

        CUpti_Profiler_Host_Deinitialize_Params hostDeinitializeParams = {CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        hostDeinitializeParams.pHostObject = m_pHostObject;
        CUptiResult cuptiStatus = cuptiProfilerHostDeinitialize(&hostDeinitializeParams);
        if (cuptiStatus != CUPTI_SUCCESS) {
            std::cerr << "ERROR!! Failed to deinitialize profiler host.\n";
            CUPTI_API_CALL(cuptiStatus);
        }
        m_pHostObject = nullptr;
    }

    void listSupportedChips(
        std::vector<std::string>& chipNames
    )
    {
        CUpti_Profiler_Host_GetSupportedChips_Params hostGetSupportedChipsParams = {CUpti_Profiler_Host_GetSupportedChips_Params_STRUCT_SIZE};
        CUptiResult cuptiStatus = cuptiProfilerHostGetSupportedChips(&hostGetSupportedChipsParams);
        if (cuptiStatus != CUPTI_SUCCESS) {
            std::cerr << "ERROR!! Failed to get supported chips.\n";
            CUPTI_API_CALL(cuptiStatus);
        }

        chipNames.resize(hostGetSupportedChipsParams.numChips);
        for (size_t i = 0; i < hostGetSupportedChipsParams.numChips; i++) {
            chipNames[i] = hostGetSupportedChipsParams.ppChipNames[i];
        }
    }

    static CUptiResult GetChipName(
        size_t deviceIndex,
        std::string& chipName
    )
    {
        CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
        getChipNameParams.deviceIndex = deviceIndex;
        CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
        chipName = getChipNameParams.pChipName;
        return CUPTI_SUCCESS;
    }

    static CUptiResult GetCounterAvailabilityImage(
        CUcontext ctx,
        std::vector<uint8_t>& counterAvailabilityImage
    )
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

protected:

    std::string m_chipName;
    std::vector<uint8_t> m_counterAvailibilityImage;
    CUpti_ProfilerType m_profilerType;
    CUpti_Profiler_Host_Object* m_pHostObject = nullptr;
};

class MetricScheduler : public ProfilerHost
{
public:
    MetricScheduler() = default;

    MetricScheduler(
        CUcontext context,
        CUpti_ProfilerType profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER
    )
    {
        CUdevice device;
        DRIVER_API_CALL(cuCtxGetDevice(&device));

        GetChipName((size_t)device, m_chipName);
        GetCounterAvailabilityImage(context, m_counterAvailibilityImage);

        setup(m_chipName, m_counterAvailibilityImage, profilerType);
    }

    ~MetricScheduler() { teardown(); }

    bool createConfigImage(
        const std::vector<std::string>& metricNames,
        std::vector<uint8_t>& configImage
    )
    {
        // Add metrics to config image
        {
            std::vector<const char*> metricNamesCStr;
            metricNamesCStr.reserve(metricNames.size());
            for (const auto& metricName : metricNames) {
                metricNamesCStr.push_back(metricName.c_str());
            }

            CUpti_Profiler_Host_ConfigAddMetrics_Params configAddMetricsParams {CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
            configAddMetricsParams.pHostObject = m_pHostObject;
            configAddMetricsParams.ppMetricNames = metricNamesCStr.data();
            configAddMetricsParams.numMetrics = metricNames.size();
            CUptiResult cuptiStatus = cuptiProfilerHostConfigAddMetrics(&configAddMetricsParams);
            if (cuptiStatus != CUPTI_SUCCESS) {
                std::cerr << "ERROR!! Failed to add (" << metricNames.size() << ") metrics [" << metricNames[0] << "] to config image.\n";
                return false;
            }
        }

        // Get Config image size and data
        {
            CUpti_Profiler_Host_GetConfigImageSize_Params getConfigImageSizeParams {CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
            getConfigImageSizeParams.pHostObject = m_pHostObject;
            CUptiResult cuptiStatus = cuptiProfilerHostGetConfigImageSize(&getConfigImageSizeParams);
            if (cuptiStatus != CUPTI_SUCCESS) {
                std::cerr << "ERROR!! Failed to get config image size.\n";
                return false;
            }
            configImage.resize(getConfigImageSizeParams.configImageSize);

            CUpti_Profiler_Host_GetConfigImage_Params getConfigImageParams = {CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
            getConfigImageParams.pHostObject = m_pHostObject;
            getConfigImageParams.pConfigImage = configImage.data();
            getConfigImageParams.configImageSize = configImage.size();
            cuptiStatus = cuptiProfilerHostGetConfigImage(&getConfigImageParams);
            if (cuptiStatus != CUPTI_SUCCESS) {
                std::cerr << "ERROR!! Failed to get config image.\n";
                return false;
            }
        }

        return true;
    }
};

class MetricEnumerator : public ProfilerHost
{
public:
    MetricEnumerator() = default;
    ~MetricEnumerator() { teardown(); }

    void listSupportedBaseMetrics(
        std::vector<std::string>& metricNames
    )
    {
        CUpti_Profiler_Host_GetBaseMetrics_Params hostGetBaseCounterMetricsParams = {CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE};
        hostGetBaseCounterMetricsParams.pHostObject = m_pHostObject;
        hostGetBaseCounterMetricsParams.metricType = CUpti_MetricType::CUPTI_METRIC_TYPE_COUNTER;
        CUptiResult cuptiStatus = cuptiProfilerHostGetBaseMetrics(&hostGetBaseCounterMetricsParams);
        if (cuptiStatus != CUPTI_SUCCESS) {
            std::cerr << "ERROR!! Failed to get base counter metrics.\n";
            CUPTI_API_CALL(cuptiStatus);
        }

        // Ratio metrics
        CUpti_Profiler_Host_GetBaseMetrics_Params hostGetBaseRatioMetricsParams = {CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE};
        hostGetBaseRatioMetricsParams.pHostObject = m_pHostObject;
        hostGetBaseRatioMetricsParams.metricType = CUpti_MetricType::CUPTI_METRIC_TYPE_RATIO;
        cuptiStatus = cuptiProfilerHostGetBaseMetrics(&hostGetBaseRatioMetricsParams);
        if (cuptiStatus != CUPTI_SUCCESS) {
            std::cerr << "ERROR!! Failed to get base ratio metrics.\n";
            CUPTI_API_CALL(cuptiStatus);
        }

        // Throughput metrics
        CUpti_Profiler_Host_GetBaseMetrics_Params hostGetBaseThroughputMetricsParams = {CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE};
        hostGetBaseThroughputMetricsParams.pHostObject = m_pHostObject;
        hostGetBaseThroughputMetricsParams.metricType = CUpti_MetricType::CUPTI_METRIC_TYPE_THROUGHPUT;
        cuptiStatus = cuptiProfilerHostGetBaseMetrics(&hostGetBaseThroughputMetricsParams);
        if (cuptiStatus != CUPTI_SUCCESS) {
            std::cerr << "ERROR!! Failed to get base throughput metrics.\n";
            CUPTI_API_CALL(cuptiStatus);
        }

        const size_t numMetrics = hostGetBaseCounterMetricsParams.numMetrics + hostGetBaseRatioMetricsParams.numMetrics + hostGetBaseThroughputMetricsParams.numMetrics;
        metricNames.resize(numMetrics);

        size_t metricOffset = 0;
        // fill counter metrics
        for (size_t i = metricOffset; i < metricOffset + hostGetBaseCounterMetricsParams.numMetrics; i++) {
            metricNames[i] = hostGetBaseCounterMetricsParams.ppMetricNames[i];
        }

        // fill ratio metrics
        metricOffset += hostGetBaseCounterMetricsParams.numMetrics;
        for (size_t i = metricOffset; i < metricOffset + hostGetBaseRatioMetricsParams.numMetrics; i++) {
            metricNames[i] = hostGetBaseRatioMetricsParams.ppMetricNames[i - metricOffset];
        }

        // fill throughput metrics
        metricOffset += hostGetBaseRatioMetricsParams.numMetrics;
        for (size_t i = metricOffset; i < metricOffset + hostGetBaseThroughputMetricsParams.numMetrics; i++) {
            metricNames[i] = hostGetBaseThroughputMetricsParams.ppMetricNames[i - metricOffset];
        }
    }

    void listMetricProperties(
        const std::string& metricName,
        CUpti_MetricType& metricType,
        std::string& description,
        std::string& hwUnit,
        std::string& dimUnits,
        std::string& numPasses,
        bool listNumPasses = false
    )
    {
        CUpti_Profiler_Host_GetMetricProperties_Params hostGetMetricPropertiesParams = {CUpti_Profiler_Host_GetMetricProperties_Params_STRUCT_SIZE};
        hostGetMetricPropertiesParams.pHostObject = m_pHostObject;
        hostGetMetricPropertiesParams.pMetricName = metricName.c_str();
        CUptiResult cuptiStatus = cuptiProfilerHostGetMetricProperties(&hostGetMetricPropertiesParams);
        if (cuptiStatus != CUPTI_SUCCESS) {
            std::cerr << "ERROR!! Failed to get metric properties for " << metricName << "\n";
            CUPTI_API_CALL(cuptiStatus);
        }

        metricType = hostGetMetricPropertiesParams.metricType;
        description = hostGetMetricPropertiesParams.pDescription;
        hwUnit = hostGetMetricPropertiesParams.pHwUnit;
        dimUnits = hostGetMetricPropertiesParams.pDimUnit;

        if (listNumPasses)
        {
            std::string metricNameWithSubmetrics = metricName;
            if (hostGetMetricPropertiesParams.metricType == CUpti_MetricType::CUPTI_METRIC_TYPE_COUNTER) {
                metricNameWithSubmetrics = metricName + ".sum";
            } else if (hostGetMetricPropertiesParams.metricType == CUpti_MetricType::CUPTI_METRIC_TYPE_RATIO) {
                metricNameWithSubmetrics = metricName + ".pct";
            } else if (hostGetMetricPropertiesParams.metricType == CUpti_MetricType::CUPTI_METRIC_TYPE_THROUGHPUT) {
                metricNameWithSubmetrics = metricName + ".sum.pct_of_peak_sustained_active";
            }

            numPasses = std::to_string(getNumOfPasses({metricNameWithSubmetrics}));
        }
    }

    void listSubmetrics(
        const std::string& metricName,
        CUpti_MetricType metricType,
        std::vector<std::string>& submetricNames
    )
    {
        CUpti_Profiler_Host_GetSubMetrics_Params hostGetSubmetricsParams = {CUpti_Profiler_Host_GetSubMetrics_Params_STRUCT_SIZE};
        hostGetSubmetricsParams.pHostObject = m_pHostObject;
        hostGetSubmetricsParams.pMetricName = metricName.c_str();
        hostGetSubmetricsParams.metricType = metricType;
        CUptiResult cuptiStatus = cuptiProfilerHostGetSubMetrics(&hostGetSubmetricsParams);
        if (cuptiStatus != CUPTI_SUCCESS) {
            std::cerr << "ERROR!! Failed to get submetrics for " << metricName << ".\n";
            CUPTI_API_CALL(cuptiStatus);
        }

        submetricNames.resize(hostGetSubmetricsParams.numOfSubmetrics);
        for (size_t i = 0; i < hostGetSubmetricsParams.numOfSubmetrics; i++) {
            submetricNames[i] = hostGetSubmetricsParams.ppSubMetrics[i];
        }
    }

    uint32_t getNumOfPasses(
        const std::vector<std::string>& metricNames
    )
    {
        std::vector<uint8_t> configImage;
        MetricScheduler metricScheduler;
        metricScheduler.setup(m_chipName, m_counterAvailibilityImage, m_profilerType);
        if (!metricScheduler.createConfigImage(metricNames, configImage)) {
            return 0;
        }

        CUpti_Profiler_Host_GetNumOfPasses_Params hostGetNumPassesParams = {CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
        hostGetNumPassesParams.pConfigImage = configImage.data();
        hostGetNumPassesParams.configImageSize = configImage.size();
        CUptiResult cuptiStatus = cuptiProfilerHostGetNumOfPasses(&hostGetNumPassesParams);
        if (cuptiStatus != CUPTI_SUCCESS) {
            std::cerr << "ERROR!! Failed to get number of passes for " << metricNames[0] << ".\n";
            return 0;
        }
        return hostGetNumPassesParams.numOfPasses;
    }

};

class MetricEvaluator : public ProfilerHost
{
public:
    struct MetricValuePair
    {
        std::string metricName;
        double value;
    };

    struct RangeInfo
    {
        std::string rangeName;
        std::vector<MetricValuePair> metricAndValues;
    };

    MetricEvaluator() = default;

    MetricEvaluator(
        CUcontext context,
        CUpti_ProfilerType profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER
    )
    {
        CUdevice device;
        DRIVER_API_CALL(cuCtxGetDevice(&device));

        GetChipName((size_t)device, m_chipName);
        GetCounterAvailabilityImage(context, m_counterAvailibilityImage);

        setup(m_chipName, m_counterAvailibilityImage, profilerType);
    }

    ~MetricEvaluator() { teardown(); }

    uint32_t getNumOfRanges(
        const std::vector<uint8_t>& counterDataImage
    )
    {
        CUpti_RangeProfiler_GetCounterDataInfo_Params getCounterDataInfoParams = {CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE};
        getCounterDataInfoParams.pCounterDataImage = counterDataImage.data();
        getCounterDataInfoParams.counterDataImageSize = counterDataImage.size();
        CUPTI_API_CALL(cuptiRangeProfilerGetCounterDataInfo(&getCounterDataInfoParams));
        return getCounterDataInfoParams.numTotalRanges;
    }

    void getRangeName(
        uint32_t rangeIndex,
        std::string& rangeName,
        const std::vector<uint8_t>& counterDataImage
    )
    {
        CUpti_RangeProfiler_CounterData_GetRangeInfo_Params getRangeInfoParams = {CUpti_RangeProfiler_CounterData_GetRangeInfo_Params_STRUCT_SIZE};
        getRangeInfoParams.counterDataImageSize = counterDataImage.size();
        getRangeInfoParams.pCounterDataImage = counterDataImage.data();
        getRangeInfoParams.rangeIndex = rangeIndex;
        getRangeInfoParams.rangeDelimiter = "/";
        CUPTI_API_CALL(cuptiRangeProfilerCounterDataGetRangeInfo(&getRangeInfoParams));
        rangeName = getRangeInfoParams.rangeName;
    }

    void evaluateMetricsForRange(
        const std::vector<uint8_t>& counterDataImage,
        std::vector<const char*>& metricNames,
        std::vector<double>& metricValues, uint32_t rangeIndex
    )
    {
        CUpti_Profiler_Host_EvaluateToGpuValues_Params evalauateToGpuValuesParams {CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
        evalauateToGpuValuesParams.pHostObject = m_pHostObject;
        evalauateToGpuValuesParams.pCounterDataImage = counterDataImage.data();
        evalauateToGpuValuesParams.counterDataImageSize = counterDataImage.size();
        evalauateToGpuValuesParams.ppMetricNames = metricNames.data();
        evalauateToGpuValuesParams.numMetrics = metricNames.size();
        evalauateToGpuValuesParams.rangeIndex = rangeIndex;
        evalauateToGpuValuesParams.pMetricValues = metricValues.data();
        CUPTI_API_CALL(cuptiProfilerHostEvaluateToGpuValues(&evalauateToGpuValuesParams));
    }

    void evaluateAllRanges(
        const std::vector<uint8_t>& counterDataImage,
        const std::vector<std::string>& metricNames,
        std::vector<RangeInfo>& rangeInfos
    )
    {
        std::vector<const char*> metricNamesCStr(metricNames.size());
        std::transform(metricNames.begin(), metricNames.end(), metricNamesCStr.begin(), [](const std::string& metricName) { return metricName.c_str(); });

        const uint32_t numOfRanges = getNumOfRanges(counterDataImage);
        rangeInfos.resize(numOfRanges);
        for (uint32_t rangeIndex = 0; rangeIndex < numOfRanges; rangeIndex++)
        {
            std::string rangeName;
            getRangeName(rangeIndex, rangeName, counterDataImage);

            std::vector<double> metricValues(metricNames.size());
            evaluateMetricsForRange(counterDataImage, metricNamesCStr, metricValues, rangeIndex);

            for (size_t i = 0; i < metricNames.size(); i++)
            {
                MetricValuePair metricValuePair;
                metricValuePair.metricName = metricNames[i];
                metricValuePair.value = metricValues[i];
                rangeInfos[rangeIndex].metricAndValues.push_back(metricValuePair);
            }
            rangeInfos[rangeIndex].rangeName = rangeName;
        }
    }

    void printMetricData(
        const std::vector<uint8_t>& counterDataImage,
        const std::vector<const char*>& metricNames
    )
    {
        std::vector<std::string> metricNamesStr;
        std::transform(metricNames.begin(), metricNames.end(), std::back_inserter(metricNamesStr), [](const char* metric) { return std::string(metric); });
        printMetricData(counterDataImage, metricNamesStr);
    }

    void printMetricData(
        const std::vector<uint8_t>& counterDataImage,
        const std::vector<std::string>& metricNames
    )
    {
        std::vector<RangeInfo> rangeInfos;
        evaluateAllRanges(counterDataImage, metricNames, rangeInfos);
        printMetricData(rangeInfos);
    }

    void printMetricData(
        const std::vector<RangeInfo>& rangeInfos
    )
    {
        std::cout << "Total num of Ranges: " << rangeInfos.size() << "\n\n";
        for(size_t rangeIndex = 0; rangeIndex < rangeInfos.size(); ++rangeIndex)
        {
            const auto& rangeInfo = rangeInfos[rangeIndex];
            std::cout << "Range Name: " << rangeInfo.rangeName << "\n";
            std::cout << "-----------------------------------------------------------------------------------\n";
            for (const auto& metric : rangeInfo.metricAndValues)
            {
                std::cout << std::fixed << std::setprecision(3);
                std::cout << std::setw(50) << std::left << metric.metricName;
                std::cout << std::setw(30) << std::right << metric.value << "\n";
            }
            std::cout << "-----------------------------------------------------------------------------------\n\n";
        }
    }
};

} } // namespace cupti::utils
