// Copyright 2020-2022 NVIDIA Corporation. All rights reserved
//
// The sample provides the generic workflow for querying various properties of metrics which are available as part of
// the Profiling APIs. In this particular case we are querying for number of passes, collection method, metric type and the 
// hardware unit associated for a list of metrics.
//
// Number of passes : It gives the number of passes required for collection of the metric as some of the metric
// cannot be collected in single pass due to hardware or software limitation, we need to replay the exact same
// set of GPU workloads multiple times.
//
// Collection method : It gives the source of the metric (HW or SW). Most of metric are provided by hardware but for
// some metric we have to instrument the kernel to collect the metric. Further these metrics cannot be combined with
// any other metrics in the same pass as otherwise instrumented code will also contribute to the metric value.
//

// System headers
#include <cstring>
#include <memory>
#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <vector>

// CUPTI headers
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include "helper_cupti.h"

// NVPW headers
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>

#include <Parser.h>
#include <Utils.h>

static std::unordered_map<NVPW_MetricType, const char*> typeToStrMap = {
                                                                        {NVPW_METRIC_TYPE_COUNTER,        "Counter"},
                                                                        {NVPW_METRIC_TYPE_RATIO,            "Ratio"},
                                                                        {NVPW_METRIC_TYPE_THROUGHPUT,  "Throughput"}
                                                                        };

static std::unordered_map<NVPW_RollupOp, const char*> rollUpToStrMap = {
                                                                            {NVPW_ROLLUP_OP_AVG, ".avg"},
                                                                            {NVPW_ROLLUP_OP_SUM, ".sum"},
                                                                            {NVPW_ROLLUP_OP_MIN, ".min"},
                                                                            {NVPW_ROLLUP_OP_MAX, ".max"}
                                                                        };

static std::unordered_map<NVPW_Submetric, const char*> submetricToStrMap = {
                                                                            {NVPW_SUBMETRIC_PEAK_SUSTAINED, ".peak_sustained"},
                                                                            {NVPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE, ".peak_sustained_active"},
                                                                            {NVPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE_PER_SECOND, ".peak_sustained_active.per_second"},
                                                                            {NVPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED, ".peak_sustained_elapsed"},
                                                                            {NVPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED_PER_SECOND, ".peak_sustained_elapsed.per_second"},
                                                                            {NVPW_SUBMETRIC_PER_SECOND, ".per_second"},
                                                                            {NVPW_SUBMETRIC_PER_CYCLE_ACTIVE, ".per_cycle_active"},
                                                                            {NVPW_SUBMETRIC_PER_CYCLE_ELAPSED, ".per_cycle_elapsed"},
                                                                            {NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ACTIVE, ".pct_of_peak_sustained_active"},
                                                                            {NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ELAPSED, ".pct_of_peak_sustained_elapsed"},
                                                                            {NVPW_SUBMETRIC_MAX_RATE, ".max_rate"},
                                                                            {NVPW_SUBMETRIC_PCT, ".pct"},
                                                                            {NVPW_SUBMETRIC_RATIO, ".ratio"},
                                                                        };

struct MetricDetails
{
    const char* name;
    const char* description;
    const char* type;
    const char* hwUnit;
    std::string collectionType;
    size_t numOfPasses;
    std::vector<std::string> submetrics;
};

struct ApplicationParams
{
    std::vector<MetricDetails> metrics;
    const char* chipName = NULL;
    std::vector<uint8_t> counterAvailabilityImage;
    bool bListSubMetrics = false;
};

class MetricEvaluator
{
public:
    explicit MetricEvaluator(const char* pChipName, uint8_t* pCounterAvailabilityImage)
    {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
        calculateScratchBufferSizeParam.pChipName = pChipName;
        calculateScratchBufferSizeParam.pCounterAvailabilityImage = pCounterAvailabilityImage;
        NVPA_Status nvpwResult = NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calculateScratchBufferSizeParam);
        if (nvpwResult != NVPA_STATUS_SUCCESS) {
            std::cerr << "Failed to calculate Scratch buffer size with error "
                      << NV::Metric::Utils::GetNVPWResultString(nvpwResult) << "\n";
            exit(EXIT_FAILURE);
        }

        m_scratchBuffer.resize(calculateScratchBufferSizeParam.scratchBufferSize);
        NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
        metricEvaluatorInitializeParams.scratchBufferSize = m_scratchBuffer.size();
        metricEvaluatorInitializeParams.pScratchBuffer = m_scratchBuffer.data();
        metricEvaluatorInitializeParams.pChipName = pChipName;
        metricEvaluatorInitializeParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
        nvpwResult = NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams);
        if (nvpwResult != NVPA_STATUS_SUCCESS) {
            std::cerr << "Failed to initialize metric evaluator "
                      << NV::Metric::Utils::GetNVPWResultString(nvpwResult) << "\n";
            exit(EXIT_FAILURE);
        }
        m_pNVPWMetricEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;
    }

    bool ListAllMetrics(std::vector<MetricDetails>& metrics)
    {
        for (auto i = 0; i < NVPW_MetricType::NVPW_METRIC_TYPE__COUNT; ++i)
        {
            NVPW_MetricType metricType = static_cast<NVPW_MetricType>(i);
            NVPW_MetricsEvaluator_GetMetricNames_Params getMetricNamesParams = { NVPW_MetricsEvaluator_GetMetricNames_Params_STRUCT_SIZE };
            getMetricNamesParams.metricType = metricType;
            getMetricNamesParams.pMetricsEvaluator = m_pNVPWMetricEvaluator;
            RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetMetricNames(&getMetricNamesParams));

            for (size_t metricIndex = 0; metricIndex < getMetricNamesParams.numMetrics; ++metricIndex)
            {
                size_t metricNameBeginIndex = getMetricNamesParams.pMetricNameBeginIndices[metricIndex];
                const char* metricName = &getMetricNamesParams.pMetricNames[metricNameBeginIndex];

                MetricDetails metric = {};
                metric.name = metricName;
                GetMetricProperties(metric, metricType, metricIndex);
                metric.collectionType = GetMetricCollectionMethod(metricName);
                metrics.push_back(metric);
            }
        }
        return true;
    }

    bool GetMetricProperties(MetricDetails& metric, NVPW_MetricType metricType, size_t metricIndex)
    {
        NVPA_Status status = NVPA_STATUS_SUCCESS;
        NVPW_HwUnit hwUnit = NVPW_HW_UNIT_INVALID;
        if (metricType == NVPW_MetricType::NVPW_METRIC_TYPE_COUNTER)
        {
            NVPW_MetricsEvaluator_GetCounterProperties_Params counterPropParams {NVPW_MetricsEvaluator_GetCounterProperties_Params_STRUCT_SIZE};
            counterPropParams.pMetricsEvaluator = m_pNVPWMetricEvaluator;
            counterPropParams.counterIndex = metricIndex;
            RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetCounterProperties(&counterPropParams));
            metric.description = counterPropParams.pDescription;
            hwUnit = (NVPW_HwUnit)counterPropParams.hwUnit;
        }
        else if (metricType == NVPW_MetricType::NVPW_METRIC_TYPE_RATIO)
        {
            NVPW_MetricsEvaluator_GetRatioMetricProperties_Params ratioPropParams {NVPW_MetricsEvaluator_GetRatioMetricProperties_Params_STRUCT_SIZE};
            ratioPropParams.pMetricsEvaluator = m_pNVPWMetricEvaluator;
            ratioPropParams.ratioMetricIndex = metricIndex;
            RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetRatioMetricProperties(&ratioPropParams));
            metric.description = ratioPropParams.pDescription;
            hwUnit = (NVPW_HwUnit)ratioPropParams.hwUnit;
        }
        else if (metricType == NVPW_MetricType::NVPW_METRIC_TYPE_THROUGHPUT)
        {
            NVPW_MetricsEvaluator_GetThroughputMetricProperties_Params throughputPropParams {NVPW_MetricsEvaluator_GetThroughputMetricProperties_Params_STRUCT_SIZE};
            throughputPropParams.pMetricsEvaluator = m_pNVPWMetricEvaluator;
            throughputPropParams.throughputMetricIndex = metricIndex;
            RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetThroughputMetricProperties(&throughputPropParams));
            metric.description = throughputPropParams.pDescription;
            hwUnit = (NVPW_HwUnit)throughputPropParams.hwUnit;
        }

        NVPW_MetricsEvaluator_HwUnitToString_Params hwUnitToStrParams {NVPW_MetricsEvaluator_HwUnitToString_Params_STRUCT_SIZE};
        hwUnitToStrParams.pMetricsEvaluator = m_pNVPWMetricEvaluator;
        hwUnitToStrParams.hwUnit = hwUnit;
        RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_HwUnitToString(&hwUnitToStrParams));
        metric.hwUnit = hwUnitToStrParams.pHwUnitName;
        metric.type = typeToStrMap[metricType];
        return true;
    }

    bool GetMetricTypeAndIndex(const char* metricName, NVPW_MetricType& metricType, size_t& metricIndex)
    {
        NVPW_MetricsEvaluator_GetMetricTypeAndIndex_Params getMetricTypeAndIndexParam {NVPW_MetricsEvaluator_GetMetricTypeAndIndex_Params_STRUCT_SIZE};
        getMetricTypeAndIndexParam.pMetricName = metricName;
        getMetricTypeAndIndexParam.pMetricsEvaluator = m_pNVPWMetricEvaluator;
        RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetMetricTypeAndIndex(&getMetricTypeAndIndexParam));
        metricType = (NVPW_MetricType)getMetricTypeAndIndexParam.metricType;
        metricIndex = getMetricTypeAndIndexParam.metricIndex;
        return true;
    }

    bool GetSubmetrics(const char* metricName, NVPW_MetricType metricType, std::vector<std::string>& submetrics)
    {
        for (size_t rollupOpIndex = 0; rollupOpIndex < NVPW_RollupOp::NVPW_ROLLUP_OP__COUNT; ++rollupOpIndex)
        {
            // Only throughput and counter metrics will have roll ups (min, max, sum, avg)
            if (metricType != NVPW_MetricType::NVPW_METRIC_TYPE_RATIO)
            {
                const char* rollupStr = rollUpToStrMap[(NVPW_RollupOp)rollupOpIndex];
                submetrics.push_back(rollupStr);
            }

            if (metricType != NVPW_MetricType::NVPW_METRIC_TYPE_RATIO)
            {
                const char* submetric = rollUpToStrMap[(NVPW_RollupOp)rollupOpIndex];
                NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params getSupportedSubmetricsParmas = { NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params_STRUCT_SIZE };
                getSupportedSubmetricsParmas.metricType = metricType;
                getSupportedSubmetricsParmas.pMetricsEvaluator = m_pNVPWMetricEvaluator;
                RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetSupportedSubmetrics(&getSupportedSubmetricsParmas));
                for (size_t submetricIndex = 0; submetricIndex < getSupportedSubmetricsParmas.numSupportedSubmetrics; ++submetricIndex)
                {
                    auto nvpwSubMetric = (NVPW_Submetric)getSupportedSubmetricsParmas.pSupportedSubmetrics[submetricIndex];
                    auto submetricNameItr = submetricToStrMap.find(nvpwSubMetric);
                    if (submetricNameItr != submetricToStrMap.end()) {
                        std::string submetricName = std::string(submetric) + std::string(submetricNameItr->second);
                        submetrics.push_back(submetricName);
                    }
                }
            }
        }
        return true;
    }

    bool GetRawMetricRequests(std::string metricName, std::vector<NVPA_RawMetricRequest>& rawMetricRequests)
    {
        std::string reqName;
        bool isolated = true;
        bool keepInstances = true;
        std::vector<const char*> rawMetricNames;
        NV::Metric::Parser::ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
        keepInstances = true;

        NVPW_MetricEvalRequest metricEvalRequest;
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertMetricToEvalRequest = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
        convertMetricToEvalRequest.pMetricsEvaluator = m_pNVPWMetricEvaluator;
        convertMetricToEvalRequest.pMetricName = reqName.c_str();
        convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
        convertMetricToEvalRequest.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        NVPA_Status nvpwResult = NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&convertMetricToEvalRequest);
        if (nvpwResult != NVPA_STATUS_SUCCESS) {
            printf("ERROR!! Failed to create metric eval request from metric name.\n");
            printf("Possibly Invalid metric name, Make sure the metric has submetric or rollups\n");
            return false;
        }

        std::vector<const char*> rawDependencies;
        NVPW_MetricsEvaluator_GetMetricRawDependencies_Params getMetricRawDependenciesParms = {NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE};
        getMetricRawDependenciesParms.pMetricsEvaluator = m_pNVPWMetricEvaluator;
        getMetricRawDependenciesParms.pMetricEvalRequests = &metricEvalRequest;
        getMetricRawDependenciesParms.numMetricEvalRequests = 1;
        getMetricRawDependenciesParms.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        getMetricRawDependenciesParms.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
        RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetMetricRawDependencies(&getMetricRawDependenciesParms));
        rawDependencies.resize(getMetricRawDependenciesParms.numRawDependencies);
        getMetricRawDependenciesParms.ppRawDependencies = rawDependencies.data();
        RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetMetricRawDependencies(&getMetricRawDependenciesParms));

        for (size_t i = 0; i < rawDependencies.size(); ++i) {
            rawMetricNames.push_back(rawDependencies[i]);
        }

        for (auto& rawMetricName : rawMetricNames)
        {
            NVPA_RawMetricRequest metricRequest = { NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE };
            metricRequest.pMetricName = rawMetricName;
            metricRequest.isolated = isolated;
            metricRequest.keepInstances = keepInstances;
            rawMetricRequests.push_back(metricRequest);
        }
        return true;
    }

    std::string GetMetricCollectionMethod(std::string metricName)
    {
        const std::string SW_CHECK = "sass";
        if (metricName.find(SW_CHECK) != std::string::npos) {
            return "SW";
        }
        return "HW";
    }

    ~MetricEvaluator()
    {
        NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = { NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE };
        metricEvaluatorDestroyParams.pMetricsEvaluator = m_pNVPWMetricEvaluator;
        NVPA_Status nvpwResult = NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams);
        if (nvpwResult != NVPA_STATUS_SUCCESS) {
            std::cerr << "Failed to destroy metric evaluator\n";
            exit(EXIT_FAILURE);
        }
    }

private:
    NVPW_MetricsEvaluator* m_pNVPWMetricEvaluator;
    std::vector<uint8_t> m_scratchBuffer;
};

class MetricConfig
{
public:
    explicit MetricConfig(const char* pChipName, uint8_t* pCounterAvailabilityImage)
    {
        NVPW_CUDA_RawMetricsConfig_Create_V2_Params rawMetricsConfigCreateParams = { NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE };
        rawMetricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
        rawMetricsConfigCreateParams.pChipName = pChipName;
        rawMetricsConfigCreateParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
        NVPA_Status nvpwResult = NVPW_CUDA_RawMetricsConfig_Create_V2(&rawMetricsConfigCreateParams);
        if (nvpwResult != NVPA_STATUS_SUCCESS) {
            std::cerr << "Failed to create raw metric config\n";
            exit(EXIT_FAILURE);
        }
        pRawMetricsConfig = rawMetricsConfigCreateParams.pRawMetricsConfig;
        mChipName = std::string(pChipName);
    }

    bool GetNumOfPasses(const std::vector<const char*>& metrics, MetricEvaluator* pMetricEvaluator, size_t& numOfPasses)
    {
        NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
        beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
        RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams));

        for (auto& metric : metrics)
        {
            std::vector<NVPA_RawMetricRequest> rawMetricRequests;
            if (!pMetricEvaluator->GetRawMetricRequests(metric, rawMetricRequests)) {
                printf("Error: Failed to get raw metrics.\n");
                return false;
            }

            NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
            addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
            addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
            addMetricsParams.numMetricRequests = rawMetricRequests.size();
            RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams));
        }

        NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
        endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
        RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams));

        NVPW_RawMetricsConfig_GetNumPasses_Params rawMetricsConfigGetNumPassesParams = { NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE };
        rawMetricsConfigGetNumPassesParams.pRawMetricsConfig = pRawMetricsConfig;
        RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GetNumPasses(&rawMetricsConfigGetNumPassesParams));

        // No Nesting of ranges in case of CUPTI_AutoRange, in AutoRange
        // the range is already at finest granularity of every kernel Launch so numNestingLevels = 1
        size_t numNestingLevels = 1;
        size_t numIsolatedPasses = rawMetricsConfigGetNumPassesParams.numIsolatedPasses;
        size_t numPipelinedPasses = rawMetricsConfigGetNumPassesParams.numPipelinedPasses;
        numOfPasses = numPipelinedPasses + numIsolatedPasses * numNestingLevels;
        return true;
    }

    ~MetricConfig()
    {
        NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
        rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
        NVPA_Status nvpwResult = NVPW_RawMetricsConfig_Destroy((NVPW_RawMetricsConfig_Destroy_Params*)&rawMetricsConfigDestroyParams);
        if (nvpwResult != NVPA_STATUS_SUCCESS) {
            std::cerr << "Failed to destroy raw metric config\n";
            exit(EXIT_FAILURE);
        }
    }

private:
    NVPA_RawMetricsConfig *pRawMetricsConfig = NULL;
    std::string mChipName;
};

void ExportMetricData(const ApplicationParams& appParams, bool listAllMetrics);

void PrintHelp()
{
    std::cout << "./cupti_metric_properties \n" <<
                                        "\t\t\t--device [device num] \n" <<
                                        "\t\t\t--chip [chip name] \n" <<
                                        "\t\t\t--metrics [list of metrics separated with comma] \n" <<
                                        "\t\t\t\t--list-submetrics\n";

    std::cout   << "Notes:\n"
                << "\t1) --list-submetrics option is only valid when a list of metrics are passed using the --metrics option \n"
                << "\t2) The --device flag will be ingroned when the --chip flag is also used for getting metric properties for chip.\n";

    std::cout   << "Usage:\n"
                << "\t ./cupti_metric_properties --help\n"
                << "\t ./cupti_metric_properties --chip GA100\n"
                << "\t ./cupti_metric_properties --metrics sm__mioc_inst_issued.sum,sm__sass_inst_executed.avg\n"
                << "\t ./cupti_metric_properties --metrics sm__mioc_inst_issued.sum,sm__sass_inst_executed.avg --list-submetrics\n";
}

const char* GetChipNameForDevice(uint32_t deviceNum)
{
    int deviceCount = 0;
    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0 || deviceCount < (deviceNum+1))
    {
        printf("Invalid CUDA device\n");
        exit(EXIT_WAIVED);
    }
    printf("CUDA Device Number: %d\n", deviceNum);

    CUdevice cuDevice;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));

    // Initialize profiler API and test device compatibility
    CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = deviceNum;
    params.api = CUPTI_PROFILER_RANGE_PROFILING;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Unable to profile on device " << deviceNum << ::std::endl;
        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
            ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
            ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
            ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
        } else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED) {
            ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
            ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
            ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
        }

        if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
            ::std::cerr << "\tWSL is not supported" << ::std::endl;
        }
        exit(EXIT_WAIVED);
    }

    /* Get chip name for the cuda  device */
    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = deviceNum;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    return getChipNameParams.pChipName;
}

CUptiResult GetCounterAvailabilityImage(std::vector<uint8_t>& counterAvailabilityImage)
{
    DRIVER_API_CALL(cuInit(0));
    CUcontext ctx;
    DRIVER_API_CALL(cuCtxCreate(&ctx, 0, 0));

    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = { CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE };
    getCounterAvailabilityParams.ctx = ctx;
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);
    getCounterAvailabilityParams.counterAvailabilityImageSize = counterAvailabilityImage.size();
    getCounterAvailabilityParams.pCounterAvailabilityImage = counterAvailabilityImage.data();
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    DRIVER_API_CALL(cuCtxDestroy(ctx));
    return CUPTI_SUCCESS;
}

void ParseArgs(int argc, char *argv[], ApplicationParams& appParams)
{
    int deviceNum = 0;
    for (int i = 1; i < argc; ++i)
    {
        char *arg = argv[i];
        if (strcmp(arg, "--help") == 0)
        {
            PrintHelp();
            exit(EXIT_SUCCESS);
        }
        if (strcmp(arg, "--device") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! Add device number for querying metrics details.\n");
                exit(EXIT_FAILURE);
            }
            deviceNum = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(arg, "--chip") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! Add chip name for querying metrics details.\n");
                exit(EXIT_FAILURE);
            }
            appParams.chipName = argv[i + 1];
            i++;
        }
        else if (strcmp(arg, "--metrics") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! Add metric names for querying metrics details.\n");
                exit(EXIT_FAILURE);
            }
            const char* metricName = strtok(argv[i+1], ",");
            while (metricName != NULL)
            {
                MetricDetails metric = {};
                metric.name = metricName;
                appParams.metrics.push_back(metric);
                metricName = strtok(NULL, ",");
            }
            i++;
        }
        else if (strcmp(arg, "--list-submetrics") == 0)
        {
            appParams.bListSubMetrics = true;
            i++;
        }
        else
        {
            PrintHelp();
            exit(EXIT_SUCCESS);
        }
    }

    if (!appParams.chipName) {
        appParams.chipName = GetChipNameForDevice(deviceNum);
        CUPTI_API_CALL(GetCounterAvailabilityImage(appParams.counterAvailabilityImage));
    }
    printf("Chip Name: %s\n", appParams.chipName);

    if (appParams.metrics.empty() && appParams.bListSubMetrics) {
        printf("ERROR!! pass metrics names in --metrics flag for listing sub-metrics\n");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    ApplicationParams appParams;
    ParseArgs(argc, argv, appParams);

    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    RETURN_IF_NVPW_ERROR(false, NVPW_InitializeHost(&initializeHostParams));

    std::unique_ptr<MetricEvaluator> metricEvaluatorPtr = std::make_unique<MetricEvaluator>(appParams.chipName, appParams.counterAvailabilityImage.data());
    const bool bListAllMetrics = (appParams.metrics.empty() ? true : false);
    if (bListAllMetrics)
    {
        if (!metricEvaluatorPtr->ListAllMetrics(appParams.metrics)) {
            std::cerr << "Failed to List all the metrics for " << appParams.chipName << " chip.\n";
            exit(EXIT_FAILURE);
        }
        std::cout << "Num of metrics supported: " << appParams.metrics.size() << ".\n";
    }
    else
    {
        for (auto& metric : appParams.metrics)
        {
            size_t metricIndex = 0;
            NVPW_MetricType metricType = NVPW_MetricType::NVPW_METRIC_TYPE__COUNT;
            metricEvaluatorPtr->GetMetricTypeAndIndex(metric.name, metricType, metricIndex);
            metricEvaluatorPtr->GetMetricProperties(metric, metricType, metricIndex);
            metric.collectionType = metricEvaluatorPtr->GetMetricCollectionMethod(metric.name);
            if (appParams.bListSubMetrics) {
                metricEvaluatorPtr->GetSubmetrics(metric.name, metricType, metric.submetrics);
            }

            std::unique_ptr<MetricConfig> metricConfigPtr = std::make_unique<MetricConfig>(appParams.chipName, appParams.counterAvailabilityImage.data());
            if (!metricConfigPtr->GetNumOfPasses({metric.name}, metricEvaluatorPtr.get(), metric.numOfPasses)) {
                printf("Failed to get Number of passes for metric.\n");
                exit(EXIT_FAILURE);
            }
        }

        std::vector<const char*> metricNames;
        for (auto& metric : appParams.metrics) {
            metricNames.push_back(metric.name);
        }

        size_t numOfPassesInTotal;
        std::unique_ptr<MetricConfig> metricConfigPtr = std::make_unique<MetricConfig>(appParams.chipName, appParams.counterAvailabilityImage.data());
        if (!metricConfigPtr->GetNumOfPasses(metricNames, metricEvaluatorPtr.get(), numOfPassesInTotal)) {
            printf("Failed to get Number of passes for metric.\n");
            exit(EXIT_FAILURE);
        }
        printf("Num of Passes required in total for collecting listed metrics in one profiling session: %d\n", (int)numOfPassesInTotal);
    }

    ExportMetricData(appParams, bListAllMetrics);
    return 0;
}

void ExportMetricData(const ApplicationParams& appParams, bool listAllMetrics)
{
    std::cout << std::setfill('-') << std::setw(200) << "" << std::setfill(' ') << std::endl;
    std::cout   << std::setw(80) << std::left << "Metric Name"  << "\t"
                    << std::setw(10) << std::left << "HW Unit" << "\t"
                    << std::setw(15) << std::left << "Type" << "\t"
                    << std::setw(10) << std::left << "Collection" << "\t";
    if (!listAllMetrics)
        std::cout   << std::setw(10) << std::left << "Passes" << "\t";
    std::cout       << std::setw(10) << std::left << "Description" << "\n";
    std::cout << std::setfill('-') << std::setw(200) << "" << std::setfill(' ') << std::endl;
    for (auto& metric : appParams.metrics)
    {
        std::cout   << std::setw(80) << std::left << metric.name  << "\t"
                    << std::setw(10) << std::left << metric.hwUnit << "\t"
                    << std::setw(15) << std::left << metric.type << "\t"
                    << std::setw(10) << std::left << metric.collectionType << "\t";
        if (!listAllMetrics)
            std::cout   << std::setw(10) << std::left << metric.numOfPasses << "\t";
        std::cout   << std::setw(10) << std::left << metric.description << "\n";

        if (appParams.bListSubMetrics)
        {
            std::cout << "\t Submetrics: \n";
            for (auto submetric : metric.submetrics) {
                std::cout << "\t\t" << submetric << "\n";
            }
        }
    }
}

