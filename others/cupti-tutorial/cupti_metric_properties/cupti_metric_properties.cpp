// Copyright 2020-2025 NVIDIA Corporation. All rights reserved
//
// The sample provides the generic workflow for querying various properties of metrics which are available as part of
// the Range Profiling APIs. In this particular case we are querying for number of passes, collection method, metric type and the
// hardware unit and sub-metrics associated for a list of metrics.
//
// Number of passes : It gives the number of passes required for collection of the metric as some of the metric
// cannot be collected in single pass due to hardware or software limitation, we need to replay the exact same
// set of GPU workloads multiple times.
//
// Collection method : It gives the source of the metric (HW or SW). Most of metric are provided by hardware but for
// some metric we have to instrument the kernel to collect the metric. Further these metrics cannot be combined with
// any other metrics in the same pass as otherwise instrumented code will also contribute to the metric value.
//

#include <iostream>
#include "table_util.h"
#include "command_line_parser_util.h"
#include "cupti_profiler_host_util.h"
#include "cupti_target.h"
#include "cupti_profiler_target.h"

using namespace cupti::utils;

struct CommandLineArgs
{
    bool listChips = false;
    bool listMetrics = false;
    bool listNumPasses = false;
    bool listSubMetrics = false;
    bool printTable = true;
    std::string outputFile = "";
    std::string chip = "";
    std::vector<std::string> metrics = {};
    bool verbose = false;
};

struct MetricProperties
{
    std::string name;
    std::string description;
    std::string hwUnit;
    std::string dimunits;
    std::string metricType;
    std::string numPasses;
    std::vector<std::string> subMetrics;
};

void ParseCommandLineArgs(int argc, char* argv[], CommandLineArgs& args);
uint32_t NumOfPassesForAllMetrics(MetricEnumerator& metricEnumerator, const std::vector<MetricProperties>& metricProperties);
void PrintOrExportMetricPropertiesToTable(const std::vector<MetricProperties>& metricProperties, const std::string& outputFile, bool printTable);
void GetMetricProperties(MetricEnumerator& metricEnumerator, const std::vector<std::string>& metrics, std::vector<MetricProperties>& metricProperties, bool listNumPasses, bool listSubMetrics);

int main(int argc, char* argv[])
{
    CommandLineArgs args = {};
    ParseCommandLineArgs(argc, argv, args);

    std::vector<uint8_t> counterAvailabilityImage = {};

    MetricEnumerator metricEnumerator;
    metricEnumerator.setup(args.chip, counterAvailabilityImage, CUpti_ProfilerType::CUPTI_PROFILER_TYPE_RANGE_PROFILER);

    std::vector<MetricProperties> metricProperties = {};
    uint32_t totalNumOfPasses = 0;
    if (args.listChips)
    {
        std::vector<std::string> chipNames = {};
        metricEnumerator.listSupportedChips(chipNames);
        std::cout << "Supported chips: " << std::endl;
        for (const auto& chipName : chipNames) {
            std::cout << chipName << " ";
        }
        std::cout << std::endl;
        return 0;
    }
    else if (!args.metrics.empty())
    {
        GetMetricProperties(metricEnumerator, args.metrics, metricProperties, args.listNumPasses, args.listSubMetrics);
        totalNumOfPasses = NumOfPassesForAllMetrics(metricEnumerator, metricProperties);
    }
    else if (args.listMetrics)
    {
        std::vector<std::string> metricNames = {};
        metricEnumerator.listSupportedBaseMetrics(metricNames);
        GetMetricProperties(metricEnumerator, metricNames, metricProperties, args.listNumPasses, args.listSubMetrics);
    }

    std::cout << "Chip: " << args.chip << std::endl;
    PrintOrExportMetricPropertiesToTable(metricProperties, args.outputFile, args.printTable);

    if (!args.metrics.empty()) {
        std::cout << "Total number of passes for queried metrics in [" << args.chip << "]: " << totalNumOfPasses << std::endl;
    } else {
        std::cout << "Total number of metrics for chip [" << args.chip << "]: " << metricProperties.size() << std::endl;
    }

    // Notes:
    if (!args.listNumPasses) {
        std::cout << "\nNotes:\nFor listing number of passes for each metric, add '-lnp' flag which is time consuming." << std::endl;
    }

    return 0;
}

void GetMetricProperties(MetricEnumerator& metricEnumerator, const std::vector<std::string>& metrics, std::vector<MetricProperties>& metricProperties, bool listNumPasses, bool listSubMetrics)
{
    for (const auto& metricName : metrics)
    {
        MetricProperties metricProperty = {};
        metricProperty.name = metricName;
        CUpti_MetricType metricType = CUpti_MetricType::CUPTI_METRIC_TYPE_COUNTER;
        metricEnumerator.listMetricProperties(
            metricName,
            metricType,
            metricProperty.description,
            metricProperty.hwUnit,
            metricProperty.dimunits,
            metricProperty.numPasses,
            listNumPasses
        );
        metricProperty.metricType = (metricType == CUpti_MetricType::CUPTI_METRIC_TYPE_COUNTER) ? "Counter" :
                                        (metricType == CUpti_MetricType::CUPTI_METRIC_TYPE_RATIO) ? "Ratio" : "Throughput";

        // List sub-metrics if requested
        if (listSubMetrics) {
            metricEnumerator.listSubmetrics(metricName, metricType, metricProperty.subMetrics);
        }

        metricProperties.push_back(metricProperty);
    }
}

uint32_t NumOfPassesForAllMetrics(MetricEnumerator& metricEnumerator, const std::vector<MetricProperties>& metricProperties)
{
    std::vector<std::string> metricNames = {};
    for (const auto& metricProperty : metricProperties)
    {
        std::string metricName = metricProperty.name;
        if (metricProperty.subMetrics.empty())
        {
            std::vector<std::string> subMetrics = {};
            CUpti_MetricType metricType = (metricProperty.metricType == "Counter") ? CUpti_MetricType::CUPTI_METRIC_TYPE_COUNTER :
                                            (metricProperty.metricType == "Ratio") ? CUpti_MetricType::CUPTI_METRIC_TYPE_RATIO :
                                            CUpti_MetricType::CUPTI_METRIC_TYPE_THROUGHPUT;
            metricEnumerator.listSubmetrics(metricProperty.name, metricType, subMetrics);
            metricName += subMetrics[0];
        }
        else
        {
            metricName += metricProperty.subMetrics[0];
        }
        metricNames.push_back(metricName);
    }
    return metricEnumerator.getNumOfPasses(metricNames);
}

std::vector<std::string> split(std::string str, char separator)
{
    std::vector<std::string> result = {};
    std::string temp = "";

    for (int i = 0; i < (int)str.size(); i++)
    {
        if (str[i] != separator) {
            temp += str[i];
        } else {
            result.push_back(temp);
            temp = "";
        }
    }

    // Add the last token if there is one
    if (!temp.empty()) {
        result.push_back(temp);
    }
    return result;
}

void ParseCommandLineArgs(int argc, char* argv[], CommandLineArgs& args)
{
    CommandLineParser parser;
    parser.addOption<size_t>("-d", "--device", "Device index (will be ignored if chip is specified)", 0);
    parser.addOption<bool>("-lc", "--list-chips", "List supported chips", false);
    parser.addOption<bool>("-lm", "--list-metrics", "List supported metrics", true);
    parser.addOption<bool>("-ls", "--list-submetrics", "List supported sub-metrics", false);
    parser.addOption<bool>("-lnp", "--list-num-passes", "List number of passes for each metric (time consuming)", false);
    parser.addOption<std::string>("-c", "--chip", "Chip name", "");
    parser.addOption<std::string>("-m", "--metrics", "Metric name", "");
    parser.addOption<bool>("-v", "--verbose", "Enable verbose mode", false);
    parser.addOption<size_t>("-pt", "--print-table", "Print table", 1);
    parser.addOption<std::string>("-o", "--output", "Output file", "");

    parser.parse(argc, argv);

    args.listChips = parser.get<bool>("--list-chips");
    args.listMetrics = parser.get<bool>("--list-metrics");
    args.listNumPasses = parser.get<bool>("--list-num-passes");
    args.listSubMetrics = parser.get<bool>("--list-submetrics");
    args.chip = parser.get<std::string>("--chip");
    size_t device = parser.get<size_t>("--device");

    if (args.chip.empty())
    {
        // For quering the chip name from device, we need to call profiler target APIs and for that we need to init CUDA first.
        cuInit(0);

        CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

        CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
        getChipNameParams.deviceIndex = device;
        CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
        args.chip = getChipNameParams.pChipName;

        CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = { CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
    }

    std::string metricsStr = parser.get<std::string>("--metrics");
    args.metrics = split(metricsStr, ',');
    args.outputFile = parser.get<std::string>("--output");
    args.printTable = parser.get<size_t>("--print-table") != 0;
    args.verbose = parser.get<bool>("--verbose");

    if (args.verbose)
    {
        std::cout << "List chips: " << args.listChips << "\n";
        std::cout << "List metrics: " << args.listMetrics << "\n";
        std::cout << "List num passes: " << args.listNumPasses << "\n";
        std::cout << "List sub-metrics: " << args.listSubMetrics << "\n";
        std::cout << "Chip: " << args.chip << "\n";
        std::cout << "Metrics: ";
        for (const auto& metric : args.metrics) {
            std::cout << metric << " ";
        }
        std::cout << "\n";
    }
}

void PrintOrExportMetricPropertiesToTable(const std::vector<MetricProperties>& metricProperties, const std::string& outputFile, bool printTable)
{
    const int passesWidth = (metricProperties[0].numPasses.size()) ? 10 : 0;
    const int subMetricsWidth = (metricProperties[0].subMetrics.size()) ? 50 : 0;
    std::vector<Table::Column> columns =
    {
        {"Metric Name", 50, Alignment::Left, OverflowMode::WrapHard},
        {"HW Unit", 10, Alignment::Left, OverflowMode::WrapHard},
        {"Type", 20, Alignment::Left, OverflowMode::WrapHard},
        {"Dim Units", 20, Alignment::Left, OverflowMode::WrapHard},
        {"Passes", passesWidth, Alignment::Left, OverflowMode::WrapHard},
        {"Description", 62, Alignment::Left, OverflowMode::WrapWord},
        {"Sub-metrics", subMetricsWidth, Alignment::Left, OverflowMode::WrapWord}
    };

    // Create a Table object with the desired columns and overflow mode
    Table table(columns);

    // Add some example rows to the table
    for (const auto& metricProperty : metricProperties)
    {
        std::string subMetricsStr = "";
        for (const auto& subMetric : metricProperty.subMetrics) {
            subMetricsStr += subMetric + ", ";
        }
        table.addRow(
        {
            metricProperty.name,
            metricProperty.hwUnit,
            metricProperty.metricType,
            metricProperty.dimunits,
            metricProperty.numPasses,
            metricProperty.description,
            subMetricsStr
        });
    }

    // Print the table to the console
    if (printTable) {
        table.print();
    }

    // Export the table to file
    if (!outputFile.empty()) {
        table.exportToTextFile(outputFile);
    }
}

