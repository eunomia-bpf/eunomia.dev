/*
 * Copyright 2025 NVIDIA Corporation. All rights reserved
 *
 * Sample demonstrating how to use CuptiParseNvtxPayload() to parse and decode
 * NVTX extended payloads from CUPTI activity records.
 *
 * To compile this sample, user should compile it with the latest NVTX headers (v3.2).
 * The NVTX headers are available here: https://github.com/NVIDIA/NVTX
 *
 * The sample uses the NVTX API to create a domain and register the payload schemas.
 * The sample also demonstrates how to use the NVTX API to create events with
 * extended payloads and push/pop ranges with these events.
 *
 * The CuptiParseNvtxPayload() function is used to interpret and print the
 * decoded payload values directly to the terminal. This allows developers to
 * verify the correctness and format of the payload data.
 *
 * For reference, the original payload data that was passed by the sample to NVTX
 * is also written to `nvtx_payload_blob.txt`. This file can be used to cross-check
 * and confirm that the decoded output from CuptiParseNvtxPayload() matches
 * the actual payload contents.
 *
 * The helper files used in this sample to parse NVTX extended payloads:
 * ../common/nvtx/nvtx_payload_parser.cpp
 * ../common/nvtx/nvtx_payload_parser.h
 * ../common/nvtx/nvtx_payload_attributes.cpp
 * ../common/nvtx/nvtx_payload_attributes.h
 *
 * Before running the sample set the NVTX_INJECTION64_PATH
 * environment variable pointing to the CUPTI Library.
 * For Linux:
 *    export NVTX_INJECTION64_PATH=<full_path>/libcupti.so
 * For Windows:
 *    set NVTX_INJECTION64_PATH=<full_path>\cupti.dll
 */

// System headers
#include <stdio.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <inttypes.h>

// CUDA Headers
#include <cuda.h>
#include <cuda_runtime.h>

// Standard NVTX headers
#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtPayload.h"

// CUPTI Headers
#include <helper_cupti_activity.h>
#include <nvtx_payload_parser.h>

// File pointer for writing NVTX payload blob
FILE *g_pFile = nullptr;

__global__ void
VectorAdd(
    const int *pA,
    const int *pB,
    int *pC,
    int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        pC[i] = pA[i] + pB[i];
    }
}

void
ParseNvtxExtendedPayloads(
    CUpti_Activity* pRecord)
{
    switch (pRecord->kind)
    {
        case CUPTI_ACTIVITY_KIND_MARKER_DATA:
        {
            CUpti_ActivityMarkerData2 *pMarkerDataRecord = (CUpti_ActivityMarkerData2 *)pRecord;
            if (pMarkerDataRecord->payloadKind == CUPTI_METRIC_VALUE_KIND_NVTX_EXTENDED_PAYLOAD)
            {
                uint64_t payloadAddress = pMarkerDataRecord->payload.metricValueNvtxExtendedPayload;
                std::cout << "\nParsing NVTX extended payload address: " << payloadAddress << std::endl;
                nvtxPayloadData_t *pPayload = (nvtxPayloadData_t *)payloadAddress;
                CuptiParseNvtxPayload(pMarkerDataRecord->cuptiDomainId, pPayload);
                std::cout << std::endl;

                // Free the payload memory
                    if (pPayload != NULL)
                    {
                        if (pPayload->payload != NULL)
                        {
                            free(const_cast<void*>(pPayload->payload));
                            pPayload->payload = NULL;
                        }
                            
                        free(pPayload);
                        pPayload = NULL;
                    }
            }
            break;
        }
        default:
            break;
    }

}

static void
InitializeVector(
    int *pVector,
    int N)
{
    for (int i = 0; i < N; i++)
    {
        pVector[i] = i;
    }
}

static void
DoVectorAddition()
{
    CUdevice device = 0;
    uint32_t deviceId = 0;
    int N = 50000;
    size_t size = N * sizeof (int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int *pHostA = 0, *pHostB = 0, *pHostC = 0;
    int *pDeviceA = 0, *pDeviceB = 0, *pDeviceC = 0;

    // Create domain "Vector Addition".
    nvtxDomainHandle_t domain = nvtxDomainCreateA("Vector Addition");

    // Register an enum schema for the vector addition operations.
    enum VectorAddOp
    {
        VECADD_ALLOC = 1,
        VECADD_MEMCPY = 2,
        VECADD_KERNEL = 3,
        VECADD_FREE = 4
    };

    nvtxPayloadEnum_t enumVectorAddSchema[] =
    {
        {"Allocate memory", VECADD_ALLOC},
        {"Copy memory", VECADD_MEMCPY},
        {"Kernel launch", VECADD_KERNEL},
        {"Free memory", VECADD_FREE}
    };

    nvtxPayloadEnumAttr_t enumSchemaAttr;
    enumSchemaAttr.name = "Vector Addition Operation Enum";
    enumSchemaAttr.fieldMask = NVTX_PAYLOAD_ENUM_ATTR_FIELD_ENTRIES |
            NVTX_PAYLOAD_ENUM_ATTR_FIELD_NUM_ENTRIES |
            NVTX_PAYLOAD_ENUM_ATTR_FIELD_SIZE;
    enumSchemaAttr.entries = enumVectorAddSchema;
    enumSchemaAttr.numEntries = std::extent<decltype(enumVectorAddSchema)>::value;
    enumSchemaAttr.sizeOfEnum = sizeof(enum VectorAddOp);

    uint64_t enumVectorAddSchemaId = nvtxPayloadEnumRegister(domain, &enumSchemaAttr);

    // Register a schema for the vector addition payload.
    typedef struct
    {
        enum VectorAddOp vectorOp;
        uint32_t deviceId;
        int vectorElements;
        size_t vectorSize;
    } VectorAddPayload_t;

    nvtxPayloadSchemaEntry_t schema[] =
    {
        {0, enumVectorAddSchemaId, "VectorAddOp"},
        {0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "Device ID"},
        {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "No. of elements"},
        {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Size of vector"}
    };

    nvtxPayloadSchemaAttr_t schemaAttr;
    schemaAttr.fieldMask = NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME |
            NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
            NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES |
            NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
            NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE;
    schemaAttr.name = "Vector Addition Schema";
    schemaAttr.type = NVTX_PAYLOAD_SCHEMA_TYPE_STATIC;
    schemaAttr.entries = schema;
    schemaAttr.numEntries = std::extent<decltype(schema)>::value;
    schemaAttr.payloadStaticSize = sizeof(VectorAddPayload_t);

    uint64_t vectorAddSchemaId = nvtxPayloadSchemaRegister(domain, &schemaAttr);

    DRIVER_API_CALL(cuDeviceGet(&device, deviceId));

    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;

    // Payload data struct
    nvtxPayloadData_t payloadData;

    // Fill payload with data.
    VectorAddPayload_t payload = {VECADD_ALLOC, 0, N, size};

    // Print the parameters of the payload in the file nvtx_payload_blob.txt.
    NVTX_FPRINTF(g_pFile, "| Entry Name: Allocate memory | Entry Value: %d |\n", payload.vectorOp);
    NVTX_FPRINTF(g_pFile, "| Entry Name: Device ID | Entry Value: %d |\n", payload.deviceId);
    NVTX_FPRINTF(g_pFile, "| Entry Name: No. of elements | Entry Value: %d |\n", payload.vectorElements);
    NVTX_FPRINTF(g_pFile, "| Entry Name: Size of vector | Entry Value: %zu |\n", payload.vectorSize);

    // Set the NVTX event attributes with extended payload.
    NVTX_PAYLOAD_EVTATTR_SET_DATA(eventAttrib, &payloadData, vectorAddSchemaId, &payload, sizeof(VectorAddPayload_t))
    eventAttrib.message.ascii = "Memory allocation";

    // Push range "Memory allocation" on domain "Vector Addition" with extended payload.
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Allocate input vectors pHostA and pHostB in host memory.
    pHostA = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostC);

    // Allocate vectors in device memory
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, size));

    // Pop range "Memory allocation" on domain "Vector Addition".
    nvtxDomainRangePop(domain);

    // Initialize input vectors.
    InitializeVector(pHostA, N);
    InitializeVector(pHostB, N);
    memset(pHostC, 0, size);

    // Fill payload with data.
    payload = {VECADD_MEMCPY, 0, N, size};

    // Print the parameters of the payload in the file nvtx_payload_blob.txt.
    NVTX_FPRINTF(g_pFile, "| Entry Name: Copy memory | Entry Value: %d |\n", payload.vectorOp);
    NVTX_FPRINTF(g_pFile, "| Entry Name: Device ID | Entry Value: %d |\n", payload.deviceId);
    NVTX_FPRINTF(g_pFile, "| Entry Name: No. of elements | Entry Value: %d |\n", payload.vectorElements);
    NVTX_FPRINTF(g_pFile, "| Entry Name: Size of vector | Entry Value: %zu |\n", payload.vectorSize);

    // Set the NVTX event attributes with extended payload.
    NVTX_PAYLOAD_EVTATTR_SET_DATA(eventAttrib, &payloadData, vectorAddSchemaId, &payload, sizeof(VectorAddPayload_t))
    eventAttrib.message.ascii = "Memcopy";

    // Push range "Memcopy" on domain "Vector Addition" with extended payload.
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Copy vectors from host memory to device memory.
    RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(pDeviceB, pHostB, size, cudaMemcpyHostToDevice));

    // Push range "Memcopy" on domain "Vector Addition".
    nvtxDomainRangePop(domain);

    // Fill payload with data.
    payload = {VECADD_KERNEL, 0, N, size};

    // Print the parameters of the payload in the file nvtx_payload_blob.txt.
    NVTX_FPRINTF(g_pFile, "| Entry Name: Kernel Launch | Entry Value: %d |\n", payload.vectorOp);
    NVTX_FPRINTF(g_pFile, "| Entry Name: Device ID | Entry Value: %d |\n", payload.deviceId);
    NVTX_FPRINTF(g_pFile, "| Entry Name: No. of elements | Entry Value: %d |\n", payload.vectorElements);
    NVTX_FPRINTF(g_pFile, "| Entry Name: Size of vector | Entry Value: %zu |\n", payload.vectorSize);

    // Set the NVTX event attributes with extended payload.
    NVTX_PAYLOAD_EVTATTR_SET_DATA(eventAttrib, &payloadData, vectorAddSchemaId, &payload, sizeof(VectorAddPayload_t))
    eventAttrib.message.ascii = "Kernel Launch";

    // Push range "Kernel Launch" on domain "Vector Addition" with extended payload.
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Invoke kernel
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    VectorAdd <<< blocksPerGrid, threadsPerBlock >>> (pDeviceA, pDeviceB, pDeviceC, N);
    RUNTIME_API_CALL(cudaGetLastError());
    DRIVER_API_CALL(cuCtxSynchronize());

    // Push range "Kernel Launch" on domain "Vector Addition".
    nvtxDomainRangePop(domain);

    // Fill payload with data.
    payload = {VECADD_MEMCPY, 0, N, size};

    // Print the parameters of the payload in the file nvtx_payload_blob.txt.
    NVTX_FPRINTF(g_pFile, "| Entry Name: Copy memory | Entry Value: %d |\n", payload.vectorOp);
    NVTX_FPRINTF(g_pFile, "| Entry Name: Device ID | Entry Value: %d |\n", payload.deviceId);
    NVTX_FPRINTF(g_pFile, "| Entry Name: No. of elements | Entry Value: %d |\n", payload.vectorElements);
    NVTX_FPRINTF(g_pFile, "| Entry Name: Size of vector | Entry Value: %zu |\n", payload.vectorSize);

    // Set the NVTX event attributes with extended payload.
    NVTX_PAYLOAD_EVTATTR_SET_DATA(eventAttrib, &payloadData, vectorAddSchemaId, &payload, sizeof(VectorAddPayload_t))
    eventAttrib.message.ascii = "Memcopy";

    // Push range "Memcopy" on domain "Vector Addition" with extended payload.
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Copy result from device memory to host memory.
    // pHostC contains the result in host memory.
    RUNTIME_API_CALL(cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost));

    // Push range "Memcopy" on domain "Vector Addition".
    nvtxDomainRangePop(domain);

    // Fill payload with data.
    payload = {VECADD_FREE, 0, N, size};

    // Print the parameters of the payload in the file nvtx_payload_blob.txt.
    NVTX_FPRINTF(g_pFile, "| Entry Name: Free memory | Entry Value: %d |\n", payload.vectorOp);
    NVTX_FPRINTF(g_pFile, "| Entry Name: Device ID | Entry Value: %d |\n", payload.deviceId);
    NVTX_FPRINTF(g_pFile, "| Entry Name: No. of elements | Entry Value: %d |\n", payload.vectorElements);
    NVTX_FPRINTF(g_pFile, "| Entry Name: Size of vector | Entry Value: %zu |\n", payload.vectorSize);

    // Set the NVTX event attributes with extended payload.
    NVTX_PAYLOAD_EVTATTR_SET_DATA(eventAttrib, &payloadData, vectorAddSchemaId, &payload, sizeof(VectorAddPayload_t))
    eventAttrib.message.ascii = "Free memory";

    // Push range "Free memory" on domain "Vector Addition" with extended payload.
    nvtxDomainRangePushEx(domain, &eventAttrib);

    // Free device memory.
    if (pDeviceA)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceA));
    }
    if (pDeviceB)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceB));
    }
    if (pDeviceC)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceC));
    }

    // Free host memory.
    if (pHostA)
    {
        free(pHostA);
    }
    if (pHostB)
    {
        free(pHostB);
    }
    if (pHostC)
    {
        free(pHostC);
    }

    // Push range "Free memory" on domain "Vector Addition".
    nvtxDomainRangePop(domain);

    DRIVER_API_CALL(cuCtxSynchronize());
}

void
PrintFileContent(
    const char *pFileName)
{
    FILE *fp = fopen(pFileName, "r");
    if (!fp)
    {
        perror(pFileName);
        return;
    }

    std::cout << "\nContent of " << pFileName << ":" << std::endl;
    int c;
    while ((c = fgetc(fp)) != EOF)
    {
        putchar(c);
    }

    fclose(fp);
    std::cout << std::endl;
}

static void
SetupCupti()
{
    UserData *pUserData = (UserData *)malloc(sizeof(UserData));
    MEMORY_ALLOCATION_CALL(pUserData);

    memset(pUserData, 0, sizeof(UserData));
    pUserData->pPostProcessActivityRecords = ParseNvtxExtendedPayloads;
    pUserData->printActivityRecords        = 1;

    // Common CUPTI Initialization
    InitCuptiTrace(pUserData, NULL, stdout);

    CUPTI_API_CALL_VERBOSE(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
    CUPTI_API_CALL_VERBOSE(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER_DATA));
}


int main(
    int argc,
    char* argv[])
{
    const char *pNvtxPath = getenv("NVTX_INJECTION64_PATH");
    if (pNvtxPath) 
    {
        std::cout << "NVTX_INJECTION64_PATH: " << pNvtxPath << std::endl;
    } 
    else 
    {
        std::cerr << "\nError: NVTX_INJECTION64_PATH is not set.\nPlease set the NVTX_INJECTION64_PATH to CUPTI library.\n" << std::endl;
        exit(EXIT_FAILURE);
    }

    SetupCupti();

    // Initialize CUDA.
    DRIVER_API_CALL(cuInit(0));

    g_pFile = fopen("nvtx_payload_blob.txt", "w");
    if (!g_pFile)
    {
        std::cerr << "Error: Unable to open nvtx_payload_blob.txt for writing." << std::endl;
        return 1;
    }

    // Do simple vector addition.
    // The sample uses NVTX to create events with extended payloads.
    DoVectorAddition();

    fclose(g_pFile);

    std::cout << std::endl;
    DeInitCuptiTrace();

    std::cout << "\nPrinting the original payload data that was passed by the sample to NVTX.";
    PrintFileContent("nvtx_payload_blob.txt");

    // Free NVTX payload attributes and schemas cached when parsing the payloads.
    FreeAllNvtxPayloadAttributes();

    return 0;
}
