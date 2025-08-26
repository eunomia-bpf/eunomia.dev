/**
 * Copyright 2025 NVIDIA Corporation. All rights reserved
 * @file nvtx_payload_parser.cpp
 * @brief Implementation of NVTX payload parsing and logging functions.
 *
 * * Typical usage:
 *   - Call CuptiParseNvtxPayload() from a CUPTI buffer completed callback to decode
 *     and print NVTX payloads.
 *   - Use GetNvtxPayloadAttributes() to retrieve payload attributes for a given schema ID.
 *   - Use ParsePayloadSchema() to parse the payload data using the schema attributes.
 *   - Use ParsePayloadEnum() to parse the payload data using the enum attributes.
 *   - Use ParsePredefinedType() to convert payload entries to string representations.
 *   - NVTX_PAYLOAD_LOG_INFO is used to log the payload data for each entry in the schema.
 *       This is a placeholder for CUPTI client to access payload data and process it as required.
 *   - Use the provided logging macros for consistent output.
 *   - Set the log level using SetLogLevel() to control verbosity.
 *      Currently log level is set to NVTX_PAYLOAD_LOG_INFO.
 *
 * Thread safety: Unless otherwise noted, global data structures are not thread-safe.
 */

#include <nvtx_payload_parser.h>
#include <sstream>
#include <iomanip>
#include <cstdarg>
#include <inttypes.h>

void
Log(
    NvtxPayloadLogLevel level,
    const char* format, ...)
{
    // Create a stringstream to build the log message.
    std::stringstream ss;

    // Only log messages that are at or above the current global log level.
    if (level > g_nvtxData.logLevel)
    {
        // If the message is below the current log level, do not log it.
        return;
    }

    // Add a standard prefix to all log messages for easy identification.
    ss << "| [NVTX Payload] | ";

    // Add a string indicating the log level.
    switch (level)
    {
        case NVTX_PAYLOAD_LOG_ERROR:
            ss << "Error | ";
            break;
        case NVTX_PAYLOAD_LOG_WARNING:
            ss << "Warning | ";
            break;
        case NVTX_PAYLOAD_LOG_INFO:
            ss << "Info | ";
            break;
        case NVTX_PAYLOAD_LOG_DEBUG_INFO:
            ss << "Debug | ";
            break;
    }

    // Prepare to process the variable argument list for formatting.
    va_list args;
    va_start(args, format);

    // Buffer to hold the formatted message.
    char buffer[1024];
    // Format the message using vsnprintf and the provided arguments.
    vsnprintf(buffer, sizeof(buffer), format, args);

    // Append the formatted message to the stringstream.
    ss << buffer;
    // Clean up the variable argument list.
    va_end(args);

    // Print errors and warnings to std::cerr, others to std::cout.
    if (level <= NVTX_PAYLOAD_LOG_WARNING)
    {
        std::cerr << ss.str() << std::endl;
    }
    else
    {
        std::cout << ss.str() << std::endl;
    }
}

void
SetLogLevel(
    NvtxPayloadLogLevel level)
{
    // Set the global log level in the singleton g_nvtxData.
    // All subsequent log messages will be filtered based on this level.
    g_nvtxData.logLevel = level;
}

void
SetPayloadDataTypesInfo()
{
    // If the data types vector is already populated, do nothing.
    if (g_nvtxData.nvtxPayloadDataTypes.size() > 0)
    {
        NVTX_PAYLOAD_LOG_DEBUG("Payload data types already initialized");
        return;
    }

    // Query CUPTI for the NVTX payload entry type information.
    const nvtxPayloadEntryTypeInfo_t* pTypeInfo = cuptiActivityGetNvtxExtPayloadEntryTypeInfo();
    if (pTypeInfo == nullptr)
    {
        // If CUPTI does not provide the type info, log and return.
        NVTX_PAYLOAD_LOG_DEBUG("Could not get NVTX payload entry type info");
        return;
    }

    // Reserve space in the vector for all available types.
    g_nvtxData.nvtxPayloadDataTypes.reserve(pTypeInfo->size);

    // Populate the vector with size and alignment for each type.
    for (uint16_t i = 0; i < pTypeInfo->size; ++i)
    {
        g_nvtxData.nvtxPayloadDataTypes.emplace_back(pTypeInfo[i].size, pTypeInfo[i].align);
    }

    // Log the number of types initialized.
    NVTX_PAYLOAD_LOG_DEBUG("Initialized %d payload data types", pTypeInfo->size);
}

void
FreeAllNvtxPayloadAttributes()
{
    // Iterate over all domain IDs in the global attribute map.
    for (auto& domainPair : g_nvtxData.nvtxPayloadAttributeMap)
    {
        // For each domain, iterate over all schema IDs.
        for (auto& schemaPair : domainPair.second)
        {
            // Free the NvtxPayloadAttributes object.
            delete schemaPair.second;
        }
        // Clear the schema ID map for this domain.
        domainPair.second.clear();
    }
    // Clear the global attribute map.
    g_nvtxData.nvtxPayloadAttributeMap.clear();
}

NvtxPayloadAttributes *
GetNvtxPayloadAttributes(
    uint32_t cuptiDomainId,
    uint64_t schemaId)
{
    // Try to find the domain in the global attribute map.
    auto domainIter = g_nvtxData.nvtxPayloadAttributeMap.find(cuptiDomainId);
    if (domainIter == g_nvtxData.nvtxPayloadAttributeMap.end())
    {
        // If not found, create a new entry for this domain ID.
        g_nvtxData.nvtxPayloadAttributeMap[cuptiDomainId] = NvtxSchemaIdMap();
        domainIter = g_nvtxData.nvtxPayloadAttributeMap.find(cuptiDomainId);
    }

    // Try to find the schema in the domain's schema map.
    auto schemaIter = domainIter->second.find(schemaId);
    if (schemaIter != domainIter->second.end())
    {
        // If found, return the cached payload attributes.
        NVTX_PAYLOAD_LOG_DEBUG("Schema ID %llu found in map", static_cast<unsigned long long>(schemaId));
        return schemaIter->second;
    }
    else
    {
        // If not found, try to fetch the payload attributes from CUPTI.
        CUpti_NvtxExtPayloadAttr cuptiPayloadAttributes = {0};
        CUptiResult result = cuptiActivityGetNvtxExtPayloadAttr(cuptiDomainId, schemaId, &cuptiPayloadAttributes);
        if (result != CUPTI_SUCCESS)
        {
            // If CUPTI fails to provide the attributes, log and return nullptr.
            NVTX_PAYLOAD_LOG_ERROR("Could not get NVTX payload attributes for schema ID %llu from CUPTI", static_cast<unsigned long long>(schemaId));
            return nullptr;
        }
        else
        {
            NVTX_PAYLOAD_LOG_DEBUG("Schema ID %llu found in CUPTI", static_cast<unsigned long long>(schemaId));
            if (cuptiPayloadAttributes.type == CUPTI_NVTX_EXT_PAYLOAD_TYPE_SCHEMA)
            {
                // Handle schema type: cast and extract fields.
                const nvtxPayloadSchemaAttr_t *pSchemaAttr = reinterpret_cast<const nvtxPayloadSchemaAttr_t *>(cuptiPayloadAttributes.attributes);
                if (pSchemaAttr == nullptr)
                {
                    NVTX_PAYLOAD_LOG_ERROR("Payload schema attribute is null");
                    return nullptr;
                }

                // Create a new NvtxPayloadSchema object and populate its fields.
                auto pSchema = new NvtxPayloadSchema();
                pSchema->schemaId = schemaId;
                pSchema->payloadType = cuptiPayloadAttributes.type;
                pSchema->domainId = cuptiDomainId;
                if (pSchemaAttr->name)
                {
                    pSchema->name = pSchemaAttr->name;
                }

                pSchema->fieldMask = pSchemaAttr->fieldMask;
                pSchema->schemaType = pSchemaAttr->type;
                pSchema->flags = pSchemaAttr->flags;
                pSchema->payloadStaticSize = pSchemaAttr->payloadStaticSize;
                pSchema->packAlign = pSchemaAttr->packAlign;
                pSchema->processed = false;

                // Populate the schema's entries from the CUPTI-provided array.
                pSchema->entries.reserve(pSchemaAttr->numEntries);
                for (size_t i = 0; i < pSchemaAttr->numEntries; ++i)
                {
                    const nvtxPayloadSchemaEntry_t &entry = pSchemaAttr->entries[i];
                    NvtxSchemaEntry schemaEntry;
                    schemaEntry.flags = entry.flags;
                    schemaEntry.type = entry.type;

                    if (entry.name)
                    {
                        schemaEntry.name = entry.name;
                    }
                    if (entry.description)
                    {
                        schemaEntry.description = entry.description;
                    }

                    schemaEntry.extent = entry.arrayOrUnionDetail;
                    schemaEntry.offset = entry.offset;

                    pSchema->entries.push_back(schemaEntry);
                }

                if (pSchemaAttr->entries != nullptr)
                {
                    // Free the CUPTI-allocated payload entries array if it was allocated.
                    free(static_cast<void *>(const_cast<nvtxPayloadSchemaEntry_t *>(pSchemaAttr->entries)));
                }

                // Free the CUPTI-allocated payload attribute memory.
                free(cuptiPayloadAttributes.attributes);

                // Store the schema in the map for future lookups.
                domainIter->second[schemaId] = pSchema;
                NVTX_PAYLOAD_LOG_DEBUG("Stored schema ID %llu in map of payload type schema", static_cast<unsigned long long>(schemaId));

                // Return the newly cached schema.
                return domainIter->second[schemaId];
            }
            else if (cuptiPayloadAttributes.type == CUPTI_NVTX_EXT_PAYLOAD_TYPE_ENUM)
            {
                // Handle enum type: cast and extract fields.
                const nvtxPayloadEnumAttr_t *pEnumAttr = reinterpret_cast<const nvtxPayloadEnumAttr_t *>(cuptiPayloadAttributes.attributes);
                if (pEnumAttr == nullptr)
                {
                    NVTX_PAYLOAD_LOG_ERROR("Payload enum attribute is null");
                    return nullptr;
                }

                // Create a new NvtxPayloadEnum object and populate its fields.
                auto pEnum = new NvtxPayloadEnum();
                pEnum->schemaId = schemaId;
                pEnum->payloadType = cuptiPayloadAttributes.type;
                pEnum->domainId = cuptiDomainId;
                if (pEnumAttr->name)
                {
                    pEnum->name = pEnumAttr->name;
                }

                pEnum->fieldMask = pEnumAttr->fieldMask;
                pEnum->sizeOfEnum = pEnumAttr->sizeOfEnum;

                // Populate the enum's entries from the CUPTI-provided array.
                pEnum->entries.reserve(pEnumAttr->numEntries);
                for (size_t i = 0; i < pEnumAttr->numEntries; ++i)
                {
                    const nvtxPayloadEnum_t &entry = pEnumAttr->entries[i];
                    NvtxEnumEntry enumEntry;

                    if (entry.name)
                    {
                        enumEntry.name = entry.name;
                    }

                    enumEntry.value = entry.value;
                    enumEntry.isFlag = entry.isFlag;

                    pEnum->entries.push_back(enumEntry);
                }

                // Free the CUPTI-allocated enum attribute memory.
                if (pEnumAttr->entries != nullptr)
                {
                    // Free the CUPTI-allocated payload entries array if it was allocated.
                    free(static_cast<void *>(const_cast<nvtxPayloadEnum_t *>(pEnumAttr->entries)));
                }

                // Free the CUPTI-allocated payload attribute memory.
                free(cuptiPayloadAttributes.attributes);

                // Store the enum in the map for future lookups.
                domainIter->second[schemaId] = pEnum;
                NVTX_PAYLOAD_LOG_DEBUG("Stored schema ID %llu in map of payload type enum", static_cast<unsigned long long>(schemaId));

                // Return the newly cached enum.
                return domainIter->second[schemaId];
            }
        }
    }

    // If we reach here, the schema was not found in the map or CUPTI.
    NVTX_PAYLOAD_LOG_ERROR("Schema ID %llu not found in map or CUPTI", static_cast<unsigned long long>(schemaId));

    return nullptr;
}

uint16_t
GetSizeOfFixedSizeTypes(
    uint64_t type)
{
    // Use a switch statement to handle each special-case type.
    switch (type)
    {
        // 2-byte types: half-precision floats, bfloat16, UTF-16 strings
        case NVTX_PAYLOAD_ENTRY_TYPE_FLOAT16:
        case NVTX_PAYLOAD_ENTRY_TYPE_BF16:
        case NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF16:
            return 2;

        // 4-byte types: single-precision floats, TF32, UTF-32 strings, categories, colors, thread/pid IDs
        case NVTX_PAYLOAD_ENTRY_TYPE_FLOAT32:
        case NVTX_PAYLOAD_ENTRY_TYPE_TF32:
        case NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF32:
        case NVTX_PAYLOAD_ENTRY_TYPE_CATEGORY:
        case NVTX_PAYLOAD_ENTRY_TYPE_COLOR_ARGB:
        case NVTX_PAYLOAD_ENTRY_TYPE_TID_UINT32:
        case NVTX_PAYLOAD_ENTRY_TYPE_PID_UINT32:
            return 4;

        // 8-byte types: double-precision floats, 64-bit thread/pid IDs, scope IDs
        case NVTX_PAYLOAD_ENTRY_TYPE_FLOAT64:
        case NVTX_PAYLOAD_ENTRY_TYPE_TID_UINT64:
        case NVTX_PAYLOAD_ENTRY_TYPE_PID_UINT64:
        case NVTX_PAYLOAD_ENTRY_TYPE_SCOPE_ID:
            return 8;

        // 16-byte types: 128-bit integers and floats
        case NVTX_PAYLOAD_ENTRY_TYPE_INT128:
        case NVTX_PAYLOAD_ENTRY_TYPE_UINT128:
        case NVTX_PAYLOAD_ENTRY_TYPE_FLOAT128:
            return 16;

        // 1-byte types: bytes, C strings, UTF-8 strings, union selectors
        case NVTX_PAYLOAD_ENTRY_TYPE_BYTE:
        case NVTX_PAYLOAD_ENTRY_TYPE_CSTRING:
        case NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF8:
        case NVTX_PAYLOAD_ENTRY_TYPE_UNION_SELECTOR:
            return 1;

        // If the type is not recognized, return InvalidTypeSize (usually 0).
        default:
            return InvalidTypeSize;
    }
}

uint16_t
GetSizeOfPayloadPredefinedType(
    uint64_t type)
{
    // If the type is within the range of the info array, use the global data types vector.
    if (type < NVTX_PAYLOAD_ENTRY_TYPE_INFO_ARRAY_SIZE)
    {
        // Check if the type index is valid for the vector.
        if (type >= g_nvtxData.nvtxPayloadDataTypes.size())
        {
            // If not, log an error and return invalid size.
            NVTX_PAYLOAD_LOG_ERROR("NVTX payload entry type %llu is not found among %u pre-defined payload types stored.", static_cast<unsigned long long>(type), g_nvtxData.nvtxPayloadDataTypes.size());
            return InvalidTypeSize;
        }
        // Return the size for this type from the vector.
        return g_nvtxData.nvtxPayloadDataTypes[type].size;
    }
    // If the type is not in the info array, but is less than the string handle type, use the fixed size types.
    else if (type < NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE)
    {
        return GetSizeOfFixedSizeTypes(type);
    }
    // If the type is the registered string handle, use the address type's size.
    else if (type == NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE)
    {
        return g_nvtxData.nvtxPayloadDataTypes[NVTX_PAYLOAD_ENTRY_TYPE_ADDRESS].size;
    }

    // If the type is not recognized, return invalid size.
    return InvalidTypeSize;
}

void
ParsePredefinedType(
    const NvtxSchemaEntry& entry,
    char* pPayloadEntry,
    std::string& output)
{
    try
    {
        // Macro to handle numeric types in a compact way.
        // It casts the payload pointer to the correct type, reads the value, and appends it as a string.
#define CASE_NUMBER(NVTX_TYPE, REAL_TYPE)                           \
case NVTX_PAYLOAD_ENTRY_TYPE_##NVTX_TYPE:                           \
{                                                                   \
    REAL_TYPE value = *reinterpret_cast<REAL_TYPE*>(pPayloadEntry); \
    output += std::to_string(value);                                \
    break;                                                          \
}
        // Switch on the entry type to handle all supported NVTX types.
        switch(entry.type)
        {
            // Handle single character types.
            case NVTX_PAYLOAD_ENTRY_TYPE_CHAR:
                output += *pPayloadEntry;
                break;

            // Handle all numeric types using the macro.
            CASE_NUMBER(UCHAR, unsigned char)
            CASE_NUMBER(SHORT, short)
            CASE_NUMBER(USHORT, unsigned short)
            CASE_NUMBER(INT, int)
            CASE_NUMBER(UINT, unsigned int)
            CASE_NUMBER(LONG, long)
            CASE_NUMBER(ULONG, unsigned long)
            CASE_NUMBER(LONGLONG, long long)
            CASE_NUMBER(ULONGLONG, unsigned long long)
            CASE_NUMBER(INT8, int8_t)
            CASE_NUMBER(UINT8, uint8_t)
            CASE_NUMBER(INT16, int16_t)
            CASE_NUMBER(UINT16, uint16_t)
            CASE_NUMBER(INT32, int32_t)
            CASE_NUMBER(UINT32, uint32_t)
            CASE_NUMBER(INT64, int64_t)
            CASE_NUMBER(UINT64, uint64_t)
            CASE_NUMBER(FLOAT, float)
            CASE_NUMBER(DOUBLE, double)
            CASE_NUMBER(LONGDOUBLE, long double)
            CASE_NUMBER(SIZE, size_t)
            CASE_NUMBER(FLOAT32, float)
            CASE_NUMBER(FLOAT64, double)

            // Handle pointer types by formatting as a hexadecimal string.
            case NVTX_PAYLOAD_ENTRY_TYPE_ADDRESS:
            {
                std::ostringstream ret;
                ret << std::hex << std::setfill('0') << std::setw(2) << std::nouppercase << *reinterpret_cast<void**>(pPayloadEntry);
                output += ret.str();
                break;
            }

            // Handle C strings and UTF-8 strings by copying the specified extent.
            case NVTX_PAYLOAD_ENTRY_TYPE_CSTRING:
            case NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF8:
                output += std::string(pPayloadEntry, entry.extent);
                break;

            // Handle single bytes as hexadecimal.
            case NVTX_PAYLOAD_ENTRY_TYPE_BYTE:
            {
                std::ostringstream ret;
                ret << std::hex << std::setfill('0') << std::setw(2) << std::nouppercase << short(*pPayloadEntry);
                output += ret.str();
                break;
            }

            // Default case: log an error for unsupported types.
            default:
                NVTX_PAYLOAD_LOG_ERROR("NVTX payload schema entry type %llu is not supported", static_cast<unsigned long long>(entry.type));
                break;
        }
        // No need to handle [unsigned] int, short, long and size_t, because
        // they have been replaced with a fixed-size type.
#undef CASE_NUMBER
    }
    catch (const std::exception& e)
    {
        // Log any exceptions that occur during parsing.
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload to string error: %s (entry type %llu)", e.what(), static_cast<unsigned long long>(entry.type));
    }
    catch (...)
    {
        // Log any unknown errors.
        NVTX_PAYLOAD_LOG_ERROR("NVTX binary payload parse error (entry type %llu)", static_cast<unsigned long long>(entry.type));
    }
}

uint64_t
GetNumber(
    const NvtxSchemaEntry& schemaEntry,
    const char *pEntryData)
{
    // Skip unsupported schema entries.
    if ((schemaEntry.flags & NVTX_PAYLOAD_ENTRY_FLAG_DEEP_COPY) || (schemaEntry.flags & NVTX_PAYLOAD_ENTRY_FLAG_POINTER))
    {
        return UINT64_MAX;
    }

#define CASE_NUMBER(NVTX_TYPE, REAL_TYPE, FORMAT_TYPE)                          \
case NVTX_PAYLOAD_ENTRY_TYPE_##NVTX_TYPE:                                       \
{                                                                               \
    const REAL_TYPE value = *reinterpret_cast<const REAL_TYPE*>(pEntryData);    \
    return static_cast<FORMAT_TYPE>(value);                                     \
}

    switch(schemaEntry.type)
    {
        // Handle all numeric types using the macro.
        CASE_NUMBER(UCHAR, unsigned char, uint64_t)
        CASE_NUMBER(SHORT, short, uint64_t)
        CASE_NUMBER(USHORT, unsigned short, uint64_t)
        CASE_NUMBER(INT, int, uint64_t)
        CASE_NUMBER(UINT, unsigned int, uint64_t)
        CASE_NUMBER(LONG, long, uint64_t)
        CASE_NUMBER(ULONG, unsigned long, uint64_t)
        CASE_NUMBER(LONGLONG, long long, uint64_t)
        CASE_NUMBER(ULONGLONG, unsigned long long, uint64_t)
        CASE_NUMBER(INT8, int8_t, uint64_t)
        CASE_NUMBER(UINT8, uint8_t, uint64_t)
        CASE_NUMBER(INT16, int16_t, uint64_t)
        CASE_NUMBER(UINT16, uint16_t, uint64_t)
        CASE_NUMBER(INT32, int32_t, uint64_t)
        CASE_NUMBER(UINT32, uint32_t, uint64_t)
        CASE_NUMBER(INT64, int64_t, uint64_t)
        CASE_NUMBER(UINT64, uint64_t, uint64_t)
        CASE_NUMBER(FLOAT, float, uint64_t)
        CASE_NUMBER(DOUBLE, double, uint64_t)
        CASE_NUMBER(LONGDOUBLE, long double, uint64_t)
        CASE_NUMBER(SIZE, size_t, uint64_t)
        CASE_NUMBER(FLOAT32, float, uint64_t)
        CASE_NUMBER(FLOAT64, double, uint64_t)
        default:
            return UINT64_MAX;
    }
#undef CASE_NUMBER
}

uint64_t
TryGetArrayLengthFromEntryValue(
    const NvtxSchemaEntry& entry,
    const char *pPayloadEntry)
{
    // Only try to get the array length if the entry is a numeric value.
    return GetNumber(entry, pPayloadEntry);
}

bool
IsCString(
    uint64_t entryType)
{
    // Return true if the entry type is either standard C string or UTF-8 C string.
    return (entryType == NVTX_PAYLOAD_ENTRY_TYPE_CSTRING) ||
           (entryType == NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF8);
}

void
ParseNestedSchema(
    const NvtxPayloadSchema *pPayloadSchema,
    const NvtxSchemaEntry& entry,
    const char *pPayloadEntry)
{
    // Fetch the attributes for the nested schema using the domain ID and entry type.
    NvtxPayloadAttributes *pNestedPayload = GetNvtxPayloadAttributes(pPayloadSchema->domainId, entry.type);
    if (pNestedPayload)
    {
        // Determine the size of the nested schema's payload.
        size_t typeSize = pPayloadSchema->GetSizeOfPayloadEntryType(entry.type);
        // Recursively parse the nested payload using the nested schema's attributes.
        ParsePayload(pNestedPayload, pPayloadEntry, typeSize);
    }
    else
    {
        // If the nested schema is not found, log an error.
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload schema not found for entry type %llu", static_cast<unsigned long long>(entry.type));
    }
}

void
ParsePayloadSchema(
    NvtxPayloadSchema *pPayloadSchema,
    const char *pPayloadBase,
    size_t payloadSize,
    NvtxSchemaEntry *pNestedSchemaEntry)
{
    // Check for null schema pointer.
    if (!pPayloadSchema)
    {
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload schema is null");
        return;
    }

    // Check for null payload pointer.
    if (!pPayloadBase)
    {
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload base pointer is null");
        return;
    }

    // Check for zero payload size.
    if (payloadSize == 0)
    {
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload size is 0");
        return;
    }

    // Only static and dynamic schema types are supported.
    if (pPayloadSchema->schemaType != NVTX_PAYLOAD_SCHEMA_TYPE_STATIC &&
        pPayloadSchema->schemaType != NVTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC)
    {
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload schema of type %llu not supported", static_cast<unsigned long long>(pPayloadSchema->schemaType));
        return;
    }

    // Ensure the schema is processed (offsets and sizes are set).
    if (!pPayloadSchema->IsProcessed())
    {
        NVTX_PAYLOAD_LOG_DEBUG("Processing schema entries");
        (const_cast<NvtxPayloadSchema*>(pPayloadSchema))->ProcessEntries();
    }

    // For static schemas, check that the payload size matches the expected static size.
    if (pPayloadSchema->schemaType == NVTX_PAYLOAD_SCHEMA_TYPE_STATIC &&
        pPayloadSchema->payloadStaticSize != payloadSize)
    {
        NVTX_PAYLOAD_LOG_ERROR("NVTX extended payload size %zu != static payload size %llu (schema %llu)",
            payloadSize, static_cast<unsigned long long>(pPayloadSchema->payloadStaticSize), static_cast<unsigned long long>(pPayloadSchema->schemaId));
        return;
    }

    // Variables for dynamic offset tracking and entry iteration.
    size_t dynamicOffset = 0;
    size_t lastEntryIdx = pPayloadSchema->entries.size() - 1;
    const char *pPayloadEnd = pPayloadBase + payloadSize;

    // Iterate over all entries in the schema.
    for (size_t entryIdx = 0; entryIdx <= lastEntryIdx; ++entryIdx)
    {
        const NvtxSchemaEntry& schemaEntry = pPayloadSchema->entries[entryIdx];

        // Lambda for printing errors with schema and entry context.
        auto PrintError = [pPayloadSchema, &schemaEntry](const char *pMsg)
        {
            NVTX_PAYLOAD_LOG_ERROR("NVTX payload schema %s (%llu) entry '%s': %s",
                pPayloadSchema->name.c_str(), static_cast<unsigned long long>(pPayloadSchema->schemaId),
                schemaEntry.name.c_str(), pMsg);
        };

        // Determine the size of the current entry.
        size_t entrySize = pPayloadSchema->GetSizeOfPayloadEntry(schemaEntry);
        if (entrySize == NvtxSchemaEntry::SizeInvalid)
        {
            PrintError("Invalid entry size = 0");
            continue;
        }

        // Determine the offset of the entry within the payload.
        uint64_t entryOffset = schemaEntry.offset;
        if (!pPayloadSchema->IsOffsetValid(schemaEntry))
        {
            if (pPayloadSchema->schemaType == NVTX_PAYLOAD_SCHEMA_TYPE_STATIC)
            {
                PrintError("Invalid entry offset");
                return;
            }

            if (pPayloadSchema->schemaType == NVTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC)
            {
                if (dynamicOffset > 0)
                {
                    pPayloadSchema->AlignOffset(dynamicOffset, schemaEntry.type);
                    entryOffset = dynamicOffset;
                }
                else
                {
                    PrintError("Entry that is nested schema and of type NVTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC is supported. Skipping entry.");
                    continue;
                }
            }
        }

        // Pointer to the entry's data within the payload.
        char *pPayloadEntry = const_cast<char*>(pPayloadBase + entryOffset);

        // Check for out-of-bounds access.
        if ((pPayloadEntry + entrySize) > pPayloadEnd)
        {
            NVTX_PAYLOAD_LOG_ERROR("NVTX payload schema %s (%llu): read out of bounds (addr: %p, Entry size: %zu, Payload End address: %p)",
                pPayloadSchema->name.c_str(), static_cast<unsigned long long>(pPayloadSchema->schemaId),
                static_cast<void*>(pPayloadEntry), entrySize,
                static_cast<void*>(const_cast<char*>(pPayloadEnd)));
            break;
        }

        // Handle array types (fixed-size, length-indexed, or zero-terminated).
        if (((schemaEntry.flags & NVTX_PAYLOAD_ENTRY_FLAG_IS_ARRAY) == NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_FIXED_SIZE) ||
            ((schemaEntry.flags & NVTX_PAYLOAD_ENTRY_FLAG_IS_ARRAY) == NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_INDEX))
        {
            uint64_t arrayExtent = schemaEntry.extent;

            // For length-indexed arrays, get the length from another entry.
            if ((schemaEntry.flags & NVTX_PAYLOAD_ENTRY_FLAG_IS_ARRAY) == NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_INDEX)
            {
                if (arrayExtent >= entryIdx)
                {
                    PrintError("Array length field must be before array field");
                    return;
                }

                const NvtxSchemaEntry& sizeEntry = pPayloadSchema->entries[arrayExtent];
                if (!pPayloadSchema->IsOffsetValid(sizeEntry))
                {
                    PrintError("Array length index entry with invalid offset");
                    return;
                }

                char *pSizeEntry = const_cast<char*>(pPayloadBase + sizeEntry.offset);
                // We assume the array length is a numeric type.
                // If the entry is not a numeric type, we cannot get the length.
                // This is a limitation of the current implementation.
                uint64_t tmpLen = TryGetArrayLengthFromEntryValue(sizeEntry, pSizeEntry);
                if (tmpLen == UINT64_MAX)
                {
                    PrintError("Array length not found");
                    return;
                }
                else
                {
                    arrayExtent = tmpLen;
                }

                dynamicOffset = static_cast<size_t>(entryOffset);
            }

            if (arrayExtent == 0)
            {
                PrintError("Array length is 0");
                continue;
            }

            size_t typeSize = pPayloadSchema->GetSizeOfPayloadEntryType(schemaEntry.type);
            if (typeSize == 0)
            {
                return;
            }

            uint64_t arraySizeBytes = arrayExtent * typeSize;
            if (pPayloadEntry + arraySizeBytes > pPayloadEnd)
            {
                PrintError("Array exceeds the payload bounds");
                dynamicOffset = 0;
                continue;
            }

            // Iterate over each element in the array.
            for (uint64_t idx = 0; idx < arrayExtent; ++idx)
            {
                if (schemaEntry.type < NVTX_PAYLOAD_SCHEMA_ID_STATIC_START)
                {
                    // NOTE: PLACEHOLDER FOR GETTING PAYLOAD DATA FOR AN ENTRY IN THE SCHEMA
                    std::string payloadString;
                    ParsePredefinedType(schemaEntry, pPayloadEntry, payloadString);

                    NVTX_PAYLOAD_LOG_INFO("Schema: %s | Schema ID %llu | Entry Name: %s | Entry Value: %s |",
                        pPayloadSchema->name.c_str(), static_cast<unsigned long long>(pPayloadSchema->schemaId),
                        schemaEntry.name.c_str(), payloadString.c_str());

                    NVTX_FPRINTF(g_nvtxData.pFile, "| Entry Name: %s | Entry Value: %s |\n", schemaEntry.name.c_str(), payloadString.c_str());
                }
                else
                {
                    ParseNestedSchema(pPayloadSchema, schemaEntry, pPayloadEntry);
                }

                pPayloadEntry += typeSize;
            }

            if (dynamicOffset > 0)
            {
                dynamicOffset += typeSize * static_cast<size_t>(arrayExtent);
            }

            continue;
        }
        else if ((schemaEntry.flags & NVTX_PAYLOAD_ENTRY_FLAG_IS_ARRAY) == NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_ZERO_TERMINATED)
        {
            if (entryIdx != lastEntryIdx)
            {
                PrintError("NULL-terminated arrays are only supported as last entry");
                break;
            }

            if (schemaEntry.flags & NVTX_PAYLOAD_ENTRY_FLAG_POINTER ||
                schemaEntry.flags & NVTX_PAYLOAD_ENTRY_FLAG_DEEP_COPY)
            {
                PrintError("Pointer types and deep copy is not supported yet");
            }
            else if (IsCString(schemaEntry.type))
            {
                size_t offset = (dynamicOffset > 0) ? dynamicOffset : schemaEntry.offset;
                size_t maxSize = payloadSize - offset;
                size_t strLength = strlen(pPayloadEnd);
                if (strLength > maxSize)
                {
                    PrintError("NVTX payload: Null-terminated string is longer than payload");
                    strLength = maxSize;
                }

                NvtxSchemaEntry tmpEntry = schemaEntry;
                tmpEntry.extent = strLength + 1;

                // NOTE: PLACEHOLDER FOR GETTING PAYLOAD DATA FOR AN ENTRY IN THE SCHEMA
                std::string payloadString;
                ParsePredefinedType(tmpEntry, pPayloadEntry, payloadString);

                NVTX_PAYLOAD_LOG_INFO("Schema: %s | Schema ID %llu | Entry Name: %s | Entry Value: %s |",
                    pPayloadSchema->name.c_str(), static_cast<unsigned long long>(pPayloadSchema->schemaId),
                    schemaEntry.name.c_str(), payloadString.c_str());

                NVTX_FPRINTF(g_nvtxData.pFile, "| Entry Name: %s | Entry Value: %s |\n", schemaEntry.name.c_str(), payloadString.c_str());
            }
            else
            {
                PrintError("NULL-terminated arrays are only valid as string and for pointer types");
            }

            break;
        }

        // For dynamic layouts, update the dynamic offset after each entry.
        if (dynamicOffset > 0)
        {
            size_t typeSize = pPayloadSchema->GetSizeOfPayloadEntryType(schemaEntry.type);
            dynamicOffset += typeSize;
        }

        // Handle predefined types and nested schemas.
        if (schemaEntry.type < NVTX_PAYLOAD_SCHEMA_ID_STATIC_START)
        {
            // NOTE: PLACEHOLDER FOR GETTING PAYLOAD DATA FOR AN ENTRY IN THE SCHEMA
            std::string payloadString;
            ParsePredefinedType(schemaEntry, pPayloadEntry, payloadString);

            NVTX_PAYLOAD_LOG_INFO("Schema: %s | Schema ID %llu | Entry Name: %s | Entry Value: %s |",
                pPayloadSchema->name.c_str(), static_cast<unsigned long long>(pPayloadSchema->schemaId),
                schemaEntry.name.c_str(), payloadString.c_str());

            NVTX_FPRINTF(g_nvtxData.pFile, "| Entry Name: %s | Entry Value: %s |\n", schemaEntry.name.c_str(), payloadString.c_str());
        }
        else
        {
            ParseNestedSchema(pPayloadSchema, schemaEntry, pPayloadEntry);
        }
    }
}

void
ParsePayloadEnum(
    NvtxPayloadEnum *pPayloadEnum,
    const char *pPayloadBase,
    size_t payloadSize)
{
    // Check for null enum schema pointer.
    if (pPayloadEnum == nullptr)
    {
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload enum is null");
        return;
    }

    // Check for null payload pointer.
    if (pPayloadBase == nullptr)
    {
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload base pointer is null");
        return;
    }

    // Check for zero payload size.
    if (payloadSize == 0)
    {
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload size is 0");
        return;
    }

    // Check for zero enum size in the schema.
    if (pPayloadEnum->sizeOfEnum == 0)
    {
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload enum size is 0");
        return;
    }

    // Check that the payload size matches the expected enum size.
    if (payloadSize != pPayloadEnum->sizeOfEnum)
    {
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload enum size %zu != enum size %llu", payloadSize, static_cast<unsigned long long>(pPayloadEnum->sizeOfEnum));
        return;
    }

    // Handle 8-byte enums (most common for 64-bit values).
    if (pPayloadEnum->sizeOfEnum == 8)
    {
        uint64_t enumValue = *reinterpret_cast<const uint64_t*>(pPayloadBase);
        for (const NvtxEnumEntry& entry : pPayloadEnum->entries)
        {
            if (entry.value == enumValue)
            {
                // NOTE: PLACEHOLDER FOR GETTING PAYLOAD DATA FOR AN ENTRY IN THE SCHEMA
                NVTX_PAYLOAD_LOG_INFO("Enum: %s | Schema ID %llu | Entry Name: %s | Entry Value: %llu |",
                    pPayloadEnum->name.c_str(), static_cast<unsigned long long>(pPayloadEnum->schemaId),
                    entry.name.c_str(), static_cast<unsigned long long>(entry.value));

                NVTX_FPRINTF(g_nvtxData.pFile, "| Entry Name: %s | Entry Value: %llu |\n", entry.name.c_str(), static_cast<unsigned long long>(entry.value));

                return;
            }
        }
        // If the value is not found in the enum entries, log an error.
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload enum value %llu not found in enum entries", static_cast<unsigned long long>(enumValue));
    }
    // Handle 4-byte enums (for 32-bit values).
    else if (pPayloadEnum->sizeOfEnum == 4)
    {
        uint32_t enumValue = *reinterpret_cast<const uint32_t*>(pPayloadBase);
        for (const NvtxEnumEntry& entry : pPayloadEnum->entries)
        {
            if (entry.value == enumValue)
            {
                // NOTE: PLACEHOLDER FOR GETTING PAYLOAD DATA FOR AN ENTRY IN THE SCHEMA
                NVTX_PAYLOAD_LOG_INFO("Enum: %s | Schema ID %llu | Entry Name: %s | Entry Value: %llu |",
                    pPayloadEnum->name.c_str(), static_cast<unsigned long long>(pPayloadEnum->schemaId),
                    entry.name.c_str(), static_cast<unsigned long long>(enumValue));

                NVTX_FPRINTF(g_nvtxData.pFile, "| Entry Name: %s | Entry Value: %llu |\n", entry.name.c_str(), static_cast<unsigned long long>(enumValue));

                return;
            }
        }
        // If the value is not found in the enum entries, log an error.
        NVTX_PAYLOAD_LOG_ERROR("NVTX payload enum value %llu not found in enum entries", static_cast<unsigned long long>(enumValue));
    }
}

void
ParsePayload(
    NvtxPayloadAttributes *pPayloadAttributes,
    const char *pPayloadBase,
    size_t payloadSize)
{
    // Check for null schema/enum pointer.
    if (pPayloadAttributes == nullptr)
    {
        NVTX_PAYLOAD_LOG_ERROR("Schema provided is null");
        return;
    }

    // Check for null payload pointer.
    if (pPayloadBase == nullptr)
    {
        NVTX_PAYLOAD_LOG_ERROR("Payload provided is null");
        return;
    }

    // Check for zero payload size.
    if (payloadSize == 0)
    {
        NVTX_PAYLOAD_LOG_ERROR("Payload size is zero");
        return;
    }

    // If the payload is a schema, parse it as a structured payload.
    if (pPayloadAttributes->payloadType == CUPTI_NVTX_EXT_PAYLOAD_TYPE_SCHEMA)
    {
        NvtxPayloadSchema *pSchema = dynamic_cast<NvtxPayloadSchema *>(pPayloadAttributes);
        if (pSchema == nullptr)
        {
            NVTX_PAYLOAD_LOG_ERROR("Payload attributes are not of type NVTX_PAYLOAD_TYPE_SCHEMA");
            return;
        }
        ParsePayloadSchema(pSchema, pPayloadBase, payloadSize, nullptr);
    }
    // If the payload is an enum, parse it as an enum value.
    else if (pPayloadAttributes->payloadType == CUPTI_NVTX_EXT_PAYLOAD_TYPE_ENUM)
    {
        NvtxPayloadEnum *pEnum = dynamic_cast<NvtxPayloadEnum *>(pPayloadAttributes);
        if (pEnum == nullptr)
        {
            NVTX_PAYLOAD_LOG_ERROR("Payload attributes are not of type NVTX_PAYLOAD_TYPE_ENUM");
            return;
        }
        ParsePayloadEnum(pEnum, pPayloadBase, payloadSize);
    }
    // If the payload is a predefined type (not a schema or enum), parse and print it directly.
    else
    {
        NVTX_PAYLOAD_LOG_ERROR("Unsupported payload type %u", pPayloadAttributes->payloadType);
        return;
    }
}

void
CuptiParseNvtxPayload(
    uint32_t cuptiDomainId,
    nvtxPayloadData_t *pPayloadData,
    FILE *pFileHandle)
{
    // Set the log level to INFO for this parsing session.
    SetLogLevel(NVTX_PAYLOAD_LOG_INFO);

    // Set the global file pointer for logging, if provided.
    if (pFileHandle != NULL)
    {
        g_nvtxData.pFile = pFileHandle;
    }
    else
    {
        g_nvtxData.pFile = NULL;
    }

    NvtxPayloadAttributes *pSchema = nullptr;

    // Check for null payload data pointer.
    if (pPayloadData == nullptr)
    {
        NVTX_PAYLOAD_LOG_ERROR("Payload data is null");
        return;
    }

    // Ensure the global payload data types info is initialized.
    SetPayloadDataTypesInfo();

    // Extract the schema ID from the payload data.
    uint64_t schemaId = pPayloadData->schemaId;

    // If the schemaId indicates a custom schema (static or enum), fetch and parse it.
    if (schemaId >= NVTX_PAYLOAD_SCHEMA_ID_STATIC_START)
    {
        pSchema = GetNvtxPayloadAttributes(cuptiDomainId, schemaId);
        if (pSchema == nullptr)
        {
            NVTX_PAYLOAD_LOG_ERROR("Could not get schema for ID %llu", static_cast<unsigned long long>(schemaId));
            return;
        }
        // Parse the payload using the fetched schema.
        ParsePayload(pSchema, reinterpret_cast<const char*>(pPayloadData->payload), pPayloadData->size);
    }
    // If the schemaId is a predefined type, parse and print it directly.
    else
    {
        // NOTE: PLACEHOLDER FOR GETTING PAYLOAD DATA FOR AN ENTRY IN THE SCHEMA
        // Payload is a predefined type. No need to lookup a schema.
        std::string payloadString;
        const char *pPayload = reinterpret_cast<const char*>(pPayloadData->payload);
        char * pPayloadEntry = const_cast<char*>(pPayload);
        NvtxSchemaEntry schemaEntry;
        schemaEntry.type = schemaId;
        ParsePredefinedType(schemaEntry, pPayloadEntry, payloadString);

        NVTX_PAYLOAD_LOG_INFO("| Entry Type: %llu | Entry Value: %s |", static_cast<unsigned long long>(schemaId), payloadString.c_str());

        NVTX_FPRINTF(g_nvtxData.pFile, "| Entry Type: %llu | Entry Value: %s |\n", static_cast<unsigned long long>(schemaId), payloadString.c_str());
    }
}
