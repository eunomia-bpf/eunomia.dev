/*
 * Copyright 2025 NVIDIA Corporation. All rights reserved
 * @file nvtx_payload_parser.h
 * @brief Declarations for parsing and interpreting NVTX extended payloads using CUPTI.
 *
 * This header provides the main interfaces, data structures, and logging utilities
 * for decoding NVTX payloads received from CUPTI activity records. It supports
 * both schema-based and enum-based payloads, and includes facilities for
 * fetching schema definitions, parsing binary payloads, and logging results.
 *
 * Typical usage:
 *   - Call CuptiParseNvtxPayload() from a CUPTI buffer completed callback to decode
 *     and print NVTX payloads.
 *   - Use the provided logging macros for consistent output.
 *
 * Thread safety: Unless otherwise noted, global data structures are not thread-safe.
 */

#pragma once

#include <stdio.h>
#include <memory>
#include <iostream>
#include <inttypes.h>
#include <unordered_map>

// NVTX headers
#include "nvtx3/nvToolsExt.h"
#include <nvtx3/nvToolsExtPayload.h>

// CUPTI header
#include <cupti_activity.h>

// NVTX payload parser headers
#include <nvtx_payload_attributes.h>

// Extern C functions to get NVTX payload attributes and entry type info

/**
 * \brief Get NVTX payload attribute
 *
 * This API returns the NVTX payload attribute for the given domainId and schemaId.
 * The payload attribute is used to parse the extended payload that is being passed
 * in the NVTX record.
 *
 * \param domainId The domain ID of the NVTX record
 * \param schemaId The schema ID of the NVTX record
 * \param pPayloadAttributes Pointer to the payload attribute structure, user will receive the payload attributes in this structure.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \param pPayloadAttributes is NULL
 */
extern "C" CUptiResult CUPTIAPI cuptiActivityGetNvtxExtPayloadAttr(uint32_t cuptiDomainId, uint64_t schemaId, CUpti_NvtxExtPayloadAttr *pPayloadAttributes);

/**
 * \brief Get NVTX payload entry type info
 *
 * This API returns the NVTX payload entry type info for pre-defined entry types in NVTX.
 * The payload entry type info is used to parse the extended payload that is being passed
 * in the NVTX record.
 *
 * Refer to nvtxDetail/nvtxExtPayloadTypeInfo.h for the definition of nvtxPayloadEntryTypeInfo_t.
 *
 * \retval Pointer to the NVTX payload entry type info
 */
extern "C" const nvtxPayloadEntryTypeInfo_t * CUPTIAPI cuptiActivityGetNvtxExtPayloadEntryTypeInfo();

// NOTE: Can switch to unique_ptr if ownership semantics are needed for NvtxPayloadAttributes
using NvtxSchemaIdMap = std::unordered_map<uint64_t, NvtxPayloadAttributes *>;

/**
 * \brief Describes a payload data type's size and alignment.
 */
struct NvtxPayloadDataType
{
    uint16_t size;  ///< Size of the data type in bytes
    uint16_t align; ///< Alignment of the data type in bytes

    NvtxPayloadDataType(uint16_t size, uint16_t align) : size(size), align(align) {}
};

static const size_t InvalidTypeSize = 0;

/**
 * \brief Global structure to store NVTX payload schema attributes and type information.
 *
 * This singleton holds all schema-related metadata for parsing NVTX payloads received from CUPTI.
 * NOTE: Cleanup of this structure is managed by the application, as it contains pointers to dynamically allocated
 *       NvtxPayloadAttributes objects. The application must ensure to free these attributes when they are no longer needed.
 *       This structure is not thread-safe and should be accessed in a single-threaded context or with proper synchronization.
 */
struct GlobalNvtxData
{
    /**
     * \brief Maps CUPTI domain IDs to schema ID maps.
     *
     * For each CUPTI domain ID, maps schema IDs to the corresponding NVTX payload attributes.
     * Attributes are owned by unique pointers to ensure proper memory management.
     */
    std::unordered_map<uint32_t, NvtxSchemaIdMap> nvtxPayloadAttributeMap;

    /**
     * \brief Stores type information for each NVTX payload entry.
     *
     * Each entry corresponds to a standard payload entry type with defined size and alignment.
     */
    std::vector<NvtxPayloadDataType> nvtxPayloadDataTypes;

    /**
     * \brief The logging level for NVTX payload parser.
     */
    uint32_t logLevel;

    /**
     * \brief File pointer for logging NVTX payload data.
     */
    FILE *pFile;
};

/**
 * \brief Singleton instance of GlobalNvtxData.
 */
static GlobalNvtxData g_nvtxData;

/**
 * @brief Initializes the global NVTX payload data types information.
 *
 * This function populates the global vector of payload data types (size and alignment)
 * by querying CUPTI for the available NVTX payload entry types. This information is
 * required for correct parsing and alignment of NVTX payloads.
 *
 * It is safe to call this function multiple times; initialization will only occur once.
 */
void SetPayloadDataTypesInfo();

/**
 * @brief Entry point for parsing NVTX extended payload marker data received from CUPTI.
 *
 * This function is typically called from the CUPTI buffer completed callback. It handles
 * records with payloadKind == CUPTI_METRIC_VALUE_KIND_NVTX_EXTENDED_PAYLOAD, parses the
 * payload data, and prints/logs the interpreted result.
 *
 * @param cuptiDomainId   The CUPTI domain ID for the NVTX record.
 * @param pPayloadData    Pointer to the payload data structure (contains schemaId, payload, size).
 * @param pFileHandle     Optional file handle for logging output (can be NULL).
 */
void CuptiParseNvtxPayload(uint32_t cuptiDomainId, nvtxPayloadData_t *pPayloadData, FILE *pFileHandle = NULL);

/**
 * @brief Frees all cached NVTX payload attributes.
 *
 * This function iterates through all stored NVTX payload attributes in the global map
 * and deallocates them. It should be called when the application is done with NVTX payload
 * parsing to prevent memory leaks.
 * 
 * This function is not thread-safe and should be called in a single-threaded context.
 */
void FreeAllNvtxPayloadAttributes();

/**
 * @brief Fetches and caches NVTX payload attributes from CUPTI.
 *
 * This function retrieves payload schema and enum attributes registered via NVTX APIs
 * such as nvtxDomainRegisterPayloadSchema() and nvtxDomainRegisterPayloadEnum(), and
 * stores them in a global map for future lookups. If the attributes are already cached,
 * it returns them directly.
 *
 * @param cuptiDomainId The CUPTI domain ID for the NVTX record.
 * @param schemaId      The schema ID for the NVTX record.
 * @return Pointer to the payload attributes, or nullptr if not found or on error.
 */
NvtxPayloadAttributes *GetNvtxPayloadAttributes(uint32_t cuptiDomainId, uint64_t schemaId);

/**
 * @brief Parses a nested schema entry within a parent NVTX payload schema.
 *
 * This function is called when a schema entry refers to another schema (i.e., a nested structure).
 * It fetches the nested schema's attributes, determines the size, and recursively parses the nested payload.
 *
 * @param pPayloadSchema   The parent schema containing the nested entry.
 * @param entry            The schema entry describing the nested type.
 * @param pPayloadEntry    Pointer to the binary payload data for the nested entry.
 */
void
ParseNestedSchema(const NvtxPayloadSchema *pPayloadSchema, const NvtxSchemaEntry& entry, const char *pPayloadEntry);

/**
 * @brief Recursively parses a payload schema and prints field values.
 *
 * This function traverses the entries of a given NVTX payload schema, interprets each field
 * according to its type, and prints the parsed values. Handles nested schemas, arrays,
 * dynamic/static layouts, and various entry flags.
 *
 * @param pPayloadSchema        The schema describing the payload structure.
 * @param pPayloadBase          Pointer to the start of the binary payload data.
 * @param payloadSize           Size of the payload data in bytes.
 * @param pNestedSchemaEntry    (Optional) Pointer to the parent schema entry if this is a nested schema.
 */
void ParsePayloadSchema(NvtxPayloadSchema *pPayloadSchema, const char *pPayloadBase, size_t payloadSize, NvtxSchemaEntry *pNestedSchemaEntry = nullptr);

/**
 * @brief Parses an NVTX payload enum and prints the corresponding entry value.
 *
 * This function interprets the binary payload as an enum value, looks up the corresponding
 * entry in the enum schema, and prints/logs the result. Handles both 4-byte and 8-byte enums.
 *
 * @param pPayloadEnum   The enum schema describing the possible values.
 * @param pPayloadBase   Pointer to the binary payload data for the enum value.
 * @param payloadSize    Size of the payload data in bytes.
 */
void ParsePayloadEnum(NvtxPayloadEnum *pPayloadEnum, const char *pPayloadBase, size_t payloadSize);

/**
 * @brief Parses a payload using the given payload attributes (schema or enum).
 *
 * This function determines the type of the payload (schema or enum), and dispatches
 * to the appropriate parsing function. If the payload is a predefined type (not a schema),
 * it parses and prints the value directly.
 *
 * @param pPayloadAttributes Pointer to the payload attributes (schema or enum).
 * @param pPayloadBase       Pointer to the binary payload data.
 * @param payloadSize        Size of the payload data in bytes.
 */
void ParsePayload(NvtxPayloadAttributes *pPayloadAttributes, const char *pPayloadBase, size_t payloadSize);

// Utility functions for size and alignment
/**
 * @brief Checks if the given entry type corresponds to a C string type.
 *
 * This function is used to determine if a payload entry type is a C string
 * (either standard or UTF-8 encoded). This is useful for special handling
 * of string types during payload parsing.
 *
 * @param entryType The NVTX payload entry type identifier.
 * @return true if the type is a C string or C string UTF-8; false otherwise.
 */
bool IsCString(uint64_t entryType);

/**
 * @brief Returns the size (in bytes) of a predefined NVTX payload entry type.
 *
 * For standard NVTX types, size and alignment data is fetched from g_nvtxData.nvtxPayloadDataTypes.
 * For special cases (not in the standard array), GetSizeOfFixedSizeTypes() is used.
 * Handles special cases for registered string handles and unknown types.
 *
 * @param type The NVTX payload entry type identifier.
 * @return The size in bytes of the type, or InvalidTypeSize (usually 0) if unknown.
 */
uint16_t GetSizeOfPayloadPredefinedType(uint64_t type);

/**
 * @brief Returns the size (in bytes) of fixed-size payload types with non-standard sizes.
 *
 * This function is used for NVTX payload entry types that are not included in the
 * standard g_nvtxData.nvtxPayloadDataTypes vector, but have well-known, fixed sizes.
 * It is typically used for special types such as 16-bit floats, 128-bit integers, and
 * various string encodings.
 *
 * @param type The NVTX payload entry type identifier.
 * @return The size in bytes of the type, or InvalidTypeSize (usually 0) if unknown.
 */
uint16_t GetSizeOfFixedSizeTypes(uint64_t type);

// Utility functions for parsing NVTX payload pre-defined entries
/**
 * @brief Parses a predefined NVTX payload entry and converts it to a string representation.
 *
 * This function interprets the binary data at pPayloadEntry according to the type specified
 * in the schema entry, and appends a human-readable string representation to the output.
 * Handles all standard NVTX types, including numbers, pointers, and strings.
 *
 * @param entry         The schema entry describing the type and extent of the payload.
 * @param pPayloadEntry Pointer to the binary payload data for this entry.
 * @param output        String to which the parsed value will be appended.
 */
void
ParsePredefinedType(const NvtxSchemaEntry& entry, char* pPayloadEntry, std::string& output);


// Logging utilities
/**
 * \brief Logging levels for NVTX payload parser.
 *
 * Use these levels to control the verbosity of log output:
 *   - NVTX_PAYLOAD_LOG_ERROR:   Only critical errors.
 *   - NVTX_PAYLOAD_LOG_WARNING: Warnings and errors.
 *   - NVTX_PAYLOAD_LOG_INFO:    Informational messages, warnings, and errors.
 *   - NVTX_PAYLOAD_LOG_DEBUG_INFO: Debug, info, warnings, and errors.
 */
typedef enum
{
    NVTX_PAYLOAD_LOG_ERROR = 0,
    NVTX_PAYLOAD_LOG_WARNING,
    NVTX_PAYLOAD_LOG_INFO,
    NVTX_PAYLOAD_LOG_DEBUG_INFO
} NvtxPayloadLogLevel;

/**
 * @brief Logs messages with the specified log level.
 *
 * This function formats and prints log messages to either std::cerr or std::cout,
 * depending on the log level. It supports printf-style formatting and prepends
 * each message with a standard NVTX payload log prefix and the log level.
 *
 * @param level   The log level for the message (error, warning, info, debug).
 * @param format  The printf-style format string for the message.
 * @param ...     Additional arguments for the format string.
 */
void Log(NvtxPayloadLogLevel level, const char* format, ...);

/**
 * @brief Sets the log level for NVTX payload parser.
 *
 * This function allows the user to set the minimum log level for messages to be printed.
 * Messages with a lower log level will be ignored by the Log() function.
 *
 * @param level The desired log level (error, warning, info, debug).
 */
void SetLogLevel(NvtxPayloadLogLevel level);

/**
 * \brief NVTX-specific convenience macros for logging.
 *
 * Use these macros for consistent logging throughout the NVTX payload parser codebase.
 * Example:
 *   NVTX_PAYLOAD_LOG_INFO("Parsed payload for schema %llu", schemaId);
 */
#define NVTX_PAYLOAD_LOG_ERROR(...)   Log(NVTX_PAYLOAD_LOG_ERROR, __VA_ARGS__)
#define NVTX_PAYLOAD_LOG_WARNING(...) Log(NVTX_PAYLOAD_LOG_WARNING, __VA_ARGS__)
#define NVTX_PAYLOAD_LOG_INFO(...)    Log(NVTX_PAYLOAD_LOG_INFO, __VA_ARGS__)
#define NVTX_PAYLOAD_LOG_DEBUG(...)   Log(NVTX_PAYLOAD_LOG_DEBUG_INFO, __VA_ARGS__)

/**
 * \brief Print the payload entry name and value in a file.
 */
#define NVTX_FPRINTF(pFile, ...)     \
    if (pFile != NULL)               \
    {                                \
        fprintf(pFile, __VA_ARGS__); \
    }
