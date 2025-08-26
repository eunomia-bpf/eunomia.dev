/*
 * Copyright 2025 NVIDIA Corporation. All rights reserved
 * @file nvtx_payload_attributes.h
 * @brief Data structures and interfaces for representing and manipulating NVTX payload attributes, schemas, and enums.
 *
 * This header defines the core types used for describing NVTX payload schemas and enums, as well as their entries and attributes.
 * It is used in conjunction with CUPTI to parse and interpret extended NVTX payloads, supporting both static and dynamic schemas.
 *
 * Typical usage:
 *   - Used by the NVTX payload parser to fetch, store, and interpret schema and enum metadata.
 *   - Structures here are filled by CUPTI APIs and then used to parse binary payloads.
 *
 * Thread safety: These data structures themselves are not thread-safe. If accessed from multiple threads, external synchronization is required.
 */

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>

// Standard NVTX headers
#include "nvtx3/nvToolsExt.h"
#include <nvtx3/nvToolsExtPayload.h>

constexpr uint64_t UnknownPayloadStaticSize = 0;

// The data related to this structure is provided by
// CUPTI API cuptiActivityGetNvtxExtPayloadAttr()
struct NvtxPayloadAttributes
{
    uint32_t payloadType;
    uint32_t domainId;
    uint64_t schemaId;
    std::string name = {};

    virtual ~NvtxPayloadAttributes() = default;
};

// NVTX_PAYLOAD_TYPE_SCHEMA
struct NvtxSchemaEntry
{
    uint64_t flags = 0;           // flag for type specialization
    uint64_t type = 0;            // predefined type or custom schema ID
    std::string name = {};        // name of field
    std::string description = {}; // description of field
    uint64_t extent = 0;          // string or array length or union selector
    uint64_t offset = 0;          // offset in the structure (in bytes)

    static const size_t SizeDynamic = SIZE_MAX;
    static const size_t SizeInvalid = 0;

    bool operator==(const NvtxSchemaEntry& other) const
    {
        return offset == other.offset && type == other.type &&
            flags == other.flags && extent == other.extent &&
            name == other.name && description == other.description;
    }

    bool operator!=(const NvtxSchemaEntry& other) const
    {
        return !operator==(other);
    }
};

using SchemaEntries = std::vector<NvtxSchemaEntry>;
struct NvtxPayloadSchema: NvtxPayloadAttributes
{
    uint64_t fieldMask;
    uint64_t schemaType;
    uint64_t flags;
    uint64_t payloadStaticSize;
    uint64_t packAlign;
    SchemaEntries entries;
    bool processed;

    // Checks if the offset of a schema entry is valid.
    // For most entries, a non-zero offset is considered valid. For the first entry, zero is also valid.
    // This function is used to ensure that entry offsets are set correctly before parsing payloads.
    // Parameters:
    //   entry - The schema entry whose offset validity is to be checked.
    // Returns:
    //   true if the offset is valid for this entry; false otherwise.
    bool IsOffsetValid(const NvtxSchemaEntry& entry) const noexcept;

    // Processes the schema entries to ensure all offsets and the static payload size are set.
    // This function is idempotent and will only process entries if they haven't been processed already.
    // It also checks for unsupported schema types and handles exceptions during offset calculation.
    void ProcessEntries();

    bool IsProcessed() const noexcept
    {
        return processed;
    }

    // Aligns the entryOffset to the required alignment for the given entryType.
    // This ensures that the entry starts at a memory address that satisfies the alignment requirements
    // of the type, which is important for correct and efficient memory access.
    // Parameters:
    //   entryOffset - (in/out) The current offset in bytes. Will be updated to the next aligned offset.
    //   entryType   - The type identifier for the entry, used to determine alignment requirements.
    void AlignOffset(size_t& entryOffset, uint64_t entryType) const noexcept;

    // This file implements the NvtxPayloadSchema class and related logic for handling NVTX payload schemas and entries.
    // It provides methods for determining the size and offsets of payload entries, handling nested schemas, and managing schema processing.

    // Returns the size (in bytes) of a payload entry type, which can be either a predefined type or a custom schema.
    // This function is central to determining how much memory is needed for a given payload entry.
    // Parameters:
    //   type - The type identifier for the payload entry. This can be a predefined type or a schema ID.
    // Returns:
    //   The size in bytes of the entry type, or NvtxSchemaEntry::SizeInvalid if the type is invalid or unsupported.
    size_t GetSizeOfPayloadEntryType(uint64_t type) const noexcept;

    // Returns the size (in bytes) of a schema entry. This function also handles nested
    // schemas and entry flags.
    size_t GetSizeOfPayloadEntry(const NvtxSchemaEntry& entry) const noexcept;

private:
    // Returns the size (in bytes) of the largest type in the schema identified by schemaId.
    // This is used for alignment and packing calculations, ensuring that the structure is properly aligned in memory.
    // Parameters:
    //   schemaId - The schema ID whose largest entry type size is to be determined.
    // Returns:
    //   The size in bytes of the largest entry type, or NvtxSchemaEntry::SizeInvalid if the schema is not found.
    uint16_t GetSizeOfLargestType(uint64_t schemaId) const noexcept;

    // Checks if the static payload size for a static schema is still pending (i.e., not set yet).
    // This is used to determine whether the schema's size and offsets need to be auto-detected.
    // Returns:
    //   true if the static payload size is unknown and the schema is static; false otherwise.
    inline bool IsStaticPayloadSizePending() const;

    // Iterates over all entries of the schema and sets their offsets and the total static payload size if not already set.
    // Entry offsets can only be determined until the first variable-size field is encountered.
    // This function is critical for preparing the schema for parsing payloads, ensuring all offsets and sizes are correct.
    void SetSizeAndOffsets();
};


// NVTX_PAYLOAD_TYPE_ENUM
struct NvtxEnumEntry
{
    std::string name = {};
    uint64_t value = 0;
    int8_t isFlag = 0;
};

using EnumEntries = std::vector<NvtxEnumEntry>;
struct NvtxPayloadEnum: NvtxPayloadAttributes
{
    uint64_t fieldMask;
    EnumEntries entries;
    uint64_t sizeOfEnum;
};
