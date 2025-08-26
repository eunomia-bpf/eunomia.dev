/**
 * Copyright 2025 NVIDIA Corporation. All rights reserved
 * @file nvtx_payload_attributes.cpp
 * @brief Implementation of NVTX payload attributes and schema handling.
 *
 * Typical usage:
 *   - Used by the NVTX payload parser to fetch, store, and interpret schema and enum metadata.
 *   - Structures here are filled by CUPTI APIs and then used to parse binary payloads.
 *
 * Thread safety: These data structures themselves are not thread-safe. If accessed from multiple threads, external synchronization is required.
 */

#include <algorithm>
#include <inttypes.h>
#include <stdio.h>

#include "nvtx_payload_attributes.h"
#include "nvtx_payload_parser.h"

size_t
NvtxPayloadSchema::GetSizeOfPayloadEntryType(
    uint64_t type) const noexcept
{
    // First, try to get the size for a predefined (standard) payload type.
    size_t sizeOfType = GetSizeOfPayloadPredefinedType(type);
    if (sizeOfType)
    {
        // If a valid size is found for a predefined type, log and return it.
        NVTX_PAYLOAD_LOG_DEBUG("Found predefined type size %zu for type %llu", sizeOfType, static_cast<unsigned long long>(type));
        return sizeOfType;
    }
    // If not a predefined type, check if the type is a custom schema (static or enum).
    else if (type >= NVTX_PAYLOAD_SCHEMA_ID_STATIC_START)
    {
        // Look up the schema for this type using the domainId and type as schemaId.
        const NvtxPayloadAttributes* schema = GetNvtxPayloadAttributes(domainId, type);
        if (schema)
        {
            // If the schema is an enum type, return the size of the enum.
            if (schema->payloadType == CUPTI_NVTX_EXT_PAYLOAD_TYPE_ENUM)
            {
                size_t enumSize = static_cast<const NvtxPayloadEnum*>(schema)->sizeOfEnum;
                NVTX_PAYLOAD_LOG_DEBUG("Found enum size %zu for type %llu", enumSize, static_cast<unsigned long long>(type));
                return enumSize;
            }
            else
            {
                // Otherwise, treat it as a nested schema (struct-like).
                const NvtxPayloadSchema* plSchema = static_cast<const NvtxPayloadSchema*>(schema);

                // Only static layouts are supported as nested schemas. Dynamic schemas are not supported.
                if (plSchema->schemaType == NVTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC)
                {
                    NVTX_PAYLOAD_LOG_ERROR("Nested dynamic payload schemas are not supported");
                    return NvtxSchemaEntry::SizeInvalid;
                }

                // Ensure the static payload size is set by processing entries if needed.
                // This is important for correct offset and size calculations.
                const_cast<NvtxPayloadSchema*>(plSchema)->ProcessEntries();
                NVTX_PAYLOAD_LOG_DEBUG("Found schema size %llu for type %llu", static_cast<unsigned long long>(plSchema->payloadStaticSize), static_cast<unsigned long long>(type));
                return plSchema->payloadStaticSize;
            }
        }
    }

    // If we reach here, the type is invalid or unsupported (e.g., not found in predefined or custom schemas).
    NVTX_PAYLOAD_LOG_ERROR("Cannot determine size for invalid or unsupported NVTX binary payload entry or schema %llu", static_cast<unsigned long long>(type));
    return NvtxSchemaEntry::SizeInvalid;
}

size_t
NvtxPayloadSchema::GetSizeOfPayloadEntry(
    const NvtxSchemaEntry& entry) const noexcept
{
    // Check if the entry is a pointer or requires deep copy. Deep copy is not supported yet.
    // In both cases, we assume the entry has the size of a pointer.
    if ((entry.flags & NVTX_PAYLOAD_ENTRY_FLAG_DEEP_COPY) || (entry.flags & NVTX_PAYLOAD_ENTRY_FLAG_POINTER))
    {
        // Check if the payload data types vector is large enough to contain the address type.
        if (g_nvtxData.nvtxPayloadDataTypes.size() >= NVTX_PAYLOAD_ENTRY_TYPE_ADDRESS)
        {
            // Use the size of the address type (pointer size) for this entry.
            size_t pointerSize = g_nvtxData.nvtxPayloadDataTypes[NVTX_PAYLOAD_ENTRY_TYPE_ADDRESS].size;
            NVTX_PAYLOAD_LOG_DEBUG("Using pointer size %zu for entry with deep copy/pointer flags", pointerSize);
            return pointerSize;
        }
        else
        {
            // If the pointer size is not found, log an error and return invalid size.
            NVTX_PAYLOAD_LOG_ERROR("Pointer size not found in payload data types");
            return NvtxSchemaEntry::SizeInvalid;
        }
    }

    // Check for dynamic length arrays. Their size cannot be determined from the schema entry alone.
    // In dynamic layouts, entry offsets are determined at parse time, not statically.
    if (NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_TYPE(entry.flags) == NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_INDEX ||
        NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_TYPE(entry.flags) == NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_ZERO_TERMINATED)
    {
        NVTX_PAYLOAD_LOG_DEBUG("Dynamic array size detected for entry '%s'", entry.name.c_str());
        return NvtxSchemaEntry::SizeDynamic;
    }

    // Handle fixed-size embedded arrays, including C strings.
    if ((NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_TYPE(entry.flags) == NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_FIXED_SIZE) ||
        (entry.type >= NVTX_PAYLOAD_ENTRY_TYPE_CSTRING && entry.type <= NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF32))
    {
        // Get the size of the underlying type.
        size_t typeSize = GetSizeOfPayloadEntryType(entry.type);
        // Total size is extent (number of elements) times the type size.
        size_t totalSize = entry.extent * typeSize;
        NVTX_PAYLOAD_LOG_DEBUG("Fixed array size %zu (extent %llu * type size %zu) for entry '%s'", totalSize, static_cast<unsigned long long>(entry.extent), typeSize, entry.name.c_str());
        return totalSize;
    }

    // For all other cases, get the size of the entry's type (could be a nested schema or a simple type).
    size_t typeSize = GetSizeOfPayloadEntryType(entry.type);
    NVTX_PAYLOAD_LOG_DEBUG("Type size %zu for entry '%s'", typeSize, entry.name.c_str());
    return typeSize;
}

uint16_t
NvtxPayloadSchema::GetSizeOfLargestType(
    uint64_t schemaId) const noexcept
{
    // Retrieve the schema attributes for the given schemaId and current domainId.
    const NvtxPayloadAttributes* schema = GetNvtxPayloadAttributes(domainId, schemaId);
    if (!schema)
    {
        // If the schema is not found, log an error and return invalid size.
        NVTX_PAYLOAD_LOG_ERROR("Schema not found for ID %llu", static_cast<unsigned long long>(schemaId));
        return NvtxSchemaEntry::SizeInvalid;
    }

    // If the schema is an enum, return the size of the enum directly.
    if (schema->payloadType == CUPTI_NVTX_EXT_PAYLOAD_TYPE_ENUM)
    {
        const NvtxPayloadEnum* enumSchema = static_cast<const NvtxPayloadEnum*>(schema);
        NVTX_PAYLOAD_LOG_DEBUG("Enum size %llu for schema %llu", static_cast<unsigned long long>(enumSchema->sizeOfEnum), static_cast<unsigned long long>(schemaId));
        return static_cast<uint16_t>(enumSchema->sizeOfEnum);
    }

    // Otherwise, treat it as a schema with entries (struct-like).
    const NvtxPayloadSchema* plSchema = static_cast<const NvtxPayloadSchema*>(schema);
    uint16_t sizeMax = 0; // Track the largest size found.
    for (const auto& entry: plSchema->entries)
    {
        uint16_t size = 0;
        // If the entry type is itself a schema, recurse to find its largest type.
        if (entry.type >= NVTX_PAYLOAD_SCHEMA_ID_STATIC_START)
        {
            size = GetSizeOfLargestType(entry.type);
        }
        else
        {
            // Otherwise, get the size of the predefined type.
            size = GetSizeOfPayloadPredefinedType(entry.type);
        }

        // Update the maximum size if this entry's size is larger.
        if (size > sizeMax)
        {
            sizeMax = size;
            NVTX_PAYLOAD_LOG_DEBUG("New max size %u found for entry '%s'", sizeMax, entry.name.c_str());
        }
    }

    return sizeMax;
}

void
NvtxPayloadSchema::AlignOffset(
    size_t& entryOffset,
    uint64_t entryType) const noexcept
{
    // If packAlign is 1, no alignment is needed (byte alignment).
    if (packAlign == 1)
    {
        NVTX_PAYLOAD_LOG_DEBUG("No alignment needed for pack align 1");
        return;
    }

    // Determine the size of the type for alignment purposes.
    // For nested schemas, use the largest type size in the schema.
    size_t typeSize = 0;
    if (entryType >= NVTX_PAYLOAD_SCHEMA_ID_STATIC_START)
    {
        typeSize = GetSizeOfLargestType(entryType);
    }
    else
    {
        typeSize = GetSizeOfPayloadEntryType(entryType);
    }

    // The alignment requirement is the smaller of the type size and the packAlign value.
    size_t alignTo = (typeSize < packAlign) ? typeSize : packAlign;
    NVTX_PAYLOAD_LOG_DEBUG("Aligning offset %zu to %zu (type size %zu, pack align %zu)", entryOffset, alignTo, typeSize, packAlign);

    // The entryOffset is treated as a pointer for alignment calculation.
    void *pAddrToAlign = reinterpret_cast<void*>(entryOffset);

    // The buffer size is not known, so use SIZE_MAX as a placeholder.
    size_t sz = SIZE_MAX;

    // Use std::align to compute the next aligned address.
    // NOTE: This requires C++17 or later.
    void *pAlignedAddr = std::align(alignTo, typeSize, pAddrToAlign, sz);
    if (pAlignedAddr)
    {
        // If alignment is successful, update entryOffset to the aligned address.
        entryOffset = reinterpret_cast<size_t>(pAlignedAddr);
        NVTX_PAYLOAD_LOG_DEBUG("Aligned offset to %zu", entryOffset);
    }
    else
    {
        // If alignment fails, log an error.
        NVTX_PAYLOAD_LOG_ERROR("Failed to align offset %zu", entryOffset);
    }
}

bool
NvtxPayloadSchema::IsStaticPayloadSizePending() const
{
    // The payloadStaticSize is set to UnknownPayloadStaticSize (usually 0) if not yet determined.
    // Only static schemas (not dynamic) should have a static payload size.
    return (payloadStaticSize == UnknownPayloadStaticSize) && (schemaType == NVTX_PAYLOAD_SCHEMA_TYPE_STATIC);
}

void
NvtxPayloadSchema::SetSizeAndOffsets()
{
    NVTX_PAYLOAD_LOG_DEBUG("Setting size and offsets for schema '%s'", name.c_str());

    // If packing alignment is not set (0), use natural alignment (largest type size in the schema).
    if (packAlign == 0)
    {
        // "A struct instance will have the alignment of its widest scalar member."
        packAlign = GetSizeOfLargestType(schemaId);
        NVTX_PAYLOAD_LOG_DEBUG("Using natural alignment %zu", packAlign);
    }

    bool autoDetectOffset = true; // Whether to automatically detect offsets for entries.
    size_t entryOffset = 0;       // Tracks the current offset for the next entry.
    for (NvtxSchemaEntry& entry: entries)
    {
        // Check for unsupported features: deep copy and pointers are not supported in schemas.
        if ((entry.flags & NVTX_PAYLOAD_ENTRY_FLAG_DEEP_COPY) || (entry.flags & NVTX_PAYLOAD_ENTRY_FLAG_POINTER))
        {
            NVTX_PAYLOAD_LOG_ERROR("Schema '%s' contains unsupported features: deep copy and pointers", name.c_str());
        }

        // Handle invalid entry types (type == 0). This is likely the last entry in a null-terminated array.
        if (entry.type == 0)
        {
            NVTX_PAYLOAD_LOG_ERROR("Schema '%s' contains invalid entry type (0). Stop processing.", name.c_str());
            // Remove the invalid entry and stop processing further entries.
            entries.pop_back();
            break;
        }

        // Determine the size of the current entry.
        size_t entrySize = GetSizeOfPayloadEntry(entry);
        NVTX_PAYLOAD_LOG_DEBUG("Processing entry '%s' with size %zu", entry.name.c_str(), entrySize);

        // For the first entry, set the offset to the entry size (or use the provided offset if set).
        if (entryOffset == 0 && autoDetectOffset)
        {
            // Set next minimum offset for subsequent entries.
            entryOffset = entrySize;
            NVTX_PAYLOAD_LOG_DEBUG("First entry offset set to %zu", entryOffset);
        }
        else
        {
            // If this entry's offset was set by the user, use it and update entryOffset accordingly.
            if (entry.offset > 0)
            {
                // Set next minimum offset based on user-specified offset and entry size.
                entryOffset = entry.offset + entrySize;
                NVTX_PAYLOAD_LOG_DEBUG("Using user-specified offset %u, next offset will be %zu", entry.offset, entryOffset);
                // Continue auto-detection after a user-specified offset.
                autoDetectOffset = true;
            }
            else
            {
                // If auto-detection is disabled, skip offset detection for this entry.
                if (!autoDetectOffset)
                {
                    NVTX_PAYLOAD_LOG_DEBUG("Skipping offset detection for entry '%s'", entry.name.c_str());
                    continue;
                }

                // Align the current offset as required for this entry's type.
                AlignOffset(entryOffset, entry.type);
                // Set the entry's offset to the aligned offset.
                entry.offset = static_cast<uint32_t>(entryOffset);
                NVTX_PAYLOAD_LOG_DEBUG("Set offset %u for entry '%s'", entry.offset, entry.name.c_str());
                // Update entryOffset for the next entry.
                entryOffset += entrySize;
            }
        }

        // If the entry size is invalid, log an error and stop auto-detection for subsequent entries.
        if (entrySize == NvtxSchemaEntry::SizeInvalid)
        {
            NVTX_PAYLOAD_LOG_ERROR("Schema '%s': Invalid entry size for '%s', flags %u, type %llu", name.c_str(), entry.name.c_str(), entry.flags, static_cast<unsigned long long>(entry.type));
            autoDetectOffset = false;
        }

        // If the entry size is dynamic, stop auto-detection for subsequent entries.
        if (entrySize == NvtxSchemaEntry::SizeDynamic)
        {
            NVTX_PAYLOAD_LOG_DEBUG("Dynamic size detected for entry '%s', stopping offset detection", entry.name.c_str());
            autoDetectOffset = false;
        }
    }

    // If the static payload size was set to 'auto-detect' and all entries were processed, set the final payload size.
    if (IsStaticPayloadSizePending() && autoDetectOffset)
    {
        // Align the final offset for the schema as a whole.
        AlignOffset(entryOffset, schemaId);
        payloadStaticSize = static_cast<uint64_t>(entryOffset);
        NVTX_PAYLOAD_LOG_DEBUG("Schema '%s' payload size set to %llu", name.c_str(), static_cast<unsigned long long>(payloadStaticSize));
    }
}

void
NvtxPayloadSchema::ProcessEntries()
{
    // If the schema has already been processed, skip further processing.
    if (processed)
    {
        NVTX_PAYLOAD_LOG_DEBUG("Schema '%s' already processed", name.c_str());
        return;
    }

    // Only static and dynamic schema types are supported. Union types are not supported yet.
    if ((schemaType != NVTX_PAYLOAD_SCHEMA_TYPE_STATIC) && (schemaType != NVTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC))
    {
        NVTX_PAYLOAD_LOG_ERROR("Schema type %u not supported", schemaType);
        return;
    }

    // Determine if offsets and static payload size need to be set.
    bool determineOffsets = false;

    // If the static payload size is pending, we need to determine offsets.
    if (IsStaticPayloadSizePending())
    {
        NVTX_PAYLOAD_LOG_DEBUG("Static payload size pending for schema '%s'", name.c_str());
        determineOffsets = true;
    }

    // If not already determined, check if any entry (other than the first) has an unset offset.
    if (!determineOffsets)
    {
        for (const NvtxSchemaEntry& entry: entries)
        {
            if (entry.offset > 0)
            {
                continue;
            }

            // The offset of the first entry is likely (and valid) to be 0.
            if (entry == entries.front())
            {
                continue;
            }

            NVTX_PAYLOAD_LOG_DEBUG("Found entry '%s' without offset, will determine offsets", entry.name.c_str());
            determineOffsets = true;
            break;
        }
    }

    // If offsets need to be determined, call SetSizeAndOffsets and handle any exceptions.
    if (determineOffsets)
    {
        try
        {
            SetSizeAndOffsets();
        }
        catch (const std::exception& e)
        {
            NVTX_PAYLOAD_LOG_ERROR("Failed to set entry offsets for schema '%s': %s", name.c_str(), e.what());
        }
    }

    // Mark the schema as processed to avoid redundant work in the future.
    processed = true;
    NVTX_PAYLOAD_LOG_DEBUG("Schema '%s' processing completed", name.c_str());
}

bool
NvtxPayloadSchema::IsOffsetValid(
    const NvtxSchemaEntry& entry) const noexcept
{
    // All offsets other than 0 are assumed to be valid (explicitly set by user or auto-detected).
    if (entry.offset != 0)
    {
        NVTX_PAYLOAD_LOG_DEBUG("Entry '%s' has valid non-zero offset %u", entry.name.c_str(), entry.offset);
        return true;
    }

    // Offset 0 is only valid for the first entry in the schema.
    if (entries.front() == entry)
    {
        NVTX_PAYLOAD_LOG_DEBUG("Entry '%s' is first entry with valid zero offset", entry.name.c_str());
        return true;
    }

    // Any other case with offset 0 is invalid.
    NVTX_PAYLOAD_LOG_DEBUG("Entry '%s' has invalid zero offset", entry.name.c_str());
    return false;
}
