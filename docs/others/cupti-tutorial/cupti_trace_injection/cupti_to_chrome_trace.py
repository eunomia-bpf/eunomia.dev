#!/usr/bin/env python3

import json
import re
import argparse

def parse_cupti_trace(filename):
    """Parse CUPTI trace data and convert to Chrome Trace Format"""
    
    events = []
    
    # Regular expressions for different trace line formats
    runtime_pattern = r'RUNTIME \[ (\d+), (\d+) \] duration (\d+), "([^"]+)", cbid (\d+), processId (\d+), threadId (\d+), correlationId (\d+)'
    driver_pattern = r'DRIVER \[ (\d+), (\d+) \] duration (\d+), "([^"]+)", cbid (\d+), processId (\d+), threadId (\d+), correlationId (\d+)'
    kernel_pattern = r'CONCURRENT_KERNEL \[ (\d+), (\d+) \] duration (\d+), "([^"]+)", correlationId (\d+)'
    overhead_pattern = r'OVERHEAD ([A-Z_]+) \[ (\d+), (\d+) \] duration (\d+), (\w+), id (\d+), correlation id (\d+)'
    memory_pattern = r'MEMORY2 \[ (\d+) \] memoryOperationType (\w+), memoryKind (\w+), size (\d+), address (\d+)'
    memcpy_pattern = r'MEMCPY "([^"]+)" \[ (\d+), (\d+) \] duration (\d+), size (\d+), copyCount (\d+), srcKind (\w+), dstKind (\w+), correlationId (\d+)'
    grid_pattern = r'\s+grid \[ (\d+), (\d+), (\d+) \], block \[ (\d+), (\d+), (\d+) \]'
    device_pattern = r'\s+deviceId (\d+), contextId (\d+), streamId (\d+)'
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line or line.startswith('Calling CUPTI') or line.startswith('Enabling') or \
           line.startswith('Disabling') or line.startswith('Found') or \
           line.startswith('Configuring') or line.startswith('It took') or \
           line.startswith('Activity buffer'):
            i += 1
            continue
            
        # Parse RUNTIME events
        match = re.search(runtime_pattern, line)
        if match:
            start_time = int(match.group(1))
            _ = int(match.group(2))  # end_time (unused)
            duration = int(match.group(3))
            name = match.group(4)
            cbid = match.group(5)
            process_id = int(match.group(6))
            thread_id = int(match.group(7))
            correlation_id = int(match.group(8))
            
            # Convert to microseconds from nanoseconds
            events.append({
                "name": f"Runtime: {name}",
                "ph": "X",  # Complete event
                "ts": start_time / 1000,  # Convert ns to µs
                "dur": duration / 1000,
                "tid": thread_id,
                "pid": process_id,
                "cat": "CUDA_Runtime",
                "args": {
                    "cbid": cbid,
                    "correlationId": correlation_id
                }
            })
            i += 1
            continue
            
        # Parse DRIVER events
        match = re.search(driver_pattern, line)
        if match:
            start_time = int(match.group(1))
            _ = int(match.group(2))  # end_time (unused)
            duration = int(match.group(3))
            name = match.group(4)
            cbid = match.group(5)
            process_id = int(match.group(6))
            thread_id = int(match.group(7))
            correlation_id = int(match.group(8))
            
            events.append({
                "name": f"Driver: {name}",
                "ph": "X",
                "ts": start_time / 1000,
                "dur": duration / 1000,
                "tid": thread_id,
                "pid": process_id,
                "cat": "CUDA_Driver",
                "args": {
                    "cbid": cbid,
                    "correlationId": correlation_id
                }
            })
            i += 1
            continue
            
        # Parse CONCURRENT_KERNEL events
        match = re.search(kernel_pattern, line)
        if match:
            start_time = int(match.group(1))
            _ = int(match.group(2))  # end_time (unused)
            duration = int(match.group(3))
            name = match.group(4)
            correlation_id = int(match.group(5))
            
            kernel_info = {
                "name": f"Kernel: {name}",
                "ph": "X",
                "ts": start_time / 1000,
                "dur": duration / 1000,
                "cat": "GPU_Kernel",
                "args": {
                    "correlationId": correlation_id
                }
            }
            
            # Check next lines for additional kernel info
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                grid_match = re.search(grid_pattern, next_line)
                if grid_match:
                    kernel_info["args"]["grid"] = [
                        int(grid_match.group(1)),
                        int(grid_match.group(2)),
                        int(grid_match.group(3))
                    ]
                    kernel_info["args"]["block"] = [
                        int(grid_match.group(4)),
                        int(grid_match.group(5)),
                        int(grid_match.group(6))
                    ]
                    i += 1
                    
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                device_match = re.search(device_pattern, next_line)
                if device_match:
                    device_id = int(device_match.group(1))
                    context_id = int(device_match.group(2))
                    stream_id = int(device_match.group(3))
                    
                    # Use stream as thread ID for GPU events
                    kernel_info["tid"] = f"GPU{device_id}_Stream{stream_id}"
                    kernel_info["pid"] = f"Device_{device_id}"
                    kernel_info["args"]["deviceId"] = device_id
                    kernel_info["args"]["contextId"] = context_id
                    kernel_info["args"]["streamId"] = stream_id
                    i += 1
                    
            events.append(kernel_info)
            i += 1
            continue
            
        # Parse OVERHEAD events
        match = re.search(overhead_pattern, line)
        if match:
            overhead_type = match.group(1)
            start_time = int(match.group(2))
            end_time = int(match.group(3))
            duration = int(match.group(4))
            overhead_target = match.group(5)
            overhead_id = int(match.group(6))
            correlation_id = int(match.group(7))
            
            events.append({
                "name": f"Overhead: {overhead_type}",
                "ph": "X",
                "ts": start_time / 1000,
                "dur": duration / 1000,
                "tid": overhead_id,
                "pid": "CUPTI_Overhead",
                "cat": "Overhead",
                "args": {
                    "type": overhead_type,
                    "target": overhead_target,
                    "correlationId": correlation_id
                }
            })
            i += 1
            continue
            
        # Parse MEMCPY events
        match = re.search(memcpy_pattern, line)
        if match:
            copy_type = match.group(1)
            start_time = int(match.group(2))
            _ = int(match.group(3))  # end_time (unused)
            duration = int(match.group(4))
            size = int(match.group(5))
            copy_count = int(match.group(6))
            src_kind = match.group(7)
            dst_kind = match.group(8)
            correlation_id = int(match.group(9))
            
            memcpy_info = {
                "name": f"MemCopy: {copy_type}",
                "ph": "X",
                "ts": start_time / 1000,
                "dur": duration / 1000,
                "cat": "MemCopy",
                "args": {
                    "type": copy_type,
                    "size": size,
                    "copyCount": copy_count,
                    "srcKind": src_kind,
                    "dstKind": dst_kind,
                    "correlationId": correlation_id
                }
            }
            
            # Check next line for device info
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                device_match = re.search(device_pattern, next_line)
                if device_match:
                    device_id = int(device_match.group(1))
                    context_id = int(device_match.group(2))
                    stream_id = int(device_match.group(3))
                    
                    memcpy_info["tid"] = f"GPU{device_id}_Stream{stream_id}"
                    memcpy_info["pid"] = f"Device_{device_id}"
                    memcpy_info["args"]["deviceId"] = device_id
                    memcpy_info["args"]["contextId"] = context_id
                    memcpy_info["args"]["streamId"] = stream_id
                    i += 1
                else:
                    # Default to host thread if no device info
                    memcpy_info["tid"] = "MemCopy_Operations"
                    memcpy_info["pid"] = "MemCopy"
                    
            events.append(memcpy_info)
            i += 1
            continue
            
        # Parse MEMORY2 events
        match = re.search(memory_pattern, line)
        if match:
            timestamp = int(match.group(1))
            operation = match.group(2)
            memory_kind = match.group(3)
            size = int(match.group(4))
            address = int(match.group(5))
            
            # Use instant event for memory operations
            events.append({
                "name": f"Memory: {operation} ({memory_kind})",
                "ph": "i",  # Instant event
                "ts": timestamp / 1000,
                "tid": "Memory_Operations",
                "pid": "Memory",
                "cat": "Memory",
                "s": "g",  # Global scope
                "args": {
                    "operation": operation,
                    "kind": memory_kind,
                    "size": size,
                    "address": hex(address)
                }
            })
            i += 1
            continue
            
        i += 1
    
    return events

def main():
    parser = argparse.ArgumentParser(description='Convert CUPTI trace data to Chrome Trace Format')
    parser.add_argument('input', help='Input CUPTI trace file')
    parser.add_argument('-o', '--output', default='trace.json', 
                       help='Output Chrome trace JSON file (default: trace.json)')
    args = parser.parse_args()
    
    print(f"Parsing CUPTI trace file: {args.input}")
    events = parse_cupti_trace(args.input)
    
    print(f"Found {len(events)} events")
    
    # Write Chrome Trace Format JSON
    trace_data = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "tool": "CUPTI to Chrome Trace Converter",
            "format": "Chrome Trace Format"
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(trace_data, f, indent=2)
    
    print(f"Chrome trace file written to: {args.output}")
    print("\nTo visualize the trace:")
    print("1. Open Chrome or Edge browser")
    print("2. Navigate to chrome://tracing or edge://tracing")
    print("3. Click 'Load' and select the generated JSON file")
    print("\nAlternatively, visit https://ui.perfetto.dev/ and drag the JSON file there")

if __name__ == '__main__':
    main()