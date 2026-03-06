// Host-side registration stub for hand-written PTX device functions.
// This replaces the nvcc-generated host code that would normally come from
// compiling inject_funcs.cu. It registers the fatbin (containing the compiled
// PTX cubins) with the CUDA runtime so NVBit can discover device functions.
//
// The fatbin data is embedded via inject_funcs_embed.S into .nv_fatbin section.
// The wrapper struct is placed in .nvFatBinSegment so cuobjdump can find it.
// Hidden visibility on the data symbol ensures R_X86_64_RELATIVE relocation
// (not R_X86_64_64), which cuobjdump can resolve statically.

#include <cstdio>

// CUDA internal registration API (stable ABI, not in public headers)
extern "C" {
    void** __cudaRegisterFatBinary(void*);
    void __cudaRegisterFatBinaryEnd(void**);
    void __cudaUnregisterFatBinary(void**);
}

// Symbol from inject_funcs_embed.S: fatbin data in .nv_fatbin section
extern "C" {
    extern unsigned char _inject_funcs_fatbin_data[]
        __attribute__((visibility("hidden")));
}

// CUDA fatbin wrapper structure (magic = 0x466243B1)
struct __fatBinC_Wrapper_t {
    int magic;
    int version;
    const void* data;
    void* filename_or_fatbins;
};

static __fatBinC_Wrapper_t __fatDeviceText
    __attribute__((aligned(8), section(".nvFatBinSegment"))) = {
    0x466243B1,  // magic number
    1,           // version
    _inject_funcs_fatbin_data,  // pointer to fatbin in .nv_fatbin section
    nullptr
};

static void** __cudaFatCubinHandle = nullptr;

// Constructor: register fatbin when .so is loaded
__attribute__((constructor))
static void __cuda_register_inject_funcs() {
    __cudaFatCubinHandle = __cudaRegisterFatBinary(&__fatDeviceText);
    if (__cudaFatCubinHandle) {
        __cudaRegisterFatBinaryEnd(__cudaFatCubinHandle);
    }
}

// Destructor: unregister fatbin when .so is unloaded
__attribute__((destructor))
static void __cuda_unregister_inject_funcs() {
    if (__cudaFatCubinHandle) {
        __cudaUnregisterFatBinary(__cudaFatCubinHandle);
    }
}
