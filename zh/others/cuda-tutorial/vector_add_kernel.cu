// Simple vector addition kernel for PTX generation
extern "C" __global__ void vector_add_ptx(int* a, int* b, int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
} 