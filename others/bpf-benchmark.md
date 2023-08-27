# BPF-Benchmark

BPF-Benchmark is a simple testsuit designed to assess the performance of different userspace eBPF (Berkeley Packet Filter) runtimes.

## Usage

To effectively utilize BPF-Benchmark, follow these steps:

1. **Install Dependencies**: Ensure that `clang` and `llvm` are installed on your system. If you are using `Debian` or `Ubuntu`, you can easily install these packages by executing the command: `sudo apt install clang llvm`.
2. **Acquire Runtimes**: BPF-Benchmark includes three pre-packaged runtimes: `llvm-jit`, `ebpf`, and `ubpf`. If the provided executables are incompatible with your system, you have the option to manually build these runtimes.
3. **Install Required Dependencies**: The toolkit relies on various dependencies, all of which are listed in the `requirements.txt` file. Install these dependencies to ensure smooth functionality.
4. **Compile BPF Programs**: Execute the command `make -C bpf_progs` to compile the BPF programs.
5. **Run Benchmark**: Launch the benchmarking process by running the `run_benchmark.py` script.
6. **View Results**: The results of the benchmark can be observed directly in the console, or you can find graphical representations in the `output` folder. Additionally, raw data in JSON format is stored in `output/data.json`.

## Adding a New Test

To include a new test within BPF-Benchmark, adhere to these guidelines:

1. **Create BPF Program**: Craft an eBPF program and name it as `XXX.bpf.c`, placing it in the `bpf_progs` directory. The program should contain a function named `unsigned long long bpf_main(void* mem)` as its entry point. Any other functions must be inlined using the `__always_inline` attribute.
2. **Optional Memory File**: You have the choice to create a file named `XXX.mem` if your eBPF program requires specific memory input during execution.

## Included Test Cases

BPF-Benchmark currently incorporates the following BPF code test cases:

- `log2_int.bpf.c`
- `memcpy.bpf.c`
- `native_wrapper.c`
- `prime.bpf.c`
- `simple.bpf.c`
- `strcmp_fail.bpf.c`
- `strcmp_full.bpf.c`
- `switch.bpf.c`

Select and execute the tests ending with the `.bpf` extension.

## Supported Runtimes

BPF-Benchmark facilitates performance comparison across the following runtimes:

- `bpftime-llvm`
- `bpftime-rbpf`
- `bpftime-ubpf`

These runtimes are involved in the benchmarking process to evaluate their relative efficiencies.

Enhance your BPF runtime evaluation experience using BPF-Benchmark! If you encounter any issues or need further assistance, feel free to reach out to us. Your feedback is invaluable in refining this toolkit.

repo: https://github.com/eunomia-bpf/bpf-benchmark