# Top-level Makefile for CUPTI samples
#
# Copyright NVIDIA Corporation 


# Define supported SM architectures. Adjust this list based on your GPU(s) and CUDA version.
# For example, for Tesla P40 (compute capability 6.1), you need sm_61.
# Common architectures: 61 70 75 80 86 87 89 90

# List of all sample directories
SAMPLE_DIRS = activity_trace \
              activity_trace_async \
              autorange_profiling \
              callback_event \
              callback_metric \
              callback_timestamp \
              cupti_query \
              event_multi_gpu \
              event_sampling \
              extensions \
              nvlink_bandwidth \
              openacc_trace \
              pc_sampling \
              pc_sampling_continuous \
              pc_sampling_start_stop \
              pc_sampling_utility \
              pm_sampling \
              sass_source_map \
              unified_memory \
              userrange_profiling

.PHONY: all clean $(SAMPLE_DIRS)

all: $(SAMPLE_DIRS)

# Rule to build each sample directory
$(SAMPLE_DIRS):
	@echo "Building $@..."
	@$(MAKE) -C $@
	@echo "Finished building $@"

# Clean all sample directories
clean:
	@for dir in $(SAMPLE_DIRS); do \
		echo "Cleaning $$dir..."; \
		$(MAKE) -C $$dir clean; \
	done
	@echo "All samples cleaned"

# Help message
help:
	@echo "CUPTI Samples Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make               - Build all samples"
	@echo "  make <sample_dir>  - Build specific sample"
	@echo "  make clean         - Clean all samples"
	@echo "  make help          - Display this help message"
	@echo ""
	@echo "Available samples:"
	@for dir in $(SAMPLE_DIRS); do \
		echo "  $$dir"; \
	done 