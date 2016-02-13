# Options: HOST, LOKI, GPU (eventually?)
PLATFORM ?= LOKI

BUILD_DIR ?= build

# Basic Linear Algorithms Subprograms library.
# Options: blas, cblas, openblas
# Host only. Usually available with `sudo apt-get install lib<option>`.
BLAS ?= openblas

# libloki - a collection of tools allowing parallelism on Loki.
LIBLOKI_DIR ?= /usr/groups/comparch-loki/tools/releases/libloki/current/

# Compiler.
HOST_CC := gcc
LOKI_CC := lokicc
CC := $($(PLATFORM)_CC)

# Linker.
HOST_LD := ld
LOKI_LD := loki-elf-ld
LD := $($(PLATFORM)_LD)

# Include paths.
# TODO: remove src
HOST_INCLUDES := include src
LOKI_INCLUDES := include src $(LIBLOKI_DIR)/include
INCLUDES := $(addprefix -I,$($(PLATFORM)_INCLUDES))

# Library paths.
HOST_LIB_PATHS := 
LOKI_LIB_PATHS := $(LIBLOKI_DIR)/lib
LIB_PATHS := $(addprefix -L,$($(PLATFORM)_LIB_PATHS))

# Libraries.
HOST_LIBS := m $(BLAS)
LOKI_LIBS := m loki
LIBS := $(addprefix -l,$($(PLATFORM)_LIBS))

# Any remaining compiler flags.
HOST_CFLAGS := -O3 -std=c99
LOKI_CFLAGS := -O3 -DLOKI -mllvm -unroll-threshold=48
CFLAGS := $($(PLATFORM)_CFLAGS)

.PHONY: all
all: debug mnist cifar

.PHONY: debug
debug:
	
$(BUILD_DIR):
	mkdir $@

# Network parameters.
%.o: %.bin
	$(LD) -r -b binary -o $@ $^


# Network models.
mnist: examples/mnist.c data/mnist/data.o data/mnist/labels.o | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIB_PATHS) $^ -o $(BUILD_DIR)/$@ $(LIBS)

cifar: examples/cifar.c data/cifar/data_batch_1.o | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIB_PATHS) $^ -o $(BUILD_DIR)/$@ $(LIBS)

german: examples/GTM.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIB_PATHS) $^ -o $(BUILD_DIR)/$@ $(LIBS)

vgg: examples/VGG.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIB_PATHS) $^ -o $(BUILD_DIR)/$@ $(LIBS)


# Tests.
.PHONY:
tests: src/mat_test.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIB_PATHS) $^ -o mat_test2 $(LIBS)
	
gemm-test: test/gemm_test.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIB_PATHS) $^ -o $(BUILD_DIR)/$@ $(LIBS)

gemv-test: test/gemv_test.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIB_PATHS) $^ -o $(BUILD_DIR)/$@ $(LIBS)


clean:
	rm -r $(BUILD_DIR)
	rm -r data/*/*.o
