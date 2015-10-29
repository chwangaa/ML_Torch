# Optimisations to apply to standard compilations.
OPTIMISATIONS=-O3

# Optimisations to apply to Loki compilations (different compiler).
LOKI_OPTIMISATIONS=-O3

# Libraries required.
LIBRARIES=-lm -Iinclude -Isrc -lcblas
LOKI_LIBRARIES = -lm -Iinclude -Isrc -lloki
LIBLOKI_DIR ?= /Workspace/libloki

DEPENDENCIES = 

all: host

host: host-mnist host-cifar

tests:
	loki-clang -ccc-host-triple loki-elf-linux $(LOKI_OPTIMISATIONS) $(DEPENDENCIES) -I$(LIBLOKI_DIR)/include -L$(LIBLOKI_DIR)/lib src/mat_test.c -o mat_test2 $(LIBRARIES) -lloki

loki: loki-mnist loki-cifar loki-german

host-mnist:
	ld -r -b binary -o data.o data/mnist/data.bin
	ld -r -b binary -o labels.o data/mnist/labels.bin
	gcc -std=c99 $(OPTIMISATIONS) examples/mnist.c $(DEPENDENCIES) data.o labels.o -o build/mnist $(LIBRARIES)
	rm -f data.o labels.o

host-cifar:
	ld -r -b binary -o data.o data/cifar/data_batch_1.bin
	gcc -std=c99 $(OPTIMISATIONS) examples/cifar.c $(DEPENDENCIES) data.o -o build/cifar $(LIBRARIES)
	rm -f data.o

host-german:
	gcc -std=c99 $(OPTIMISATIONS) examples/GTM.c $(DEPENDENCIES) -o build/german $(LIBRARIES)

host-vgg:
	gcc -std=c99 $(OPTIMISATIONS) examples/VGG.c $(DEPENDENCIES) -o build/vgg $(LIBRARIES)


loki-mnist:
	loki-elf-ld -r -b binary -o data.o data/mnist/data.bin
	loki-elf-ld -r -b binary -o labels.o data/mnist/labels.bin
	lokicc -ccc-host-triple loki-elf-linux $(LOKI_OPTIMISATIONS) $(DEPENDENCIES) -I$(LIBLOKI_DIR)/include -L$(LIBLOKI_DIR)/lib examples/mnist.c data.o labels.o -o mnist-loki $(LOKI_LIBRARIES) -DLOKI
	rm -f data.o labels.o

loki-german:
	lokicc -ccc-host-triple loki-elf-linux $(LOKI_OPTIMISATIONS) $(DEPENDENCIES) -I$(LIBLOKI_DIR)/include -L$(LIBLOKI_DIR)/lib examples/GTM.c -o loki-german $(LIBRARIES) -lloki -DLOKI
	rm -f data.o labels.o

loki-gemm-test:
	lokicc -ccc-host-triple loki-elf-linux $(LOKI_OPTIMISATIONS) $(DEPENDENCIES) -I$(LIBLOKI_DIR)/include -L$(LIBLOKI_DIR)/lib test/gemm_test.c -o build/loki-gemm-test $(LOKI_LIBRARIES) -DLOKI

loki-gemv-test:
	lokicc -ccc-host-triple loki-elf-linux $(LOKI_OPTIMISATIONS) $(DEPENDENCIES) -I$(LIBLOKI_DIR)/include -L$(LIBLOKI_DIR)/lib test/gemv_test.c -o build/loki-gemv-test $(LOKI_LIBRARIES) -DLOKI


loki-cifar:
	loki-elf-ld -r -b binary -o data.o data/cifar/data_batch_1.bin
	lokicc $(LOKI_OPTIMISATIONS) $(DEPENDENCIES) -I$(LIBLOKI_DIR)/include -L$(LIBLOKI_DIR)/lib examples/cifar.c data.o -o cifar-loki $(LOKI_LIBRARIES) -DLOKI
	rm -f data.o


clean:
	rm -f mnist mnist-loki cifar cifar-loki
