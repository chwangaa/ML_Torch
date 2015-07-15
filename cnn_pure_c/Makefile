# Optimisations to apply to standard compilations.
OPTIMISATIONS=-O3

# Optimisations to apply to Loki compilations (different compiler).
LOKI_OPTIMISATIONS=-O3

# Libraries required.
LIBRARIES=-lm

all: host

host: host-mnist host-cifar

loki: loki-mnist loki-cifar

host-mnist:
	ld -r -b binary -o data.o data/mnist/data.bin
	ld -r -b binary -o labels.o data/mnist/labels.bin
	gcc -std=c99 $(OPTIMISATIONS) mnist.c data.o labels.o -o mnist $(LIBRARIES)
	rm -f data.o labels.o

host-cifar:
	ld -r -b binary -o data.o data/cifar/data_batch_1.bin
	gcc -std=c99 $(OPTIMISATIONS) cifar.c data.o -o cifar $(LIBRARIES)
	rm -f data.o

loki-mnist:
	loki-elf-ld -r -b binary -o data.o data/mnist/data.bin
	loki-elf-ld -r -b binary -o labels.o data/mnist/labels.bin
	lokicc -std=c99 $(LOKI_OPTIMISATIONS) mnist.c data.o labels.o -o mnist-loki $(LIBRARIES)
	rm -f data.o labels.o

loki-cifar:
	loki-elf-ld -r -b binary -o data.o data/cifar/data_batch_1.bin
	lokicc -std=c99 $(LOKI_OPTIMISATIONS) cifar.c data.o -o cifar-loki $(LIBRARIES)
	rm -f data.o

clean:
	rm -f mnist mnist-loki cifar cifar-loki

