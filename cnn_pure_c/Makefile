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
	gcc -std=c99 $(OPTIMISATIONS) mnist.c -o mnist $(LIBRARIES)

host-cifar:
	gcc -std=c99 $(OPTIMISATIONS) cifar.c -o cifar $(LIBRARIES)

loki-mnist:
	lokicc -std=c99 $(LOKI_OPTIMISATIONS) mnist.c -o mnist-loki $(LIBRARIES)

loki-cifar:
	lokicc -std=c99 $(LOKI_OPTIMISATIONS) cifar.c -o cifar-loki $(LIBRARIES)

clean:
	rm -f mnist mnist-loki cifar cifar-loki

