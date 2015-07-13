all: host

host: host-mnist host-cifar

loki: loki-mnist loki-cifar

host-mnist:
	gcc -std=c99 -O3 mnist.c -o mnist -lm

loki-mnist:
	lokicc -std=c99 -O3 mnist.c -o mnist-loki -lm

host-cifar:
	gcc -std=c99 -O3 cifar.c -o cifar -lm

loki-cifar:
	lokicc -std=c99 -O3 cifar.c -o cifar-loki -lm

clean:
	rm mnist mnist-loki cifar cifar-loki

