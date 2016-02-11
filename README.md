# Branch for testing LokiCNN on MNIST

this branch is for testing MNIST on loki simulator

a compiled working version is in ``LokiCNN/mnist-loki``

1) The current version works for 1 core
2) The current version, when using 8 cores, can do one inference, but then crushes on the 2nd inference with error message '.. been idle for 127 cycles'
