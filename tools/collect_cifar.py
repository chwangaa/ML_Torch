import collection

solver_file = 'examples/mnist/lenet_solver.prototxt'
net_file = "examples/mnist/lenet_train_test.prototxt"
log_file = "data_collections/logs/Chihang/lenet_mnist.csv"
batch_sizes = [32]
collection.collectDatas(solver_file, net_file, log_file, batch_sizes)


# solver_file = 'examples/cifar10/cifar10_full_solver.prototxt'
# net_file = "examples/cifar10/cifar10_full_train_test.prototxt"
# log_file = "data_collections/logs/Chihang/cifarnet_cifar10.csv"
# batch_sizes = [32, 64, 128]

# collection.collectDatas(solver_file, net_file, log_file, batch_sizes)
