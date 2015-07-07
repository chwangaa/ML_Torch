import csv
import os.path

# import caffe
import sys
sys.path.append('/home/tom/Workspace/caffe/python/')
import caffe
import numpy
import time
import datetime


LOG_FILE_NAME = 'log.csv' # not used

MACHINE_NAME = 'Chihang Laptop'
DEFAULT_FIELDS = ['MACHINE', 'MODE', 'BATCH_SIZE', 'OPERATION', 'RUNNING TIME', 'LOG TIME', 'NOTE']

class LogFile:
	def __init__(self, file_name, fields=DEFAULT_FIELDS):
		# if the file exists, then read the header and construct an instance
		if os.path.isfile(file_name):
			print "File already exists, will append onwards"
			
			with open(file_name, 'r') as log_file:
				reader = csv.DictReader(log_file)
				self.file_name = file_name
				self.fields = reader.fieldnames
				if self.fields == None:
					# if the header is absent, use the default header
					self.fields = fields
					self._create_header()
		# else, build a new instance
		else:
			self.file_name = file_name
			self.fields = fields
			# initialize the log file
			self._create_header()

	def writeLog(self, log):
		file_name = self.file_name
		fields = self.fields
		with open(file_name, 'a') as log_file:
			writer = csv.DictWriter(log_file, fieldnames=fields)
			writer.writerow(log)

	def writeLogs(self, logs):
		file_name = self.file_name
		fields = self.fields
		with open(file_name, 'a') as log_file:
			writer = csv.DictWriter(log_file, fieldnames=fields)
			writer.writerows(logs)

	def _create_header(self):
		file_name = self.file_name
		with open(file_name, 'w') as log_file:
			writer = csv.DictWriter(log_file, fieldnames=self.fields)
			writer.writeheader()

	def readLogAsDict(self, fields=None):
		# if fields is specified, only return those data
		pass



from caffe import layers as L
from caffe import params as P

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    # return n.to_proto()
    return n.to_proto()


def createLenet(batch_size):
	with open('examples/mnist/lenet_auto_train.prototxt', 'w') as f:
		f.write(str(lenet('examples/mnist/mnist_train_lmdb', batch_size)))
	# with open('examples/mnist/lenet_auto_test.prototxt', 'w') as f:
	# 	f.write(str(lenet('examples/mnist/mnist_test_lmdb', 100)))


def cifarnet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(mean_file="examples/cifar10/mean.binaryproto"), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=32, pad=2, stride=1, weight_filler=dict(type='gaussian', std=0.0001),
    						bias_filler=dict(type='constant'))

    n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.relu1 = L.ReLU(n.pool1, in_place=True)
    n.norm1 = L.LRN(n.pool1, lrn_param=dict(local_size=3, alpha=5e-05, beta=0.75, norm_region=P.LRN.WITHIN_CHANNEL))
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, pad=2, num_output=32, weight_filler=dict(type='gaussian', std=0.01),
    						bias_filler=dict(type='constant'))
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.pool2 = L.Pooling(n.conv2, kernel_size=3, stride=2, pool=P.Pooling.AVE)
    n.norm2 = L.LRN(n.pool2, lrn_param=dict(local_size=3, alpha=5e-05, beta=0.75, norm_region=P.LRN.WITHIN_CHANNEL))
    n.conv3 = L.Convolution(n.norm2, kernel_size=5, pad=2, num_output=64, stride=1, weight_filler=dict(type='gaussian', std=0.01),
    						bias_filler=dict(type='constant'))
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    n.pool3 = L.Pooling(n.conv3, kernel_size=3, stride=2, pool=P.Pooling.AVE)

    n.ip1 = L.InnerProduct(n.pool3, num_output=10, weight_filler=dict(type='gaussian', std=0.01),
    						bias_filler=dict(type='constant'))
    n.loss = L.SoftmaxWithLoss(n.ip1, n.label)
    return n.to_proto()


def createCifarnet(batch_size):
	with open('examples/cifar10/cifar10_auto_train.prototxt', 'w') as f:
		f.write(str(cifarnet('examples/cifar10/cifar10_train_lmdb', batch_size)))
	# with open('examples/cifar10/cifar10_auto_test.prototxt', 'w') as f:
	# 	f.write(str(cifarnet('examples/cifar10/cifar10_test_lmdb', 32)))

def makeLog(log_raw, machine, mode, batch_size):
	logs = []
	log_time = datetime.datetime.now()
	for entry in log_raw:
		log_entry = {}
		log_entry['MACHINE'] = machine
		log_entry['MODE'] = mode
		log_entry['BATCH_SIZE'] = batch_size
		log_entry['OPERATION'] = entry
		log_entry['RUNNING TIME'] = log_raw[entry]
		log_entry['LOG TIME'] = log_time
		logs.append(log_entry)
	return logs


def run(solver_file, use_gpu=True):
	NUM_REPEAT = 2
	UNITS = 1000 # change s to ms

	if use_gpu:
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()
	#TODO: potentially needs to set_device() if multiple processors

	# load the solver
	solver = caffe.SGDSolver(solver_file)
	# initialize log dictionary
	log = {}
	
	# record step time (with backpropagation)
	# record the first time
	step_start_time = time.time()
	solver.step(1)
	step_time = time.time() - step_start_time
	log['first step time'] = step_time * UNITS
	
	# record the new 3 times and take the average
	step_time_record = []
	for i in range(NUM_REPEAT):
		step_start_time = time.time()
		solver.step(1)
		step_time = time.time() - step_start_time
		step_time_record.append(step_time)
	mean_step_time = numpy.mean(step_time_record)
	log['mean step time'] = mean_step_time * UNITS

	# find all the layers
	layers = list(solver.net._layer_names)
	# record the mean forward time from each layer
	for layer in layers:
		layer_time_record = []
		for i in range(NUM_REPEAT):
			layer_start_time = time.time()
			solver.net.forward(start=layer)
			layer_time = time.time() - layer_start_time
			layer_time_record.append(layer_time)
		mean_layer_time = numpy.mean(layer_time_record)
		log[layer] = mean_layer_time * UNITS

	return log


def collectLenetData(batch_sizes, log_file_name):
	log_file = LogFile(log_file_name)
	for batch_size in batch_sizes:
		createLenet(batch_size)
		# run in GPU
		log_raw = run('examples/mnist/lenet_auto_solver.prototxt')
		log_entries = makeLog(log_raw, machine=MACHINE_NAME, mode='GPU', batch_size=batch_size)
		log_file.writeLogs(log_entries)
		# run in CPU
		log_raw = run('examples/mnist/lenet_auto_solver.prototxt', False)
		log_entries = makeLog(log_raw, machine=MACHINE_NAME, mode='CPU', batch_size=batch_size)
		log_file.writeLogs(log_entries)


def collectCifar10Data(batch_sizes, log_file_name):
	log_file = LogFile(log_file_name)
	for batch_size in batch_sizes:
		createCifarnet(batch_size)
		# run in GPU
		log_raw = run('examples/cifar10/cifar10_auto_solver.prototxt')
		log_entries = makeLog(log_raw, machine=MACHINE_NAME, mode='GPU', batch_size=batch_size)
		log_file.writeLogs(log_entries)
		# run in CPU
		log_raw = run('examples/cifar10/cifar10_auto_solver.prototxt', False)
		log_entries = makeLog(log_raw, machine=MACHINE_NAME, mode='CPU', batch_size=batch_size)
		log_file.writeLogs(log_entries)