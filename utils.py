import csv
import os.path

# import caffe
import sys
CAFFE_PATH = '/home/tom/Workspace/caffe/python/'
sys.path.append(CAFFE_PATH)
import caffe
import numpy
import time
import datetime


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


def collectData(solver_file, gpu=True):
	NUM_REPEAT = 2
	UNITS = 1000 # change s to ms

	if gpu:
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()
	#TODO: potentially needs to set_device() if multiple processors

	# load the solver
	solver = caffe.SGDSolver(solver_file)
	# initialize log dictionary
	log = {}
	print "here"
	
	# record step time (with backpropagation)
	# record the first time
	step_start_time = time.time()
	solver.step(1)
	step_time = time.time() - step_start_time
	log['first step time'] = step_time * UNITS
	print "hereerere"
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


def setBatchSize(input_file_name, batch_size):
	filedata = None
	# make a dummy output_file_path
	import os.path
	directory_name = os.path.dirname(input_file_name)
	file_name = os.path.basename(input_file_name)
	model_path = directory_name + '/dummy_'+file_name
	with open(input_file_name, 'r') as f:
		filedata = f.read()

	batch_size = str(batch_size)
	filedata = filedata.replace('TRAIN_BATCH_SIZE', batch_size)

	with open(model_path, 'w') as f:
		f.write(filedata)

	return model_path
