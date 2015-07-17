import os.path
import utils
import log


MACHINE_NAME = 'Chihang Laptop'

def collectDatas(solver_file_name, net_file, log_file_name, batch_sizes):
	if not os.path.isfile(solver_file_name):
		raise Exception("Solver file does not exist")

	log_file = log.LogFile(log_file_name)
	if type(batch_sizes) == list:
		for batch_size in batch_sizes:
			utils.setBatchSize(net_file, batch_size)
			# run in GPU
			print "load GPU data"
			log_raw = utils.collectData(solver_file_name, gpu=True)
			log_entries = utils.makeLog(log_raw, machine=MACHINE_NAME, mode='GPU', batch_size=batch_size)
			log_file.writeLogs(log_entries)
			# run in CPU
			log_raw = utils.collectData(solver_file_name, gpu=False)
			log_entries = utils.makeLog(log_raw, machine=MACHINE_NAME, mode='CPU', batch_size=batch_size)
			log_file.writeLogs(log_entries)
