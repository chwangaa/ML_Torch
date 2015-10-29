import os.path
import csv

DEFAULT_FIELDS = ['M', 'N', 'K', 'Cycle', 'Instruction', 'Core', 'Error']


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

	def readLogAsDict(self, core, x, y):
		# value should be either Cycle or Instruction
		data = {}
		with open(self.file_name, 'r') as log_file:
			reader = csv.DictReader(log_file)
			for row in reader:
				if row['Core'] == str(core):
					label = row[x]
					data[label] = float(row[y])
		return data

def scatter(file_name, cores, x, y):
	log_file = LogFile(file_name)
	series = {}
	for core in cores:
		label = str(core) + 'Core ' + x + " against " + y
		data = log_file.readLogAsDict(core, x, y)
		series[label] = data
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	import numpy as np
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	num = len(cores) + 1
	colours = cm.rainbow(np.linspace(0, 1, num))
	c = 0
    # these are for setting up colours

	for label, data in series.iteritems():
		xs = list(data.iterkeys())
		ys = list(data.itervalues())
		ax1.scatter(xs, ys, label=label, color=colours[c])
		c += 1
	legend = ax1.legend(loc='upper left', shadow=True)
	plt.show()