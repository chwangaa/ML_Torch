def saveConvLayerWeights(net, layer_name, file_name):
	'''
	save the weights & biases in the layer specified by
	(net, layer_name) to the file_name
	'''
	# first, save weights
	data_list = net.params[layer_name][0].data.tolist()
	num_filters = len(data_list)
	num_channels = len(data_list[0])
	height = len(data_list[0][0])
	width = len(data_list[0][0][0])

	file_handler = open(file_name, 'w')
	file_handler.writelines("%s %s %s %s \n"%(height, width, num_channels, num_filters))

	for f in data_list:
		for n in f:
			for h in n:
				for w in h:
					print >> file_handler, w
	# then, save bias
	bias_list = net.params[layer_name][1].data.tolist()
	for b in bias_list:
		print >> file_handler, b

	file_handler.close()

def saveFCLayerWeights(net, layer_name, file_name):
	'''
	save the weights & biases in the layer specified by
	(net, layer_name) to the file_name
	'''
	# first, save weights
	data_list = net.params[layer_name][0].data.tolist()
	num_outputs = len(data_list)
	num_inputs = len(data_list[0])

	file_handler = open(file_name, 'w')
	file_handler.writelines("%s %s \n"%(num_inputs, num_outputs))

	for f in data_list:
		for weight in f:
			print >> file_handler, weight

	# then, save bias
	bias_list = net.params[layer_name][1].data.tolist()
	for b in bias_list:
		print >> file_handler, b

	file_handler.close()