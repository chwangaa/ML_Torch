import caffe


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


def exportWeight(net_prototxt, net_weight, save_directory_name):
	assert(os.path.exists(save_directory_name))
	net = caffe.Net(net_prototxt, net_weight, caffe.TEST)
	# here a little buggy, we assume any layer start with 'conv' is a conv layer, 'ip' is a ip layer
	for i in net.params.keys():
		file_name = os.path.join(save_directory_name, i);
		if i.startswith('ip'):
			saveFCLayerWeights(net, i, file_name)
		else:
			if i.startswith('conv'):
				saveConvLayerWeights(net, i, file_name)
			else:
				raise Exception("Unknown layer definition")