def parse_model_def(prototxt_path):
	f = open(prototxt_path)
	full = ''
	for line in f:
		full += line.rstrip('\n') + ' '

	# split on "layers"
	split = full.split('layers')
	if len(split) ==1:
		split = full.split('layer')
	
	# get input dimension
	if split[1].split()[0] == '{':
		input_splits = split[0].split()
		start = 0
	else:
		input_splits = split[1].split()
		start = 1
	input_dims = []
	last_idx=0
	while(True):
		try:
			idx = input_splits.index('input_dim:', last_idx+1)
			print input_splits
			input_dims.append(int(input_splits[idx+1]))
			last_idx=idx
		except:
			break

	# now go thru layers
	layers= []
	for layer_string in split[start+1:]:
		lsplit = layer_string.split()
		params = {}
		params['name'] = find_params('name',lsplit)
		params['type'] = find_params('type',lsplit)

		if params['type'] == 'CONVOLUTION' or params['type']=='Convolution':
			params['num_output'] = find_params('num_output', lsplit)
			params['pad'] = find_params('pad',lsplit,default=0)
			params['stride'] = find_params('stride',lsplit, default=1)
			params['kernel_size'] = find_params('kernel_size',lsplit)
			params['group'] = find_params('group',lsplit,default=1)

		elif params['type'] == 'INNER_PRODUCT' or params['type'] == 'InnerProduct':
			params['num_output'] = find_params('num_output',lsplit)

		elif params['type'] == 'DROPOUT' or params['type'] == 'Dropout':
			params['dropout_ratio'] = find_params('dropout_ratio',lsplit)
		elif params['type'] == 'POOLING' or params['type'] == 'Pooling':
			params['pool'] = find_params('pool',lsplit)
			params['kernel_size'] = find_params('kernel_size',lsplit)
			params['stride'] = find_params('stride',lsplit)

		elif params['type'] == 'LRN':
			params['local_size'] = find_params('local_size', lsplit, default=5)
			params['alpha'] = find_params('alpha', lsplit, default=1)
			params['beta'] = find_params('beta', lsplit, default=5)


		layers.append(params)

	return input_dims, layers


def find_params(string, lsplit, default='DEFAULT'):
	try:
		idx = lsplit.index(string+':')
		return lsplit[idx+1].rstrip('"').lstrip('"')
	except:
		return default

if __name__ == '__main__':
	split = parse_model_def('VGG_ILSVRC_16_layers_deploy.prototxt')
