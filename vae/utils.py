import yaml
import os

def load_config(default_fp, custom_fp):

	assert os.path.isfile(default_fp), default_fp
	if custom_fp:
		assert os.path.isfile(custom_fp), custom_fp
	with open(default_fp, "r") as default_file:
		config_d = yaml.load(default_file, Loader=yaml.FullLoader)
	# overwrite parts of the config
	if custom_fp:
		with open(custom_fp, "r") as custom_file:
			custom_d = yaml.load(custom_file, Loader=yaml.FullLoader)
		assert all([k in config_d for k in custom_d]), set(custom_d.keys()) - set(config_d.keys())
		for k in custom_d.keys():
			if k not in config_d:
				raise ValueError(f"Key {k} not in default config")
			config_d[k] = custom_d[k]
	return config_d

if __name__ == '__main__':
	config_d = load_config(default_fp='config/default.yml', custom_fp='config/test.yml')
	print(config_d)