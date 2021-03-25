# 1D CNN - Human Activity Recognition
import numpy as np

def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

def load_group(filename, prefix="://Mateus//Neural_Networks//0001---mnistTensorFlow//"+
    "1D CNN//UCI HAR Dataset//UCI HAR Dataset//"):

    loaded = list()
    for name in filename:
        data = load_file(prefix+name)
        loaded.append()
    loaded = dstack(loaded)
    return loaded

def load_dataset_group(group, prefix="://Mateus//Neural_Networks//0001---mnistTensorFlow//"+
    "1D CNN//UCI HAR Dataset//UCI HAR Dataset//"):

    filepath = prefix + group + "//Inertial Signals"

    filenames = list()
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']

    x = load_group(filenames, filepath)
    y = load_file(prefix + group + 'y_'+ group + ".txt")

    return x, y

