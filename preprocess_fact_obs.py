from multiprocessing import Pool
import photon_stream as ps
import numpy as np
import operator
import random
import pickle
import gzip
import json
import os

# Important variables
mc_data_path = '/run/media/jbieker/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/obs/'
id_position_path = '/run/media/jbieker/SSD/Development/CNN_Classification_with_FACT_images/position_dict.p'
temporary_path = '/run/media/jbieker/Seagate/D_test'
processed_data_path = '/run/media/jbieker/Seagate/C_test'

def getMetadata():
    '''
    Gathers the file paths of the training data
    '''
    # Iterate over every file in the subdirs and check if it has the right file extension
    file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(os.path.expanduser(mc_data_path)) for file in fileName if '.json' in file]
    return file_paths


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 46, 45, 1)).astype(np.float32)
    labels = (np.arange(2) == labels[:,None]).astype(np.float32)
    return dataset, labels

file_paths = getMetadata()
id_position = pickle.load(open(id_position_path, "rb"))

data = []
num = 0
for path in file_paths:
    with gzip.open(path) as file:
        # Gamma=True, Proton=False
        label = True if 'gamma' in path else False
        print(path)
        for line in file:
            try:
                event_photons = json.loads(line.decode('utf-8'))['PhotonArrivals_500ps']

                event = []
                input_matrix = np.zeros([46,45,100])
                for i in range(1440):
                    event.extend(event_photons[i])

                    x, y = id_position[i]
                    for value in event_photons[i]:
                        input_matrix[int(x)][int(y)][value-30] += 1

                input_matrix = np.sum(input_matrix[:,:,10:30], axis=2)
                data.append([input_matrix, label])



                if len(data)%1000 == 0 and len(data)!=0:
                    pic, lab = zip(*data)
                    pic, lab = reformat(np.array(pic), np.array(lab))
                    data_dict={'Image':pic, 'Label':lab}

                    with gzip.open( temporary_path + "/PhotonArrivals_500ps_"+str(num)+".p", "wb" ) as data_file:
                        pickle.dump(data_dict, data_file)
                    data = []
                    num += 1

            except:
                pass

# Load pickled data and split it into pictures and labels
def load_data(file):
    with gzip.open(temporary_path+'/'+file, 'rb') as f:
        data_dict = pickle.load(f)
    pic = data_dict['Image']
    lab = data_dict['Label']
    return (pic, lab)

# Pool-load pickled data and split it into pictures and labels (list)
p = Pool()
data = p.map(load_data, os.listdir(temporary_path))
pics, labs = zip(*data)
del data, p

# Concatenate the data to a single np.array
pic = np.concatenate(pics)
lab = np.concatenate(labs)
del pics, labs


# Values to standardize the data
mean = np.mean(pic)
std = np.std(pic)
print(mean, std)


# Randomize and split the data into train/validation/test dataset
p = np.random.permutation(len(pic))
all_pics = pic[p]
all_labels = lab[p]
del p, pic, lab

def save_data(i):
    pics_batch = all_pics[(i-1)*1000:i*1000]
    labels_batch = all_labels[(i-1)*1000:i*1000]

    data_dict={'Image':(pics_batch-mean)/std, 'Label':labels_batch}
    with gzip.open(processed_data_path + '/PhotonArrivals_500ps_{}.p'.format(i), 'wb') as f:
        pickle.dump(data_dict, f)

num_files = len(os.listdir(temporary_path))
p = Pool()
data = p.map(save_data, range(1,num_files+1))


