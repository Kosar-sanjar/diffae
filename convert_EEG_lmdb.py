# Imports
import os
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np
import multiprocessing
import lmdb
import numpy as np
from tqdm import tqdm
import os
import pickle

subject=0

# Dataset class
class EEGDataset:

    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        else:
            self.data=loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]

        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float()
        # eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[:,20:460]
    
        # if opt.model_type == "model10":
        #     eeg = eeg.t()
        #     eeg = eeg.view(1,128,opt.time_high-opt.time_low)
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label


dataset = EEGDataset("datasets/eeg_5_95_std.pth")
with open ("datasets/EEG_images.pickle","wb") as handle:
   pickle.dump(dataset.images,handle,protocol=pickle.HIGHEST_PROTOCOL)
with open ("datasets/EEG_labels.pickle","wb") as handle:
   pickle.dump(dataset.labels,handle,protocol=pickle.HIGHEST_PROTOCOL)

indexes = [i for i in range(len(dataset)) if 460 <= dataset.data[i]["eeg"].size(1) <= 600]
np.random.shuffle(indexes)

test_size = 100
test_index= indexes[:test_size]
train_size = 3000
train_index= indexes[test_size:train_size+test_size]

def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size), 'valid') / window_size

window_size = 5

def process_data_train(index_and_i):
    i, index = index_and_i
    data = np.array(dataset[index][0].tolist())
    
    # apply moving average
    data = np.apply_along_axis(moving_average, axis=1, arr=data, window_size=window_size)
    
    # 3d EEG data
    # data = np.stack((data[:,:128],data[:,128:256],data[:,256:384]),axis=2)
    
    # 2d EEG data 128x128
    # data = data[:,:128]

    # 2d EEG data 128x400
    # data = data[:,:400]

    key = f"data-{str(i).zfill(5)}".encode("utf-8")
    with env.begin(write=True) as txn:
        storedata= data.tobytes()
        txn.put(key, storedata)


def process_data_test(index_and_i):
    i, index = index_and_i
    data = np.array(dataset[index][0].tolist())
    
    # apply moving average
    data = np.apply_along_axis(moving_average, axis=1, arr=data, window_size=window_size)
    
    label = dataset[index][1]
    image = dataset.data[index]["image"]
    subject = dataset.data[index]["subject"]

    key = f"data-{str(i).zfill(5)}".encode("utf-8")
    with env.begin(write=True) as txn:
        storedata={"data":data,"image":image,"label":label,"subject":subject}
        storedata=bytearray(pickle.dumps(storedata))
        txn.put(key, storedata)

def prepare(env,select_indexes,n_worker=1,store_type="train"):
    """
    Function to prepare the LMDB database.
    Generates 11965 data samples, each with shape (128, 500).
    """

    # with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    if store_type =="train":
        with multiprocessing.Pool(n_worker) as pool:
            list(tqdm(pool.imap(process_data_train, enumerate(select_indexes)), total=len(select_indexes)))
    
    if store_type =="test":
        with multiprocessing.Pool(n_worker) as pool:
            list(tqdm(pool.imap(process_data_test, enumerate(select_indexes)), total=len(select_indexes)))
            

    with env.begin(write=True) as txn:
        txn.put("length".encode("utf-8"), str(len(select_indexes)).encode("utf-8"))


if __name__ == "__main__":
    """
    Generate 11965 data samples with shape (128, 500) and save to LMDB
    """
    num_workers = 16
    train_out_path = 'datasets/ffhq256.lmdb'
    test_out_path = 'datasets/EEGtest.lmdb'


    if not os.path.exists(train_out_path):
        os.makedirs(train_out_path)

    with lmdb.open(train_out_path, map_size=1024**4, readahead=False) as env:
        prepare(env,train_index,num_workers,"train")
    
    if not os.path.exists(test_out_path):
        os.makedirs(test_out_path)

    with lmdb.open(test_out_path, map_size=1024**4, readahead=False) as env:
        prepare(env,test_index,num_workers,"test")

