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
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[opt.time_low:opt.time_high,:]

        if opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1,128,opt.time_high-opt.time_low)
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label


dataset = EEGDataset("datasets/eeg_5_95_std.pth")



def prepare(env, n_worker=1):
    """
    Function to prepare the LMDB database.
    Generates 11965 data samples, each with shape (128, 500).
    """
#    total = len(dataset.data)  # Number of data samples
    total = 100  # Number of data samples


    with multiprocessing.Pool(n_worker) as pool:
        for i in tqdm(range(total)):
            data = np.array(dataset.data[i]["eeg"].tolist())

            # 3d EEG data
            #data = np.stack((data[:,:128],data[:,128:256],data[:,256:384]),axis=2)
            
            # 2d EEG data 128x128
            # data = data[:,:128]

            # 2d EEG data 128x496
            data = data[:,:496]

            key = f"data-{str(i).zfill(5)}".encode("utf-8")

            with env.begin(write=True) as txn:
                txn.put(key, data.tobytes())

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    """
    Generate 11965 data samples with shape (128, 500) and save to LMDB
    """
    num_workers = 16
    out_path = 'datasets/ffhq256.lmdb'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with lmdb.open(out_path, map_size=1024**4, readahead=False) as env:
        prepare(env, num_workers)

