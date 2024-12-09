import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import multiprocessing
import lmdb
from tqdm import tqdm
import pickle
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enable benchmark mode in cuDNN for optimized performance
cudnn.benchmark = True

# Configuration Constants
SUBJECT = 0  # 0 means no filtering; set to a specific subject ID to filter
WINDOW_SIZE = 5
START_TIME = 20
END_TIME = 420

# Paths to Data Files
EEG_SIGNALS_PATH = "datasets/eeg_5_95_std.pth"
EEG_IMAGES_PICKLE = "datasets/EEG_images.pickle"
EEG_LABELS_PICKLE = "datasets/EEG_labels.pickle"

# LMDB Output Paths
ENCODER_LMDB_PATH = 'datasets/eeg_encoder.lmdb'         # For Semantic Encoder (EEG only)
DDIM_LMDB_PATH = 'datasets/eeg_ddim_ffhq.lmdb'          # For Conditional DDIM (EEG + Image)

# Function to Compute Moving Average
def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size), 'valid') / window_size

# Dataset Class Defined Within This Script
class EEGDataset:
    def __init__(self, eeg_signals_path):
        """
        Initialize the EEGDataset by loading EEG signals, labels, and images.
        """
        # Load EEG signals from the .pth file
        loaded = torch.load(eeg_signals_path)
        
        # Filter data based on the subject if SUBJECT != 0
        if SUBJECT != 0:
            self.data = [
                loaded['dataset'][i] 
                for i in range(len(loaded['dataset'])) 
                if loaded['dataset'][i]['subject'] == SUBJECT
            ]
        else:
            self.data = loaded['dataset']
        
        # Extract labels and images
        self.labels = loaded["labels"]
        self.images = loaded["images"]

        # Compute dataset size
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        """
        Retrieve the EEG signal and label at index i after preprocessing.
        """
        # Process EEG signal
        eeg = self.data[i]["eeg"].float()
        eeg = eeg[:, START_TIME:END_TIME + WINDOW_SIZE - 1]

        # Get label
        label = self.data[i]["label"]

        return eeg, label

# Initialize the Dataset
dataset = EEGDataset(EEG_SIGNALS_PATH)

# Optionally, save images and labels as pickled files (useful for reference)
with open(EEG_IMAGES_PICKLE, "wb") as handle:
    pickle.dump(dataset.images, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(EEG_LABELS_PICKLE, "wb") as handle:
    pickle.dump(dataset.labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Prepare Training and Testing Indices
def prepare_indices(dataset, test_samples_per_label=3):
    """
    Split the dataset into training and testing indices ensuring a balanced test set across labels.
    """
    # Identify all valid indices where EEG data has sufficient length
    valid_indices = [
        i for i in range(len(dataset)) 
        if (END_TIME + WINDOW_SIZE - 1) <= dataset.data[i]["eeg"].size(1)
    ]
    
    # Shuffle the indices to ensure randomness
    np.random.shuffle(valid_indices)

    # Dictionary to hold test indices per label for balanced sampling
    test_dict = {}
    train_indices = valid_indices.copy()

    for index in valid_indices[::-1]:  # Iterate in reverse for efficiency
        label = dataset[index][1]
        if len(test_dict.get(label, [])) < test_samples_per_label:
            if label not in test_dict:
                test_dict[label] = [index]
            else:
                test_dict[label].append(index)
            train_indices.remove(index)

    # Compile test indices from the dictionary
    test_indices = []
    for label_indices in test_dict.values():
        test_indices.extend(label_indices)

    return train_indices, test_indices

# Get Training and Testing Indices
train_index, test_index = prepare_indices(dataset)

# Processing Functions for LMDB Entries
def process_data_encoder(i, index):
    """
    Process and serialize EEG data for the Semantic Encoder LMDB.
    """
    eeg_tensor, _ = dataset[index]  # Retrieve EEG and label (label unused here)
    eeg_np = eeg_tensor.numpy()
    eeg_ma = moving_average(eeg_np, WINDOW_SIZE)
    serialized_data = bytearray(pickle.dumps(eeg_ma))
    key = f"data-{str(i).zfill(5)}".encode("utf-8")
    return key, serialized_data

def process_data_ddim(i, index):
    """
    Process and serialize EEG data along with corresponding images for the Conditional DDIM LMDB.
    """
    eeg_tensor, label = dataset[index]
    eeg_np = eeg_tensor.numpy()
    eeg_ma = moving_average(eeg_np, WINDOW_SIZE)
    
    # Retrieve corresponding image and subject information
    image = dataset.data[index]["image"]  # Ensure 'image' is a PIL Image or convert it
    subject = dataset.data[index]["subject"]
    
    # Ensure image is in PIL format
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Optionally, convert image to a standardized format (e.g., RGB)
    image = image.convert("RGB")
    
    store_data = {
        "data": eeg_ma,      # Processed EEG data
        "image": image,      # Corresponding image
        "label": label,      # Label
        "subject": subject   # Subject ID
    }
    serialized_data = bytearray(pickle.dumps(store_data))
    key = f"data-{str(i).zfill(5)}".encode("utf-8")
    return key, serialized_data

# Function to Populate LMDB Databases Sequentially
def prepare_lmdb_sequential(env, selected_indices, process_func):
    """
    Populate the LMDB database with processed data sequentially.

    Args:
        env (lmdb.Environment): The LMDB environment to populate.
        selected_indices (list): List of dataset indices to process.
        process_func (function): Function to process each data point.
    """
    with env.begin(write=True) as txn:
        for i, index in tqdm(enumerate(selected_indices), total=len(selected_indices), desc="Writing LMDB"):
            key, serialized_data = process_func(i, index)
            txn.put(key, serialized_data)
    env.put("length".encode("utf-8"), str(len(selected_indices)).encode("utf-8"))
    logging.info(f"LMDB populated with {len(selected_indices)} entries.")

# Main Execution Block
if __name__ == "__main__":
    """
    Main function to create LMDB databases for Semantic Encoder and Conditional DDIM.
    """
    num_workers = 16  # Adjust based on your CPU cores and system capabilities

    # Ensure LMDB output directories exist
    if not os.path.exists(ENCODER_LMDB_PATH):
        os.makedirs(ENCODER_LMDB_PATH)
    if not os.path.exists(DDIM_LMDB_PATH):
        os.makedirs(DDIM_LMDB_PATH)

    # Open LMDB environments
    env_encoder = lmdb.open(ENCODER_LMDB_PATH, map_size=1024**4, readahead=False)
    env_ddim = lmdb.open(DDIM_LMDB_PATH, map_size=1024**4, readahead=False)

    try:
        # Preprocess data in parallel (optional)
        # Consider implementing a producer-consumer pattern if preprocessing is time-consuming

        # Populate Semantic Encoder LMDB with training data
        logging.info("Populating Semantic Encoder LMDB with training data...")
        prepare_lmdb_sequential(env_encoder, train_index, process_data_encoder)
        logging.info("Semantic Encoder LMDB populated successfully.")

        # Populate Conditional DDIM LMDB with training data
        logging.info("Populating Conditional DDIM LMDB with training data...")
        prepare_lmdb_sequential(env_ddim, train_index, process_data_ddim)
        logging.info("Conditional DDIM LMDB populated successfully.")

        # Optional: Populate Test LMDBs
        # Uncomment the following block if you decide to create test LMDBs in the future

        """
        # Paths for Test LMDBs
        TEST_ENCODER_LMDB_PATH = 'datasets/eeg_encoder_test.lmdb'
        TEST_DDIM_LMDB_PATH = 'datasets/eeg_ddim_ffhq_test.lmdb'

        # Ensure Test LMDB directories exist
        if not os.path.exists(TEST_ENCODER_LMDB_PATH):
            os.makedirs(TEST_ENCODER_LMDB_PATH)
        if not os.path.exists(TEST_DDIM_LMDB_PATH):
            os.makedirs(TEST_DDIM_LMDB_PATH)

        # Open Test LMDB environments
        env_test_encoder = lmdb.open(TEST_ENCODER_LMDB_PATH, map_size=1024**4, readahead=False)
        env_test_ddim = lmdb.open(TEST_DDIM_LMDB_PATH, map_size=1024**4, readahead=False)

        # Populate Test LMDBs
        logging.info("Populating Semantic Encoder Test LMDB with test data...")
        prepare_lmdb_sequential(env_test_encoder, test_index, process_data_encoder)
        logging.info("Semantic Encoder Test LMDB populated successfully.")

        logging.info("Populating Conditional DDIM Test LMDB with test data...")
        prepare_lmdb_sequential(env_test_ddim, test_index, process_data_ddim)
        logging.info("Conditional DDIM Test LMDB populated successfully.")

        # Close Test LMDB environments
        env_test_encoder.close()
        env_test_ddim.close()
        """

    finally:
        # Close LMDB environments
        env_encoder.close()
        env_ddim.close()

    logging.info("All LMDB databases have been successfully created.")
