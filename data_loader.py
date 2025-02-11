from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import pytorch_lightning as pl
import random 

# Seed all relevant libraries and frameworks
SEED = 42
random.seed(SEED)                     # Python random module
np.random.seed(SEED)                  # NumPy
torch.manual_seed(SEED)               # PyTorch (CPU)
torch.cuda.manual_seed(SEED)          # PyTorch (GPU, if available)
torch.cuda.manual_seed_all(SEED)      # PyTorch for multi-GPU
pl.seed_everything(SEED)              # PyTorch Lightning seed utility
torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
torch.backends.cudnn.benchmark = False     # Avoid non-deterministic optimizations

# STEP 1: Dataset Class Definition
class MemmapDataset(Dataset):
    """PyTorch Dataset for memory-mapped numpy array."""

    def __init__(self, data_path, shape=(1100, 2, 1024, 1024), transform=None):
        """
        Args:
            data_path (str): Path to the .npy memmap file.
            shape (tuple): Shape of the dataset (samples, channels, height, width).
            transform (callable, optional): Transformations for augmentation.
        """
        self.data_path = data_path
        self.shape = shape
        self.transform = transform
        self.memmap_array = np.memmap(data_path, dtype=np.float32, mode='r', shape=shape)

    def __len__(self):
        return self.shape[0]  # Number of samples

    def __getitem__(self, idx):
        image = self.memmap_array[idx, 0, :, :]  # Extract image
        label = self.memmap_array[idx, 1, :, :]  # Extract label

        # Example threshold value; adjust as needed.
        threshold = 200

        # Modify the label array as follows:
        # - If label == 0, keep it as 0.
        # - Else, if label < threshold, set it to 1.
        # - Else (label >= threshold), set it to 2.
        label = np.where(label == 0, 0, np.where(label < threshold, 1, 2))

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1024, 1024)
        label = torch.tensor(label, dtype=torch.long)  # Shape: (1024, 1024)

        if self.transform:
            image = self.transform(image)

        return image, label

class MemmapDataModule(pl.LightningDataModule):
    """Lightning DataModule for memory-mapped datasets with pre-split train, val, and test sets."""

    def __init__(self, train_path, val_path, test_path, train_shape, val_shape, test_shape, batch_size=8, num_workers=4, transform=None):
        """
        Args:
            train_path (str): Path to the training memmap file.
            val_path (str): Path to the validation memmap file.
            test_path (str): Path to the test memmap file.
            train_shape (tuple): Shape of the training dataset.
            val_shape (tuple): Shape of the validation dataset.
            test_shape (tuple): Shape of the test dataset.
            batch_size (int): Batch size for training and validation.
            num_workers (int): Number of workers for DataLoader.
            transform (callable, optional): Transformations for augmentation.
        """
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.train_shape = train_shape
        self.val_shape = val_shape
        self.test_shape = test_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        """Initialize datasets."""
        self.train_dataset = MemmapDataset(self.train_path, self.train_shape, self.transform)
        self.val_dataset = MemmapDataset(self.val_path, self.val_shape, self.transform)
        self.test_dataset = MemmapDataset(self.test_path, self.test_shape, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    