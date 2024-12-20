from torch.utils.data import Dataset
import torch

class AmsICUSepticShock(Dataset):
        def __init__(self, data, transform=None):
            """
            Args:
                data (pd.DataFrame): Your data in a DataFrame format.
                transform (callable, optional): Optional transforms to apply to the data.
            """
            self.data = data
            self.transform = transform

        def __len__(self):
            # Return the number of samples in the dataset
            return len(self.data)

        def __getitem__(self, idx):
            # Get the sample at index idx
            sample = self.data.iloc[idx]

            # Convert sample to a tensor
            input_data = torch.tensor(sample.values, dtype=torch.float32)

            # Apply transformations if any
            if self.transform:
                input_data = self.transform(input_data)

            return input_data