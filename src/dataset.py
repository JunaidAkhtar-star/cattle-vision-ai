import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CattleBreedDataset(Dataset):
    def __init__(self, images_dir, csv_file, transform=None):
        self.images_dir = images_dir
        self.labels_df = pd.read_csv(csv_file)
        self.transform = transform
        # Breed columns (all except filename)
        self.breed_columns = self.labels_df.columns[1:]
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]  # filename column
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label_vector = self.labels_df.iloc[idx, 1:].values.astype('float32')
        label_idx = torch.tensor(label_vector).argmax().item()  # Convert one-hot to class index
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx

# Usage example
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
train_dataset = CattleBreedDataset('data/train/images', 'data/train/_classes.csv', transform=data_transforms)


from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
