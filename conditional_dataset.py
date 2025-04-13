import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ConditionalImageDataset(Dataset):
    def __init__(self, image_folder, metadata_path, transform=None):
        self.transform = transform
        self.image_folder = image_folder
        self.root_dir = os.path.dirname(image_folder)  # Parent directory of image folder
        
        # Load metadata
        self.metadata = []
        with open(metadata_path, 'r') as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        print(f"Loaded {len(self.metadata)} images with metadata")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get metadata
        item = self.metadata[idx]
        
        # Extract parameter from text
        text = item["text"]
        # Extract parameter like "11.2798" from text
        param_value = float(text.split()[-1])
        
        # Load image
        img_path = os.path.join(self.root_dir, item["file_name"])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, param_value
