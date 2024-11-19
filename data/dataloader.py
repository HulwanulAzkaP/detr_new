import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from config.config import CONFIG

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Args:
            root_dir (str): Directory containing all the images.
            annotation_file (str): Path to the COCO annotation file.
            transform (callable, optional): Transformations to apply to images.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Load COCO annotations
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        # Parse image and annotation data
        self.images = {img["id"]: img for img in coco_data["images"]}
        self.annotations = coco_data["annotations"]

        # Map category IDs to class indices (0-based)
        self.categories = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get annotation and corresponding image
        annotation = self.annotations[idx]
        img_id = annotation["image_id"]
        img_info = self.images[img_id]

        # Load image
        img_path = os.path.join(self.root_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Get bounding box and label
        bbox = annotation["bbox"]  # [x, y, width, height]
        label = self.categories[annotation["category_id"]]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert bounding box to [x_min, y_min, x_max, y_max]
        bbox = torch.tensor([
            bbox[0],
            bbox[1],
            bbox[0] + bbox[2],
            bbox[1] + bbox[3]
        ])

        return {"image": image, "label": label, "bbox": bbox}

def get_dataloader(data_dir, annotation_file):
    """
    Create a DataLoader for the COCO dataset.

    Args:
        data_dir (str): Directory containing the images.
        annotation_file (str): Path to the COCO annotations file.
    """
    batch_size = CONFIG["training"]["batch_size"]

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Create dataset and dataloader
    dataset = COCODataset(data_dir, annotation_file, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def collate_fn(batch):
    """
    Custom collate function to handle batches of variable-sized images and annotations.
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    bboxes = torch.stack([item["bbox"] for item in batch])
    return {"images": images, "labels": labels, "bboxes": bboxes}
