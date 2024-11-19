import torch
from models.backbone import DETRBackbone
from data.dataloader import get_dataloader
from config.config import CONFIG
from utils.utils import print_config, save_checkpoint

def train():
    # Print konfigurasi
    print_config(CONFIG)

    # Load DataLoader
    train_loader = get_dataloader(
        CONFIG["dataset"]["train"]["images"],
        CONFIG["dataset"]["train"]["annotations"]
    )

    # Load Model
    model = DETRBackbone(pretrained=CONFIG["model"]["pretrained"])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["training"]["learning_rate"])

    # Training Loop
    num_epochs = CONFIG["training"]["epochs"]
    device = CONFIG["training"]["device"]
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs, labels, bboxes = batch["images"], batch["labels"], batch["bboxes"]
            inputs, labels, bboxes = inputs.to(device), labels.to(device), bboxes.to(device)

            optimizer.zero_grad()
            class_preds, bbox_preds = model(inputs)
            class_loss = torch.nn.functional.cross_entropy(class_preds, labels)
            bbox_loss = torch.nn.functional.mse_loss(bbox_preds, bboxes)
            loss = class_loss + bbox_loss

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Save model
    save_checkpoint(model, f"{CONFIG['output']['checkpoint_path']}/model_final.pth")

if __name__ == "__main__":
    train()
