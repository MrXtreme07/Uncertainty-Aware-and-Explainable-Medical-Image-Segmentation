import torch
from torch.utils.data import DataLoader
from utils.dataset import ISICDataset
from models.unet import UNet
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_latest_checkpoint():
    if not os.path.exists("outputs"):
        return None

    files = [f for f in os.listdir("outputs") if f.endswith(".pth")]
    if not files:
        return None

    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join("outputs", files[-1])


def main():
    dataset = ISICDataset("data/training_input", "data/train_masks")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = UNet().to(device)

    # 🔁 Load latest checkpoint if exists
    checkpoint = get_latest_checkpoint()
    start_epoch = 0

    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint))
        start_epoch = int(checkpoint.split("_")[-1].split(".")[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    num_epochs = 29

    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()

        model.train()
        total_loss = 0

        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Time: {epoch_time:.2f}s")

        # 💾 Save model
        os.makedirs("outputs", exist_ok=True)
        torch.save(model.state_dict(), f"outputs/model_epoch_{epoch+1}.pth")

        # 📝 Append loss
        with open("loss.txt", "a") as f:
            f.write(f"{total_loss:.4f}\n")


if __name__ == "__main__":
    main()