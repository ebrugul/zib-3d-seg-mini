import os
import torch
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped,
)

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader


# ---------- Device ----------
def get_device():
    # 3D Conv is not supported on MPS in this setup -> use CPU
    return torch.device("cpu")


# ---------- Simple overlay ----------
def save_overlay(image, label, pred, path):
    """
    image, label, pred: tensors with shape [B, C, D, H, W]
    We visualize the middle axial slice.
    """
    # ensure cpu + numpy
    image = image.detach().cpu()
    label = label.detach().cpu()
    pred = pred.detach().cpu()

    # pick middle slice along depth
    d = image.shape[2] // 2

    img2d = image[0, 0, d, :, :].numpy()
    gt2d  = label[0, 0, d, :, :].numpy()
    pr2d  = pred[0, 0, d, :, :].numpy()

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img2d, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("GT")
    plt.imshow(img2d, cmap="gray")
    plt.imshow(gt2d > 0.5, alpha=0.4)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(img2d, cmap="gray")
    plt.imshow(pr2d > 0.5, alpha=0.4)
    plt.axis("off")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    device = get_device()
    print("Using device:", device)

    # ---------- Transforms ----------
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-100,
            a_max=300,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(48, 48, 48),
            pos=1,
            neg=1,
            num_samples=1,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.3),
        EnsureTyped(keys=["image", "label"], device=device)
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-100,
            a_max=300,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"], device=device),
    ])

    # ---------- Dataset ----------
    data_dir = "./data"

    train_ds = DecathlonDataset(
        root_dir=data_dir,
        task="Task09_Spleen",
        section="training",
        transform=train_transforms,
        download=True,
    )

    val_ds = DecathlonDataset(
        root_dir=data_dir,
        task="Task09_Spleen",
        section="validation",
        transform=val_transforms,
        download=False,
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # ---------- Model ----------
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 96),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_fn = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metric = DiceMetric(include_background=True, reduction="mean")

    # ---------- Training ----------
    epochs = 20
    best_dice = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            if isinstance(batch, list):
                batch = batch[0]

            x = batch["image"]
            y = batch["label"]

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, mean train loss = {epoch_loss:.4f}")


        # ---------- Validation ----------
        if (epoch + 1) % 2 != 0:
            continue

        model.eval()
        metric.reset()

        with torch.no_grad():
            
            roi_size = (96, 96, 96)   # sliding window patch
            sw_batch_size = 1

            for vbatch in val_loader:
                if isinstance(vbatch, list):
                    vbatch = vbatch[0]

                vx = vbatch["image"]
                vy = vbatch["label"]

                v_logits = sliding_window_inference(vx, roi_size, sw_batch_size, model)
                v_pred = (torch.sigmoid(v_logits) > 0.5).float()
                metric(v_pred, vy)

            dice = metric.aggregate().item()
            print(f"Validation Dice: {dice:.4f}")

            if dice > best_dice:
                best_dice = dice
                torch.save(model.state_dict(), "artifacts/best_model.pt")
                save_overlay(vx, vy, v_pred, "artifacts/overlay.png")

    print("Best Dice:", best_dice)


if __name__ == "__main__":
    main()
