# This script trains a detection model using the provided dataset.
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from datasets.road_dataset import load_data
from metrics import DetectionMetric
from models import load_model, save_model
#The metrics are provided in `metrics.py`, so no need to implement them yourself.
#You can see how to use them in `grader/tests.py`.

def train_detection(
    exp_dir="logs",
    model_name="detector",
    num_epoch=50,
    lr=1e-3,
    seed=2024,
    optimizer="adamw",
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / (
        f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)

    train_data = load_data(
        "drive_data/train",
        transform_pipeline="default",
        shuffle=True,
        num_workers=0,
    )
    val_data = load_data(
        "drive_data/val",
        transform_pipeline="default",
        shuffle=False,
    )

    model.eval()

    with torch.no_grad():
        batch = next(iter(val_data))
        img = batch["image"].to(device)
        label = batch["track"].to(device).long()

        seg_logits, depth_pred = model(img)
        pred = seg_logits.argmax(dim=1)

    #seg_loss_fn = nn.CrossEntropyLoss()
    seg_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0, 3.0], device=device))
    depth_loss_fn = nn.L1Loss()
    depth_weight = 2

    if optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-2
        )
    elif optimizer.lower() == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    metric = DetectionMetric(num_classes=3)
    global_step = 0

    for epoch in range(num_epoch):
        model.train()

        train_losses = []
        train_seg_losses = []
        train_depth_losses = []
        train_accs = []

        # Training
        for batch in train_data:
            img = batch["image"].to(device)
            label = batch["track"].to(device).long()
            depth = batch["depth"].to(device).float()

            # Dataset depth: (B,H,W)
            # Model depth:   (B,1,H,W)

            seg_logits, raw_depth = model(img)

            #probs = torch.softmax(seg_logits, dim=1)

            # print("Mean probability per class:",
            #     probs.mean(dim=(0, 2, 3)))

            # print("Mean logit per class:",
            #     seg_logits.mean(dim=(0, 2, 3)))

            # pixel_loss = torch.nn.functional.cross_entropy(
            # seg_logits, label, reduction="none"
            # )

            # for c in range(3):
            #     mask = label == c
            #     print(f"Class {c} loss:", pixel_loss[mask].mean().item())

            seg_loss = seg_loss_fn(seg_logits, label)
            depth_loss = depth_loss_fn(raw_depth, depth)
            loss = seg_loss + depth_weight * depth_loss
            #+ depth_weight * depth_loss
            # print(
            #     f"seg_loss={seg_loss.item():.4f}, "
            #     f"depth_loss={depth_loss.item():.4f}"
            # )
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())
            train_seg_losses.append(seg_loss.item())
            train_depth_losses.append(depth_loss.item())
            train_accs.append(
                (seg_logits.argmax(dim=1) == label).float().mean().item()
            )

            global_step += 1

        # Validation
        model.eval()
        metric.reset()

        val_losses = []
        val_seg_losses = []
        val_depth_losses = []

        with torch.inference_mode():
            for batch in val_data:
                img = batch["image"].to(device)
                label = batch["track"].to(device).long()
                depth = batch["depth"].to(device).float()

                seg_logits, raw_depth = model(img)

                seg_loss = seg_loss_fn(seg_logits, label)
                depth_loss = depth_loss_fn(raw_depth, depth)
                val_loss = seg_loss + depth_weight * depth_loss

                val_losses.append(val_loss.item())
                val_seg_losses.append(seg_loss.item())
                val_depth_losses.append(depth_loss.item())

                preds = seg_logits.argmax(dim=1)

                # DetectionMetric expects depth tensors shaped (B,H,W)
                metric.add(
                    preds,
                    label,
                    raw_depth,
                    depth,
                )

        val_metrics = metric.compute()

        # TensorBoard logging
        logger.add_scalar(
            "train/loss", np.mean(train_losses), epoch
        )
        logger.add_scalar(
            "train/seg_loss", np.mean(train_seg_losses), epoch
        )
        logger.add_scalar(
            "train/depth_loss", np.mean(train_depth_losses), epoch
        )
        logger.add_scalar(
            "train/accuracy", np.mean(train_accs), epoch
        )

        logger.add_scalar(
            "val/loss", np.mean(val_losses), epoch
        )
        logger.add_scalar(
            "val/seg_loss", np.mean(val_seg_losses), epoch
        )
        logger.add_scalar(
            "val/depth_loss", np.mean(val_depth_losses), epoch
        )
        logger.add_scalar(
            "val/accuracy", val_metrics["accuracy"], epoch
        )
        logger.add_scalar(
            "val/iou", val_metrics["iou"], epoch
        )
        logger.add_scalar(
            "val/abs_depth_error", val_metrics["abs_depth_error"], epoch
        )
        logger.add_scalar(
            "val/tp_depth_error", val_metrics["tp_depth_error"], epoch
        )

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{num_epoch}: "
                f"train_acc={np.mean(train_accs):.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_iou={val_metrics['iou']:.4f} "
                f"val_depth_mae={val_metrics['abs_depth_error']:.4f}"
            )

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    logger.close()

    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--optimizer", type=str, default="adam")
    
    #parser.add_argument("--batch_size", type=int, default=128)
    # optional: additional model hyperparamters
    #parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train_detection(**vars(parser.parse_args()))