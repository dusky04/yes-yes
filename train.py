from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    c,
    exp_name: str,
    weights_dir: Path,
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.Optimizer] = None,
):
    weights_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    loss_fn = loss_fn.to(device)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)

    # training loop
    for epoch in range(c.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_iterator: tqdm[DataLoader[Any]] = tqdm(
            train_dataloader, desc=f"Processing Epoch: {epoch + 1:02d}"
        )
        for batch_idx, (videos, labels) in enumerate(batch_iterator):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(videos)
            loss = loss_fn(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            batch_iterator.set_postfix(
                {
                    "loss": loss.item(),
                    "acc": 100.0 * correct / total if total > 0 else 0,
                }
            )

        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = 100.0 * correct / total if total > 0 else 0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for videos, labels in test_dataloader:
                videos = videos.to(device)
                labels = labels.to(device)
                outputs = model(videos)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * videos.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_epoch_loss = val_loss / val_total if val_total > 0 else 0
        val_epoch_acc = 100.0 * val_correct / val_total if val_total > 0 else 0

        if scheduler:
            scheduler.step()

        print(
            f"Epoch {epoch + 1}/{c.NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%"
        )

        # Save model checkpoint
        torch.save(
            model.state_dict(),
            Path(weights_dir) / f"{exp_name}_epoch{epoch + 1}.pth",
        )

    print("Training complete.")
