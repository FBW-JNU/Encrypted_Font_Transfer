import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import wandb
from tqdm import tqdm

from data.dataset import TextDataset, TextDatasetval
from models.model import TRGAN
from params import *


def make_loader(dataset, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
    )


def main():
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "Encrypted-Font-Transfer"),
        name=EXP_NAME,
        mode=os.environ.get("WANDB_MODE", "disabled"),
    )
    init_project()

    train_loader = make_loader(TextDataset(num_examples=NUM_EXAMPLES))
    validation_loader = make_loader(TextDatasetval(num_examples=NUM_EXAMPLES))

    model = TRGAN()
    model_path = SAVED_MODELS_DIR / EXP_NAME
    user_key = load_or_create_key(model_path, resume=RESUME)
    checkpoint_path = model_path / "model.pth"
    if RESUME:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    model.keyHashNet.requires_grad_(False)
    total_steps = EPOCHS * len(train_loader)
    progress = tqdm(total=total_steps, desc="training", unit="batch", dynamic_ncols=True)

    for epoch in range(EPOCHS):
        model.epoch = epoch
        started = time.time()

        for batch_index, batch in enumerate(train_loader):
            if batch_index % NUM_CRITIC_DWL_TRAIN == 0:
                model._set_input(batch, user_key)
                model.optimize_Ext()
                model.optimize_Ext_step()

            progress.update(1)
            if batch_index % 10 == 0 or batch_index == len(train_loader) - 1:
                progress.set_postfix(
                    epoch=f"{epoch + 1}/{EPOCHS}",
                    loss_E=f"{model.loss_E.detach().item():.6f}",
                    refresh=False,
                )

        losses = model.get_current_losses()
        wandb.log({
            "loss-E": losses["ext"],
            "epoch": epoch,
            "time-per-epoch": time.time() - started,
        })

        if epoch % SAVE_IMAGE == 0:
            validation_batch = next(iter(validation_loader))
            training_page = model._generate_page(model.sdata, model.input["swids"])
            validation_page = model._generate_page(
                validation_batch["simg"].to(DEVICE), validation_batch["swids"]
            )
            images = [
                wandb.Image(training_page, caption="training page"),
                wandb.Image(validation_page, caption="validation page"),
            ]
            if SECRET_TYPE == 0:
                images.extend([
                    wandb.Image(model.secret[0], caption="secret"),
                    wandb.Image(model.ext[0], caption="extraction"),
                ])
            wandb.log({"results": images})

        if epoch % SAVE_MODEL == 0:
            torch.save(model.state_dict(), checkpoint_path)
        if epoch % SAVE_MODEL_HISTORY == 0:
            torch.save(model.state_dict(), model_path / f"model-{epoch}.pth")
            torch.save(model.keyHashNet.state_dict(), model_path / "hashNet.pth")

    progress.close()


if __name__ == "__main__":
    main()
