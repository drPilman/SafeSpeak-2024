import argparse
import math
from copy import deepcopy

import torch
import wandb
from IPython.core.display_functions import clear_output
from torch import nn

from custom_dataset import get_datasets
from dataset import get_dataloaders
from loss import CapsuleLoss
from metrics import validate, calculate_eer
from model.models import get_model
from custom_scheduler import get_cosine_warm_up
from utils import progressbar, get_optimizer, load_checkpoint


def main(config):
    wandb.init(
        project=config["wandb_project"], config=config, entity=config["wandb_entity"]
    )
    datasets = get_datasets(config)
    dataloaders = get_dataloaders(datasets, config)
    best_score = math.inf
    data_len = len(dataloaders["train"])

    model = get_model(config)
    model= nn.DataParallel(model).to(config["device"])

    optimizer = get_optimizer(model, config)
    all_step = config["epoches"]*data_len
    scheduler = get_cosine_warm_up(optimizer, int(all_step*0.01), all_step)
    # torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=wandb.config["step_size"], gamma=wandb.config["gamma"]
    # )
    if config["model"] == "ResCapsGuard":
        loss_fn = CapsuleLoss(gpu_id=wandb.config['gpu_id'], weight=torch.FloatTensor([0.1, 0.9]))
    elif config["model"] == "Res2TCNGuard":
        loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(config["device"]))
    elif config["model"] == "Wav2Vec2":
        loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(config["device"])).to(config["device"])
    else:
        raise NotImplementedError



    for epoch in range(config["epoches"]):
        # train part
        train_loss = 0
        model.train()
        prefix = "%s / %s, best_score %s " % (epoch + 1, config["epoches"], best_score)
        for i, (data, label) in progressbar(dataloaders["train"], prefix=prefix):
            data, label = data.to(config["device"]), label.to(config["device"])
            optimizer.zero_grad()
            classes, class_ = model(
                data,
                random=wandb.config["random"],
                dropout=wandb.config["dropout"],
                random_size=wandb.config["random_size"],
            )
            if config["model"] == "ResCapsGuard":
                loss = loss_fn(classes, label)
            elif config["model"] in ["Res2TCNGuard", "Wav2Vec2"]:
                loss = loss_fn(class_, label)
            else:
                raise NotImplementedError
            train_loss += loss.item() / data_len
            loss.backward()
            optimizer.step()
            scheduler.step()
            wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
            if i!=0 and i%config["val_step"]==0:
                val_loss, eer = validate(dataloaders["validate"], model, config["device"], loss_fn)
                wandb.log({
                    "val_loss": val_loss,
                    "val_eer": eer
                })
                if config["test"]:
                    break
                if best_score > eer:
                    best_score = eer
                    torch.save(model.state_dict(), "best_checkpoint.pth")
                    with open("best.txt", "w") as f:
                        f.write(f"{epoch} {i} {best_score}")
        val_loss, eer = validate(dataloaders["validate"], model, config["device"], loss_fn)
        wandb.log({
            "epoch_train_loss": train_loss,
            "val_loss_epoch": val_loss,
            "val_eer_epoch": eer
        })
        if best_score > eer:
            best_score = eer
            torch.save(model.state_dict(), "best_checkpoint.pth")
            with open("best.txt", "w") as f:
                f.write(f"{epoch} end {best_score}")
        clear_output()


    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_wav2vec.json")
    args = parser.parse_args()
    config = load_checkpoint(args.config)
    main(config)
