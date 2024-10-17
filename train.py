import argparse
import math
from copy import deepcopy

import torch
import wandb
from IPython.core.display_functions import clear_output
from torch import nn

from dataset import get_datasets, get_dataloaders
from loss import CapsuleLoss
from metrics import produce_evaluation_file, calculate_eer_tdcf
from model.models import get_model

from utils import progressbar, get_optimizer, load_checkpoint


def main(config):
    wandb.init(project=config["wandb_project"],
               config=config)
    datasets = get_datasets(config)
    dataloaders = get_dataloaders(datasets, config)

    model = get_model(config).to(config["device"])

    optimizer = get_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=wandb.config['step_size'],
        gamma=wandb.config['gamma'])
    if config["model"] == "ResCapsGuard":
        loss_fn = CapsuleLoss(gpu_id=wandb.config['gpu_id'], weight=torch.FloatTensor([0.1, 0.9]))
    elif config["model"] == "Res2TCNGuard":
        loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(config["device"]))

    best_score = math.inf
    best_state = None

    for epoch in range(config["epoches"]):
        # train part
        train_loss = 0
        prefix = '%s / %s, best_score %s ' % (epoch + 1, config["epoches"], best_score)
        for data, label, _ in progressbar(dataloaders["train"], prefix=prefix):
            data, label = data.to(config["device"]), label.to(config["device"])
            optimizer.zero_grad()
            classes, class_ = model(
                data,
                random=wandb.config['random'],
                dropout=wandb.config['dropout'],
                random_size=wandb.config['random_size']
            )
            if config["model"] == "ResCapsGuard":
                loss = loss_fn(classes, label)
            elif config["model"] == "Res2TCNGuard":
                loss = loss_fn(class_, label)
            else:
                raise NotImplementedError
            train_loss += loss.item() / len(dataloaders["train"])
            loss.backward()
            optimizer.step()
        scheduler.step()

        # val_part
        dev_loss = produce_evaluation_file(
            dataloaders["dev"],
            model, config["device"],
            loss_fn,
            config["produced_file"],
            config["dev_label_path"])
        eer, tdcf = calculate_eer_tdcf(cm_scores_file=config["produced_file"],
                                       asv_score_file=config["asv_score_filename"],
                                       output_file=None,
                                       printout=False)

        if best_score > eer:
            best_score = eer
            best_state = deepcopy(model.state_dict())
            path = 'best_checkpoint' + str(best_score) + ".pth"
            torch.save(best_state, path)

        clear_output()

        metrics = {
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_eer": eer,
            "dev_tdcf": tdcf
        }
        wandb.log(metrics)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='configs/config_rescapsguard.json')
    args = parser.parse_args()
    config = load_checkpoint(args.config)
    main(config)

