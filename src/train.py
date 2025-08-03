import argparse
import logging
import os
import sys
import time

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.data import decollate_batch
from monai.inferers import SliceInferer, SlidingWindowInfererAdapt
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose


from utils import read_config
from model_zoo import get_network
from data_generator import FetalDataLoader

torch.backends.cudnn.benchmark = True


def train(args):
    config = read_config(args.cfg, mode="train")

    if not os.path.exists(config.saved_model_path):
        os.makedirs(config.saved_model_path)

    logging.basicConfig(
        filename=config.saved_model_path + "/log_train.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(str(config))

    # Load data
    fetal_data = FetalDataLoader(config, Train=True)
    train_dataloader, val_dataloader = fetal_data.load_data()

    # Load the model
    model = get_network(config)
    device = args.device
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    if config.optimizer == "SGD":
        optimizer = SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=0.0001,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(
        #     model.parameters(),
        #     lr=config.learning_rate,
        #     weight_decay=0.00004
        # )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

    loss_function = DiceCELoss(
        include_background=config.include_background,
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        batch=False,
        smooth_nr=0.00001,
        smooth_dr=0.00001,
        lambda_dice=0.6,
        lambda_ce=0.4,
    )

    dice_metric = DiceMetric(
        include_background=config.include_background,
        reduction="mean",
        get_not_nans=False,
        ignore_empty=True)

    # Validation setting
    if config.spatial_dims == "2d":
        infer = SliceInferer(
            roi_size=(config.img_size[0], config.img_size[1]),
            sw_batch_size=1,
            cval=-1,
            spatial_dim=2,
            progress=False
        )
    else:
        infer = SlidingWindowInfererAdapt(
            roi_size=(config.img_size[0], config.img_size[1], config.img_size[2]),
            sw_batch_size=1,
        )

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=config.out_channels)])
    post_label = Compose([AsDiscrete(to_onehot=config.out_channels)])

    max_epochs = config.max_epochs
    val_interval = config.val_interval

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    # Start training
    logging.info("-" * 30 + "training starts" + "-" * 30)
    step_start = time.time()
    for epoch in range(max_epochs):
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()

        epoch_loss = 0
        step = 0
        for batch_data in train_dataloader:
            step += 1
            inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(f"{step}/{len(train_dataloader)}, " f"train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            val_epoch_loss = 0
            model.eval()
            with torch.no_grad():
                for val_data in val_dataloader:
                    val_inputs, val_labels = (val_data["image"].to(device), val_data["label"].to(device))

                    val_outputs = infer(val_inputs, model)

                    val_loss = loss_function(val_outputs, val_labels)
                    val_epoch_loss += val_loss.item()

                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                val_epoch_loss /= len(val_dataloader)
                scheduler.step(val_epoch_loss)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                metric_values.append(metric)

                # reset the status for next validation round
                dice_metric.reset()
                logging.info(f"epoch {epoch + 1}"
                             f" average validation dice: {metric:.4f}"
                             f" average validation loss: {val_epoch_loss:.4f}")

                # Save the model
                if metric > best_metric:
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        save_mode_path = os.path.join(
                            config.saved_model_path, 'checkpoint_best.pth'
                        )
                        torch.save(model.state_dict(), save_mode_path)
                        logging.info(
                            f"saved model at current epoch: {epoch + 1},"
                            f" current best mean dice: {metric:.4f}"
                            f" at epoch: {best_metric_epoch}"
                            f" validation_loss:{val_epoch_loss:.4f}"
                        )
                if (epoch + 1) == max_epochs:
                    save_mode_path = os.path.join(
                        config.saved_model_path, 'checkpoint_last.pth'
                    )
                    torch.save(model.state_dict(), save_mode_path)


        print("-" * 30)
        # end of one epoch

    train_time = time.time() - step_start
    logging.info(
        f"train completed in {train_time:.4f} seconds "  f"best_metric: {best_metric:.4f} "
        f"" f"at epoch: {best_metric_epoch}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        type=str,
                        default='./config.yml',
                        help='path to config file')

    parser.add_argument('--n_gpu',
                        type=int,
                        default=2,
                        help='total gpu')

    parser.add_argument('--deterministic',
                        type=int,
                        default=1,
                        help='whether use deterministic training')

    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        help='random seed')

    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='what device to use')

    args = parser.parse_args()

    train(args)
