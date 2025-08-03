import os
import sys
import time
import logging

import monai
import pandas as pd
import argparse
import numpy as np

import torch
from fvcore.nn import FlopCountAnalysis
import monai.transforms as tr
from monai.data import decollate_batch
from monai.handlers import from_engine
from monai.inferers import SliceInferer, SlidingWindowInferer
from monai.metrics import MeanIoU, DiceMetric
from monai.utils import set_determinism
from monai.metrics import HausdorffDistanceMetric

from model_zoo import get_network
from data_generator import FetalDataLoader
from utils import visualize_sample, read_config

def get_memory_consumption():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6  # Convert to MB
    return None


def test(args):
    config = read_config(args.cfg, mode="test")

    if not os.path.exists(config.save_results_path):
        os.makedirs(config.save_results_path)

    logging.basicConfig(
        filename=config.save_results_path + "/log_test_" + config.modality + ".txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(str(config))

    device = args.device
    model = get_network(config)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    model_path = os.path.join(config.saved_model_path, config.saved_model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"total_trainable_parameters: {pytorch_total_params}")

    model.eval()

    fetal_test_data = FetalDataLoader(config, Train=False)
    test_dataloader, test_files, test_org_transforms_list = fetal_test_data.load_data()

    if config.spatial_dims == "2d":
        inferer = SliceInferer(
            roi_size=(config.img_size[0], config.img_size[1]),
            sw_batch_size=1,
            cval=-1,
            spatial_dim=2,
            progress=False
        )
        # inferer = SliceInferer(
        #     roi_size=(config.img_size"], config.img_size"]),
        #     spatial_dim=2,
        #     sw_batch_size=4,
        #     overlap=0.50,
        #     progress=False
        # )

    else:
        inferer = SlidingWindowInferer(
            roi_size=(config.img_size[0],
                      config.img_size[1],
                      config.img_size[2]),
            sw_batch_size=4
        )

    dice_metric = DiceMetric(
        include_background=config.include_background,
        reduction="mean",
        get_not_nans=False,
        ignore_empty=True
    )
    dice_metric_perclass = DiceMetric(
        include_background=config.include_background,
        reduction="none",  # <<<< crucial: returns one value per class
        get_not_nans=False,
        ignore_empty=True,
    )
    iou_metric = MeanIoU(
        include_background=config.include_background,
        reduction="mean",
        get_not_nans=False,
        ignore_empty=True
    )
    iou_metric_perclass = MeanIoU(
        include_background=config.include_background,
        reduction="none",  # <<<< crucial: returns one value per class
        get_not_nans=False,
        ignore_empty=True,
    )

    hd_metric = HausdorffDistanceMetric(percentile=95)
    hd_metric_perclass = HausdorffDistanceMetric(reduction="none", percentile=95)



    test_time = []
    test_memory = []
    subj_list = []
    mask_list = []
    data_type = []

    post_transforms_list = [
        tr.Invertd(
            keys="pred",
            transform=tr.Compose(test_org_transforms_list),
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        tr.Activationsd(keys="pred", softmax=True),
        tr.AsDiscreted(keys="pred", argmax=True, to_onehot=None),
        # tr.KeepLargestConnectedComponentd(keys="pred", applied_labels=1, num_components=1, is_onehot=False),
        # tr.RemoveSmallObjectsd(keys="pred", min_size=12, connectivity=1),
    ]

    if config.save_predictions:
        post_transforms_list.append(
            tr.SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=config.save_results_path,
                separate_folder=False,
                data_root_dir=config.data_root_dir,
                output_postfix="pred",
                resample=True
            )
        )

    post_transforms = tr.Compose(post_transforms_list)

    with torch.no_grad():
        for i, test_data in enumerate(test_dataloader):
            test_inputs, test_labels = test_data["image"].to(device), test_data["label"].to(device)

            start_memory = get_memory_consumption()
            start_time = time.time()

            test_data["pred"] = inferer(test_inputs, model)
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

            test_time.append(time.time() - start_time) #/ test_inputs.shape[-1])
            end_memory = get_memory_consumption()

            if start_memory is not None and end_memory is not None:
                test_memory.append(end_memory - start_memory)

            # Calculate Loss
            original_label = tr.LoadImage()(test_labels[0].meta["filename_or_obj"])[None, None]
            one_hot_label = monai.networks.utils.one_hot(
                torch.as_tensor(original_label, device=device),
                num_classes=config.out_channels
            )
            one_hot_pred = monai.networks.utils.one_hot(
                torch.as_tensor(from_engine(["pred"])(test_data)[0][None], device=device),
                num_classes=config.out_channels
            )

            dice_metric(y_pred=one_hot_pred, y=one_hot_label.cuda())
            dice_metric_perclass(y_pred=one_hot_pred, y=one_hot_label.cuda())
            dice = dice_metric.aggregate()
            dice_per_class = dice_metric_perclass.aggregate()

            iou_metric(y_pred=one_hot_pred, y=one_hot_label.cuda())
            iou_metric_perclass(y_pred=one_hot_pred, y=one_hot_label.cuda())
            iou = iou_metric.aggregate()
            iou_per_class = iou_metric_perclass.aggregate()

            hd_metric(y_pred=one_hot_pred, y=one_hot_label.cuda(), spacing=test_labels.meta["pixdim"][0][1].item())
            hd_metric_perclass(y_pred=one_hot_pred, y=one_hot_label.cuda(), spacing=test_labels.meta["pixdim"][0][1].item())
            hd = hd_metric.aggregate()
            hd_per_class = hd_metric_perclass.aggregate()

            if config.modality == "T2W" or config.modality == "otherscanners":
                subject_id = os.path.basename(os.path.dirname(test_inputs.meta["filename_or_obj"][0]))
                if subject_id.endswith('s1') or subject_id.endswith('s2'):
                    subject_id = subject_id[:-2]
                subj_list.append(subject_id)
                data_type.append(os.path.basename(os.path.dirname(os.path.dirname(
                    test_inputs.meta["filename_or_obj"][0]))))

            elif config.modality == "DWI":
                subject_id = os.path.basename(test_inputs.meta["filename_or_obj"][0]).split('_')[0]
                if subject_id.endswith('s1') or subject_id.endswith('s2'):
                    subject_id = subject_id[:-2]
                subj_list.append(subject_id)
                data_type.append(os.path.basename(os.path.dirname(test_inputs.meta["filename_or_obj"][0])))

            elif config.modality == "fMRI":
                subject_id = os.path.dirname(test_inputs.meta["filename_or_obj"][0]).split('/')[-2][:-2]
                subj_list.append(subject_id)
                data_type.append('fMRI')

            elif config.modality == "fMRI-ON":
                subject_id = os.path.dirname(test_inputs.meta["filename_or_obj"][0]).split('/')[6]
                subj_list.append(subject_id)
                data_type.append('fMRI-ON')

            elif config.modality == "T2MIDL":
                subject_id = os.path.dirname(test_inputs.meta["filename_or_obj"][0]).split('/')[4]
                subj_list.append(subject_id)
                data_type.append('T2MIDL')

            if config.plot_results:
                # if dice.item() < 0.85:
                test_output = from_engine(["pred"])(test_data)
                original_image = tr.LoadImage()(test_output[0].meta["filename_or_obj"])
                original_label = tr.LoadImage()(test_labels[0].meta["filename_or_obj"])

                filename_without_extension = os.path.splitext(os.path.basename
                                                              (test_inputs.meta["filename_or_obj"][0]))[0]
                parent_folder = os.path.basename(os.path.dirname(test_inputs.meta["filename_or_obj"][0]))

                new_filename = f"{parent_folder}_{filename_without_extension}.png"
                save_dir = os.path.join(config.save_results_path, 'bad_dice_figs')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_name = os.path.join(save_dir, new_filename)

                visualize_sample(
                    original_image,
                    test_output[0].detach().cpu()[0],
                    gt=original_label,
                    volume_dice=dice.item(),
                    slice_dice=None,
                    save_name=save_name,
                    display=False
                )
                # print(test_inputs.meta["filename_or_obj"][0])
                # print(dice.item())
                # print(iou.item())
                # print(dice_per_class

    if config.save_metrics:
        header = ["Method", "Modality", "Type", "Subject", "Dice", "IoU", "HD"]
        dice_list = (dice_metric.get_buffer().detach().cpu().numpy()).tolist()
        iou_list = (iou_metric.get_buffer().detach().cpu().numpy()).tolist()
        hd_list = (hd_metric.get_buffer().detach().cpu().numpy()).tolist()
        modality = [config.modality] * len(dice_list)
        method = [config.model_name] * len(dice_list)

        data = list(zip(method, modality, data_type, subj_list, dice_list, iou_list, hd_list))

        file_path = os.path.join(config.save_results_path, config.modality + "_" + method[0] + ".csv")
        df = pd.DataFrame(data, columns=header)
        df.to_csv(file_path, index=False)

        logging.info(f"evaluation metric dice mean: {np.mean(dice_list)}")
        logging.info(f"evaluation metric dice std: {np.std(dice_list)}")

        logging.info(f"evaluation metric iou mean: {np.mean(iou_list)}")
        logging.info(f"evaluation metric iou std: {np.std(iou_list)}")

        logging.info(f"evaluation metric hd mean: {np.mean(hd_list)}")
        logging.info(f"evaluation metric hd std: {np.std(hd_list)}")

    flops = FlopCountAnalysis(model, test_inputs)
    logging.info(f"GFLOPs: {flops.total() / 1e9:.2f}")

    # logging.info(f"evaluation metric dice: {dice_metric.aggregate()}")
    # logging.info(f"evaluation metric iou: {iou_metric.aggregate()}")

    logging.info(f"latency mean: {np.mean(test_time[1:])}")
    logging.info(f"latency std: {np.std(test_time[1:])}")

    logging.info(f"memory mean: {np.mean(test_memory[1:])}")
    logging.info(f"memory std: {np.std(test_memory[1:])}")

    dice_metric.reset()
    iou_metric.reset()
    dice_metric_perclass.reset()


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
                        default=torch.device("cuda" if torch.cuda.is_available() else "copu"),
                        help='what device to use')

    args = parser.parse_args()

    if args.deterministic:
        set_determinism(seed=12345)
    else:
        set_determinism(seed=None)

    test(args)
