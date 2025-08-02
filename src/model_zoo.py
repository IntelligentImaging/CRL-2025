"""
Model Factory Module for Medical Image Segmentation

This module provides a factory function to create various neural network models
for medical image segmentation tasks, supporting both 2D and 3D architectures.
"""

import monai


def get_network(config):
    model_name = config.model_name
    in_ch = config.in_channels
    out_ch = config.out_channels
    dropout = config.dropout
    spatial_dims = int(config.spatial_dims[0])
    img_size = config.img_size

    if model_name == "UNet":
        model = monai.networks.nets.UNet(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=(24, 48, 96, 192, 384),
            strides=(2, 2, 2, 2),
            num_res_units=0,
            kernel_size=3,
            up_kernel_size=3,
            dropout=dropout,
        )

    elif model_name == "SegResNet":
        model = monai.networks.nets.SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=out_ch,
            dropout_prob=dropout,
        )

    elif model_name == "DynUNet":
        patch_size = [128, 128, 128]
        spacing = (1.0, 1.0, 1.0)
        spacings = spacing
        sizes = patch_size
        strides, kernels = [], []
        while True:
            spacing_ratio = [sp / min(spacings) for sp in spacings]
            stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])
        print("strides",strides)
        print("kernels",kernels)
        # initialise the network
        model = monai.networks.nets.DynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            # deep_supervision=True,
            # deep_supr_num=2,
            dropout=dropout,
            res_block=False,
        )

    elif model_name == 'AttUNet':
        model = monai.networks.nets.AttentionUnet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            dropout=dropout,
        )

    elif model_name == 'UNetr':
        model = monai.networks.nets.UNETR(
            in_channels=in_ch,
            out_channels=out_ch,
            img_size=img_size,
            spatial_dims=spatial_dims,
            feature_size=24,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            # pos_embed="perceptron",
            norm_name='instance',  # "instance",
            res_block=False,
            dropout_rate=dropout,
        )

    elif model_name == 'SwinUNETR':
        model = monai.networks.nets.SwinUNETR(
            img_size=img_size,
            in_channels=in_ch,
            out_channels=out_ch,
            spatial_dims=spatial_dims,
            feature_size=24,
            # norm_name='batch',  # "instance",
            drop_rate=dropout,
        )

    elif model_name == 'UAT':
        from models.TransAttUnet.TransAttUnet import UNet_Attention_Transformer_Multiscale
        model = UNet_Attention_Transformer_Multiscale(
            n_channels=1,
            n_classes=2,
            bilinear=True
        )

    elif model_name == "UMAMBA":
        from models.mamba.LightMUNet import LightMUNetModel
        model = LightMUNetModel(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=out_ch,
            feature_size=24,
            dropout=dropout,
 #           blocks_down= (1, 2, 2, 4),
   #         blocks_up= (1, 1, 1),            
        )
    elif model_name == "ATTUMAMBA":
        from models.mamba.attlightmunet import ATTLightMUNetModel
        model = ATTLightMUNetModel(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=out_ch,
            feature_size=12,
            dropout=dropout
        )

    elif model_name == "SlimUNETR":
        from models.SlimUNETR.SlimUNETR import SlimUNETR
        model = SlimUNETR(
            in_channels=1,
            out_channels=2,
            embed_dim=256,
            embedding_dim=8 * 8 * 4,  # w*h*d at the last stage
            channels=(32, 64, 128),
            blocks=(1, 2, 3, 2),
            heads=(1, 2, 4, 4),
            r=(2, 2, 2, 2),
            # distillation=False,
            dropout=0.15,
        )
    elif model_name == "SlimUNETR":
        from models.SlimUNETREQ.SlimUNETR import SlimUNETREQ
        model = SlimUNETREQ(
            in_channels=1,
            out_channels=2,
            embed_dim=256,
            embedding_dim=8 * 8 * 4,
            channels=(32, 64, 128),
            blocks=(1, 2, 3, 2),
            heads=(1, 2, 4, 4),
            r=(4, 2, 2, 1),
            dropout=0.15,
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model