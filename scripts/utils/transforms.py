import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    MapTransform,
    ToTensord,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    SpatialPadd,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
)

class ConvertToMultiChannelMSDd(MapTransform):
    """
    Трансформация для конвертации меток MSD в многоканальный формат (BraTS regions):
    Label 1: Edema (ED)
    Label 2: Non-enhancing tumor (NET)
    Label 3: Enhancing tumor (ET)

    Channels:
    0: WT (Whole Tumor) = ED + NET + ET (1+2+3)
    1: TC (Tumor Core) = NET + ET (2+3)
    2: ET (Enhancing Tumor) = ET (3)
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # WT: ED(1) + NET(2) + ET(3)
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 1, d[key] == 2), 
                    d[key] == 3
                )
            )
            # TC: NET(2) + ET(3)
            result.append(
                torch.logical_or(d[key] == 2, d[key] == 3)
            )
            # ET: ET(3)
            result.append(d[key] == 3)
            
            d[key] = torch.stack(result, axis=0).float()
        return d

class AddMultiScaleLabelsd(MapTransform):
    """
    Трансформация для предварительного вычисления меток разных разрешений (для Deep Supervision).
    """
    def __init__(self, keys, levels=5):
        super().__init__(keys)
        self.levels = levels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            if not isinstance(label, torch.Tensor):
                label = torch.as_tensor(label)
            
            for i in range(1, self.levels):
                size = [max(1, s // (2**i)) for s in label.shape[1:]]
                low_res_label = torch.nn.functional.interpolate(
                    label.unsqueeze(0).float(),
                    size=size,
                    mode='nearest'
                ).squeeze(0)
                d[f"{key}_level_{i}"] = low_res_label
        return d

def get_transforms(config):
    train_transforms_list = [
        #Препроцессинг
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ScaleIntensityRangePercentilesd(
            keys="image",
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            channel_wise=True,
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ConvertToMultiChannelMSDd(keys="label"),
        SpatialPadd(keys=["image", "label"], spatial_size=config["img_size"]),
        #Аугментации
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config["img_size"],
            pos=1,
            neg=1,
            num_samples=4,
        ),
        RandAffined(
            keys=["image", "label"],
            prob=0.2,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
        ),
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            prob=0.1,
            spatial_size=config["img_size"],
            mode=("bilinear", "nearest"),
        ),
        RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.7, 1.5)),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ToTensord(keys=["image", "label"]),
    ]

    if config.get("deep_supervision", False):
        train_transforms_list.append(AddMultiScaleLabelsd(keys=["label"], levels=5))

    train_transforms = Compose(train_transforms_list)

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ScaleIntensityRangePercentilesd(
            keys="image",
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            channel_wise=True,
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ConvertToMultiChannelMSDd(keys="label"),
        SpatialPadd(keys=["image", "label"], spatial_size=config["img_size"]),
        ToTensord(keys=["image", "label"]),
    ])
    return train_transforms, val_transforms
