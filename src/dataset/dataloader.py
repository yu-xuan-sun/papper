# main data-loader
import os
from typing import Any, Callable, Dict, Optional
import numpy as np
import torch
from torchvision import transforms as trsfs
from PIL import Image

from src.dataset.geo import VisionDataset
from src.dataset.utils import get_subset, load_file, encode_loc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EbirdVisionDataset(VisionDataset):

    def __init__(self,
                 df_paths,
                 data_base_dir,
                 bands,
                 env,
                 env_var_sizes,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 mode: Optional[str] = "train",
                 datatype="refl",
                 target="probs",
                 targets_folder="targets",
                 env_data_folder="environmental_bounded",
                 images_folder="images",
                 subset=None,
                 use_loc=False,
                 res=[],
                 loc_type=None,
                 num_species=684,
                 concat_env_to_sat: bool = True,
                 drop_to_rgb: bool = False) -> None:
        """
        df_paths: dataframe with paths to data for each hotspot
        data_base_dir: base directory for data
        bands: list of bands to include, anysubset of  ["r", "g", "b", "nir"] or  "rgb" (for image dataset) 
        env: list eof env data to take into account [ped, bioclim]
        transforms:
        mode : train|val|test
        datatype: "refl" (reflectance values ) or "img" (image dataset)
        target : "probs" or "binary"
        subset : None or list of indices of the indices of species to keep 
        """

        super().__init__()
        self.df = df_paths
        self.data_base_dir = data_base_dir
        self.total_images = len(df_paths)
        self.transform = transforms
        self.bands = bands
        self.env = env
        self.env_var_sizes = env_var_sizes
        self.mode = mode
        self.type = datatype
        self.target = target
        self.targets_folder = targets_folder
        self.env_data_folder = env_data_folder
        self.images_folder = images_folder
        self.subset = get_subset(subset, num_species)
        self.use_loc = use_loc
        self.loc_type = loc_type
        self.res = res
        self.num_species = num_species
        # 是否将 env 栅格在通道维拼接到卫星图像；dinov2 场景推荐 False，单独作为向量特征
        self.concat_env_to_sat = concat_env_to_sat
        # 针对 ViT/DINOv2 等 3 通道模型，将卫星图像压到 RGB 3 通道
        self.drop_to_rgb = drop_to_rgb

    def __len__(self):
        return self.total_images

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_ = {}

        hotspot_id = self.df.iloc[index]['hotspot_id']

        # satellite image
        if self.type == 'img':
            img_path = os.path.join(self.data_base_dir, self.images_folder + "_visual", hotspot_id + '_visual.tif')
        else:
            img_path = os.path.join(self.data_base_dir, self.images_folder, hotspot_id + '.tif')

        img = load_file(img_path)
        # 将输入稳健地转换为 CHW 张量
        if isinstance(img, Image.Image):
            img = np.array(img)
        if isinstance(img, np.ndarray):
            if img.ndim == 3:
                # 若已是 (C,H,W) 且通道数在前，则直接转换
                if img.shape[0] in (3, 4):
                    sats = torch.from_numpy(img).float()
                else:
                    # 视为 (H,W,C) -> (C,H,W)
                    sats = torch.from_numpy(img).permute(2, 0, 1).float()
            elif img.ndim == 2:
                # 单通道 (H,W) -> (1,H,W)
                sats = torch.from_numpy(img).unsqueeze(0).float()
            else:
                raise ValueError(f"Unexpected satellite image ndim={img.ndim} for {img_path}")
        elif torch.is_tensor(img):
            sats = img
            if sats.dim() == 3 and sats.shape[0] not in (3, 4) and sats.shape[-1] in (3, 4):
                sats = sats.permute(2, 0, 1)
        else:
            raise TypeError(f"Unsupported image type: {type(img)} from {img_path}")
        item_["sat"] = sats

        assert len(self.env) == len(self.env_var_sizes), "env variables sizes must be equal to the size of env vars specified`"
        # env rasters
        for i, env_var in enumerate(self.env):
            env_npy = os.path.join(self.data_base_dir, self.env_data_folder, hotspot_id + '.npy')
            env_data = load_file(env_npy)

            # 计算累计通道偏移，按指定 env 顺序切片
            ch_offset = int(sum(self.env_var_sizes[:i]))
            ch_count = int(self.env_var_sizes[i])
            ch_end = ch_offset + ch_count

            # 适配 env 数据存储格式：(C,H,W) 或 (H,W,C)
            if env_data.ndim != 3:
                raise ValueError(f"Env data must be 3D, got shape {env_data.shape} at {env_npy}")

            # 判定通道所在维度
            total_channels = int(sum(self.env_var_sizes))
            tensor: torch.Tensor
            if env_data.shape[0] == total_channels:
                # (C,H,W)
                arr = env_data[ch_offset:ch_end, :, :]
                tensor = torch.from_numpy(arr).float()  # (C,H,W)
                # 若切片为空，尝试使用全部通道或占位填充
                if tensor.shape[0] == 0:
                    if env_data.shape[0] == self.env_var_sizes[i]:
                        tensor = torch.from_numpy(env_data).float()
                    else:
                        tensor = torch.zeros(self.env_var_sizes[i], 1, 1, dtype=torch.float32)
            elif env_data.shape[-1] == total_channels:
                # (H,W,C)
                arr = env_data[:, :, ch_offset:ch_end]
                if arr.shape[-1] == 0:
                    if env_data.shape[-1] == self.env_var_sizes[i]:
                        arr = env_data  # 使用全部通道
                    else:
                        tensor = torch.zeros(self.env_var_sizes[i], 1, 1, dtype=torch.float32)
                        item_[env_var] = tensor
                        continue
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float()  # (C,H,W)
            else:
                # 文件可能只包含当前 env 的通道数（而非拼接）
                if env_data.shape[0] == self.env_var_sizes[i]:
                    # (C,H,W)
                    tensor = torch.from_numpy(env_data).float()
                elif env_data.shape[-1] == self.env_var_sizes[i]:
                    # (H,W,C)
                    tensor = torch.from_numpy(env_data).permute(2, 0, 1).float()
                else:
                    raise ValueError(
                        f"Env data channels mismatch. shape={env_data.shape}, expected total={total_channels} or current={self.env_var_sizes[i]}"
                    )

            # 最终保证维度为 (C,H,W)，且 C>0
            if tensor.dim() != 3:
                raise ValueError(f"Env tensor for '{env_var}' must be 3D (C,H,W), got {tuple(tensor.shape)}")
            if tensor.shape[0] == 0:
                tensor = torch.zeros(self.env_var_sizes[i], 1, 1, dtype=torch.float32)

            item_[env_var] = tensor

        t = trsfs.Compose(self.transform)
        item_ = t(item_)

        # 统一确保卫星图为 4D NCHW（N=1）
        if isinstance(item_["sat"], torch.Tensor):
            x = item_["sat"]
            if x.dim() == 2:  # (H,W)
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 3:  # (C,H,W) 或 (H,W,C)
                # 尝试规范为 (C,H,W)
                if x.shape[0] not in (3, 4) and x.shape[-1] in (3, 4):
                    x = x.permute(2, 0, 1)
                # 扩 batch 维
                x = x.unsqueeze(0)
            elif x.dim() == 4:
                pass
            else:
                raise ValueError(f"Unexpected 'sat' dims: {tuple(x.shape)}")
            # 如果需要，将卫星图像裁剪为 RGB 3 通道（优先保留前 3 个通道 r,g,b）
            if self.drop_to_rgb and x.shape[1] >= 3:
                x = x[:, 0:3, :, :]
            item_["sat"] = x

        # 在拼接前，确保各 env 的空间尺寸与 sat 一致（统一为 4D NCHW）
        if isinstance(item_["sat"], torch.Tensor) and item_["sat"].dim() == 4:
            _, _, Hs, Ws = item_["sat"].shape
            for e in self.env:
                if e in item_ and isinstance(item_[e], torch.Tensor):
                    ten = item_[e]
                    # 统一到 4D
                    if ten.dim() == 2:
                        ten = ten.unsqueeze(0).unsqueeze(0)
                    elif ten.dim() == 3:
                        ten = ten.unsqueeze(0)
                    elif ten.dim() == 4:
                        pass
                    else:
                        raise ValueError(f"Env tensor '{e}' must be 3D/4D before concat, got {tuple(ten.shape)}")
                    # 现在 ten 应该是 (1, C, H, W)
                    _, Ch, He, We = ten.shape
                    # 若通道为 0（异常），用零张量填充
                    if Ch == 0:
                        try:
                            idx = self.env.index(e)
                            Ch = int(self.env_var_sizes[idx])
                        except Exception:
                            Ch = 1
                        ten = torch.zeros((1, Ch, max(He, 1), max(We, 1)), dtype=torch.float32)
                    # 对齐空间尺寸
                    if He != Hs or We != Ws:
                        ten = torch.nn.functional.interpolate(ten.float(), size=(Hs, Ws), mode='nearest')
                    item_[e] = ten

        if self.concat_env_to_sat:
            # 旧行为：在通道维拼接 env 到 sat（4D NCHW，沿 dim=1 拼接）
            for e in self.env:
                if e not in item_:
                    continue
                # 最后一次检查尺寸
                if item_[e].shape[-2:] != item_["sat"].shape[-2:]:
                    item_[e] = torch.nn.functional.interpolate(item_[e].float(), size=item_["sat"].shape[-2:], mode='nearest')
                # 对齐 batch 维
                if item_[e].dim() == 3:
                    item_[e] = item_[e].unsqueeze(0)
                if item_["sat"].dim() == 3:
                    item_["sat"] = item_["sat"].unsqueeze(0)
                item_["sat"] = torch.cat([item_["sat"], item_[e]], dim=1).float()
        else:
            # 新行为：将 env 栅格做空间均值，形成向量特征，单独输出在 'env'
            env_feats = []
            for e in self.env:
                if e not in item_:
                    continue
                ten = item_[e]
                # 确保 4D NCHW
                if ten.dim() == 2:
                    ten = ten.unsqueeze(0).unsqueeze(0)
                elif ten.dim() == 3:
                    ten = ten.unsqueeze(0)
                # 对 H,W 求均值 -> (N,C)
                pooled = ten.float().mean(dim=(-2, -1))
                # 去掉 N 维（N=1） -> (C,)
                pooled = pooled.squeeze(0)
                env_feats.append(pooled)
                # 为节省内存，移除大 tensor
                del item_[e]
            if len(env_feats) > 0:
                item_["env"] = torch.cat(env_feats, dim=0).float()

        # target labels
        species = load_file(os.path.join(self.data_base_dir, self.targets_folder, hotspot_id + '.json'))
        if self.target == "probs":
            if not self.subset is None:
                item_["target"] = np.array(species["probs"])[self.subset]
            else:
                item_["target"] = species["probs"]
            item_["target"] = torch.Tensor(item_["target"])

        elif self.target == "binary":
            if self.subset is not None:
                targ = np.array(species["probs"])[self.subset]
            else:
                targ = species["probs"]
            item_["original_target"] = torch.Tensor(targ)
            targ[targ > 0] = 1
            item_["target"] = torch.Tensor(targ)

        elif self.target == "log":
            if not self.subset is None:
                item_["target"] = np.array(species["probs"])[self.subset]
            else:
                item_["target"] = species["probs"]

        else:
            raise NameError("type of target not supported, should be probs or binary")

        item_["num_complete_checklists"] = species["num_complete_checklists"]

        if self.use_loc:
            if self.loc_type == "latlon":
                lon, lat = torch.Tensor([item_["lon"]]), torch.Tensor([item_["lat"]])
                loc = torch.cat((lon, lat)).unsqueeze(0)
                loc = encode_loc(loc)
                item_["loc"] = loc

        item_["hotspot_id"] = hotspot_id

        return item_
