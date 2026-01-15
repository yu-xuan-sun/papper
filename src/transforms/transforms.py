from typing import Dict
import os
from math import ceil, floor
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
import torchvision
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import collections
import numbers
import random

Module.__module__ = "torch.nn"

sat = ["sat"]
env = ["bioclim", "ped"]
landuse = ["landuse"]
all_data = sat + env + landuse


class RandomHorizontalFlip:  # type: ignore[misc,name-defined]
    """Horizontally flip the given sample randomly with a given probability."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize a new transform instance.
        Args:
            p: probability of the sample being flipped
        """
        super().__init__()
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Randomly flip the image and target tensors.
        Args:
            sample: a single data sample
        Returns:
            a possibly flipped sample
        """
        if torch.rand(1) < self.p:
            for s in sample:
                if s in all_data:
                    sample[s] = sample[s].flip(-1)

                elif s == "boxes":
                    height, width = sample[s].shape[-2:]
                    sample["boxes"][:, [0, 2]] = width - sample["boxes"][:, [2, 0]]

        return sample


class RandomVerticalFlip:  # type: ignore[misc,name-defined]
    """Vertically flip the given sample randomly with a given probability."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize a new transform instance.
        Args:
            p: probability of the sample being flipped
        """
        super().__init__()
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Randomly flip the image and target tensors.
        Args:
            sample: a single data sample
        Returns:
            a possibly flipped sample
        """
        if torch.rand(1) < self.p:
            for s in sample:
                if s in all_data:
                    sample[s] = sample[s].flip(-2)

                elif s == "boxes":
                    height, width = sample[s].shape[-2:]
                    sample["boxes"][:, [1, 3]] = height - sample["boxes"][:, [3, 1]]

            # if "mask" in sample:  #    sample["mask"] = sample["mask"].flip(-2)

        return sample


def normalize_custom(t, mini=0, maxi=1):
    if len(t.shape) == 3:
        return mini + (maxi - mini) * (t - t.min()) / (t.max() - t.min())

    batch_size = t.shape[0]
    min_t = t.reshape(batch_size, -1).min(1)[0].reshape(batch_size, 1, 1, 1)
    t = t - min_t
    max_t = t.reshape(batch_size, -1).max(1)[0].reshape(batch_size, 1, 1, 1)
    t = t / max_t
    return mini + (maxi - mini) * t


class Normalize:
    def __init__(self, maxchan=True, custom=None, subset=sat, normalize_by_255=False, verbose: bool = False):
        """
        custom : ([means], [std])
        means =[r: 894.6719, g: 932.5726, b:693.2768, nir: 2817.9849]
        std = [r:883.9763, g:747.6857, b:749.3098, nir: 1342.6334]
        subset: set of inputs on which to apply the normalization (typically env variables and sat would require different normalizations)
        """
        self.maxchan = maxchan
        # TODO make this work with the values of the normalization values computed over the whole dataset
        self.subset = subset
        self.custom = custom
        self.normalize_by_255 = normalize_by_255
        # 允许通过环境变量临时开启一次性调试：SATBIRD_NORMALIZE_DEBUG=1
        env_verbose = os.environ.get("SATBIRD_NORMALIZE_DEBUG", "0") == "1"
        self.verbose = verbose or env_verbose

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        d = {}
        if self.maxchan:
            for task in self.subset:
                tensor = sample[task]
                sample[task] = tensor / torch.amax(tensor, dim=(-2, -1), keepdims=True)
        # 标准化
        if self.normalize_by_255:
            for task in self.subset:
                sample[task] = sample[task] / 255
        else:
            if self.custom:
                means, std = self.custom
                # 允许 numpy 列表，转换为 Python list
                if hasattr(means, 'tolist'):
                    means = means.tolist()
                if hasattr(std, 'tolist'):
                    std = std.tolist()
                # 若自定义均值/方差为空，则跳过该标准化，避免将通道裁剪为0
                if means is None or std is None or len(means) == 0 or len(std) == 0:
                    # 不进行基于 custom 的归一化
                    expected_c = None
                    means = None
                    std = None
                else:
                    try:
                        expected_c = int(len(means)) if len(means) > 0 else None
                    except Exception:
                        expected_c = None
                # 一次性调试打印
                if self.verbose:
                    if not hasattr(self, "_debug_once"):
                        self._debug_once = False
                    do_debug = False
                    for task in self.subset:
                        if task in sample and isinstance(sample[task], torch.Tensor):
                            do_debug = True
                            break
                    if do_debug and not self._debug_once and expected_c is not None:
                        for task in self.subset:
                            if task in sample and isinstance(sample[task], torch.Tensor):
                                print(f"[Normalize][{task}] before shape={tuple(sample[task].shape)} expected_c={expected_c}")
                        self._debug_once = True
                for task in self.subset:
                    t = sample[task]
                    # 若没有有效的均值方差，跳过该 task 的 custom 标准化
                    if means is None or std is None:
                        continue
                    # 若输入为 2D (H,W)，先扩展为 (1,H,W)
                    if t.dim() == 2:
                        t = t.unsqueeze(0)
                        sample[task] = t
                    # 自动纠正通道维度位置，确保为 (C,H,W) 或 (N,C,H,W)
                    if expected_c is not None:
                        if t.dim() == 3:  # (C,H,W) 或其它顺序
                            if t.shape[0] != expected_c:
                                dims = list(t.shape)
                                if expected_c in dims:
                                    idx = dims.index(expected_c)
                                    if idx == 1:
                                        t = t.permute(1, 0, 2)
                                    elif idx == 2:
                                        t = t.permute(2, 0, 1)
                                else:
                                    # 默认将最后一维作为通道维
                                    t = t.permute(2, 0, 1)
                            # 若通道数仍大于期望，裁剪前 expected_c 个通道
                            if t.shape[0] > expected_c:
                                t = t[:expected_c, ...]
                        elif t.dim() == 4:  # (N,C,H,W) 或其它顺序
                            if t.shape[1] != expected_c:
                                dims = list(t.shape)
                                # 在非 batch 维中寻找 expected_c
                                candidate_idx = None
                                for idx in [1, 2, 3]:
                                    if dims[idx] == expected_c:
                                        candidate_idx = idx
                                        break
                                if candidate_idx is not None:
                                    if candidate_idx == 2:
                                        t = t.permute(0, 2, 1, 3)
                                    elif candidate_idx == 3:
                                        t = t.permute(0, 3, 1, 2)
                                    # candidate_idx==1 无需调整
                                else:
                                    # 默认将最后一维作为通道维
                                    t = t.permute(0, 3, 1, 2)
                            # 若通道数仍大于期望，裁剪前 expected_c 个通道
                            if t.shape[1] > expected_c:
                                t = t[:, :expected_c, ...]
                        # 回写修正后的张量
                        sample[task] = t
                    # 若为 4D 且 batch=1，先 squeeze 到 3D 做归一化，再还原
                    was_batched = False
                    if sample[task].dim() == 4 and sample[task].shape[0] == 1:
                        sample[task] = sample[task].squeeze(0)
                        was_batched = True
                    # 最终防守：确保为 3D 且通道数 = expected_c；必要时重排/裁剪/填充
                    x = sample[task]
                    if expected_c is not None and x.dim() == 3:
                        if x.shape[0] != expected_c:
                            # 若其它维等于 expected_c，则移动到通道维
                            if x.shape[1] == expected_c:
                                x = x.permute(1, 0, 2)
                            elif x.shape[2] == expected_c:
                                x = x.permute(2, 0, 1)
                            # 调整后若通道仍不对，进行裁剪或填充
                            if x.shape[0] > expected_c:
                                x = x[:expected_c, ...]
                            elif x.shape[0] < expected_c:
                                pad_c = expected_c - x.shape[0]
                                pad = torch.zeros(pad_c, x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)
                                x = torch.cat([x, pad], dim=0)
                                if self.verbose:
                                    if not hasattr(self, "_warn_padded") or not self._warn_padded:
                                        print(f"[Normalize] Padded channels for '{task}' from {x.shape[0]-pad_c} to {expected_c}")
                                        self._warn_padded = True
                        sample[task] = x
                    # 手动按通道归一化，兼容 CHW/NCHW 与 HWC/NHWC
                    x = sample[task]
                    # 如果没有有效的 custom 统计，跳过归一化
                    if means is None or std is None:
                        if was_batched:
                            sample[task] = sample[task].unsqueeze(0)
                        continue
                    m = torch.tensor(means, dtype=x.dtype, device=x.device)
                    s = torch.tensor(std, dtype=x.dtype, device=x.device)
                    if x.dim() == 3:
                        # 选择通道轴：优先找等于 expected_c 的轴，否则默认最后一轴
                        if expected_c is not None and expected_c in list(x.shape):
                            if x.shape[0] == expected_c:
                                c_axis = 0
                            elif x.shape[1] == expected_c:
                                c_axis = 1
                            elif x.shape[2] == expected_c:
                                c_axis = 2
                            else:
                                c_axis = 2
                        else:
                            # 兜底用首维为通道维
                            c_axis = 0
                        # 若通道数与均值长度不一致，按通道 loop 以避免 view 失败
                        ch = x.shape[c_axis]
                        if expected_c is not None and ch != expected_c:
                            # 调整到 CHW，再逐通道归一化
                            if c_axis == 1:
                                x = x.permute(1, 0, 2)
                            elif c_axis == 2:
                                x = x.permute(2, 0, 1)
                            ch = x.shape[0]
                            out = x.clone()
                            for ci in range(min(ch, expected_c)):
                                out[ci] = (x[ci] - m[ci]) / s[ci]
                            if ch < expected_c:
                                # 额外通道补零（数值已在前面填充为 0）
                                pass
                            x = out
                            # 还原回原通道轴
                            if c_axis == 1:
                                x = x.permute(1, 0, 2)
                            elif c_axis == 2:
                                x = x.permute(2, 0, 1)
                        else:
                            shape = [1, 1, 1]
                            shape[c_axis] = ch
                            m_ = m[:ch].view(*shape) if m.numel() >= ch else torch.zeros_like(x[:ch])
                            s_ = s[:ch].view(*shape) if s.numel() >= ch else torch.ones_like(x[:ch])
                            x = (x - m_) / s_
                    elif x.dim() == 4:
                        # 选择通道轴：优先 1 或 3
                        if expected_c is not None and (x.shape[1] == expected_c or x.shape[3] == expected_c):
                            c_axis = 1 if x.shape[1] == expected_c else 3
                        else:
                            c_axis = 1
                        ch = x.shape[c_axis]
                        if expected_c is not None and ch != expected_c:
                            # 调整到 NCHW，再逐通道归一化
                            if c_axis == 3:
                                x = x.permute(0, 3, 1, 2)
                            ch = x.shape[1]
                            out = x.clone()
                            for ci in range(min(ch, expected_c)):
                                out[:, ci] = (x[:, ci] - m[ci]) / s[ci]
                            x = out
                            # 若原来是 NHWC，恢复
                            if c_axis == 3:
                                x = x.permute(0, 2, 3, 1)
                        else:
                            shape = [1, 1, 1, 1]
                            shape[c_axis] = ch
                            m_ = m[:ch].view(*shape) if m.numel() >= ch else torch.zeros_like(x[:, :ch])
                            s_ = s[:ch].view(*shape) if s.numel() >= ch else torch.ones_like(x[:, :ch])
                            x = (x - m_) / s_
                    else:
                        raise ValueError(f"Normalize expects 3D/4D tensor, got shape {tuple(x.shape)} for '{task}'")
                    sample[task] = x
                    if was_batched:
                        sample[task] = sample[task].unsqueeze(0)
        return sample


class MatchRes:
    def __init__(self, target_size, custom):
        self.ped_res = 250
        self.bioclim_res = 1000
        self.sat_res = 10
        self.target_size = target_size
        self.custom = custom

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:

        H, W = self.target_size
        if "bioclim" in list(sample.keys()):
            # align bioclim with ped
            Hb, Wb = sample["bioclim"].shape[-2:]
            h = floor(Hb * self.sat_res / self.bioclim_res)
            w = floor(Wb * self.sat_res / self.bioclim_res)
            top = max(0, Hb // 2 - h // 2)
            left = max(0, Wb // 2 - w // 2)
            h, w = max(ceil(h), 1), max(ceil(w), 1)
            # Handle 3D/4D: keep channels intact, crop spatial dims only
            if sample["bioclim"].dim() == 3:
                sample["bioclim"] = sample["bioclim"][:, int(top): int(top + h), int(left): int(left + w)]
            elif sample["bioclim"].dim() == 4:
                sample["bioclim"] = sample["bioclim"][:, :, int(top): int(top + h), int(left): int(left + w)]
        if "ped" in list(sample.keys()):
            # align bioclim with ped
            Hb, Wb = sample["ped"].shape[-2:]
            # print(Hb,Wb)
            h = floor(Hb * self.sat_res / self.ped_res)
            w = floor(Wb * self.sat_res / self.ped_res)
            top = max(0, Hb // 2 - h // 2)
            left = max(0, Wb // 2 - w // 2)
            h, w = max(ceil(h), 1), max(ceil(w), 1)
            if sample["ped"].dim() == 3:
                sample["ped"] = sample["ped"][:, int(top): int(top + h), int(left): int(left + w)]
            elif sample["ped"].dim() == 4:
                sample["ped"] = sample["ped"][:, :, int(top): int(top + h), int(left): int(left + w)]

        means_bioclim, means_ped = self.custom

        for elem in list(sample.keys()):
            if elem in env:
                # 获取当前张量并检查空间与通道尺寸
                t = sample[elem]

                # 统一检查空间维是否为空（适配 3D/4D）
                if t.shape[-1] == 0 or t.shape[-2] == 0:
                    fill_means = means_bioclim if elem == "bioclim" else means_ped
                    filled = torch.tensor(fill_means, dtype=t.dtype, device=t.device).unsqueeze(-1).unsqueeze(-1)
                    t = filled  # (C,1,1)

                # 检查通道是否为 0，并按 3D/4D 分别处理
                if t.dim() == 3:  # (C,H,W)
                    c = t.shape[0]
                    if c == 0:
                        fill_means = means_bioclim if elem == "bioclim" else means_ped
                        t = torch.tensor(fill_means, dtype=t.dtype, device=t.device).unsqueeze(-1).unsqueeze(-1)
                    t = t.unsqueeze(0)  # (1,C,H,W)
                elif t.dim() == 4:  # (N,C,H,W)
                    c = t.shape[1]
                    if c == 0:
                        fill_means = means_bioclim if elem == "bioclim" else means_ped
                        filled = torch.tensor(fill_means, dtype=t.dtype, device=t.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                        # 目标形状 (1,C,1,1)
                        t = filled
                else:
                    raise ValueError(f"Expected 3D/4D tensor for env '{elem}', got shape {tuple(t.shape)}")

                # 插值到目标大小
                t = F.interpolate(t.float(), size=(H, W), mode='nearest')  # (N,C,H,W)
                sample[elem] = t.squeeze(0)  # (C,H,W)
        return sample


class RandomCrop:  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""

    def __init__(self, size, center=False, ignore_band=[], p=0.5):
        assert isinstance(size, (int, tuple, list))
        if not isinstance(size, int):
            assert len(size) == 2
            self.h, self.w = size
        else:
            self.h = self.w = size

        self.h = int(self.h)
        self.w = int(self.w)
        self.center = center
        self.ignore_band = ignore_band
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            sample: the input
        Returns:
            the cropped input
        """

        H, W = (sample["sat"].shape[-2:] if "sat" in sample else list(sample.values())[0].shape[-2:])
        for key in sample.keys():

            if (len(sample[key].shape) == 3):
                sample[key] = torch.unsqueeze(sample[key], 0)

        if torch.rand(1) > self.p:
            return (sample)
        else:
            if not self.center:

                top = max(0, np.random.randint(0, max(H - self.h, 1)))
                left = max(0, np.random.randint(0, max(W - self.w, 1)))
            else:
                top = max(0, (H - self.h) // 2)
                left = max(0, (W - self.w) // 2)

            item_ = {}
            for task, tensor in sample.items():
                if task in all_data and not task in self.ignore_band:
                    cropped = tensor[:, :, top: top + self.h, left: left + self.w]
                    # 若原图更小，裁剪结果可能 < 目标尺寸，这里进行零填充到 (self.h,self.w)
                    ch, h_cur, w_cur = cropped.shape[1], cropped.shape[-2], cropped.shape[-1]
                    if h_cur != self.h or w_cur != self.w:
                        pad_h = max(self.h - h_cur, 0)
                        pad_w = max(self.w - w_cur, 0)
                        # pad=(pad_left, pad_right, pad_top, pad_bottom) 这里右和下方向补齐
                        cropped = F.pad(cropped, (0, pad_w, 0, pad_h), mode='constant', value=0)
                        # 防御性裁剪，避免超过目标大小
                        cropped = cropped[:, :, : self.h, : self.w]
                    item_.update({task: cropped})
                else:
                    item_.update({task: tensor})

            return item_


class Resize:
    def __init__(self, size):
        """
        size = (height, width) target size
        """
        self.h, self.w = size

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for s in sample:
            if s in sat:
                sample[s] = F.interpolate(sample[s].float(), size=(self.h, self.w), mode='bilinear')
            elif s in env or s in landuse:

                sample[s] = F.interpolate(sample[s].float(), size=(self.h, self.w), mode='nearest')
        return (sample)


class RandomGaussianNoise:  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""

    def __init__(self, prob=0.5, max_noise=5e-2, std=1e-2):

        self.max = max_noise
        self.std = std
        self.prob = prob

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            sample: the input
        Returns:
            theinput with added gaussian noise
        """
        if random.random() < self.prob:
            for s in sample:
                if s in sat:
                    # sample Gaussian noise and clamp the noise magnitude, not the image
                    noise = torch.normal(0, self.std, sample[s].shape)
                    noise = torch.clamp(noise, min=-self.max, max=self.max)
                    sample[s] = sample[s] + noise
        return sample


class RandBrightness:
    def __init__(self, prob=0.5, max_value=0):
        self.value = max_value
        self.prob = prob

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if random.random() < self.prob:
            for s in sample:
                if s in sat:
                    sample[s][:, 0:3, :, :] = torchvision.transforms.functional.adjust_brightness(
                        sample[s][:, 0:3, :, :], self.value)
        return sample


class RandContrast:
    def __init__(self, prob=0.5, max_factor=0):
        self.factor = max_factor
        self.prob = prob

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if random.random() < self.prob:
            for s in sample:
                if s in sat:
                    sample[s][:, 0:3, :, :] = torchvision.transforms.functional.adjust_contrast(sample[s][:, 0:3, :, :],
                                                                                                self.factor)
        return sample


class RandRotation:
    """random rotate the ndarray image with the degrees.

    Args:
        degrees (number or sequence): the rotate degree.
                                  If single number, it must be positive.
                                  if squeence, it's length must 2 and first number should small than the second one.

    Raises:
        ValueError: If degrees is a single number, it must be positive.
        ValueError: If degrees is a sequence, it must be of len 2.

    Returns:
        ndarray: return rotated ndarray image.
    """

    def __init__(self, degrees, prob=0.5, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.center = center
        self.prob = prob

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if random.random() < self.prob:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            for s in sample:
                if s in sat:
                    sample[s] = torchvision.transforms.functional.rotate(sample[s], angle=angle, center=self.center)
        return sample


class GaussianBlurring:
    """Convert the input ndarray image to blurred image by gaussian method.

    Args:
        kernel_size (int): kernel size of gaussian blur method. (default: 3)

    Returns:
        ndarray: the blurred image.
    """

    def __init__(self, prob=0.5, kernel_size=3):
        self.kernel_size = kernel_size
        self.prob = prob

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if random.random() < self.prob:
            for s in sample:
                if s in sat:
                    sample[s] = torchvision.transforms.functional.gaussian_blur(sample[s], kernel_size=self.kernel_size)
        return sample


def get_transform(transform_item, mode):
    """Returns the transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if transform_item.name == "crop" and not (transform_item.ignore is True or transform_item.ignore == mode):
        return RandomCrop((transform_item.height, transform_item.width),
            center=(transform_item.center == mode or transform_item.center == True),
            ignore_band=transform_item.ignore_band or [], p=transform_item.p)
    elif transform_item.name == "matchres" and not (transform_item.ignore is True or transform_item.ignore == mode):
        return MatchRes(transform_item.target_size, transform_item.custom_means)

    elif transform_item.name == "hflip" and not (transform_item.ignore is True or transform_item.ignore == mode):
        return RandomHorizontalFlip(p=transform_item.p or 0.5)

    elif transform_item.name == "vflip" and not (transform_item.ignore is True or transform_item.ignore == mode):
        return RandomVerticalFlip(p=transform_item.p or 0.5)

    elif transform_item.name == "randomnoise" and not (transform_item.ignore is True or transform_item.ignore == mode):
        return RandomGaussianNoise(max_noise=transform_item.max_noise or 5e-2, std=transform_item.std or 1e-2)

    elif transform_item.name == "normalize" and not (transform_item.ignore is True or transform_item.ignore == mode):

        return Normalize(maxchan=transform_item.maxchan, custom=transform_item.custom or None,
                         subset=transform_item.subset, normalize_by_255=transform_item.normalize_by_255)

    elif transform_item.name == "resize" and not (transform_item.ignore is True or transform_item.ignore == mode):

        return Resize(size=transform_item.size)

    elif transform_item.name == "blur" and not (transform_item.ignore is True or transform_item.ignore == mode):
        return GaussianBlurring(prob=transform_item.p)
    elif transform_item.name == "rotate" and not (transform_item.ignore is True or transform_item.ignore == mode):
        return RandRotation(prob=transform_item.p, degrees=transform_item.val)

    elif transform_item.name == "randomcontrast" and not (
            transform_item.ignore is True or transform_item.ignore == mode):
        return RandContrast(prob=transform_item.p, max_factor=transform_item.val)

    elif transform_item.name == "randombrightness" and not (
            transform_item.ignore is True or transform_item.ignore == mode):
        return RandBrightness(prob=transform_item.p, max_value=transform_item.val)

    elif transform_item.ignore is True or transform_item.ignore == mode:
        return None

    raise ValueError("Unknown transform_item {}".format(transform_item))


def get_transforms(opts, mode):
    """Get all the transform functions listed in opts.data.transforms
    using get_transform(transform_item, mode)
    """
    crop_transforms = []
    transforms = []

    for t in opts.data.transforms:
        if t.name == "normalize":
            if t.subset == ["sat"] and opts.data.datatype == "refl":
                t.custom = opts.variables.rgbnir_means, opts.variables.rgbnir_std
            if t.subset == ["sat"] and opts.data.datatype == "img":
                t.custom = opts.variables.visual_means, opts.variables.visual_stds
            elif t.subset == ["bioclim"]:
                t.custom = opts.variables.bioclim_means, opts.variables.bioclim_std
            elif t.subset == ["ped"]:
                t.custom = opts.variables.ped_means, opts.variables.ped_std
        if t.name == "matchres":
            t.custom_means = opts.variables.bioclim_means, opts.variables.ped_means

        # account for multires
        if t.name == 'crop' and len(opts.data.multiscale) > 1:
            for res in opts.data.multiscale:
                # adapt hight and width to vars in multires
                t.hight, t.width = res, res
                crop_transforms.append(get_transform(t, mode))
        else:
            transforms.append(get_transform(t, mode))
    transforms = [t for t in transforms if t is not None]
    if crop_transforms:
        crop_transforms = [t for t in crop_transforms if t is not None]
        print('crop transforms ', crop_transforms)
        return crop_transforms, transforms
    else:
        return transforms
