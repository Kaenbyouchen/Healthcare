# -*- coding: utf-8 -*-
"""
spatial_sampler.py
在训练时对每个病例按概率图抽样patch中心。
- guided_ratio: 0~1，引导采样比例（建议0.4）
- patch_size: (Z, Y, X)
- 对每个训练step：若命中引导分支，从 priors 载入 prob_map，按概率选择中心；否则走随机分支
- 对于 has_prior=0 或缺失的病例：回退到随机采样；（可选）使用 atlas_prior
"""

import os
import numpy as np
from pathlib import Path

class SpatialAwareSampler:
    def __init__(self, priors_dir, guided_ratio=0.4, patch_size=(64, 160, 160),
                 atlas_prior=None, rng=None):
        self.priors_dir = Path(priors_dir)
        self.guided_ratio = float(guided_ratio)
        self.patch_size = tuple(patch_size)
        self.rng = np.random.default_rng() if rng is None else rng
        self.atlas_prior = atlas_prior  # 可选: (Z,Y,X) 归一化概率图（数据级先验）

    def _load_prior(self, case_id):
        f = self.priors_dir / f"{case_id}_priors.npz"
        if not f.exists():
            return None
        data = np.load(f)
        if int(data["has_prior"][0]) == 0:
            return None
        pm = data["prob_map"].astype(np.float32)
        # 防止与 patch_size 不匹配：若需要，可在此插值到当前体素尺寸
        return pm

    def _sample_center_from_prob(self, prob_map):
        # 从 prob_map 中按概率取一个体素索引 (z,y,x)
        flat = prob_map.ravel()
        idx = self.rng.choice(len(flat), p=flat)
        zyx = np.unravel_index(idx, prob_map.shape)
        return tuple(int(i) for i in zyx)

    def _clip_center(self, center, img_shape):
        # 确保以 center 截 patch 不越界
        cz, cy, cx = center
        pz, py, px = self.patch_size
        Z, Y, X = img_shape
        z1 = np.clip(cz - pz // 2, 0, max(0, Z - pz))
        y1 = np.clip(cy - py // 2, 0, max(0, Y - py))
        x1 = np.clip(cx - px // 2, 0, max(0, X - px))
        return (int(z1), int(y1), int(x1))

    def propose_patch_start(self, case_id, img_shape):
        """
        给定病例ID与图像体素形状，返回建议的 patch 起点 (z1,y1,x1)
        由上层 dataloader 用于切 patch
        """
        use_guided = (self.rng.random() < self.guided_ratio)
        if use_guided:
            pm = self._load_prior(case_id)
            if pm is None:
                pm = self.atlas_prior
            if pm is not None and pm.shape == img_shape:
                center = self._sample_center_from_prob(pm)
                return self._clip_center(center, img_shape)
        # 回退：均匀随机
        Z, Y, X = img_shape
        pz, py, px = self.patch_size
        z1 = self.rng.integers(0, max(1, Z - pz + 1))
        y1 = self.rng.integers(0, max(1, Y - py + 1))
        x1 = self.rng.integers(0, max(1, X - px + 1))
        return (int(z1), int(y1), int(x1))
