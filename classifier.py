"""
Four image-classification methods demo backend with dataset evaluation.

Implemented methods:
1. DenseNet-161       (torchvision)
2. ConvNeXt-XLarge    (timm)
3. ConvNeXt V2-Large  (timm)
4. EVA-02 Large       (timm)

DINOv2 has been removed because torch.hub/GitHub authorization and download
problems are common in classroom environments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from data_utils import load_image
from label_mapping import canonical_dataset_label, imagenet_to_coarse, topk_contains_label


@dataclass
class ModelSpec:
    key: str
    display_name: str
    source: str
    description: str


@dataclass
class ModelBundle:
    model: torch.nn.Module
    transform: Callable[[Image.Image], torch.Tensor]
    categories: List[str]
    display_name: str
    source: str
    description: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "densenet161": ModelSpec(
        key="densenet161",
        display_name="DenseNet-161",
        source="torchvision.models.densenet161",
        description="最大常用 DenseNet 配置之一，使用 ImageNet-1K 监督训练权重。",
    ),
    "convnext_xlarge": ModelSpec(
        key="convnext_xlarge",
        display_name="ConvNeXt-XLarge",
        source="timm/convnext_xlarge.fb_in22k_ft_in1k_384",
        description="ConvNeXt XLarge，使用 ImageNet-22K 预训练后再 ImageNet-1K 微调的高分辨率权重。",
    ),
    "convnextv2_large": ModelSpec(
        key="convnextv2_large",
        display_name="ConvNeXt V2-Large",
        source="timm/convnextv2_large.fcmae_ft_in22k_in1k_384",
        description="ConvNeXt V2 Large，使用 ImageNet-22K 预训练后再 ImageNet-1K 微调的高分辨率权重。",
    ),
    "eva02_large": ModelSpec(
        key="eva02_large",
        display_name="EVA-02 Large",
        source="timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        description="EVA-02 Large，使用 MIM 大规模预训练后再 ImageNet-22K 和 ImageNet-1K 微调的权重。",
    ),
}


def _get_imagenet_categories() -> List[str]:
    try:
        from torchvision.models import DenseNet161_Weights

        return list(DenseNet161_Weights.IMAGENET1K_V1.meta["categories"])
    except Exception:
        return [f"class_{i}" for i in range(1000)]


def _build_timm_transform(model: torch.nn.Module):
    from timm.data import create_transform

    try:
        from timm.data import resolve_model_data_config

        data_config = resolve_model_data_config(model)
    except Exception:
        from timm.data import resolve_data_config

        data_config = resolve_data_config({}, model=model)
    return create_transform(**data_config, is_training=False)


class FourMethodClassifier:
    """Lazy model loader, predictor, and dataset evaluator."""

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.categories = _get_imagenet_categories()
        self._cache: Dict[str, ModelBundle] = {}

    @property
    def available_models(self) -> Dict[str, str]:
        return {k: v.display_name for k, v in MODEL_SPECS.items()}

    def load(self, key: str) -> ModelBundle:
        if key in self._cache:
            return self._cache[key]
        if key not in MODEL_SPECS:
            raise KeyError(f"Unknown model key: {key}")

        if key == "densenet161":
            bundle = self._load_densenet161()
        elif key == "convnext_xlarge":
            bundle = self._load_convnext_xlarge()
        elif key == "convnextv2_large":
            bundle = self._load_convnextv2_large()
        elif key == "eva02_large":
            bundle = self._load_eva02_large()
        else:
            raise KeyError(key)

        bundle.model.to(self.device)
        bundle.model.eval()
        self._cache[key] = bundle
        return bundle

    def _load_densenet161(self) -> ModelBundle:
        from torchvision.models import DenseNet161_Weights, densenet161

        weights = DenseNet161_Weights.IMAGENET1K_V1
        model = densenet161(weights=weights)
        return ModelBundle(
            model=model,
            transform=weights.transforms(),
            categories=list(weights.meta["categories"]),
            display_name=MODEL_SPECS["densenet161"].display_name,
            source=MODEL_SPECS["densenet161"].source,
            description=MODEL_SPECS["densenet161"].description,
        )

    def _load_convnext_xlarge(self) -> ModelBundle:
        candidates = [
            "convnext_xlarge.fb_in22k_ft_in1k_384",
            "convnext_xlarge.fb_in22k_ft_in1k",
            "convnext_large.fb_in22k_ft_in1k_384",
            "convnext_large.fb_in22k_ft_in1k",
            "convnext_small.fb_in22k_ft_in1k",
            "convnext_small.fb_in1k",
        ]
        return self._load_timm_first_available(candidates, "convnext_xlarge")

    def _load_timm_first_available(self, candidates: List[str], key: str) -> ModelBundle:
        import timm

        last_error: Exception | None = None
        for name in candidates:
            try:
                model = timm.create_model(name, pretrained=True)
                transform = _build_timm_transform(model)
                return ModelBundle(
                    model=model,
                    transform=transform,
                    categories=self.categories,
                    display_name=MODEL_SPECS[key].display_name,
                    source=f"timm/{name}",
                    description=MODEL_SPECS[key].description,
                )
            except Exception as exc:
                last_error = exc

        try:
            fallback_patterns = {
                "convnext_xlarge": "convnext_xlarge*in1k",
                "convnextv2_large": "convnextv2_large*in1k",
                "eva02_large": "eva02_large*in1k",
            }
            pattern = fallback_patterns.get(key, "*in1k")
            found = timm.list_models(pattern, pretrained=True)
            if found:
                model = timm.create_model(found[0], pretrained=True)
                transform = _build_timm_transform(model)
                return ModelBundle(
                    model=model,
                    transform=transform,
                    categories=self.categories,
                    display_name=MODEL_SPECS[key].display_name,
                    source=f"timm/{found[0]}",
                    description=MODEL_SPECS[key].description + f"（自动回退到 {found[0]}）",
                )
        except Exception as exc:
            last_error = exc

        raise RuntimeError(
            f"Could not load {MODEL_SPECS[key].display_name}. "
            f"Please upgrade timm or check model availability. Last error: {last_error}"
        )

    def _load_convnextv2_large(self) -> ModelBundle:
        candidates = [
            "convnextv2_large.fcmae_ft_in22k_in1k_384",
            "convnextv2_large.fcmae_ft_in22k_in1k",
            "convnextv2_base.fcmae_ft_in22k_in1k",
            "convnextv2_base.fcmae_ft_in1k",
        ]
        return self._load_timm_first_available(candidates, "convnextv2_large")

    def _load_eva02_large(self) -> ModelBundle:
        candidates = [
            "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
            "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
            "eva02_large_patch14_448.mim_in22k_ft_in1k",
            "eva_large_patch14_336.in22k_ft_in1k",
            "eva02_base_patch14_448.mim_m38m_ft_in22k_in1k",
            "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
            "eva02_base_patch14_448.mim_in22k_ft_in1k",
            "eva02_small_patch14_336.mim_in22k_ft_in1k",
            "eva02_tiny_patch14_336.mim_in22k_ft_in1k",
        ]
        return self._load_timm_first_available(candidates, "eva02_large")

    def _logits_to_results(self, logits: torch.Tensor, bundle: ModelBundle, topk: int) -> list[dict]:
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1, logits.shape[-1]).mean(dim=1)
        probs = F.softmax(logits, dim=-1)
        k = min(topk, probs.shape[-1])
        values, indices = torch.topk(probs, k=k, dim=-1)
        batch_results = []
        for row_values, row_indices in zip(values.cpu().tolist(), indices.cpu().tolist()):
            results = []
            for rank, (idx, score) in enumerate(zip(row_indices, row_values), start=1):
                label = bundle.categories[idx] if idx < len(bundle.categories) else f"class_{idx}"
                results.append(
                    {
                        "rank": rank,
                        "class_id": int(idx),
                        "label": label,
                        "probability": float(score),
                        "probability_percent": round(float(score) * 100, 3),
                    }
                )
            batch_results.append(results)
        return batch_results

    @torch.inference_mode()
    def predict_one(self, image: Image.Image, key: str, topk: int = 5, dataset_labels: Iterable[str] | None = None) -> Dict[str, Any]:
        bundle = self.load(key)
        image = image.convert("RGB")
        x = bundle.transform(image).unsqueeze(0).to(self.device)
        logits = bundle.model(x)
        results = self._logits_to_results(logits, bundle, topk=topk)[0]
        if results:
            top1 = results[0]
            top1["coarse_label"] = imagenet_to_coarse(top1["class_id"], top1["label"], dataset_labels)
        for item in results:
            item["coarse_label"] = imagenet_to_coarse(item["class_id"], item["label"], dataset_labels)
        return {
            "key": key,
            "model": bundle.display_name,
            "source": bundle.source,
            "description": bundle.description,
            "results": results,
        }

    def predict_many(self, image: Image.Image, keys: List[str], topk: int = 5, dataset_labels: Iterable[str] | None = None) -> List[Dict[str, Any]]:
        outputs = []
        for key in keys:
            try:
                outputs.append(self.predict_one(image, key, topk=topk, dataset_labels=dataset_labels))
            except Exception as exc:
                spec = MODEL_SPECS.get(key)
                outputs.append(
                    {
                        "key": key,
                        "model": spec.display_name if spec else key,
                        "source": spec.source if spec else "unknown",
                        "description": spec.description if spec else "",
                        "error": str(exc),
                        "results": [],
                    }
                )
        return outputs

    @torch.inference_mode()
    def evaluate_dataset(
        self,
        samples: list[dict],
        dataset_labels: list[str],
        keys: list[str],
        topk: int = 5,
        batch_size: int = 8,
        progress=None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate selected models on class-folder samples.

        Returns:
            metrics_df: per-model accuracy summary.
            pred_df: per-image predictions.
        """
        if not samples:
            raise ValueError("samples is empty")
        if not keys:
            raise ValueError("请选择至少一个模型。")

        rows: list[dict] = []
        total_model_count = len(keys)
        for model_idx, key in enumerate(keys, start=1):
            spec = MODEL_SPECS.get(key)
            try:
                bundle = self.load(key)
            except Exception as exc:
                rows.append(
                    {
                        "model_key": key,
                        "model": spec.display_name if spec else key,
                        "status": "ERROR",
                        "error": str(exc),
                    }
                )
                continue

            n_batches = math.ceil(len(samples) / batch_size)
            for b in range(n_batches):
                if progress is not None:
                    progress((model_idx - 1) / total_model_count + (b / max(n_batches, 1)) / total_model_count,
                             desc=f"{bundle.display_name}: batch {b + 1}/{n_batches}")
                chunk = samples[b * batch_size : (b + 1) * batch_size]
                images = [bundle.transform(load_image(s["path"])).to(self.device) for s in chunk]
                x = torch.stack(images, dim=0)
                logits = bundle.model(x)
                batch_results = self._logits_to_results(logits, bundle, topk=topk)

                for sample, results in zip(chunk, batch_results):
                    true_label = canonical_dataset_label(sample["label"])
                    top1 = results[0]
                    pred_coarse = imagenet_to_coarse(top1["class_id"], top1["label"], dataset_labels)
                    topk_hit = topk_contains_label(results, true_label, dataset_labels)
                    rows.append(
                        {
                            "image_path": sample["path"],
                            "file": sample["file"],
                            "true_label": true_label,
                            "model_key": key,
                            "model": bundle.display_name,
                            "status": "OK",
                            "pred_imagenet_label": top1["label"],
                            "pred_imagenet_id": top1["class_id"],
                            "pred_coarse_label": pred_coarse,
                            "top1_confidence_percent": top1["probability_percent"],
                            "top1_correct": pred_coarse == true_label,
                            f"top{topk}_contains_true": bool(topk_hit),
                            "topk_text": "; ".join(
                                [
                                    f"{r['rank']}. {r['label']} → {r['coarse_label']} ({r['probability_percent']:.2f}%)"
                                    for r in [
                                        {
                                            **item,
                                            "coarse_label": imagenet_to_coarse(item["class_id"], item["label"], dataset_labels),
                                        }
                                        for item in results
                                    ]
                                ]
                            ),
                            "error": "",
                        }
                    )
            if progress is not None:
                progress(model_idx / total_model_count, desc=f"{bundle.display_name}: done")

        pred_df = pd.DataFrame(rows)
        metrics = []
        for key in keys:
            spec = MODEL_SPECS.get(key)
            sub = pred_df[(pred_df["model_key"] == key) & (pred_df["status"] == "OK")]
            err = pred_df[(pred_df["model_key"] == key) & (pred_df["status"] == "ERROR")]
            if sub.empty:
                metrics.append(
                    {
                        "model": spec.display_name if spec else key,
                        "status": "ERROR",
                        "num_samples": 0,
                        "top1_accuracy": None,
                        f"top{topk}_accuracy": None,
                        "avg_top1_confidence_percent": None,
                        "error": err["error"].iloc[0] if not err.empty else "No predictions",
                    }
                )
                continue
            metrics.append(
                {
                    "model": sub["model"].iloc[0],
                    "status": "OK",
                    "num_samples": len(sub),
                    "top1_accuracy": round(float(sub["top1_correct"].mean()), 4),
                    "top1_accuracy_percent": round(float(sub["top1_correct"].mean()) * 100, 2),
                    f"top{topk}_accuracy": round(float(sub[f"top{topk}_contains_true"].mean()), 4),
                    f"top{topk}_accuracy_percent": round(float(sub[f"top{topk}_contains_true"].mean()) * 100, 2),
                    "avg_top1_confidence_percent": round(float(sub["top1_confidence_percent"].mean()), 3),
                    "error": "",
                }
            )
        metrics_df = pd.DataFrame(metrics)
        return metrics_df, pred_df


def _entropy_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


def analyze_disagreements(pred_df: pd.DataFrame, topn: int = 8) -> pd.DataFrame:
    """Find samples on which models disagree most and attach heuristic reasons."""
    ok = pred_df[pred_df["status"] == "OK"].copy()
    if ok.empty:
        return pd.DataFrame()

    rows = []
    for image_path, group in ok.groupby("image_path"):
        true_label = group["true_label"].iloc[0]
        preds = list(group["pred_coarse_label"].astype(str))
        confs = list(group["top1_confidence_percent"].astype(float))
        labels = list(group["pred_imagenet_label"].astype(str))
        models = list(group["model"].astype(str))
        correct_count = int((group["top1_correct"] == True).sum())
        counts = {p: preds.count(p) for p in sorted(set(preds))}
        unique_count = len(counts)
        max_vote = max(counts.values()) if counts else 0
        entropy = _entropy_from_counts(counts)
        disagreement_score = round((unique_count - 1) + entropy + (1 - max_vote / max(len(preds), 1)), 4)

        if correct_count == 0:
            reason = "所有模型 Top-1 都没有映射到真实类别，可能是主体较小、背景干扰强，或数据集粗标签与 ImageNet 细粒度标签不完全对应。"
        elif unique_count >= max(3, len(preds)):
            reason = "各模型分别关注了不同视觉区域，说明图像可能包含多个显著目标或局部纹理很强的干扰物。"
        elif max(confs) - min(confs) > 35:
            reason = "模型置信度差异较大，可能有的模型抓住了主体，有的模型受背景、姿态或裁剪影响。"
        elif correct_count < len(preds):
            reason = "部分模型正确、部分模型错误，通常反映该样本边界特征不典型，或类别之间外观相近。"
        else:
            reason = "虽然粗类别一致，但 ImageNet 细粒度标签或置信度存在差异，可用于观察不同模型的表征偏好。"

        model_pred_text = " | ".join(
            [f"{m}: {p}/{lab} ({c:.1f}%)" for m, p, lab, c in zip(models, preds, labels, confs)]
        )
        rows.append(
            {
                "image_path": image_path,
                "file": group["file"].iloc[0],
                "true_label": true_label,
                "num_models": len(group),
                "unique_predicted_coarse_labels": unique_count,
                "vote_distribution": ", ".join([f"{k}:{v}" for k, v in counts.items()]),
                "correct_model_count": correct_count,
                "confidence_range_percent": round(max(confs) - min(confs), 3) if confs else 0,
                "disagreement_score": disagreement_score,
                "model_predictions": model_pred_text,
                "analysis_reason": reason,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["unique_predicted_coarse_labels", "disagreement_score", "confidence_range_percent"],
        ascending=[False, False, False],
    ).head(int(topn))
    return df.reset_index(drop=True)
