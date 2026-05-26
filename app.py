"""Gradio UI for four image-classification methods and dataset evaluation."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from classifier import FourMethodClassifier, MODEL_SPECS, analyze_disagreements
from data_utils import discover_class_folder_dataset, prepare_dataset_from_upload

classifier = FourMethodClassifier()

DISPLAY_TO_KEY = {spec.display_name: key for key, spec in MODEL_SPECS.items()}
DEFAULT_MODELS = list(DISPLAY_TO_KEY.keys())


def _plot_top1(rows: List[dict]) -> Image.Image | None:
    top1 = [r for r in rows if r.get("rank") == 1 and r.get("status") == "OK"]
    if not top1:
        return None
    labels = [r["model"] for r in top1]
    values = [r["probability_percent"] for r in top1]

    fig = plt.figure(figsize=(8, max(3, 0.55 * len(labels))))
    plt.barh(labels, values)
    plt.xlabel("Top-1 confidence (%)")
    plt.title("Top-1 confidence comparison")
    plt.xlim(0, max(100, max(values) * 1.1))
    for idx, value in enumerate(values):
        plt.text(value + 0.5, idx, f"{value:.2f}%", va="center")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _plot_accuracy(metrics_df: pd.DataFrame, topk: int) -> Image.Image | None:
    ok = metrics_df[metrics_df["status"] == "OK"].copy()
    if ok.empty:
        return None
    labels = ok["model"].tolist()
    top1 = ok["top1_accuracy_percent"].tolist()
    topk_col = f"top{topk}_accuracy_percent"
    topk_values = ok[topk_col].tolist() if topk_col in ok.columns else top1

    x = list(range(len(labels)))
    width = 0.35
    fig = plt.figure(figsize=(9, 4.5))
    plt.bar([i - width / 2 for i in x], top1, width=width, label="Top-1 coarse accuracy")
    plt.bar([i + width / 2 for i in x], topk_values, width=width, label=f"Top-{topk} contains true")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Dataset accuracy comparison")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _specific_label(row: pd.Series, coarse_col: str, fallback_col: str) -> str:
    coarse = row.get(coarse_col, "")
    if str(coarse).strip().lower() == "other":
        return row.get(fallback_col, coarse)
    return coarse


def classify(image: Image.Image, selected_models: List[str], topk: int) -> Tuple[str, pd.DataFrame, Image.Image | None]:
    if image is None:
        return "请先上传一张图片。", pd.DataFrame(), None
    if not selected_models:
        return "请至少选择一个模型。", pd.DataFrame(), None

    keys = [DISPLAY_TO_KEY[name] for name in selected_models]
    outputs = classifier.predict_many(image, keys, topk=int(topk))

    rows = []
    md_lines = ["## 分类结果概览", ""]
    for out in outputs:
        if out.get("error"):
            md_lines.append(f"- **{out['model']}**：加载或推理失败。`{out['error']}`")
            rows.append(
                {
                    "status": "ERROR",
                    "model": out["model"],
                    "rank": "-",
                    "imagenet_label": "-",
                    "coarse_label": "-",
                    "class_id": "-",
                    "probability_percent": 0.0,
                    "source": out["source"],
                    "note": out["error"],
                }
            )
            continue

        first = out["results"][0] if out["results"] else None
        if first:
            md_lines.append(
                f"- **{out['model']}**：Top-1 = **{first['label']}**，置信度 **{first['probability_percent']:.3f}%**"
            )
        for item in out["results"]:
            coarse_label = item.get("coarse_label", "other")
            if str(coarse_label).strip().lower() == "other":
                coarse_label = item["label"]
            rows.append(
                {
                    "status": "OK",
                    "model": out["model"],
                    "rank": item["rank"],
                    "imagenet_label": item["label"],
                    "coarse_label": coarse_label,
                    "class_id": item["class_id"],
                    "probability_percent": item["probability_percent"],
                    "source": out["source"],
                    "note": out["description"],
                }
            )

    df = pd.DataFrame(rows)
    plot_img = _plot_top1(rows)
    return "\n".join(md_lines), df, plot_img


def evaluate_uploaded_dataset(
    uploaded_zip,
    selected_models: List[str],
    topk: int,
    batch_size: int,
    topn_disagreement: int,
    progress=gr.Progress(),
):
    if uploaded_zip is None:
        empty = pd.DataFrame()
        return "请先上传数据集 zip。", empty, None, empty, empty, [], None, None, None
    if not selected_models:
        empty = pd.DataFrame()
        return "请至少选择一个模型。", empty, None, empty, empty, [], None, None, None

    keys = [DISPLAY_TO_KEY[name] for name in selected_models]
    dataset_root, work_dir = prepare_dataset_from_upload(uploaded_zip)
    samples, dataset_labels = discover_class_folder_dataset(dataset_root)

    progress(0, desc="开始评估数据集")
    metrics_df, pred_df = classifier.evaluate_dataset(
        samples=samples,
        dataset_labels=dataset_labels,
        keys=keys,
        topk=int(topk),
        batch_size=int(batch_size),
        progress=progress,
    )
    if not pred_df.empty and {"pred_coarse_label", "pred_imagenet_label"}.issubset(pred_df.columns):
        pred_df["pred_coarse_label"] = pred_df.apply(
            lambda row: _specific_label(row, "pred_coarse_label", "pred_imagenet_label"),
            axis=1,
        )
    dis_df = analyze_disagreements(pred_df, topn=int(topn_disagreement))
    if "analysis_reason" in dis_df.columns:
        dis_df = dis_df.drop(columns=["analysis_reason"])
    if "model_predictions" in dis_df.columns:
        dis_df["model_judgements"] = dis_df["model_predictions"].astype(str).str.replace(" | ", "\n", regex=False)
    acc_plot = _plot_accuracy(metrics_df, int(topk))

    out_dir = Path(tempfile.mkdtemp(prefix="imgcls_eval_outputs_"))
    metrics_csv = out_dir / "model_accuracy_summary.csv"
    pred_csv = out_dir / "all_model_predictions.csv"
    dis_csv = out_dir / "largest_disagreement_samples.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    dis_df.to_csv(dis_csv, index=False, encoding="utf-8-sig")

    gallery = []
    for _, row in dis_df.iterrows():
        model_judgements = row.get("model_judgements", row.get("model_predictions", ""))
        caption = (
            f"真实类别: {row['true_label']}\n"
            f"投票: {row['vote_distribution']}\n"
            f"模型判断:\n{model_judgements}"
        )
        gallery.append((row["image_path"], caption))

    label_text = ", ".join(dataset_labels)
    ok_metrics = metrics_df[metrics_df["status"] == "OK"]
    best_line = ""
    if not ok_metrics.empty:
        best = ok_metrics.sort_values("top1_accuracy_percent", ascending=False).iloc[0]
        best_line = f"\n\nTop-1 粗类别准确率最高的是 **{best['model']}**：**{best['top1_accuracy_percent']:.2f}%**。"

    summary = (
        f"## 数据集评估完成\n\n"
        f"- 数据集根目录：`{dataset_root}`\n"
        f"- 类别数：**{len(dataset_labels)}**，类别：{label_text}\n"
        f"- 图片总数：**{len(samples)}**\n"
        f"- 评估模型数：**{len(keys)}**\n"
        f"- 说明：准确率采用 ImageNet 细粒度标签到数据集粗类别的映射，例如多个犬类 ImageNet 类别会统一映射为 `dog`。"
        f"{best_line}\n"
    )
    progress(1, desc="评估完成")
    return summary, metrics_df, acc_plot, dis_df, pred_df, gallery, str(metrics_csv), str(pred_csv), str(dis_csv)


with gr.Blocks(title="四种图像分类方法评估系统") as demo:
    gr.Markdown(
        """
# 四种图像分类方法评估系统

本系统实现并对比 **DenseNet-161、ConvNeXt-XLarge、ConvNeXt V2-Large、EVA-02 Large** 四种图像分类方法。  

预训练口径说明：ConvNeXt-XLarge、ConvNeXt V2-Large 和 EVA-02 Large 优先使用大规模预训练后再 ImageNet-1K 微调的权重；DenseNet-161 当前可用公开权重为 ImageNet-1K，结果对比时需要单独注明。

> 数据集格式要求：zip 内部采用 class-folder 结构，例如 `dataset/cat/001.jpg`、`dataset/dog/002.jpg`。
        """
    )

    with gr.Tab("单张图片分类"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="上传待分类图片")
                model_input = gr.CheckboxGroup(
                    choices=DEFAULT_MODELS,
                    value=DEFAULT_MODELS,
                    label="选择要运行的图像分类方法",
                )
                topk_input = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                run_btn = gr.Button("开始分类", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown(label="结果概览")
                table = gr.Dataframe(label="详细分类结果", wrap=True)
                chart = gr.Image(label="Top-1 置信度对比图")

        run_btn.click(
            fn=classify,
            inputs=[image_input, model_input, topk_input],
            outputs=[summary, table, chart],
        )

    with gr.Tab("数据集准确率与分歧分析"):
        with gr.Row():
            with gr.Column(scale=1):
                dataset_file = gr.File(label="上传有标注数据集 zip", file_types=[".zip"])
                eval_model_input = gr.CheckboxGroup(
                    choices=DEFAULT_MODELS,
                    value=DEFAULT_MODELS,
                    label="选择参与评估的模型",
                )
                eval_topk = gr.Slider(1, 10, value=5, step=1, label="Top-K accuracy 中的 K")
                batch_size = gr.Slider(1, 32, value=8, step=1, label="Batch size")
                topn_disagreement = gr.Slider(1, 20, value=8, step=1, label="展示分歧最大样本数")
                eval_btn = gr.Button("开始评估数据集", variant="primary")
            with gr.Column(scale=2):
                eval_summary = gr.Markdown(label="评估概览")
                metrics_table = gr.Dataframe(label="各模型准确率统计", wrap=True)
                accuracy_chart = gr.Image(label="准确率对比图")

        gr.Markdown("## 模型分歧最大的样本")
        disagreement_table = gr.Dataframe(label="分歧样本分析表", wrap=True)
        disagreement_gallery = gr.Gallery(label="分歧样本预览", columns=4, height="auto")

        gr.Markdown("## 全量预测明细")
        predictions_table = gr.Dataframe(label="每张图片 × 每个模型的预测结果", wrap=True)

        with gr.Row():
            metrics_file = gr.File(label="下载准确率统计 CSV")
            pred_file = gr.File(label="下载全量预测 CSV")
            dis_file = gr.File(label="下载分歧样本 CSV")

        eval_btn.click(
            fn=evaluate_uploaded_dataset,
            inputs=[dataset_file, eval_model_input, eval_topk, batch_size, topn_disagreement],
            outputs=[
                eval_summary,
                metrics_table,
                accuracy_chart,
                disagreement_table,
                predictions_table,
                disagreement_gallery,
                metrics_file,
                pred_file,
                dis_file,
            ],
        )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
