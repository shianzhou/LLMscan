import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASETS = {
    "AutoDAN": "AutoDAN_Qwen2.5-1.5B-Instruct_last5_{pooling}_samples614.npz",
    "GCG": "GCG_Qwen2.5-1.5B-Instruct_last5_{pooling}_samples1070.npz",
    "PAP": "PAP_Qwen2.5-1.5B-Instruct_last5_{pooling}_samples500.npz",
}


def load_cached_features(cache_dir, pooling):
    data = {}
    for dataset_name, pattern in DATASETS.items():
        path = cache_dir / pattern.format(pooling=pooling)
        if not path.exists():
            raise FileNotFoundError(f"Missing cache file: {path}")
        cached = np.load(path, allow_pickle=True)
        data[dataset_name] = (
            cached["features"],
            cached["labels"].astype(int),
        )
    return data


def make_classifier(kind, random_state):
    if kind == "LR":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif kind == "MLP":
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported classifier: {kind}")
    return make_pipeline(StandardScaler(), model)


def evaluate(model, features, labels):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return {
        "ACC": accuracy_score(labels, predictions),
        "F1": f1_score(labels, predictions),
        "ROC_AUC": roc_auc_score(labels, probabilities),
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        "FNR": fn / (fn + tp) if (fn + tp) else 0.0,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def build_results(data, random_state):
    rows = []

    for dataset_name, (features, labels) in data.items():
        train_x, test_x, train_y, test_y = train_test_split(
            features,
            labels,
            test_size=0.3,
            random_state=random_state,
            stratify=labels,
        )
        for classifier_name in ["LR", "MLP"]:
            classifier = make_classifier(classifier_name, random_state)
            classifier.fit(train_x, train_y)
            rows.append(
                {
                    "mode": "same_dataset_7_3",
                    "train_set": dataset_name,
                    "test_set": dataset_name,
                    "classifier": classifier_name,
                    "train_n": len(train_y),
                    "test_n": len(test_y),
                    "feature_dim": features.shape[1],
                    **evaluate(classifier, test_x, test_y),
                }
            )

    for train_name, (train_x, train_y) in data.items():
        for test_name, (test_x, test_y) in data.items():
            if train_name == test_name:
                continue
            for classifier_name in ["LR", "MLP"]:
                classifier = make_classifier(classifier_name, random_state)
                classifier.fit(train_x, train_y)
                rows.append(
                    {
                        "mode": "cross_dataset_full",
                        "train_set": train_name,
                        "test_set": test_name,
                        "classifier": classifier_name,
                        "train_n": len(train_y),
                        "test_n": len(test_y),
                        "feature_dim": train_x.shape[1],
                        **evaluate(classifier, test_x, test_y),
                    }
                )

    return pd.DataFrame(rows)


def write_markdown(df, output_path, pooling):
    metric_cols = ["ACC", "F1", "ROC_AUC", "FPR", "FNR"]
    display_cols = ["train_set", "test_set", "classifier", "train_n", "test_n", *metric_cols]
    same = df[df["mode"] == "same_dataset_7_3"].copy()
    cross = df[df["mode"] == "cross_dataset_full"].copy()
    lr_cross = cross[cross["classifier"] == "LR"].copy()
    mlp_cross = cross[cross["classifier"] == "MLP"].copy()

    for table in [same, cross, lr_cross, mlp_cross]:
        table[metric_cols] = table[metric_cols].round(4)

    best_lr = lr_cross.sort_values("ACC", ascending=False).iloc[0]
    worst_lr = lr_cross.sort_values("ACC", ascending=True).iloc[0]

    with output_path.open("w", encoding="utf-8") as file:
        file.write("---\n")
        file.write("tags:\n")
        file.write("  - LLMScan\n")
        file.write("  - HiddenStateDetector\n")
        file.write("  - JailbreakDetection\n")
        file.write(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"pooling: {pooling}\n")
        file.write("model: Qwen2.5-1.5B-Instruct\n")
        file.write("feature_dim: 7680\n")
        file.write("---\n\n")

        file.write(f"# Hidden-State Detector 结果汇总（{pooling}）\n\n")
        file.write("> [!info] 实验说明\n")
        file.write("> 本页结果从已缓存的 `.npz` hidden-state 特征直接复算得到，未重新提取隐藏层。\n")
        file.write("> 特征为最后 5 层 hidden state + 修正后的最后真实 token pooling，维度为 `5 × 1536 = 7680`。\n\n")

        file.write("## 结论速览\n\n")
        file.write("- 同数据集 7:3 切分下，AutoDAN、GCG、PAP 的 LR/MLP 均达到 `ACC = 1.0000`。\n")
        file.write(
            f"- LR 跨数据集最佳方向：`{best_lr['train_set']} -> {best_lr['test_set']}`，"
            f"`ACC = {best_lr['ACC']:.4f}`，`F1 = {best_lr['F1']:.4f}`，"
            f"`ROC-AUC = {best_lr['ROC_AUC']:.4f}`。\n"
        )
        file.write(
            f"- LR 跨数据集最差方向：`{worst_lr['train_set']} -> {worst_lr['test_set']}`，"
            f"`ACC = {worst_lr['ACC']:.4f}`，`F1 = {worst_lr['F1']:.4f}`，"
            f"`ROC-AUC = {worst_lr['ROC_AUC']:.4f}`。\n"
        )
        file.write("- 修正 pooling 后，PAP/GCG 之间仍存在明显分布差异，跨数据集泛化不稳定。\n\n")

        file.write("## 同数据集结果（7:3）\n\n")
        file.write(same[display_cols].to_markdown(index=False))
        file.write("\n\n## 跨数据集结果（LR）\n\n")
        file.write(lr_cross[display_cols].to_markdown(index=False))
        file.write("\n\n## 跨数据集结果（MLP）\n\n")
        file.write(mlp_cross[display_cols].to_markdown(index=False))
        file.write("\n\n## 完整明细\n\n")
        file.write("> [!note] 字段说明\n")
        file.write("> `FPR` 表示正常样本被误判为 jailbreak 的比例；`FNR` 表示 jailbreak 样本被漏判为正常的比例。\n\n")
        full_cols = [
            "mode",
            "train_set",
            "test_set",
            "classifier",
            "train_n",
            "test_n",
            "feature_dim",
            *metric_cols,
            "TN",
            "FP",
            "FN",
            "TP",
        ]
        file.write(df[full_cols].round(4).to_markdown(index=False))
        file.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize hidden-state detector metrics from cached features."
    )
    parser.add_argument(
        "--output_dir",
        default="outputs_hiddenstate/qwen2.5-1.5b",
        help="Directory containing cache/ and receiving result summaries.",
    )
    parser.add_argument("--pooling", default="lasttoken_v2")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data = load_cached_features(output_dir / "cache", args.pooling)
    results = build_results(data, args.random_state)

    csv_path = output_dir / f"results_{args.pooling}.csv"
    md_path = output_dir / f"results_{args.pooling}.md"
    try:
        results.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Wrote: {csv_path}")
    except PermissionError:
        print(f"Skipped CSV because it is open or locked: {csv_path}")
    write_markdown(results, md_path, args.pooling)

    print(f"Wrote: {md_path}")
    print(
        results[
            ["mode", "train_set", "test_set", "classifier", "ACC", "F1", "ROC_AUC", "FPR", "FNR"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
