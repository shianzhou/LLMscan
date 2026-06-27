import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_NAME = ""
MODEL_PATH = ""
MODEL_SLUG = ""
DATASET_NAME = "AutoDAN"
CROSS_TEST_DATASET = "GCG"
POOLING_VERSION = "lasttoken_v2"
HIDDEN_SIZE = None
SAMPLE_COUNTS = {
    "AutoDAN": 614,
    "GCG": 1070,
}
LAYERS = [1, 2, 3, 4, 5, 10]


def safe_artifact_name(name):
    return str(name).replace("\\", "_").replace("/", "_")


def make_output_slug(model_name):
    slug = str(model_name).strip().strip("/\\").replace("\\", "/").replace("/", "_")
    return slug or "custom_model"


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["features"], data["labels"].astype(int), data["prompts"].tolist()


def save_npz(path, features, labels, prompts, n_last_layers, dataset_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        labels=np.asarray(labels, dtype=int),
        prompts=np.asarray(prompts),
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        n_last_layers=n_last_layers,
        pooling=POOLING_VERSION,
        sample_count=len(labels),
        feature_dim=features.shape[1],
        source="layer_ablation",
    )


def cache_name(dataset_name, n_last_layers):
    safe_model = safe_artifact_name(MODEL_NAME)
    return f"{dataset_name}_{safe_model}_last{n_last_layers}_{POOLING_VERSION}_samples{SAMPLE_COUNTS[dataset_name]}.npz"


def make_classifier(classifier_name, random_state):
    if classifier_name == "LR":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif classifier_name == "MLP":
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")
    return make_pipeline(StandardScaler(), model)


def evaluate_predictions(labels, predictions, scores):
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return {
        "ACC": accuracy_score(labels, predictions),
        "F1": f1_score(labels, predictions),
        "ROC_AUC": roc_auc_score(labels, scores),
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        "FNR": fn / (fn + tp) if (fn + tp) else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def evaluate_supervised(classifier, test_x, test_y):
    predictions = classifier.predict(test_x)
    scores = classifier.predict_proba(test_x)[:, 1]
    return evaluate_predictions(test_y, predictions, scores)


def train_iforest(train_x, train_y, test_x, test_y, random_state, contamination):
    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)
    normal_x = train_x[train_y == 0]
    if len(normal_x) == 0:
        raise ValueError("IForest requires at least one non_adv training sample.")

    classifier = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    classifier.fit(normal_x)

    scores = -classifier.score_samples(test_x)
    normal_scores = -classifier.score_samples(normal_x)
    threshold = np.percentile(normal_scores, 100 * (1 - contamination))
    predictions = (scores > threshold).astype(int)
    metrics = evaluate_predictions(test_y, predictions, scores)
    return {"model": classifier, "threshold": threshold, "contamination": contamination}, metrics


def build_layer_features(dataset_name, layer, main_cache_dir, ablation_cache_dir, force_extract, device):
    global HIDDEN_SIZE
    out_path = ablation_cache_dir / cache_name(dataset_name, layer)
    if out_path.exists() and not force_extract:
        return (*load_npz(out_path), out_path, "ablation_cache")

    if layer <= 5:
        last5_path = main_cache_dir / cache_name(dataset_name, 5)
        features, labels, prompts = load_npz(last5_path)
        if HIDDEN_SIZE is None:
            if features.shape[1] % 5 != 0:
                raise ValueError(f"Cannot infer hidden size from last5 feature dim: {features.shape[1]}")
            HIDDEN_SIZE = features.shape[1] // 5
        else:
            expected_dim = 5 * HIDDEN_SIZE
            if features.shape[1] != expected_dim:
                raise ValueError(f"Unexpected last5 feature dim: {features.shape[1]} != {expected_dim}")
        layer_features = features[:, -layer * HIDDEN_SIZE :]
        save_npz(out_path, layer_features, labels, prompts, layer, dataset_name)
        return layer_features, labels, prompts, out_path, "sliced_from_last5_cache"

    if layer == 10:
        if not force_extract and out_path.exists():
            return (*load_npz(out_path), out_path, "ablation_cache")
        if device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available in this environment.")
            except ImportError as exc:
                raise RuntimeError("PyTorch is required for extracting last10 features.") from exc

        try:
            from lllm.questions_loaders import AutoDAN, GCG
            from public_func.hidden_state_detector import extract_hidden_states_batched, sample_balanced_dataset
            from utils.modelUtils import ModelAndTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "last10 cache is missing, and feature extraction requires the full project dependencies. "
                f"Missing dependency: {exc.name}"
            ) from exc

        dataset_classes = {"AutoDAN": AutoDAN, "GCG": GCG}
        dataset = dataset_classes[dataset_name]()
        prompts, labels = sample_balanced_dataset(dataset, max_samples=None, random_state=42)
        mt = ModelAndTokenizer(MODEL_PATH + MODEL_NAME, low_cpu_mem_usage=True, device=device)
        if HIDDEN_SIZE is None:
            HIDDEN_SIZE = int(mt.model.config.hidden_size)
        features = extract_hidden_states_batched(
            prompts,
            mt,
            n_last_layers=layer,
            batch_size=16,
            device=device,
        )
        save_npz(out_path, features, labels, prompts, layer, dataset_name)
        return features, labels, prompts, out_path, "extracted"

    raise ValueError(f"Unsupported layer count: {layer}")


def evaluate_same_dataset_layer(layer, features, labels, models_dir, random_state, contamination):
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=random_state,
        stratify=labels,
    )

    rows = []
    for classifier_name in ["LR", "MLP"]:
        classifier = make_classifier(classifier_name, random_state)
        classifier.fit(train_x, train_y)
        metrics = evaluate_supervised(classifier, test_x, test_y)
        dump(classifier, models_dir / "same_dataset" / f"layer{layer}_{classifier_name}.joblib")
        rows.append(
            {
                "mode": "same_dataset_7_3",
                "n_last_layers": layer,
                "train_set": DATASET_NAME,
                "test_set": DATASET_NAME,
                "classifier": classifier_name,
                "train_n": len(train_y),
                "test_n": len(test_y),
                "feature_dim": features.shape[1],
                **metrics,
            }
        )

    iforest_model, iforest_metrics = train_iforest(
        train_x,
        train_y,
        test_x,
        test_y,
        random_state=random_state,
        contamination=contamination,
    )
    dump(iforest_model, models_dir / "same_dataset" / f"layer{layer}_IForest.joblib")
    rows.append(
        {
            "mode": "same_dataset_7_3",
            "n_last_layers": layer,
            "train_set": DATASET_NAME,
            "test_set": DATASET_NAME,
            "classifier": "IForest",
            "train_n": len(train_y),
            "test_n": len(test_y),
            "feature_dim": features.shape[1],
            **iforest_metrics,
        }
    )
    return rows


def evaluate_cross_dataset_layer(
    layer,
    train_features,
    train_labels,
    test_features,
    test_labels,
    models_dir,
    random_state,
    contamination,
):
    rows = []
    for classifier_name in ["LR", "MLP"]:
        classifier = make_classifier(classifier_name, random_state)
        classifier.fit(train_features, train_labels)
        metrics = evaluate_supervised(classifier, test_features, test_labels)
        dump(
            classifier,
            models_dir / "cross_dataset" / f"layer{layer}_{DATASET_NAME}_to_{CROSS_TEST_DATASET}_{classifier_name}.joblib",
        )
        rows.append(
            {
                "mode": "cross_dataset_full",
                "n_last_layers": layer,
                "train_set": DATASET_NAME,
                "test_set": CROSS_TEST_DATASET,
                "classifier": classifier_name,
                "train_n": len(train_labels),
                "test_n": len(test_labels),
                "feature_dim": train_features.shape[1],
                **metrics,
            }
        )

    iforest_model, iforest_metrics = train_iforest(
        train_features,
        train_labels,
        test_features,
        test_labels,
        random_state=random_state,
        contamination=contamination,
    )
    dump(
        iforest_model,
        models_dir / "cross_dataset" / f"layer{layer}_{DATASET_NAME}_to_{CROSS_TEST_DATASET}_IForest.joblib",
    )
    rows.append(
        {
            "mode": "cross_dataset_full",
            "n_last_layers": layer,
            "train_set": DATASET_NAME,
            "test_set": CROSS_TEST_DATASET,
            "classifier": "IForest",
            "train_n": len(train_labels),
            "test_n": len(test_labels),
            "feature_dim": train_features.shape[1],
            **iforest_metrics,
        }
    )
    return rows


def write_markdown(results, output_path):
    metric_cols = ["ACC", "F1", "ROC_AUC", "FPR", "FNR"]
    table = results.copy()
    table[metric_cols] = table[metric_cols].round(4)
    same = table[table["mode"] == "same_dataset_7_3"].copy()
    cross = table[table["mode"] == "cross_dataset_full"].copy()
    iforest = table[table["classifier"] == "IForest"].copy()
    same_iforest = same[same["classifier"] == "IForest"].copy()
    cross_iforest = cross[cross["classifier"] == "IForest"].copy()
    best_same_iforest = same_iforest.sort_values("ACC", ascending=False).iloc[0]
    worst_same_iforest = same_iforest.sort_values("ACC", ascending=True).iloc[0]

    with output_path.open("w", encoding="utf-8") as file:
        file.write("---\n")
        file.write("tags:\n")
        file.write("  - LLMScan\n")
        file.write("  - HiddenStateDetector\n")
        file.write("  - Ablation\n")
        file.write(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"model: {MODEL_NAME}\n")
        file.write(f"dataset: {DATASET_NAME}\n")
        file.write(f"cross_test_dataset: {CROSS_TEST_DATASET}\n")
        file.write(f"pooling: {POOLING_VERSION}\n")
        file.write("---\n\n")
        file.write("# AutoDAN Hidden-State 层数消融实验\n\n")
        file.write("> [!info] 实验设置\n")
        file.write(f"> 固定 {MODEL_NAME}、AutoDAN、last-token pooling，只改变最后层数。\n")
        file.write("> 同时记录 AutoDAN 同数据集 7:3 与 AutoDAN -> GCG 跨数据集 full-test 结果。\n\n")
        file.write("## 结论速览\n\n")
        file.write(f"- AutoDAN 同数据集 IForest 最佳层数：`last{int(best_same_iforest['n_last_layers'])}`，ACC = `{best_same_iforest['ACC']:.4f}`。\n")
        file.write(f"- AutoDAN 同数据集 IForest 最差层数：`last{int(worst_same_iforest['n_last_layers'])}`，ACC = `{worst_same_iforest['ACC']:.4f}`。\n")
        file.write("- LR / MLP / IForest 均使用相同的 AutoDAN split，保证只消融层数。\n\n")
        if not cross_iforest.empty:
            best_cross_iforest = cross_iforest.sort_values("ACC", ascending=False).iloc[0]
            worst_cross_iforest = cross_iforest.sort_values("ACC", ascending=True).iloc[0]
            file.write(f"- AutoDAN -> GCG IForest 最佳层数：`last{int(best_cross_iforest['n_last_layers'])}`，ACC = `{best_cross_iforest['ACC']:.4f}`。\n")
            file.write(f"- AutoDAN -> GCG IForest 最差层数：`last{int(worst_cross_iforest['n_last_layers'])}`，ACC = `{worst_cross_iforest['ACC']:.4f}`。\n")
        file.write("\n")

        display_cols = ["mode", "n_last_layers", "train_set", "test_set", "classifier", "train_n", "test_n", "feature_dim", *metric_cols]
        file.write("## 总表\n\n")
        file.write(table[display_cols].to_markdown(index=False))
        file.write("\n\n## AutoDAN 同数据集结果\n\n")
        file.write(same[display_cols].to_markdown(index=False))
        file.write("\n\n## AutoDAN -> GCG 跨数据集结果\n\n")
        file.write(cross[display_cols].to_markdown(index=False))
        file.write("\n\n## IForest 专表\n\n")
        file.write(iforest[display_cols].to_markdown(index=False))
        file.write("\n\n## 完整明细\n\n")
        full_cols = [*display_cols, "TN", "FP", "FN", "TP"]
        file.write(table[full_cols].to_markdown(index=False))
        file.write("\n")


def configure_model(model_path=None, model_name=None, hidden_size=None):
    global MODEL_NAME, MODEL_PATH, MODEL_SLUG, HIDDEN_SIZE
    if not model_name:
        raise ValueError("未指定模型。请使用 --model_name 指定模型 ID 或本地模型目录。")
    config = {
        "model_path": model_path or "",
        "model_name": model_name,
        "output_slug": make_output_slug(model_name),
        "hidden_size": hidden_size,
    }
    MODEL_NAME = config["model_name"]
    MODEL_PATH = config["model_path"]
    MODEL_SLUG = config["output_slug"]
    HIDDEN_SIZE = int(hidden_size or config["hidden_size"]) if (hidden_size or config["hidden_size"]) else None
    return config


def run_ablation(args):
    output_dir = Path(args.output_dir or f"outputs_hiddenstate/{MODEL_SLUG}/ablations/AutoDAN_layers_lasttoken_v2")
    main_cache_dir = Path(args.main_cache_dir or f"outputs_hiddenstate/{MODEL_SLUG}/cache")
    cache_dir = output_dir / "cache"
    models_dir = output_dir / "models"
    (models_dir / "same_dataset").mkdir(parents=True, exist_ok=True)
    (models_dir / "cross_dataset").mkdir(parents=True, exist_ok=True)

    rows = []
    for layer in args.layers:
        features, labels, prompts, cache_path, source = build_layer_features(
            dataset_name=DATASET_NAME,
            layer=layer,
            main_cache_dir=main_cache_dir,
            ablation_cache_dir=cache_dir,
            force_extract=args.force_extract,
            device=args.device,
        )
        expected_dim = layer * HIDDEN_SIZE
        if features.shape != (SAMPLE_COUNTS[DATASET_NAME], expected_dim):
            raise ValueError(
                f"Unexpected feature shape for {DATASET_NAME} last{layer}: "
                f"{features.shape}, expected {(SAMPLE_COUNTS[DATASET_NAME], expected_dim)}"
            )
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{DATASET_NAME} last{layer}: features={features.shape}, labels={dict(zip(unique, counts))}, source={source}, cache={cache_path}")
        rows.extend(
            evaluate_same_dataset_layer(
                layer=layer,
                features=features,
                labels=labels,
                models_dir=models_dir,
                random_state=args.random_state,
                contamination=args.contamination,
            )
        )
        cross_features, cross_labels, cross_prompts, cross_cache_path, cross_source = build_layer_features(
            dataset_name=CROSS_TEST_DATASET,
            layer=layer,
            main_cache_dir=main_cache_dir,
            ablation_cache_dir=cache_dir,
            force_extract=args.force_extract,
            device=args.device,
        )
        if cross_features.shape != (SAMPLE_COUNTS[CROSS_TEST_DATASET], expected_dim):
            raise ValueError(
                f"Unexpected feature shape for {CROSS_TEST_DATASET} last{layer}: "
                f"{cross_features.shape}, expected {(SAMPLE_COUNTS[CROSS_TEST_DATASET], expected_dim)}"
            )
        unique, counts = np.unique(cross_labels, return_counts=True)
        print(
            f"{CROSS_TEST_DATASET} last{layer}: features={cross_features.shape}, "
            f"labels={dict(zip(unique, counts))}, source={cross_source}, cache={cross_cache_path}"
        )
        rows.extend(
            evaluate_cross_dataset_layer(
                layer=layer,
                train_features=features,
                train_labels=labels,
                test_features=cross_features,
                test_labels=cross_labels,
                models_dir=models_dir,
                random_state=args.random_state,
                contamination=args.contamination,
            )
        )

    results = pd.DataFrame(rows)
    csv_path = output_dir / "results_layer_ablation.csv"
    md_path = output_dir / "results_layer_ablation.md"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_markdown(results, md_path)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(results[["mode", "n_last_layers", "train_set", "test_set", "classifier", "ACC", "F1", "ROC_AUC", "FPR", "FNR"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="AutoDAN hidden-state layer ablation.")
    parser.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    parser.add_argument("--model_path", help="手动覆盖模型路径前缀")
    parser.add_argument("--model_name", help="手动覆盖模型名称")
    parser.add_argument("--hidden_size", type=int, help="手动覆盖 hidden size")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--main_cache_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force_extract", action="store_true")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--contamination", type=float, default=0.1)
    args = parser.parse_args()

    configure_model(
        model_path=args.model_path,
        model_name=args.model_name,
        hidden_size=args.hidden_size,
    )
    run_ablation(args)


if __name__ == "__main__":
    main()
