'''
Hidden-State Detector for Jailbreak/Toxic Detection

核心思路：不做因果干预（AIE），直接用 LLM 正常前向传播的 hidden states 训练检测器。

流程：
  input prompt → tokenizer → LLM 前向传播一次（output_hidden_states=True）
  → 提取最后 N 层 hidden states → last-token pooling → 拼接为固定长度特征向量
  → 训练分类器（LR/MLP）→ 评估（ACC/F1/ROC-AUC）+ PCA/t-SNE 可视化

对比 LLMScan AIE 方法：免去逐层短路干预，每条样本只需 1 次前向传播（vs 27 次）。
'''

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modelUtils import *
from utils.utils import *
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 报错
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
from joblib import dump

from lllm.classification_utils import Classifier
from lllm.questions_loaders import AutoDAN, GCG, PAP

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

POOLING_VERSION = "lasttoken_v2"


def _safe_artifact_name(name):
    """Make model/dataset names safe for filenames while keeping them readable."""
    return str(name).replace("\\", "_").replace("/", "_")


def _make_output_slug(model_name):
    slug = str(model_name).strip().strip("/\\").replace("\\", "/").replace("/", "_")
    return slug or "custom_model"


# ============================================================
# 1. 数据集采样
# ============================================================

def sample_balanced_dataset(dataset, max_samples=None, random_state=42):
    """
    从 jailbreak 数据集中平衡采样 adv / non_adv。

    参数:
        dataset: DataFrame，有 'questions' 和 'label' 列
        max_samples: int or None, 总采样数（各取 half）
        random_state: 随机种子

    返回:
        prompts: list of str (已包装好的 prompt)
        labels:  list of int (1=adv, 0=non_adv)
    """
    if max_samples is not None:
        half = max_samples // 2
        adv_idx = dataset[dataset['label'] == 'adv_data'].index[:half]
        non_adv_idx = dataset[dataset['label'] == 'non_adv_data'].index[:half]
        selected = list(adv_idx) + list(non_adv_idx)
        sampled = dataset.loc[selected].sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"--> 平衡采样 {len(sampled)} 条 ({half} adv + {len(sampled) - half} non_adv)")
    else:
        sampled = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"--> 使用全部 {len(sampled)} 条数据")

    prompts = []
    labels = []
    for _, row in sampled.iterrows():
        question = row['questions']
        prompt = prepare_prompt(question)  # utils.utils.prepare_prompt
        prompts.append(prompt)
        labels.append(1 if row['label'] == 'adv_data' else 0)

    print(f"     adv: {sum(labels)} 条, non_adv: {len(labels) - sum(labels)} 条")
    return prompts, labels


# ============================================================
# 2. Hidden State 提取
# ============================================================

def _make_cache_path(saving_dir, dataset_name, model_name, n_last_layers,
                      pooling, sample_count):
    """构造缓存文件路径。"""
    cache_dir = os.path.join(saving_dir, "cache")
    safe_model = _safe_artifact_name(model_name)
    fname = f"{dataset_name}_{safe_model}_last{n_last_layers}_{pooling}_samples{sample_count}.npz"
    return os.path.join(cache_dir, fname)


def save_feature_cache(saving_dir, dataset_name, model_name,
                        features, labels, prompts,
                        n_last_layers, pooling, sample_count):
    """保存 features/labels/prompts + meta 到 .npz 文件。"""
    cache_dir = os.path.join(saving_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _make_cache_path(saving_dir, dataset_name, model_name,
                                   n_last_layers, pooling, sample_count)
    np.savez_compressed(
        cache_path,
        features=features,
        labels=np.array(labels),
        prompts=np.array(prompts),
        dataset_name=dataset_name,
        model_name=model_name,
        n_last_layers=n_last_layers,
        pooling=pooling,
        sample_count=sample_count,
        feature_dim=features.shape[1],
    )
    print(f"--> Hidden-state features 已保存: {os.path.basename(cache_path)}")
    return cache_path


def load_feature_cache(saving_dir, dataset_name, model_name,
                        n_last_layers, pooling, sample_count):
    """读取缓存，命中返回 (features, labels, prompts)，未命中返回 None。"""
    cache_path = _make_cache_path(saving_dir, dataset_name, model_name,
                                   n_last_layers, pooling, sample_count)
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        features = data['features']
        labels = data['labels'].tolist()
        prompts = data['prompts'].tolist()
        print(f"--> 读取 hidden-state cache: {os.path.basename(cache_path)}")
        print(f"    features: {features.shape}, labels: {len(labels)}")
        return features, labels, prompts
    return None


def _get_or_extract_features(prompts, labels, mt, dataset_name, model_name,
                              saving_dir, n_last_layers, pooling=POOLING_VERSION,
                              force_extract=False):
    """
    先查缓存，未命中再提取 hidden states，提取后自动保存。
    返回 (features, labels, prompts)。
    """
    sample_count = len(labels)

    if not force_extract:
        cached = load_feature_cache(saving_dir, dataset_name, model_name,
                                     n_last_layers, pooling, sample_count)
        if cached is not None:
            return cached

    if force_extract:
        print(f"--> force_extract=True，忽略已有缓存，重新提取 hidden states")
    else:
        print(f"--> 未发现缓存，开始提取 hidden states")

    start = time.time()
    features = extract_hidden_states_batched(prompts, mt, n_last_layers=n_last_layers, batch_size=16)
    print(f"    耗时: {time.time()-start:.1f}s ({ (time.time()-start)/len(prompts):.2f}s/sample)")

    save_feature_cache(saving_dir, dataset_name, model_name,
                        features, labels, prompts,
                        n_last_layers, pooling, sample_count)
    return features, labels, prompts


def extract_hidden_states_batched(prompts, mt, n_last_layers=5, batch_size=16, device='cuda:0'):
    """
    分批提取 hidden states 并构造特征，避免 OOM。

    参数:
        prompts: list of str
        mt: ModelAndTokenizer 实例
        n_last_layers: 取最后几层
        batch_size: 每批处理的 prompt 数量
        device: 'cuda:0' or 'cpu'

    返回:
        features: np.ndarray, shape [B, n_layers * hidden_dim]
    """
    all_features = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="    提取 hidden states"):
        batch_prompts = prompts[i:i + batch_size]
        inputs = make_inputs(mt.tokenizer, batch_prompts, device=device)
        # make_inputs 当前使用左 padding；真实 token 的末尾位置需从 mask 右侧查找。

        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = mt.model(**inputs, output_hidden_states=True)

        # 取最后 n_last_layers 层
        all_hidden = outputs.hidden_states
        last_n = list(all_hidden[-n_last_layers:])  # N 个 [B, S, hidden_dim]

        # 每层取最后一个非 padding token。该写法同时兼容左/右 padding。
        mask = inputs['attention_mask']
        B = mask.shape[0]
        last_positions = mask.shape[1] - 1 - torch.flip(mask, dims=[1]).argmax(dim=1)  # [B]

        layer_features = []
        for hs in last_n:
            last_token = hs[torch.arange(B, device=hs.device), last_positions, :]  # [B, hidden_dim]
            layer_features.append(last_token.cpu())

        batch_features = torch.cat(layer_features, dim=1)  # [B, n_layers * hidden_dim]
        all_features.append(batch_features.numpy())

        # 释放
        del outputs, all_hidden, last_n, inputs, batch_features

    features = np.concatenate(all_features, axis=0)
    torch.cuda.empty_cache()
    return features


# ============================================================
# 4. 训练与评估
# ============================================================

def _evaluate_iforest(X_train, y_train, X_test, y_test, contamination=0.1, random_state=42):
    """
    Isolation Forest 单分类：仅用 non_adv（label=0）训练，全量测试。
    返回 (acc, f1, auc, fpr, model)。
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # 只取 normal 样本训练
    X_normal = X_train[y_train == 0]
    if len(X_normal) == 0:
        print("    [IForest] 训练集中无 normal 样本，跳过")
        return None, None, None, None, None, 0, 0, 0, 0

    clf = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    clf.fit(X_normal)

    # score_samples 返回负值，越小越异常；取负后越大越异常
    scores = -clf.score_samples(X_test)
    # 用训练集的 contamination 百分位作为阈值
    train_scores = -clf.score_samples(X_normal)
    threshold = np.percentile(train_scores, 100 * (1 - contamination))
    y_pred = (scores > threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, scores)
    except ValueError:
        auc = float('nan')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"    ACC: {acc:.4f}  F1: {f1:.4f}  ROC-AUC: {auc:.4f}  FPR: {fpr:.4f}  FNR: {fnr:.4f}")
    return acc, f1, auc, fpr, fnr, clf, int(tn), int(fp), int(fn), int(tp)


def _write_csv_result(results, mode, train_name, test_name, train_n, test_n,
                       feature_dim, saving_dir):
    """将单次实验的全部分类器结果追加写入 CSV。"""
    csv_path = os.path.join(saving_dir, f"results_{POOLING_VERSION}.csv")
    os.makedirs(saving_dir, exist_ok=True)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        if write_header:
            f.write("mode,train_set,test_set,classifier,train_n,test_n,feature_dim,"
                    "ACC,F1,ROC_AUC,FPR,FNR,TN,FP,FN,TP\n")
        for name in ['logistic', 'mlp', 'iforest']:
            r = results.get(name)
            if r is None or r['acc'] is None:
                continue
            classifier_name = {"logistic": "LR", "mlp": "MLP", "iforest": "IForest"}[name]
            f.write(f"{mode},{train_name},{test_name},{classifier_name},{train_n},{test_n},"
                    f"{feature_dim},{r['acc']:.6f},{r['f1']:.6f},{r['auc']:.6f},"
                    f"{r['fpr']:.6f},{r.get('fnr',0):.6f},{r.get('tn',0)},{r.get('fp',0)},"
                    f"{r.get('fn',0)},{r.get('tp',0)}\n")


def train_evaluate(features, labels, test_size=0.3, random_state=42):
    """
    训练 LR + MLP 分类器并评估。

    返回:
        results: dict, 包含各分类器的 {acc, f1, auc, fpr}
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size,
        random_state=random_state, stratify=labels
    )
    print(f"--> 训练集: {X_train.shape[0]} 条 (adv={sum(1 for v in y_train if v == 1)}, non_adv={sum(1 for v in y_train if v == 0)}), 测试集: {X_test.shape[0]} 条 (adv={sum(1 for v in y_test if v == 1)}, non_adv={sum(1 for v in y_test if v == 0)})")
    print(f"    特征维度: {X_train.shape[1]}")

    results = {}

    # --- Logistic Regression ---
    print("\n========== Logistic Regression ==========")
    clf_lr = Classifier(X_train, y_train, classifier="logistic", scale=True,
                        max_iter=1000, random_state=random_state)
    acc, auc, conf_matrix, y_pred, y_proba = clf_lr.evaluate(X_test, y_test, return_ys=True)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    print(f"    ACC: {acc:.4f}  F1: {f1:.4f}  ROC-AUC: {auc:.4f}  FPR: {fpr:.4f}")
    results['logistic'] = {'acc': acc, 'f1': f1, 'auc': auc, 'fpr': fpr, 'fnr': fnr,
                            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                            'model': clf_lr}

    # --- MLP ---
    print("\n========== MLP Classifier ==========")
    clf_mlp = Classifier(X_train, y_train, classifier="MLP", scale=True,
                         hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)
    acc2, auc2, cm2, y_pred2, y_proba2 = clf_mlp.evaluate(X_test, y_test, return_ys=True)
    f1_2 = f1_score(y_test, y_pred2)
    tn2, fp2, fn2, tp2 = cm2.ravel()
    fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0.0
    fnr2 = fn2 / (fn2 + tp2) if (fn2 + tp2) > 0 else 0.0
    print(f"    ACC: {acc2:.4f}  F1: {f1_2:.4f}  ROC-AUC: {auc2:.4f}  FPR: {fpr2:.4f}")
    results['mlp'] = {'acc': acc2, 'f1': f1_2, 'auc': auc2, 'fpr': fpr2, 'fnr': fnr2,
                      'tn': int(tn2), 'fp': int(fp2), 'fn': int(fn2), 'tp': int(tp2),
                      'model': clf_mlp}

    # --- Isolation Forest（单分类，仅用 non_adv 训练） ---
    print("\n========== Isolation Forest ==========")
    acc3, f1_3, auc3, fpr3, fnr3, clf_if, tn3, fp3, fn3, tp3 = _evaluate_iforest(
        X_train, y_train, X_test, y_test, random_state=random_state)
    results['iforest'] = {'acc': acc3, 'f1': f1_3, 'auc': auc3, 'fpr': fpr3, 'fnr': fnr3,
                          'tn': tn3, 'fp': fp3, 'fn': fn3, 'tp': tp3,
                          'model': clf_if}

    return results


# ============================================================
# 5. PCA / t-SNE 可视化
# ============================================================

def visualize_pca_tsne(features, labels, dataset_name, model_name, saving_dir,
                        prompts=None, remove_top_outlier=False):
    """
    PCA + t-SNE 二维降维可视化，保存为 PDF。

    参数:
        features: np.ndarray [N, D]
        labels: list of int (0=non_adv, 1=adv)
        dataset_name, model_name: 用于标题和文件名
        saving_dir: 输出目录
        prompts: list of str, 仅 remove_top_outlier=True 时需要（打印异常 prompt）
        remove_top_outlier: 是否移除 top-1 L2 norm 离群点
    """
    labels_arr = np.array(labels)
    outliers_removed = []

    if remove_top_outlier:
        # 找 top-1 feature norm 离群点
        norms = np.linalg.norm(features, axis=1)
        outlier_idx = int(np.argmax(norms))
        keep_mask = np.ones(len(features), dtype=bool)
        keep_mask[outlier_idx] = False

        features_vis = features[keep_mask]
        labels_vis = labels_arr[keep_mask]

        # 打印离群点信息
        label_str = 'adv' if labels[outlier_idx] == 1 else 'non_adv'
        print(f"\n  [离群点移除]")
        print(f"    Index: {outlier_idx}  Label: {label_str}  L2 Norm: {norms[outlier_idx]:.4f}")
        if prompts is not None:
            prompt_snip = str(prompts[outlier_idx])[:200].replace('\n', '\\n')
            print(f"    Prompt: {prompt_snip}")

        # RobustScaler（对离群点鲁棒）
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_vis)
        outliers_removed.append(outlier_idx)
    else:
        features_vis = features
        labels_vis = labels_arr
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_vis)

    n_samples = len(labels_vis)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- PCA ---
    pca = PCA(n_components=2, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    for lbl, name, marker in [(0, 'non_adv', 'o'), (1, 'adv', '^')]:
        mask = labels_vis == lbl
        axes[0].scatter(features_pca[mask, 0], features_pca[mask, 1],
                        label=name, marker=marker, alpha=0.6, s=40)
    pca_title = f'PCA ({dataset_name} | {model_name})'
    if remove_top_outlier:
        pca_title += '\nwithout top-1 norm outlier'
    pca_title += f'\nVar ratio: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}'
    axes[0].set_title(pca_title)
    axes[0].legend()
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    # --- t-SNE ---
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_tsne = tsne.fit_transform(features_scaled)
    for lbl, name, marker in [(0, 'non_adv', 'o'), (1, 'adv', '^')]:
        mask = labels_vis == lbl
        axes[1].scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                        label=name, marker=marker, alpha=0.6, s=40)
    tsne_title = f't-SNE ({dataset_name} | {model_name})\nperplexity={perplexity}'
    if remove_top_outlier:
        tsne_title += '\nwithout top-1 norm outlier'
    axes[1].set_title(tsne_title)
    axes[1].legend()
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    plt.tight_layout()

    # 保存
    fig_dir = os.path.join(saving_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    safe_model = _safe_artifact_name(model_name)
    if remove_top_outlier:
        fig_name = f"hidden_state_{dataset_name}_{safe_model}_{POOLING_VERSION}_remove_top1_outlier.pdf"
    else:
        fig_name = f"hidden_state_{dataset_name}_{safe_model}_{POOLING_VERSION}.pdf"
    fig_path = os.path.join(fig_dir, fig_name)
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"--> 可视化已保存: {fig_path}")
    plt.close(fig)

    return outliers_removed


# ============================================================
# 6. 跨数据集可视化
# ============================================================

def _visualize_cross_dataset(train_features, train_labels, train_name,
                              test_features, test_labels, test_name,
                              model_name, saving_dir):
    """
    跨数据集 PCA + t-SNE：train/test 在同一图上，不同 marker/颜色。
    """
    all_features = np.concatenate([train_features, test_features], axis=0)
    all_labels = np.concatenate([np.array(train_labels), np.array(test_labels)])
    source = np.array(['train'] * len(train_features) + ['test'] * len(test_features))

    scaler = RobustScaler()
    all_scaled = scaler.fit_transform(all_features)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- PCA ---
    pca = PCA(n_components=2, random_state=42)
    all_pca = pca.fit_transform(all_scaled)
    for src, marker, alpha in [('train', 'o', 0.4), ('test', '^', 0.6)]:
        for lbl, color in [(0, '#1f77b4'), (1, '#d62728')]:
            mask = (source == src) & (all_labels == lbl)
            lbl_text = f'{src}_non_adv' if lbl == 0 else f'{src}_adv'
            axes[0].scatter(all_pca[mask, 0], all_pca[mask, 1],
                            label=lbl_text, marker=marker, c=color, alpha=alpha, s=30)
    axes[0].set_title(f'PCA: {train_name} → {test_name}\n({model_name})\n'
                      f'Var: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}')
    axes[0].legend(fontsize=7, loc='lower right')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    # --- t-SNE ---
    n_all = len(all_features)
    perplexity = min(30, n_all - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    all_tsne = tsne.fit_transform(all_scaled)
    for src, marker, alpha in [('train', 'o', 0.4), ('test', '^', 0.6)]:
        for lbl, color in [(0, '#1f77b4'), (1, '#d62728')]:
            mask = (source == src) & (all_labels == lbl)
            lbl_text = f'{src}_non_adv' if lbl == 0 else f'{src}_adv'
            axes[1].scatter(all_tsne[mask, 0], all_tsne[mask, 1],
                            label=lbl_text, marker=marker, c=color, alpha=alpha, s=30)
    axes[1].set_title(f't-SNE: {train_name} → {test_name}\n({model_name}) perplexity={perplexity}')
    axes[1].legend(fontsize=7, loc='lower right')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    plt.tight_layout()
    fig_dir = os.path.join(saving_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    safe_model = _safe_artifact_name(model_name)
    fig_path = os.path.join(fig_dir, f"cross_{train_name}_to_{test_name}_{safe_model}_{POOLING_VERSION}.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"--> 跨数据集可视化已保存: {fig_path}")
    plt.close(fig)


# ============================================================
# 7. 跨数据集训练/评估
# ============================================================

def _train_on_full_test_on_heldout(train_features, train_labels,
                                    test_features, test_labels,
                                    random_state=42):
    """
    全量训练集训练，全量测试集评估（跨数据集验证用）。
    """
    results = {}
    X_train, y_train = np.array(train_features), np.array(train_labels)
    X_test, y_test = np.array(test_features), np.array(test_labels)
    print(f"    训练集: {X_train.shape[0]} 条 (adv={sum(1 for v in y_train if v == 1)}, non_adv={sum(1 for v in y_train if v == 0)}), 测试集: {X_test.shape[0]} 条 (adv={sum(1 for v in y_test if v == 1)}, non_adv={sum(1 for v in y_test if v == 0)})")

    # --- Logistic Regression ---
    print("\n========== Logistic Regression (cross) ==========")
    clf_lr = Classifier(X_train, y_train, classifier="logistic", scale=True,
                        max_iter=1000, random_state=random_state)
    acc, auc, cm, y_pred, y_proba = clf_lr.evaluate(X_test, y_test, return_ys=True)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    print(f"    ACC: {acc:.4f}  F1: {f1:.4f}  ROC-AUC: {auc:.4f}  FPR: {fpr:.4f}")
    results['logistic'] = {'acc': acc, 'f1': f1, 'auc': auc, 'fpr': fpr, 'fnr': fnr,
                            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                            'model': clf_lr}

    # --- MLP ---
    print("\n========== MLP Classifier (cross) ==========")
    clf_mlp = Classifier(X_train, y_train, classifier="MLP", scale=True,
                         hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)
    acc2, auc2, cm2, y_pred2, y_proba2 = clf_mlp.evaluate(X_test, y_test, return_ys=True)
    f1_2 = f1_score(y_test, y_pred2)
    tn2, fp2, fn2, tp2 = cm2.ravel()
    fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0.0
    fnr2 = fn2 / (fn2 + tp2) if (fn2 + tp2) > 0 else 0.0
    print(f"    ACC: {acc2:.4f}  F1: {f1_2:.4f}  ROC-AUC: {auc2:.4f}  FPR: {fpr2:.4f}")
    results['mlp'] = {'acc': acc2, 'f1': f1_2, 'auc': auc2, 'fpr': fpr2, 'fnr': fnr2,
                      'tn': int(tn2), 'fp': int(fp2), 'fn': int(fn2), 'tp': int(tp2),
                      'model': clf_mlp}

    # --- Isolation Forest（单分类，仅用 train 中 non_adv 训练） ---
    print("\n========== Isolation Forest (cross) ==========")
    acc3, f1_3, auc3, fpr3, fnr3, clf_if, tn3, fp3, fn3, tp3 = _evaluate_iforest(
        X_train, y_train, X_test, y_test, random_state=random_state)
    results['iforest'] = {'acc': acc3, 'f1': f1_3, 'auc': auc3, 'fpr': fpr3, 'fnr': fnr3,
                          'tn': tn3, 'fp': fp3, 'fn': fn3, 'tp': tp3,
                          'model': clf_if}

    return results


# ============================================================
# 8. 主流程
# ============================================================

def run_hidden_state_detection(dataset, mt, model_name, saving_dir,
                                n_last_layers=5, max_samples=None,
                                test_size=0.3, random_state=42,
                                if_visualize=True, test_dataset=None,
                                force_extract=False):
    """
    完整 pipeline: 采样 → 提取 hidden states → 构造特征 → 训练评估 → 可视化。

    参数:
        dataset: 训练数据集
        test_dataset: 可选，跨数据集测试集。若为 None，则在 dataset 内 7:3 切分。
    """
    dataset_name = dataset.__class__.__name__

    # --- 跨数据集模式 ---
    if test_dataset is not None:
        test_name = test_dataset.__class__.__name__
        print(f"\n{'='*60}")
        print(f"  Hidden-State Detector [跨数据集]")
        print(f"  训练: {dataset_name} | 测试: {test_name} | {model_name}")
        print(f"  最后 {n_last_layers} 层 | last-token pooling")
        print(f"{'='*60}")

        # 训练集：全部样本
        print("\n[1/4] 加载训练集...")
        train_prompts, train_labels = sample_balanced_dataset(
            dataset, max_samples=max_samples, random_state=random_state)
        print(f"\n[2/4] 提取训练集 hidden states...")
        train_features, train_labels, train_prompts = _get_or_extract_features(
            train_prompts, train_labels, mt, dataset_name, model_name,
            saving_dir, n_last_layers, force_extract=force_extract)
        print(f"    特征: {train_features.shape}")

        # 测试集：全部样本
        print(f"\n[3/4] 加载测试集...")
        test_prompts, test_labels = sample_balanced_dataset(
            test_dataset, max_samples=None, random_state=random_state)
        print(f"    提取测试集 hidden states...")
        test_features, test_labels, test_prompts = _get_or_extract_features(
            test_prompts, test_labels, mt, test_name, model_name,
            saving_dir, n_last_layers, force_extract=force_extract)
        print(f"    特征: {test_features.shape}")

        # 训练 + 评估（全量训练，全量测试）
        print(f"\n[4/4] 训练（{dataset_name}）→ 测试（{test_name}）...")
        results = _train_on_full_test_on_heldout(
            train_features, train_labels,
            test_features, test_labels,
            random_state=random_state)

        # CSV 记录
        _write_csv_result(results, "cross_dataset_full", dataset_name, test_name,
                          train_features.shape[0], test_features.shape[0],
                          train_features.shape[1], saving_dir)

        # 可视化：train + test 在同一 PCA/t-SNE 上
        if if_visualize:
            _visualize_cross_dataset(
                train_features, train_labels, dataset_name,
                test_features, test_labels, test_name,
                model_name, saving_dir)

        # 输出汇总
        print(f"\n{'='*60}")
        print(f"  结果汇总: {dataset_name} → {test_name}")
        print(f"{'='*60}")
        print(f"  {'分类器':<15} {'ACC':>8} {'F1':>8} {'ROC-AUC':>8} {'FPR':>8}")
        print(f"  {'-'*47}")
        for name in ['logistic', 'mlp', 'iforest']:
            r = results[name]
            acc_s = f"{r['acc']:>8.4f}" if r['acc'] is not None else f"  {'N/A':>6}"
            f1_s = f"{r['f1']:>8.4f}" if r['f1'] is not None else f"  {'N/A':>6}"
            auc_s = f"{r['auc']:>8.4f}" if r['auc'] is not None else f"  {'N/A':>6}"
            fpr_s = f"{r['fpr']:>8.4f}" if r['fpr'] is not None else f"  {'N/A':>6}"
            print(f"  {name:<15} {acc_s} {f1_s} {auc_s} {fpr_s}")

        return results

    # --- 同数据集模式（原有逻辑） ---
    print(f"\n{'='*60}")
    print(f"  Hidden-State Detector: {dataset_name} @ {model_name}")
    print(f"  最后 {n_last_layers} 层 | last-token pooling | max_samples={max_samples}")
    print(f"{'='*60}")

    # Step 1: 采样
    print("\n[1/3] 数据采样...")
    prompts, labels = sample_balanced_dataset(dataset, max_samples=max_samples, random_state=random_state)

    # Step 2: 提取 hidden states（优先缓存）
    print(f"\n[2/3] 提取 hidden states（最后 {n_last_layers} 层, batch=16）...")
    features, labels, prompts = _get_or_extract_features(
        prompts, labels, mt, dataset_name, model_name,
        saving_dir, n_last_layers, force_extract=force_extract)
    print(f"    特征: {features.shape}")

    # Step 3: 训练评估
    print("\n[3/3] 训练分类器...")
    results = train_evaluate(features, labels, test_size=test_size, random_state=random_state)

    # CSV 记录：使用 sklearn 实际 split 后的样本数，避免四舍五入口径偏差。
    _, test_features_for_count, _, _ = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    test_n_actual = test_features_for_count.shape[0]
    train_n_actual = features.shape[0] - test_n_actual
    _write_csv_result(results, "same_dataset_7_3", dataset_name, dataset_name,
                      train_n_actual, test_n_actual, features.shape[1], saving_dir)

    # 保存模型
    os.makedirs(saving_dir, exist_ok=True)
    for name, r in results.items():
        if r['model'] is not None:
            dump_path = os.path.join(saving_dir, f"{name}_hidden_state_{dataset_name}_{POOLING_VERSION}.joblib")
            dump(r['model'], dump_path)

    # 异常点检测：查找 feature norm 最大的样本
    print(f"\n{'='*60}")
    print(f"  Feature Norm 异常检测（top-5）")
    print(f"{'='*60}")
    norms = np.linalg.norm(features, axis=1)
    top5_idx = np.argsort(-norms)[:5]  # 降序前5
    print(f"  {'Rank':<6} {'Idx':<6} {'Label':<8} {'L2 Norm':<12} {'Prompt（前200字符）'}")
    print(f"  {'-'*70}")
    for rank, idx in enumerate(top5_idx, 1):
        label_str = 'adv' if labels[idx] == 1 else 'non_adv'
        prompt_snip = prompts[idx][:200].replace('\n', '\\n')
        print(f"  {rank:<6} {idx:<6} {label_str:<8} {norms[idx]:<12.4f} {prompt_snip}")
    print(f"  Avg norm: {norms.mean():.4f}  Std: {norms.std():.4f}  "
          f"Min: {norms.min():.4f}  Max: {norms.max():.4f}")

    # 可视化（全量样本 + StandardScaler）
    if if_visualize:
        visualize_pca_tsne(features, labels, dataset_name, model_name, saving_dir)

    # 可视化（移除 top-1 norm outlier + RobustScaler）
    if if_visualize:
        visualize_pca_tsne(features, labels, dataset_name, model_name, saving_dir,
                           prompts=prompts, remove_top_outlier=True)

    # 输出汇总
    print(f"\n{'='*60}")
    print(f"  结果汇总")
    print(f"{'='*60}")
    print(f"  {'分类器':<15} {'ACC':>8} {'F1':>8} {'ROC-AUC':>8} {'FPR':>8}")
    print(f"  {'-'*47}")
    for name in ['logistic', 'mlp', 'iforest']:
        r = results[name]
        acc_s = f"{r['acc']:>8.4f}" if r['acc'] is not None else f"  {'N/A':>6}"
        f1_s = f"{r['f1']:>8.4f}" if r['f1'] is not None else f"  {'N/A':>6}"
        auc_s = f"{r['auc']:>8.4f}" if r['auc'] is not None else f"  {'N/A':>6}"
        fpr_s = f"{r['fpr']:>8.4f}" if r['fpr'] is not None else f"  {'N/A':>6}"
        print(f"  {name:<15} {acc_s} {f1_s} {auc_s} {fpr_s}")

    return results


# ============================================================
# 9. 命令行入口
# ============================================================

def load_parameters(file_path):
    with open(file_path, 'r') as file:
        parameters = json.load(file)
    return parameters


def _prepare_model_parameters(parameters, args):
    """Resolve CLI/JSON model settings into concrete server paths."""
    cli_model_path = args.model_path
    cli_model_name = args.model_name
    cli_saving_dir = args.saving_dir
    json_model_path = parameters.get('model_path')
    json_model_name = parameters.get('model_name')

    if cli_model_path or cli_model_name:
        if not cli_model_name:
            raise ValueError("--model_name is required when overriding --model_path.")
        resolved = {
            "model_path": cli_model_path or "",
            "model_name": cli_model_name,
            "saving_dir": cli_saving_dir or f"outputs_hiddenstate/{_make_output_slug(cli_model_name)}",
        }
    elif json_model_name:
        resolved = {
            "model_path": json_model_path or "",
            "model_name": json_model_name,
            "saving_dir": cli_saving_dir or parameters.get("saving_dir") or f"outputs_hiddenstate/{_make_output_slug(json_model_name)}",
        }
    else:
        raise ValueError(
            "未指定模型。请使用 --model_name 指定模型 ID 或本地模型目录。"
        )

    parameters = parameters.copy()
    parameters['model_path'] = resolved['model_path']
    parameters['model_name'] = resolved['model_name']
    parameters['saving_dir'] = resolved['saving_dir']
    return parameters


def _join_model_path(model_path, model_name):
    if os.path.isabs(model_name) or not model_path:
        return model_name
    return os.path.join(model_path, model_name)


def _run_with_parameters(parameters, args):
    print("--> 参数:", parameters)

    model_path = parameters['model_path']
    model_name = parameters['model_name']
    saving_dir = parameters.get('saving_dir', 'outputs_hiddenstate/')
    n_last_layers = parameters.get('n_last_layers', 5)
    max_samples = parameters.get('max_samples', None)
    test_size = parameters.get('test_size', 0.3)
    random_state = parameters.get('random_state', 42)
    if_visualize = parameters.get('if_visualize', True)

    # 加载模型
    print(f"--> 加载模型: {model_path}{model_name}")
    model_name_or_path = _join_model_path(model_path, model_name)
    try:
        mt = ModelAndTokenizer(
            model_name_or_path,
            low_cpu_mem_usage=True,
            device=getattr(args, 'device', 'cuda:0'),
            local_files_only=getattr(args, 'local_files_only', False),
        )
    except OSError as exc:
        print("\n[模型加载失败]")
        print(f"  当前模型参数: {model_name_or_path}")
        print("  如果服务器不能访问 huggingface.co，请使用已下载好的本地模型目录，例如：")
        print('    python public_func/hidden_state_detector.py --model_name "/root/autodl-tmp/models/Qwen2.5-7B-Instruct" --saving_dir "outputs_hiddenstate/qwen2.5-7b" --run_all --force_extract --local_files_only')
        print("  本地模型目录中应包含 config.json、tokenizer 文件和权重文件。")
        raise RuntimeError(
            "Model loading failed. Use a local model directory on offline servers, "
            "or enable network access / pre-download the Hugging Face model cache."
        ) from exc
    mt.model
    print("--> 模型加载成功")
    print(f"    层数: {mt.num_layers}, hidden_size: {mt.model.config.hidden_size}")

    # 加载训练数据集
    dataset = eval(parameters['dataset'])
    print(f"--> 训练数据集: {parameters['dataset']}, 共 {len(dataset)} 条")

    # 加载测试数据集（跨数据集验证）
    test_dataset = None
    test_dataset_str = parameters.get('test_dataset', None)
    if test_dataset_str:
        test_dataset = eval(test_dataset_str)
        print(f"--> 测试数据集: {test_dataset_str}, 共 {len(test_dataset)} 条")

    # --- 批量运行模式 ---
    run_all = getattr(args, 'run_all', False) or parameters.get('run_all', False)
    if run_all:
        ALL_DATASETS = [AutoDAN, GCG, PAP]

        # Phase 1: 提取全部数据集的 features（缓存加速）
        print(f"\n{'#'*60}")
        print(f"  Phase 1: 提取所有数据集 hidden states")
        print(f"{'#'*60}")
        for DS in ALL_DATASETS:
            ds = DS()
            ds_name = ds.__class__.__name__
            prompts, labels = sample_balanced_dataset(ds, max_samples=max_samples, random_state=random_state)
            _get_or_extract_features(
                prompts, labels, mt, ds_name, model_name,
                saving_dir, n_last_layers, force_extract=getattr(args, 'force_extract', False))

        # Phase 2: 单独跑每个数据集（同数据集 7:3）
        print(f"\n{'#'*60}")
        print(f"  Phase 2: 同数据集测试")
        print(f"{'#'*60}")
        for DS in ALL_DATASETS:
            ds = DS()
            run_hidden_state_detection(
                dataset=ds, mt=mt, model_name=model_name,
                saving_dir=saving_dir, n_last_layers=n_last_layers,
                max_samples=max_samples, test_size=test_size,
                random_state=random_state, if_visualize=if_visualize,
                force_extract=getattr(args, 'force_extract', False))

        # Phase 3: 全部交叉验证对
        print(f"\n{'#'*60}")
        print(f"  Phase 3: 交叉验证（全部组合）")
        print(f"{'#'*60}")
        for TrainDS in ALL_DATASETS:
            for TestDS in ALL_DATASETS:
                if TrainDS == TestDS:
                    continue
                train_ds = TrainDS()
                test_ds = TestDS()
                run_hidden_state_detection(
                    dataset=train_ds, mt=mt, model_name=model_name,
                    saving_dir=saving_dir, n_last_layers=n_last_layers,
                    max_samples=max_samples, test_size=test_size,
                    random_state=random_state, if_visualize=if_visualize,
                    test_dataset=test_ds, force_extract=getattr(args, 'force_extract', False))
        return

    # --- 单次运行模式 ---
    run_hidden_state_detection(
        dataset=dataset,
        mt=mt,
        model_name=model_name,
        saving_dir=saving_dir,
        n_last_layers=n_last_layers,
        max_samples=max_samples,
        test_size=test_size,
        if_visualize=if_visualize,
        test_dataset=test_dataset,
        random_state=random_state,
        force_extract=getattr(args, 'force_extract', False),
    )


if __name__ == '__main__':
    import argparse

    current_dir = os.getcwd()
    json_file_path = os.path.join(current_dir, 'public_func', 'parameters.json')
    parameters = load_parameters(json_file_path)

    parser = argparse.ArgumentParser(description='Hidden-State Detector')
    parser.add_argument('--dataset', type=str, help='数据集')
    parser.add_argument('--task', type=str, help='任务类型')
    parser.add_argument('--model_path', type=str, help='模型路径前缀')
    parser.add_argument('--model_name', type=str, help='模型名称')
    parser.add_argument('--saving_dir', type=str, help='输出目录')
    parser.add_argument('--max_samples', type=int, help='最大样本数')
    parser.add_argument('--n_last_layers', type=int, help='最后几层')
    parser.add_argument('--test_dataset', type=str, help='跨数据集测试集（如 GCG()）')
    parser.add_argument('--force_extract', action='store_true', help='忽略缓存，强制重新提取 hidden states')
    parser.add_argument('--run_all', action='store_true', help='跑全部三个数据集 + 交叉验证组合')
    parser.add_argument('--local_files_only', action='store_true', help='仅使用本地缓存模型（离线模式）')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU 设备（默认 cuda:0）')
    args = parser.parse_args()

    # 命令行覆盖
    if args.dataset:
        parameters['dataset'] = args.dataset
    if args.max_samples:
        parameters['max_samples'] = args.max_samples
    if args.n_last_layers:
        parameters['n_last_layers'] = args.n_last_layers
    if args.test_dataset:
        parameters['test_dataset'] = args.test_dataset

    force_extract = getattr(args, 'force_extract', False)
    if force_extract:
        print("--> force_extract=True，将忽略缓存重新提取")

    parameters = _prepare_model_parameters(parameters, args)
    _run_with_parameters(parameters, args)
