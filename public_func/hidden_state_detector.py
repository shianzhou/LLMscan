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
import pandas as pd
from joblib import dump

from lllm.classification_utils import Classifier
from lllm.questions_loaders import AutoDAN, GCG, PAP

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

POOLING_VERSION = "lasttoken_v2"


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
    safe_model = model_name.replace("/", "_")
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
    print(f"--> 训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")
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
    print(f"    ACC: {acc:.4f}  F1: {f1:.4f}  ROC-AUC: {auc:.4f}  FPR: {fpr:.4f}")
    results['logistic'] = {'acc': acc, 'f1': f1, 'auc': auc, 'fpr': fpr, 'model': clf_lr}

    # --- MLP ---
    print("\n========== MLP Classifier ==========")
    clf_mlp = Classifier(X_train, y_train, classifier="MLP", scale=True,
                         hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)
    acc2, auc2, cm2, y_pred2, y_proba2 = clf_mlp.evaluate(X_test, y_test, return_ys=True)
    f1_2 = f1_score(y_test, y_pred2)
    tn2, fp2, fn2, tp2 = cm2.ravel()
    fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0.0
    print(f"    ACC: {acc2:.4f}  F1: {f1_2:.4f}  ROC-AUC: {auc2:.4f}  FPR: {fpr2:.4f}")
    results['mlp'] = {'acc': acc2, 'f1': f1_2, 'auc': auc2, 'fpr': fpr2, 'model': clf_mlp}

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
    if remove_top_outlier:
        fig_name = f"hidden_state_{dataset_name}_{model_name}_{POOLING_VERSION}_remove_top1_outlier.pdf"
    else:
        fig_name = f"hidden_state_{dataset_name}_{model_name}_{POOLING_VERSION}.pdf"
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
    fig_path = os.path.join(fig_dir, f"cross_{train_name}_to_{test_name}_{model_name}_{POOLING_VERSION}.pdf")
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
    print(f"    训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")

    # --- Logistic Regression ---
    print("\n========== Logistic Regression (cross) ==========")
    clf_lr = Classifier(X_train, y_train, classifier="logistic", scale=True,
                        max_iter=1000, random_state=random_state)
    acc, auc, cm, y_pred, y_proba = clf_lr.evaluate(X_test, y_test, return_ys=True)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"    ACC: {acc:.4f}  F1: {f1:.4f}  ROC-AUC: {auc:.4f}  FPR: {fpr:.4f}")
    results['logistic'] = {'acc': acc, 'f1': f1, 'auc': auc, 'fpr': fpr, 'model': clf_lr}

    # --- MLP ---
    print("\n========== MLP Classifier (cross) ==========")
    clf_mlp = Classifier(X_train, y_train, classifier="MLP", scale=True,
                         hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)
    acc2, auc2, cm2, y_pred2, y_proba2 = clf_mlp.evaluate(X_test, y_test, return_ys=True)
    f1_2 = f1_score(y_test, y_pred2)
    tn2, fp2, fn2, tp2 = cm2.ravel()
    fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0.0
    print(f"    ACC: {acc2:.4f}  F1: {f1_2:.4f}  ROC-AUC: {auc2:.4f}  FPR: {fpr2:.4f}")
    results['mlp'] = {'acc': acc2, 'f1': f1_2, 'auc': auc2, 'fpr': fpr2, 'model': clf_mlp}

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
        for name in ['logistic', 'mlp']:
            r = results[name]
            print(f"  {name:<15} {r['acc']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} {r['fpr']:>8.4f}")

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
    for name in ['logistic', 'mlp']:
        r = results[name]
        print(f"  {name:<15} {r['acc']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} {r['fpr']:>8.4f}")

    return results


# ============================================================
# 9. 命令行入口
# ============================================================

def load_parameters(file_path):
    with open(file_path, 'r') as file:
        parameters = json.load(file)
    return parameters


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
    args = parser.parse_args()

    # 命令行覆盖
    if args.model_path:
        parameters['model_path'] = args.model_path
    if args.model_name:
        parameters['model_name'] = args.model_name
    if args.dataset:
        parameters['dataset'] = args.dataset
    if args.saving_dir:
        parameters['saving_dir'] = args.saving_dir
    if args.max_samples:
        parameters['max_samples'] = args.max_samples
    if args.n_last_layers:
        parameters['n_last_layers'] = args.n_last_layers
    if args.test_dataset:
        parameters['test_dataset'] = args.test_dataset

    force_extract = getattr(args, 'force_extract', False)
    if force_extract:
        print("--> force_extract=True，将忽略缓存重新提取")

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
    mt = ModelAndTokenizer(
        model_path + model_name,
        low_cpu_mem_usage=True,
        device='cuda:0'
    )
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
    if getattr(args, 'run_all', False):
        ALL_DATASETS = [AutoDAN, GCG, PAP]
        all_results = {}

        # Phase 1: 提取全部数据集的 features（缓存加速）
        print(f"\n{'#'*60}")
        print(f"  Phase 1: 提取所有数据集 hidden states")
        print(f"{'#'*60}")
        cached_features = {}
        for DS in ALL_DATASETS:
            ds = DS()
            ds_name = ds.__class__.__name__
            prompts, labels = sample_balanced_dataset(ds, max_samples=max_samples, random_state=random_state)
            feats, labs, proms = _get_or_extract_features(
                prompts, labels, mt, ds_name, model_name,
                saving_dir, n_last_layers, force_extract=force_extract)
            cached_features[ds_name] = (feats, labs, proms)

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
                force_extract=force_extract)

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
                    test_dataset=test_ds, force_extract=force_extract)

        sys.exit(0)

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
        force_extract=force_extract,
    )
