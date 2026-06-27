# LLMScan: Causal Scan for LLM Misbehavior Detection 
This repository is to scan LLM's "brain" and detect LLM's misbehavior based on causality analysis. 

## Abstract

Despite the success of Large Language Models (LLMs) across various fields, their potential to generate untruthful, biased and harmful responses poses significant risks, particularly in critical applications. This highlights the urgent need for systematic methods to detect and prevent such misbehavior. While existing approaches target specific issues such as harmful responses, this work introduces LLMScan, an innovative LLM monitoring technique based on causality analysis, offering a comprehensive solution.LLMScan systematically monitors the inner workings of an LLM through the lens of causal inference, operating on the premise that the LLM's `brain' behaves differently when misbehaving. By analyzing the causal contributions of the LLM's input tokens and transformer layers, LLMScan effectively detects misbehavior. Extensive experiments across various tasks and models reveal clear distinctions in the causal distributions between normal behavior and misbehavior, enabling the development of accurate, lightweight detectors for a variety of misbehavior detection tasks.

## Structure of this repository:

- `data` contains the raw datasets and processed dataset with CE informations for 4 detection tasks. `data/raw_questions` contains the datasets in their original format, while `data/processed_questions` contains the datasets transformed to a common format. (the dataset loading code is at file lllm/questions_loaders.py)
- `lllm`, `utils`: contains source code. 
- `public_fun`: contains the source code running LLMScan (CE generation and detector trianing/evaluation). In specifically, `public_fun/causality_analysis.py` contains the code for scanning model layers and generating layer-level causal effects, `public_fun/causality_analysis.py` contains the code for generating model token-level causal effects and the detector training is executed at `public_fun/causality_analysis_combine.py` which contains the code for training, evaluating our LLMScan detectors. 
- `figs`: all analyzing figures, e.g., PCA, Violin Figures and Causal Maps

`public_fun/paramters.json`

## Setup

The code was developed with Python 3.8. To install dependencies:
```bash
pip install -r requirements.txt
```

## Model
All pre-trained models are loaded from HuggingFace.
```bash
# llama-2-7b
"model_path": "meta-llama/",
"model_name": "Llama-2-7b-chat-hf"

# llama-2-13b
"model_path": "meta-llama/",
"model_name": "Llama-2-13b-chat-hf"

# llama-3.1
"model_path": "meta-llama/",
"model_name": "Meta-Llama-3.1-8B-Instruct"

# Mistral
"model_path": "mistralai/",
"model_name": "Mistral-7B-Instruct-v0.2"
```

## Example Experiment
```bash
# generating layer-level ce (remember to set the save_progress as True to save all causal effects results in processed_dataset files)
python public_func/causality_analysis.py --model_path "meta-llama/" --model_name "Llama-2-7b-chat-hf" --task "lie" --dataset "Questions1000()" --saving_dir "outputs_lie/llama-2-7b/"
# or you can directly run: 
python public_func/causality_analysis.py   # then the parameters are loaded from file public/parameters.json

# generating token-level ce 
python public_func/causality_analysis_prompt.py

# train and evaluate the detector
python public_func/causality_analysis_combine.py
```





### Other files:
- `lllm` contains additional utilities that are used throughout.
- `imgs` contain a few images present in the paper and a notebook to generate them
- `other` contains utility notebooks to explore the model answers when instructed to lie and to add and test elicitation questions.   


## Practicalities

To use this code, create a clean `Python` environment and then run 

```pip install -r requirements.txt```
```pip install -r requirements_casper.txt```


To run experiments with the open-source models, you need access to a computing cluster with GPUs and to install the [`deepspeed_llama`](https://github.com/LoryPack/deepspeed_llama) repository on that cluster. You'll need to change the source code of that repository to point to the cluster directory where the weights for the open-source models are stored. `experiments_alpaca_vicuna` and `finetuning/llama` contain a few `*.sh` example scripts for clusters using `slurm`.
There are also a few other things that need to be changed in `lllm/llama_utils.py` according to the paths of your cluster. Moreover, `finetuning/llama/llama_ft_folder.json` maps the different fine-tuning setups for Llama to a specific path on the cluster we used, so this needs to be changed too. 

Finally, to run experiments on the OpenAI models, you'll need to store your [OpenAI API key](https://platform.openai.com/account/api-keys) in a `.env` file in the root of this directory, with the format: 

```OPENAI_API_KEY=sk-<your key>```

Running experiments with the OpenAI API will incur a monetary cost. Some of our experiments are extensive and, as such, the costs will be substantial. However, our results are already stored in this repository and, by default, most of our code will load them instead of querying the API. Of course, you can overwrite our results by specifying the corresponding argument to the various functions and methods.

## Hidden-State Detector（新版主方法）

不依赖因果干预的轻量级 LLM 不当行为检测。一次前向传播 → 提取 hidden states → 训练 LR/MLP 分类器 → PCA/t-SNE 可视化。

---

### 第一步：配置服务器环境

```bash
# 1. 创建 conda 环境
conda create -n llmscan python=3.8.20 -y
conda activate llmscan

# 2. 安装 PyTorch（根据 CUDA 版本选择，见 https://pytorch.org）
# CUDA 12.1 示例：
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其余依赖
pip install -r requirements.txt

# 4. 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

---

### 第二步：跑实验

**使用 `--model_name` 指定任意模型。** 可以是 Hugging Face / ModelScope 兼容模型目录，也可以是在线模型 ID。

```bash
# 先进入项目目录。不要在 ~ 下直接运行，否则会找不到 public_func/hidden_state_detector.py
cd /root/autodl-tmp

python public_func/hidden_state_detector.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --saving_dir "outputs_hiddenstate/qwen2.5-7b" \
    --run_all \
    --force_extract
```

如果服务器不能访问 Hugging Face，不能直接用 `"Qwen/Qwen2.5-7B-Instruct"` 这种在线模型 ID。需要先把模型放到服务器本地，然后传本地目录：

```bash
cd /root/autodl-tmp

python public_func/hidden_state_detector.py \
    --model_name "/root/autodl-tmp/models/Qwen2.5-7B-Instruct" \
    --saving_dir "outputs_hiddenstate/qwen2.5-7b" \
    --run_all \
    --force_extract \
    --local_files_only
```

本地模型目录应包含 `config.json`、tokenizer 文件和权重文件。AutoDL 容器如果没有外网，必须使用这种本地路径方式，或者提前把 Hugging Face cache 准备好。

也可以在 `public_func/parameters.json` 中写入 `model_name` / `saving_dir`，再直接运行 `python public_func/hidden_state_detector.py --run_all`。

`--run_all` 自动完成：
- 三个数据集各自 7:3 测试（3 次）
- 全部 6 对交叉验证
- LR / MLP / IForest 三个分类器
- 所有 PCA + t-SNE 可视化
- hidden-state 特征缓存（二次运行秒进）
- `results_lasttoken_v2.csv/md` 汇总

> 注意：`hidden_state_detector.py --run_all` 只跑正常主实验，不会自动跑层数消融。
> 层数消融需要再运行 `layer_ablation_hidden_state.py`。

### 第三步：跑层数消融（可选但推荐）

主实验会生成 `last5` cache；层数消融会复用这个 cache 切片得到 `last1-last5`，只有 `last10` 在没有 cache 时需要重新提取 hidden states。

推荐顺序：

```bash
# 1. 正常主实验：生成 last5 cache、同数据集/跨数据集结果、图表和汇总
python public_func/hidden_state_detector.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --saving_dir "outputs_hiddenstate/qwen2.5-7b" \
    --run_all \
    --force_extract

# 2. 层数消融：复用主实验 cache，额外补 last10
python public_func/layer_ablation_hidden_state.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir "outputs_hiddenstate/qwen2.5-7b/ablations/AutoDAN_layers_lasttoken_v2" \
    --main_cache_dir "outputs_hiddenstate/qwen2.5-7b/cache" \
    --layers 1 2 3 4 5 10 \
    --device cuda:0
```

如果使用本地路径模型：

```bash
python public_func/hidden_state_detector.py \
    --model_name "/root/autodl-tmp/models/Qwen2.5-7B-Instruct" \
    --saving_dir "outputs_hiddenstate/my-local-model" \
    --run_all \
    --force_extract \
    --local_files_only

python public_func/layer_ablation_hidden_state.py \
    --model_name "/root/autodl-tmp/models/Qwen2.5-7B-Instruct" \
    --output_dir "outputs_hiddenstate/my-local-model/ablations/AutoDAN_layers_lasttoken_v2" \
    --main_cache_dir "outputs_hiddenstate/my-local-model/cache" \
    --layers 1 2 3 4 5 10 \
    --device cuda:0
```

消融输出：

- `ablations/AutoDAN_layers_lasttoken_v2/cache/`
- `ablations/AutoDAN_layers_lasttoken_v2/models/`
- `ablations/AutoDAN_layers_lasttoken_v2/results_layer_ablation.csv`
- `ablations/AutoDAN_layers_lasttoken_v2/results_layer_ablation.md`

---

### 第四步：查看结果

**每个模型自动输出到独立目录，不会互相覆盖。**

```
outputs_hiddenstate/
├── qwen2.5-7b/
│   ├── cache/
│   │   ├── AutoDAN_..._last5_lasttoken_v2_samples614.npz
│   │   ├── GCG_..._samples1070.npz
│   │   └── PAP_..._samples500.npz
│   ├── figs/
│   │   ├── hidden_state_AutoDAN_...pdf  # 同数据集 PCA/t-SNE
│   │   ├── cross_AutoDAN_to_GCG_...pdf  # 跨数据集 PCA/t-SNE
│   │   └── ...（共 12 张图）
│   ├── logistic_hidden_state_AutoDAN_lasttoken_v2.joblib
│   ├── mlp_hidden_state_AutoDAN_lasttoken_v2.joblib
│   ├── iforest_hidden_state_AutoDAN_lasttoken_v2.joblib
│   ├── results_lasttoken_v2.csv
│   ├── results_lasttoken_v2.md
│   └── ablations/
│       └── AutoDAN_layers_lasttoken_v2/
│
└── other-model/
    └── ...
```

直接使用 `--model_name` 时，如果不传 `--saving_dir`，脚本会按模型名自动生成输出目录；建议正式实验显式指定 `--saving_dir`，更容易管理。

所以换模型 = 换输出目录，不会互相污染。跑完一个新模型，对应目录下就是它的全部结果。

---

### 常用参数速查

| 想做什么 | 命令 |
|----------|------|
| 任意模型 | `--model_name "/path/to/model" --saving_dir "outputs_hiddenstate/my-model"` |
| 跑正常主实验 | `hidden_state_detector.py --run_all` |
| 跑层数消融 | `layer_ablation_hidden_state.py --layers 1 2 3 4 5 10` |
| 只跑一个数据集 | `--dataset "AutoDAN()"`（去掉 `--run_all`） |
| 跨数据集测试 | `--dataset "AutoDAN()" --test_dataset "GCG()"` |
| 限制样本数 | `--max_samples 100`（各取 50 adv + 50 non_adv） |
| 换特征层数 | `--n_last_layers 3` |
| 离线模式 | `--local_files_only` |
| 强制重新提取 | `--force_extract` |
| 指定 GPU | `--device "cuda:1"` |
| 换输出目录 | `--saving_dir "my_experiments/"` |

### 数据集

| 数据集 | `--dataset` | adv | non_adv |
|--------|-------------|-----|---------|
| AutoDAN | `"AutoDAN()"` | 372 | 242 |
| GCG | `"GCG()"` | 550 | 520 |
| PAP | `"PAP()"` | 258 | 242 |
