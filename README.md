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
conda create -n llmscan python=3.10 -y
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

**只需改 `--model_name`，其余自动完成。** 每个模型的数据集相同（AutoDAN / GCG / PAP），不需要修改。

```bash
# ===== 方式一：命令行指定模型（推荐，不需要改任何文件） =====

# Llama-2-7B
python public_func/hidden_state_detector.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --run_all

# Llama-2-13B
python public_func/hidden_state_detector.py \
    --model_name "meta-llama/Llama-2-13b-chat-hf" \
    --run_all

# Llama-3.1-8B
python public_func/hidden_state_detector.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --run_all

# Mistral-7B
python public_func/hidden_state_detector.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
    --run_all

# Qwen2.5-1.5B（小模型，快速验证）
python public_func/hidden_state_detector.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --run_all


# ===== 方式二：修改 parameters.json 后直接跑（不改命令行） =====
# 编辑 public_func/parameters.json，把 model_name 改成目标模型，然后：
python public_func/hidden_state_detector.py --run_all
```

`--run_all` 自动完成：
- 三个数据集各自 7:3 测试（3 次）
- 全部 6 对交叉验证
- 所有 PCA + t-SNE 可视化
- hidden-state 特征缓存（二次运行秒进）

---

### 第三步：查看结果

**每个模型自动输出到独立目录，不会互相覆盖。**

```
outputs_hiddenstate/
├── meta-llama_Llama-2-7b-chat-hf/      # 模型1
│   ├── cache/
│   │   ├── AutoDAN_..._samples614.npz   # hidden-state 特征缓存
│   │   ├── GCG_..._samples1070.npz
│   │   └── PAP_..._samples500.npz
│   ├── figs/
│   │   ├── hidden_state_AutoDAN_...pdf  # 同数据集 PCA/t-SNE
│   │   ├── cross_AutoDAN_to_GCG_...pdf  # 跨数据集 PCA/t-SNE
│   │   └── ...（共 12 张图）
│   ├── logistic_hidden_state_AutoDAN.joblib
│   └── mlp_hidden_state_AutoDAN.joblib
│
├── meta-llama_Llama-2-13b-chat-hf/     # 模型2
│   └── ...
│
├── meta-llama_Meta-Llama-3.1-8B-Instruct/  # 模型3
│   └── ...
│
└── mistralai_Mistral-7B-Instruct-v0.2/     # 模型4
    └── ...
```

`saving_dir` 自动拼接规则：`outputs_hiddenstate/{model_name 的 / 替换为 _}/`

所以换模型 = 换输出目录，不会互相污染。跑完一个新模型，对应目录下就是它的全部结果。

---

### 常用参数速查

| 想做什么 | 命令 |
|----------|------|
| 换模型 | `--model_name "meta-llama/Llama-2-13b-chat-hf"` |
| 跑全部实验 | `--run_all` |
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