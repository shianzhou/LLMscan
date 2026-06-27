import numpy as np
for name in ['AutoDAN', 'GCG', 'PAP']:
    if name == 'AutoDAN': n = 614
    elif name == 'GCG': n = 1070
    else: n = 500
    path = f'outputs_hiddenstate/qwen2.5-1.5b/cache/{name}_Qwen2.5-1.5B-Instruct_last5_lasttoken_v2_samples{n}.npz'
    d = np.load(path, allow_pickle=True)
    labels = d['labels']
    print(f'{name}: total={len(labels)}, adv={(labels==1).sum()}, non_adv={(labels==0).sum()}, dtype={labels.dtype}')
