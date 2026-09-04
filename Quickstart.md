
## 🛠 Installation
We use PDM to manage the dependencies. To install PDM, please refer to the [official documentation](https://pdm-project.org/en/latest/).

```bash
git clone https://github.com/BerkinChen/AlphaSAGE.git
cd AlphaSAGE
pdm install
```

## 📁 Data Preparation

We use the data from [Qlib](https://github.com/microsoft/qlib) to train the model. Please refer to the [official documentation](https://qlib.readthedocs.io/en/latest/?badge=latest) to download the data.


## 🚀 Quick Start

AlphaSAGE operates in two main stages:

### Stage 1: Generate Alpha Pool with GFlowNets

Generate a diverse pool of alpha expressions using our structure-aware GFlowNets framework:

```bash
python train_gfn.py \
        --seed 0 \
        --instrument csi300 \
        --pool_capacity 50 \
        --log_freq 500 \
        --update_freq 64 \
        --n_episodes 10000 \
        --encoder_type gnn \
        --entropy_coef 0.01 \
        --entropy_temperature 1.0 \
        --mask_dropout_prob 1.0 \
        --ssl_weight 1.0 \
        --nov_weight 0.3 \
        --weight_decay_type linear \
        --final_weight_ratio 0.0
```

**Key Parameters:**
- `--encoder_type gnn`: Uses our structure-aware RGCN encoder
- `--pool_capacity 50`: Maximum number of alphas to maintain in the pool
- `--entropy_coef 0.01`: Controls exploration vs exploitation balance
- `--ssl_weight 1.0`: Self-supervised learning weight for structure awareness
- `--nov_weight 0.3`: Novelty reward weight for diversity

### Stage 2: Evaluate and Combine Alpha Pool

Following [AlphaForge](https://github.com/DulyHao/AlphaForge), we use adaptive combination to create the final alpha portfolio:

```bash
python run_adaptive_combination.py \
    --expressions_file results_dir\
    --instruments csi300 \
    --threshold_ric 0.015 \
    --threshold_ricir 0.15 \
    --chunk_size 400 \
    --window inf \
    --n_factors 20 \
    --cuda 2 \
    --train_end_year 2021 \
    --seed 0 \
```

## 🔬 Baselines and Comparisons

### AlphaQCM and AlphaGen Baselines

For comparison with [AlphaGen](https://github.com/RL-MLDM/alphagen) and [AlphaQCM](https://github.com/ZhuZhouFan/AlphaQCM), run the following commands:

#### AlphaQCM:

```bash
# Train AlphaQCM
python train_qcm.py \
    --instruments csi300 \
    --pool 20 \
    --seed 0

# Evaluate AlphaQCM results
python run_adaptive_combination.py \
    --expressions_file results_dir \
    --instruments csi300 \
    --cuda 2 \
    --train_end_year 2021 \
    --seed 0 \
    --use_weights True
```

#### AlphaGen (PPO):
```bash
# Train AlphaGen with PPO
python train_ppo.py \
    --instruments csi300 \
    --pool 20 \
    --seed 0

# Evaluate AlphaGen results  
python run_adaptive_combination.py \
    --expressions_file results_dir \
    --instruments csi300 \
    --cuda 2 \
    --train_end_year 2021 \
    --seed 0 \
    --use_weights True
```

### Other Baselines
For AlphaForge and other ML baselines, please refer to the [AlphaForge documentation](https://github.com/DulyHao/AlphaForge).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.