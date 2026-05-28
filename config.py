import cupy as cp

config = {
    'lr': 0.001, 
    'weight_decay': 5e-4,
    'drop_rate': 0.5,
    'num_epochs': 20,
    'batch_size': 256,
    'milestones': [10, 15],
    'seed': 0,
    'data_path': r"E:\Code\data\emnist-letters-train.csv"  # 建议改成环境变量或相对路径
}

eps = 1e-8
lr_decay = 0.1

cp.random.seed(config['seed'])
