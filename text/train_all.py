import os

cmds = [
    'python train.py --decoy a the --num_epochs 50 --signal_strength 200 --noise_type bias --data_path ./data/bias/',

    'python train.py --decoy a the --num_epochs 50 --signal_strength 0 --noise_type bias --data_path ./data/bias/',

    'python train.py --decoy text video --num_epochs 50 --signal_strength 200 --noise_type random --data_path ./data/decoy/',

    'python train.py --decoy text video --num_epochs 50 --signal_strength 0 --noise_type random --data_path ./data/decoy/',

    'python train.py --decoy he she --num_epochs 50 --signal_strength 200 --noise_type gender --data_path ./data/gender/',

    'python train.py --decoy he she --num_epochs 50 --signal_strength 0 --noise_type gender --data_path ./data/gender/'
]

cmds_test = [
    'python train.py --decoy a the --num_epochs 1 --signal_strength 200 --noise_type bias --data_path ./data/bias/ --quick_run',

    'python train.py --decoy a the --num_epochs 1 --signal_strength 0 --noise_type bias --data_path ./data/bias/ --quick_run',

    'python train.py --decoy text video --num_epochs 1 --signal_strength 200 --noise_type random --data_path ./data/decoy/ --quick_run',

    'python train.py --decoy text video --num_epochs 1 --signal_strength 0 --noise_type random --data_path ./data/decoy/ --quick_run',

    'python train.py --decoy he she --num_epochs 1 --signal_strength 200 --noise_type gender --data_path ./data/gender/ --quick_run',

    'python train.py --decoy he she --num_epochs 1 --signal_strength 0 --noise_type gender --data_path ./data/gender/ --quick_run'
]

# iterate
for command in cmds:
    print(command)
    os.system(command)
