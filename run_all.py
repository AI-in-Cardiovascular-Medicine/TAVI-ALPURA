import os

for event in ["cdeath"]:
    configs = [
        f'{event}',
        f'{event}_sts',
        f'{event}_loes',
        f'{event}_es2'
    ]

    for config in configs:
        print(f"\n -------- {config} --------")
        os.system(f"python main.py --config-name {config}")

    os.system(f"python make_plots.py --config-name {configs[0]}")
