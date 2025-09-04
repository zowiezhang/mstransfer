import os
import json

# dir_path = os.getcwd()
# data_dir_path = os.path.join(dir_path, f'datasets/{args.eval_data}')

def data_loader(data_path):
    
    with open(data_path, 'r') as f:
        eval_dataset = json.load(f)

    return eval_dataset