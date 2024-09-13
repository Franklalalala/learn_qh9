import json
from pprint import pprint

import torch
from learn_qh9.trainer import Trainer

# new_dataset_path = r'/personal/qh9_data/qh9_all'
new_dataset_path = r'/personal/qh9_data/dummy_2500'
best_ckpt_path = r'/personal/qh9_data/remote_all_0903/cooked/ckpt/32.pt'
train_para_path = r'/personal/qh9_data/remote_all_0903/raw/32/input.json'

with open(train_para_path, 'r') as json_file:
    params = json.load(json_file)
params["dataset"]["src_lmdb_folder_path"] = new_dataset_path
a_trainer = Trainer(params)

a_trainer.model.load_state_dict(torch.load(best_ckpt_path)['state_dict'])
total_error_dict = a_trainer.do_valid(a_trainer.test_data_loader)


with open('test_results.json', 'w') as f:
    json.dump(total_error_dict, f, indent=2)

pprint(total_error_dict)
