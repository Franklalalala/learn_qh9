import json
from learn_qh9.trainer import Trainer

# Load parameters from input.json
with open('input.json', 'r') as json_file:
    params = json.load(json_file)

a_trainer = Trainer(params)
a_trainer.train()