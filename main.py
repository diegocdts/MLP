import os

from config import load
from model.training import CrossValidation

if not 'MLP' in os.getcwd():
    pwd, data, train = '/home/src', 'data1', 'train1'
else:
    pwd, data, train = os.getcwd(), 'data2', 'train2'
print(pwd, data, train)

data_info = load(f'{pwd}/config/data.json', data)
model_info = load(f'{pwd}/config/model.json', 'mlp3a')
train_info = load(f'{pwd}/config/train.json', train)

training_validation = CrossValidation(data_info, model_info, train_info)
training_validation.train()

data_path = data_info["blended"]
training_validation.predict(f'{data_path}SISMO_BLENDED_inline_0500_NEW.sgy')
