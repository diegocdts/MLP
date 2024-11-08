import os

from arguments import args

from config import load
from model.training import CrossValidation

args = args()
pwd = '/home/src' if not 'MLP' in os.getcwd() else os.getcwd()
data, model, train = args.data, args.model, args.train
print(pwd, data, model, train)

data_info = load(f'{pwd}/config/data.json', data)
model_info = load(f'{pwd}/config/model.json', model)
train_info = load(f'{pwd}/config/train.json', train)

training_validation = CrossValidation(data_info, model_info, train_info)
training_validation.train()

blended_test = data_info["blended_test"]
single_test = data_info["single_test"]
training_validation.predict(blended_test, single_test)
