import json
import os

file_path = 'output'
files = os.listdir(file_path)
for file in files:
    subfile_path = file_path + '/' + file
    dev = subfile_path + '/dev_result.txt'
    pre = subfile_path + '/predict_result.txt'
    f1 = open(dev,'r',encoding='utf-8')
    d1 = len(json.load(f1))
    f2 = open(pre, 'r', encoding='utf-8')
    d2 = len(json.load(f2))
    print(f'{subfile_path}:dev{d1} per{d2} ')
