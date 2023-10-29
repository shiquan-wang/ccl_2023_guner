import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import config
import data_loader
import utils
from model import Model
from train_to_raw_view import toRawView
from context_finder import getPredictContextJson
import os

class Predictor():
    def __init__(self, model) -> None:
        self.model = model

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, pre_loader, origin_data):
        self.model.eval()

        result = []
        i = 0
        with torch.no_grad():
            for data_batch in pre_loader:
                sentence_batch = origin_data[i:i + config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities, predict_sets = utils.decode(outputs.cpu().numpy(), entity_text,
                                                                                  length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}

                    for ent in ent_list:
                        instance["entity"].append({"text": ''.join([sentence[x] for x in ent[0]]),
                                       "index": ent[0],
                                       "type": config.vocab.id_to_label(ent[1])})
                    result.append(instance)
                i += config.batch_size
        return result
def zhuan_fan(result, fan_file):
    f1 = open(fan_file, 'r', encoding='utf-8')
    data1 = json.load(f1)
    ret = []
    for d1,d2 in zip(data1, result):
        cur = {}
        cur['sentence'] = d1['sentence'].copy()
        cur['entity'] = d2['entity'].copy()
        for en in cur['entity']:
            st = en['index'][0]
            ed = en['index'][-1]
            en['text'] = cur['sentence'][st:ed+1]
        ret.append(cur)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/bisai_context.json')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--model_version', type=int, default=299)
    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--sub_outpath', type=str)

    args = parser.parse_args()

    config = config.Config(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # flag表示简体还是繁体，为0是繁体，为1是简体，需要在最后转化为繁体
    flag = 1
    version = "v1"
    target_file_jianti = "D:/pyworkspace/ccl_ner-main/data/jian_data_context/jian_target_entity_{}.txt".format(version)
    out_json_file = "D:/pyworkspace/ccl_ner-main/data/jian_data_context/predict_context_{}.json".format(version)
    getPredictContextJson(target_file_jianti, out_json_file)

    predict_dataset, predict_data = data_loader.prepare_predict_input(config, out_json_file)

    pre_loader = DataLoader(dataset=predict_dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle = False,
                   num_workers=0,
                   drop_last = False)

    model_save_path = config.sub_outpath + '{}model.pt'.format(config.model_version)
    model = Model(config)
    model = model.cuda()

    inference = Predictor(model)
    inference.load(model_save_path)
    pre_result = inference.predict(pre_loader, predict_data)

    # if flag:
    #     fan_dev = 'data/bisai/dev.json'
    #     fan_predict = 'data/bisai/predict.json'
    #     pre_result = zhuan_fan(pre_result,fan_predict)
    #     dev_result = zhuan_fan(dev_result, fan_dev)

    pre_output_file = "D:/pyworkspace/ccl_ner-main/data/jian_data_context/" + '{}_context_predict_result_{}.txt'.format(config.model_version, version)

    with open(pre_output_file, 'w', encoding='utf-8') as f:
        json.dump(pre_result, f, ensure_ascii=False)

    predicate_raw_out_file = "D:/pyworkspace/ccl_ner-main/data/jian_data_context/" + '{}_context_predict_result_raw_{}.txt'.format(config.model_version, version)

    # in_file = "D:/PycharmProjects/ccl_ner-main/output/all_jian_bert_wwm_ext_1/no_trick/299_predict_result_final.txt"
    # out_file = "D:/PycharmProjects/ccl_ner-main/output/all_jian_bert_wwm_ext_1/no_trick/PREDICT_299_9587.txt"

    toRawView(pre_output_file, predicate_raw_out_file, True)