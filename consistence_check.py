#!coding=utf-8
import json
import codecs
import json
import os
import re


def findPredictionTrainConflict(predict_file, train_file, config_file, final_file):
    pred_map = {}
    with codecs.open(predict_file, "r", "utf-8") as in_stream:
        for line in in_stream.readlines():
            entity_types = re.findall(r"([{].+?[}])", line.strip())
            for entity_type in entity_types:
                entity_type = entity_type.replace("{", "").replace("}", "")
                entity, type = entity_type.split("|")
                if entity not in pred_map:
                    pred_map[entity] = []
                pred_map[entity].append(type.upper())
    train_map = {}
    with codecs.open(train_file, "r", "utf-8") as in_stream:
        for line in in_stream.readlines():
            entity_types = re.findall(r"([{].+?[}])", line.strip())
            for entity_type in entity_types:
                entity_type = entity_type.replace("{", "").replace("}", "")
                entity, type = entity_type.split("|")
                if entity not in train_map:
                    train_map[entity] = []
                train_map[entity].append(type)

    consistence_map = {}
    for entity, pred_labels in pred_map.items():
        if entity in train_map:
            ##如果测试集的预测标签，和train_set中不一样 % train_set中该实体的标签只有一个&且多次出现&该实体不在不置信实体中，则可能预测错误
            if len(pred_labels) == 1 and len(set(train_map[entity])) == 1 and len(train_map[entity]) > 1:
                if pred_labels[0] not in train_map[entity] and notInNotConfidenceFile(entity, config_file):
                    print("candidate entity not in training:{},current label:{}, training label: {}".format(entity, pred_labels[0], train_map[entity][0]))
                    consistence_map["{" + entity + "|" + pred_labels[0] + "}"] = "{" + entity + "|" + train_map[entity][0] + "}"
    with codecs.open(predict_file, "r", "utf-8") as in_stream:
        with codecs.open(final_file, "w", "utf-8") as out_stream:
            for line in in_stream.readlines():
                to_write = line.upper()
                for k, v in consistence_map.items():
                    to_write = to_write.replace(k, v)
                if to_write != line.upper():
                    print("change sentence:{} using key: {} and value: {}".format(line.strip(), k, v))
                out_stream.write(to_write)

def notInNotConfidenceFile(word, in_file):
    with codecs.open(in_file, "r", "utf-8") as in_stream:
        for line in in_stream.readlines():
            units = line.strip().split("=>")
            if units[0].find(word) >= 0:
                return False
    return True

def calculateNotConsistentProb(train_file):
    train_map = {}
    total_norecall_sentence = []
    with codecs.open(train_file, "r", "utf-8") as in_stream:
        for line in in_stream.readlines():
            entity_types = re.findall(r"([{].+?[}])", line.strip())
            norecall_sentence = re.sub(r"([{].+?[}])", "", line.strip())
            total_norecall_sentence.append(norecall_sentence)
            for entity_type in entity_types:
                entity_type = entity_type.replace("{", "").replace("}", "")
                entity, type = entity_type.split("|")
                if entity not in train_map:
                    train_map[entity] = {}
                if type not in train_map[entity]:
                    train_map[entity][type] = 0
                train_map[entity][type] = train_map[entity][type] + 1
        for key in train_map.keys():
            key_cnt = 0
            for sen in total_norecall_sentence:
                key_cnt += sen.count(key)
            train_map[key]["O"] = key_cnt
        print(train_map)
    return train_map


if __name__ == "__main__":
    predict_file = "./output/all_jian_bert_wwm_ext_1/no_trick/299_predict_result_raw_view.txt"
    train_file = "./data/bisai/train_raw_view.json"
    config_file = "./compare/config_vote_ensamble_counts.txt"
    final_output = "./output/all_jian_bert_wwm_ext_1/no_trick/299_final_submit.txt"
    findPredictionTrainConflict(predict_file, train_file, config_file, final_output)

