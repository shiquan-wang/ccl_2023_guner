#!coding=utf-8
import json
import codecs
import os
import re
import context_finder

def locateLine(dir, text):
    files = os.listdir(dir)
    result = []
    for file in files:
        with codecs.open(os.path.join(dir, file), "r", "utf-8") as in_stream:
            i = 0
            for line in in_stream:
                i += 1
                if line.find(text) >= 0:
                    result.append((file, i))
    return result

def findInFiles(dir, text):
    files = os.listdir(dir)
    result = []
    for file in files:
        with codecs.open(os.path.join(dir, file), "r", "utf-8") as in_stream:
            for line in in_stream:
                if line.find(text) >= 0:
                    print("{}:={}".format(file, line))
                    result.append("{}:={}".format(file, line))
    return result

CONTEXT_LINE_THRESHOLD = 500
PREFIX_CONTEXT_NUM = 3
import math
def findInFilesClose(dir, text, entity_phrase, sentence, locations):
    files = os.listdir(dir)
    result = []
    for file in files:
        if len(locations) > 0:
            file_names = [x[0] for x in locations]
            if file not in file_names:
                continue
            index = file_names.index(file)
            line_num_location = locations[index][1]
            total_lines = []
            with codecs.open(os.path.join(dir, file), "r", "utf-8") as in_stream:
                for line in in_stream:
                    total_lines.append(line.strip())
                for line_num, line in enumerate(total_lines):
                    if line.find(entity_phrase) >= 0:
                        ##segment level
                        phrase_list = line.split("。")
                        context_phrase = []
                        for phrase in phrase_list:
                            if text in phrase and phrase not in sentence and entity_phrase not in phrase:
                                context_phrase.append(phrase)
                        if len(context_phrase) > 0:
                            context_phrase.append(sentence)
                            result.append("。".join(context_phrase))
                        ##document level
                        else:
                            context_phrase = []
                            if abs(line_num_location - line_num + 1) < CONTEXT_LINE_THRESHOLD:
                                context_phrase.extend(total_lines[line_num + 1: line_num + PREFIX_CONTEXT_NUM])
                            result.append("".join(context_phrase))

    files = os.listdir(dir)
    if len(text) >= 3:
        for file in files:
            if len(locations) > 0:
                file_names = [x[0] for x in locations]
                if file not in file_names:
                    continue
                index = file_names.index(file)
                line_num_location = locations[index][1]
                with codecs.open(os.path.join(dir, file), "r", "utf-8") as in_stream:
                    target_line_num = 0
                    context = []
                    for line in in_stream:
                        target_line_num += 1
                        if line.find(text) >= 0:
                            target_index = line.index(text)
                            begin_target_index = findBefore(line, target_index)
                            end_target_index = findEnd(line, target_index)
                            context.append(line.strip())
                    context.append(sentence)
                    result.append(sentence)
            else:
                with codecs.open(os.path.join(dir, file), "r", "utf-8") as in_stream:
                    context = []
                    for line in in_stream:
                        if line.find(text) >= 0:
                            target_index = line.index(text)
                            begin_target_index = findBefore(line, target_index)
                            end_target_index = findEnd(line, target_index)
                            context.append(line.strip())
                    result.append(sentence)
    else:
        for file in files:
            file_names = [x[0] for x in locations]
            if file not in file_names:
                continue
            index = file_names.index(file)
            line_num_location = locations[index][1]
            with codecs.open(os.path.join(dir, file), "r", "utf-8") as in_stream:
                target_line_num = 0
                context = []
                for line in in_stream:
                    target_line_num += 1
                    if line.find(text) >= 0 and abs(target_line_num-line_num_location) <= CONTEXT_LINE_THRESHOLD:
                        target_index = line.index(text)
                        begin_target_index = findBefore(line, target_index)
                        end_target_index = findEnd(line, target_index)
                        # print("{}:={}".format(file, line))
                        context.append(line.strip())
                context.append(sentence)
                result.append(sentence)
    if len(result) == 0:
        result.append(sentence)
        return result
    else:
        split = 0
        if len(result) > 20:
            split = 20
        else:
            split = len(result)
        return result[0:split]

def findPredictionTrainConflict(predict_file, train_file):
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

    for entity, pred_labels in pred_map.items():
        if entity in train_map:
            # if len(pred_labels) == 1 and len(train_map[entity]) == 1:
            #     if pred_labels[0] != train_map[entity][0]:
            #         print("candidate entity:{}".format(entity))
            if len(pred_labels) == 1 and len(train_map[entity]) > 1:
                if pred_labels[0] not in train_map[entity]:
                    print("candidate entity not in training labels:{}".format(entity))

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
import unicodedata


class DuckType:
    def __contains__(self, s):
        return unicodedata.category(s).startswith("P")
punct = DuckType()
# print("'" in punct, '"' in punct, "a" in punct)
def findBefore(text, anchor_index):
    for i in range(anchor_index, -1, -1):
        if i < len(text):
            if text[i] in punct or i == 0:
                if i == 0:
                    return i
                return i + 1
    if i == 0:
        return i
    else:
        return anchor_index

def findEnd(text, anchor_index):
    for i in range(anchor_index, len(text)):
        if i < len(text):
            if text[i] in punct or i == len(text) - 1:
                return i
    return anchor_index

def findTwoBefore(text, anchor_index):
    skip = 0
    for i in range(anchor_index, -1, -1):
        if i < len(text):
            if text[i] in punct or i == 0:
                if i != 0:
                    if skip == 0:
                        skip += 1
                    else:
                        return i + 1
                else:
                    None
    if i == 0:
        return i
    else:
        return anchor_index

def findTwoEnd(text, anchor_index):
    skip = 0
    for i in range(anchor_index, len(text)):
        if i < len(text):
            if text[i] in punct or i == len(text) - 1:
                if i != len(text) - 1:
                    if skip == 0:
                        skip += 1
                    else:
                        return i
                else:
                    return i
            else:
                None
    return anchor_index


def getJiantiEntity(text, index):
    index_list = [int(x) for x in index.split(",")]
    st = [text[i] for i in index_list]
    phrase_begin = findTwoBefore(text, index_list[0])
    phrase_end = findTwoEnd(text, index_list[-1])
    return "".join(st), text[phrase_begin:phrase_end]

def loadVotingResult(config_file, fanti_to_jianti_map, out_file):
    in_dir = "./data/origin_corpus/total/"
    result = []
    final_map = {}
    with codecs.open(out_file, "w", "utf-8") as out_stream:
        total_line_num = 0
        with codecs.open(config_file, "r", "utf-8") as in_stream:
            for line in in_stream.readlines():
                total_line_num += 1

        with codecs.open(config_file, "r", "utf-8") as in_stream:
            line_num = 0
            for line in in_stream.readlines():
                line_num += 1
                if line_num % 10 == 0:
                    print("now in progress:%.2f...\n" % (line_num / total_line_num))
                text, entity, index = line.strip().split("=>")[0].split("^#^")
                sentence_jianti = fanti_to_jianti_map[text]
                entity_jianti, entity_two_phrase = getJiantiEntity(sentence_jianti, index)
                target_location = locateLine(in_dir, entity_two_phrase)
                candidate_context = findInFilesClose(in_dir, entity_jianti, entity_two_phrase, sentence_jianti,
                                                     target_location)
                temp = {}
                temp["text"] = sentence_jianti
                temp["context"] = candidate_context[0]
                temp["entity"] = entity_jianti
                temp["original_type"] = None
                temp["original_index"] = [int(x) for x in index.split(",")]
                result.append(temp)
                final_map[candidate_context[0]] = sentence_jianti
        out_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        print("now in progress:100%!\n")
    return final_map

if __name__ == "__main__":
    fanti_file = "./data/bisai/predict.json"
    jianti_file = "./data/jian_all/predict.json"
    fanti_to_jianti_map = context_finder.getFantiToJiantiMap(fanti_file, jianti_file)
    config_file = "./compare/config_vote_ensamble_counts.txt"
    out_file = "./compare/config_vote_ensamble_counts_context.txt"
    loadVotingResult(config_file, fanti_to_jianti_map, out_file)