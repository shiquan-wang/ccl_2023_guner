import logging
import pickle
import time
import numpy as np
from collections import defaultdict, deque
from collections import defaultdict, deque, Counter

def get_logger(dataset, config):
    pathname = "{}{}_{}.txt".format(config.sub_outpath, dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


def decode(outputs, entities, length):
    class Node:
        def __init__(self):
            self.THW = []                # [(tail, type)]
            self.NNW = defaultdict(set)   # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    batch_predict_set = []
    decode_entities = []
    q = deque()
    for instance, ent_set, l in zip(outputs, entities, length):
        predicts = []
        nodes = [Node() for _ in range(l)]
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur+1):
                # THW
                if instance[cur, pre] > 1: 
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    heads.append(pre)
                # NNW
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head,cur)].add(cur)
                    # post nodes
                    for head,tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head,tail)].add(cur)
            # entity
            for tail,type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur,tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])
        
        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])
        batch_predict_set.append(predicts)
        ent_r += len(ent_set)
        ent_p += len(predicts)
        ent_c += len(predicts.intersection(ent_set))
    return ent_c, ent_p, ent_r, decode_entities, batch_predict_set


def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r

def cal_predict_score(predict_all, origin_all):
    origin_counter = Counter()
    pred_counter = Counter()
    right_counter = Counter()

    for batch_origin, batch_predict in zip(origin_all, predict_all):
        for ent_set, predict_set in zip(batch_origin, batch_predict):
            origin_counter.update([x.split('-#-')[1] for x in ent_set])
            pred_counter.update([x.split('-#-')[1] for x in predict_set])
            right_counter.update([x.split('-#-')[1] for x in predict_set.intersection(ent_set)])

    # 计算每种类型的指标
    entity_f1_info = {}
    for label, count in origin_counter.items():
        origin = count
        found = pred_counter.get(label, 0)
        right = right_counter.get(label, 0)
        f1, precision, recall = cal_f1(right, found, origin)
        entity_f1_info[label] = {"acc": precision, 'recall': recall, 'f1': f1}
    f1, precision, recall = cal_f1(sum(right_counter.values()), sum(pred_counter.values()), sum(origin_counter.values()))
    overall_score = {'acc': precision, 'recall': recall, 'f1': f1}
    return overall_score, entity_f1_info


def vote_token_BIO(cand_ners, sentence, model_index_mapping, vote_threshold):
    tag_matrix = [['O'] * len(sentence) for i in range(len(model_index_mapping))]
    for model_name, label_entity in cand_ners:
        index = model_index_mapping[model_name]
        for label, entities in label_entity.items():
            for entity in entities:
                for i, e_i in enumerate(entity['index']):
                    if i == 0:
                        tag_matrix[index][e_i] = '{}-{}'.format('B', entity['type']).upper()
                    else:
                        tag_matrix[index][e_i] = '{}-{}'.format('I', entity['type']).upper()

    # 投票
    voted_res = []
    for i in range(len(sentence)):
        token_tags = [x[i] for x in tag_matrix]
        voted_res.append(vote_token_one_tag(token_tags, vote_threshold))

    # 解析bio序列，找出实体下标
    entities = []
    cur_index = []
    cur_type = None
    sentence_length = len(sentence)
    for i, tag_etype in enumerate(voted_res):
        if tag_etype == 'O':
            tag = tag_etype
        else:
            tag, etype = tag_etype.split('-')
        if tag == 'B':
            if len(cur_index) > 0:
                entities.append([cur_index, cur_type])
                cur_index, cur_type = [], None
            cur_index = [i]
            cur_type = etype
            if i + 1 == sentence_length:
                entities.append([cur_index, cur_type])
        elif tag == 'I':
            if len(cur_index) == 0 or etype != cur_type:
                continue
            cur_index.append(i)
            if i + 1 == sentence_length:
                entities.append([cur_index, cur_type])
        elif tag == 'O':
            if len(cur_index) > 0:
                entities.append([cur_index, cur_type])
                cur_index, cur_type = [], None
    selected_ner = []
    for indexs, etype in entities:
        selected_ner.append({'text': [sentence[i] for i in indexs],
                             'index': indexs,
                             'type': etype.lower()})
    return selected_ner


def vote_entity(cand_ners, label_top_model):
    selected_ner = []
    all_cand = {}
    label_model_mapping = {}
    for model_name, f1 in label_top_model:
        label_model_mapping[model_name] = f1
    for model_name, entities in cand_ners:
        if model_name not in label_model_mapping:
            continue
        f1 = label_model_mapping[model_name]
        for entity in entities:
            entity_index_text = convert_index_to_text(entity['index'], entity['type'])
            if entity_index_text in all_cand:
                pre_score = all_cand[entity_index_text][1]
                all_cand[entity_index_text][1] = pre_score + f1
            else:
                all_cand[entity_index_text] = [entity, f1]
    
    sorted_cand = sorted(all_cand.items(), key=lambda x: (x[1][1]), reverse=True)
    # index没有重叠的实体作为目标
    index_occured = set()  # 已有实体的下标
    for entity_text, [entity, f1] in sorted_cand:
        # 判断是否有重复，没重复是新实体，把index加入已有实体下标
        if len(index_occured.intersection(set(entity['index']))) == 0:
            selected_ner.append(entity)
            for index in entity['index']:
                index_occured.add(index)
    return selected_ner


def calc_ent(str_list):
    """计算信息熵"""
    x = np.array(str_list)
    # 得到数组x的不重复元素
    x_value_list = set(str_list)
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]#计算每个元素出现的概率
        logp = np.log2(p)
        ent -= p * logp
    return ent


def vote_token_one_tag(tag_list, vote_threshold):
    """对一个token所有model预测的tag进行投票"""
    cand = {}
    for tag in tag_list:
        cand.setdefault(tag, 0)
        cand[tag] += 1
    cand['O'] = 0  # 'O'不参与投票
    sorted_cand = sorted(cand.items(), key=lambda x: x[1], reverse=True)
    if sorted_cand[0][1] >= vote_threshold:
        return sorted_cand[0][0]
    else:
        return 'O'


def vote_token_BIO_analysis(cand_ners, sentence, model_index_mapping, vote_threshold):
    # 汇总所有模型结果
    index_2_model_mapping = {value: key for key, value in model_index_mapping.items()}
    tag_matrix = [['O'] * len(sentence) for i in range(len(model_index_mapping))]

    for model_name, label_entity in cand_ners:
        index = model_index_mapping[model_name]
        for label, entities in label_entity.items():
            for entity in entities:
                for i, e_i in enumerate(entity['index']):
                    if i == 0:
                        tag_matrix[index][e_i] = '{}-{}'.format('B', entity['type']).upper()
                    else:
                        tag_matrix[index][e_i] = '{}-{}'.format('I', entity['type']).upper()
    
    # 投票并输出
    output_res = []
    equal_sign = True  # 多个模型输出是否相同，用来输出时过滤各模型预测相同的
    for i in range(len(sentence)):
        token_tags = [x[i] for x in tag_matrix]
        voted_res = vote_token_one_tag(token_tags, vote_threshold)
        ent_score = calc_ent(token_tags)
        if ent_score != 0.0:
            equal_sign = False

        output_row = [''.join(sentence), sentence[i]]
        output_row.extend(token_tags)
        output_row.append(str(ent_score))
        output_row.append(voted_res)

        output_res.append(output_row)

    return output_res, equal_sign


if __name__ == '__main__':
    cand_ners = [('model_1', [{'index': [1, 2, 3], 'type': 'per'}]),
                 ('model_2', [{'index': [1, 2], 'type': 'per'}])]
    assert vote_token_BIO(cand_ners, '这是一个测试样本', 'per', 2)[0]['index'] == [1, 2]
    assert vote_entity(cand_ners, [['model_1', 0.9], ['model_2', 1.0]])[0]['index'] == [1, 2]

    print(calc_ent(['B-per', 'B-per', 'O', 'I-per']))
