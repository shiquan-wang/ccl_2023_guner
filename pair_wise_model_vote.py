#!coding=utf-8
import json
import codecs

class PredictObject:
    def __init__(self):
        file_name = None
        sentence_label = {}
        sentence_entity_map = {}
        score = 0.0

entity_f1_boost_gradient_map = {}

def constructObject(file_name, score):
    object = PredictObject()
    object.in_file = file_name
    object.score = score
    f = open(object.in_file,'r',encoding='utf-8')
    jo = json.load(f)
    sentence_entity_map = {}
    sentence_label_map = {}
    for d in jo:
        sentence = "".join(d["sentence"])
        labels = ["O" for i in range(0, len(sentence))]
        for a in d["entity"]:
            index = a["index"]
            index_str = ",".join([str(x) for x in a["index"]])
            text = "".join(a["text"])
            type = a["type"]
            entity = "{}$#${}".format(text, index_str)
            sentence_entity_map["{}^#^{}".format(sentence, entity)] = type
            if len(index) == 1:
                labels[index[0]] = "S_" + type
            else:
                labels[index[0]] = "B_" + type
                for i in range(1, len(index) - 1):
                    labels[index[i]] = "M_" + type
                labels[index[len(index) - 1]] = "E_" + type
        sentence_label_map[sentence] = labels
    object.sentence_entity_map = sentence_entity_map
    object.sentence_label_map = sentence_label_map
    return object

"""
得到预测的实体列表
"""
def updateSentenceLabelDistribution(a, sentence_entity_label_distribution):
    sentence_map_a = a.sentence_entity_map
    """计算b相比a的增加，或者a->b的变换"""
    for k, v in sentence_map_a.items():
        if k not in sentence_entity_label_distribution:
            sentence_entity_label_distribution[k] = {}
        if v not in sentence_entity_label_distribution[k]:
            sentence_entity_label_distribution[k][v] = 0
        sentence_entity_label_distribution[k][v] = sentence_entity_label_distribution[k][v] + 1


def getChangeEntitiesFromAToB(a, b):
    sentence_map_a = a.sentence_entity_map
    sentence_map_b = b.sentence_entity_map

    result = []
    """计算b相比a的增加，或者a->b的变换"""
    for k, v in sentence_map_b.items():
        if k not in sentence_map_a:
            sentence_b = k.split("^#^")[0]
            entity_b = k.split("^#^")[1].split("$#$")[0]
            entity_b_index_list = [int(x) for x in k.split("^#^")[1].split("$#$")[1].split(",")]
            new_flag = True
            sentece_a_labels = a.sentence_label_map[sentence_b]
            for ind in entity_b_index_list:
                if sentece_a_labels[ind] != "O":
                    ##如果a对应的区域索引不是O
                    new_flag = False
                    break
            if new_flag:
                result.append("add:={}$#${}=>{}$#${}".format(k, "O", k, v))
            else:
                entity_a_label = []
                for ind in entity_b_index_list:
                    entity_a_label.append(sentece_a_labels[ind])
                a_key = "{}^#^{}$#${}".format(sentence_b, entity_b, ",".join([str(x) for x in entity_b_index_list]))
                result.append("change:={}$#${}=>{}$#${}".format(a_key, ",".join(entity_a_label), k, v))
        else:
            if sentence_map_a[k] != sentence_map_b[k]:
                result.append("change:={}$#${}=>{}$#${}".format(k, sentence_map_a[k], k, sentence_map_b[k]))

    """计算b相比a的减少，或者a->b的变换"""
    for k, v in sentence_map_a.items():
        if k not in sentence_map_b:
            sentence_a = k.split("^#^")[0]
            entity_a = k.split("^#^")[1].split("$#$")[0]
            entity_a_index_list = [int(x) for x in k.split("^#^")[1].split("$#$")[1].split(",")]
            new_flag = True
            sentece_b_labels = b.sentence_label_map[sentence_a]
            for ind in entity_a_index_list:
                if sentece_b_labels[ind] != "O":
                    ##如果a对应的区域索引不是O
                    new_flag = False
                    break
            if new_flag:
                result.append("delete:={}$#${}=>{}$#${}".format(k, v, k, "O"))
            else:
                entity_b_label = []
                for ind in entity_a_index_list:
                    entity_b_label.append(sentece_b_labels[ind])
                b_key = "{}^#^{}$#${}".format(sentence_a, entity_a, ",".join([str(x) for x in entity_a_index_list]))
                result.append("change:={}$#${}=>{}$#${}".format(k, v, b_key, ",".join(entity_b_label)))
    return result


def getChangeEntitiesFromAToBUniderection (a, b):
    sentence_map_a = a.sentence_entity_map
    sentence_map_b = b.sentence_entity_map

    result = []
    """计算b相比a的增加，或者a->b的变换"""
    for k, v in sentence_map_b.items():
        if k not in sentence_map_a:
            sentence_b = k.split("^#^")[0]
            entity_b = k.split("^#^")[1].split("$#$")[0]
            entity_b_index_list = [int(x) for x in k.split("^#^")[1].split("$#$")[1].split(",")]
            new_flag = True
            sentece_a_labels = a.sentence_label_map[sentence_b]
            for ind in entity_b_index_list:
                if sentece_a_labels[ind] != "O":
                    ##如果a对应的区域索引不是O
                    new_flag = False
                    break
            if new_flag:
                result.append("add:={}$#${}=>{}$#${}".format(k, "O", k, v))
            else:
                entity_a_label = []
                for ind in entity_b_index_list:
                    entity_a_label.append(sentece_a_labels[ind])
                a_key = "{}^#^{}$#${}".format(sentence_b, entity_b, ",".join([str(x) for x in entity_b_index_list]))
                result.append("change:={}$#${}=>{}$#${}".format(a_key, ",".join(entity_a_label), k, v))
        else:
            if sentence_map_a[k] != sentence_map_b[k]:
                result.append("change:={}$#${}=>{}$#${}".format(k, sentence_map_a[k], k, sentence_map_b[k]))

    """计算b相比a的减少，或者a->b的变换"""
    for k, v in sentence_map_a.items():
        if k not in sentence_map_b:
            sentence_a = k.split("^#^")[0]
            entity_a = k.split("^#^")[1].split("$#$")[0]
            entity_a_index_list = [int(x) for x in k.split("^#^")[1].split("$#$")[1].split(",")]
            new_flag = True
            sentece_b_labels = b.sentence_label_map[sentence_a]
            for ind in entity_a_index_list:
                if sentece_b_labels[ind] != "O":
                    ##如果a对应的区域索引不是O
                    new_flag = False
                    break
            if new_flag:
                result.append("delete:={}$#${}=>{}$#${}".format(k, v, k, "O"))
            else:
                None
    return result

"""传入两个object a和b，要求a的score大于b的score，
   计算a和b的diff，
   b->a的变化，导致delta(a.score - b.score)
   a->b的变化，导致delta(b.score - a.score)
   """
def comparePredictObject(a, b, variable_weights):
    a_to_b_change_entity = getChangeEntitiesFromAToB(a, b)
    for x in a_to_b_change_entity:
        if x in variable_weights:
            variable_weights[x] = variable_weights[x] - (a.score - b.score)
        else:
            variable_weights[x] = - (a.score - b.score)

    b_to_a_change_entity = getChangeEntitiesFromAToB(b, a)
    for x in b_to_a_change_entity:
        if x in variable_weights:
            variable_weights[x] = variable_weights[x] + (a.score - b.score)
        else:
            variable_weights[x] =  (a.score - b.score)

"""传入两个object a和b，要求a的score大于b的score，
   计算a和b的diff，
   b->a的变化，导致delta(+1)
   a->b的变化，导致delta(+1)
   """
def comparePredictObjectCounts(a, b, variable_change_counts):
    a_to_b_change_entity = getChangeEntitiesFromAToBUniderection(a, b)
    for x in a_to_b_change_entity:
        if x in variable_change_counts:
            variable_change_counts[x] = variable_change_counts[x] + 1
        else:
            variable_change_counts[x] = 1
    #
    # b_to_a_change_entity = getChangeEntitiesFromAToB(b, a)
    # for x in b_to_a_change_entity:
    #     if x in variable_change_counts:
    #         variable_change_counts[x] = variable_change_counts[x] + 1
    #     else:
    #         variable_change_counts[x] = 1

def constructCompareGroupUsingConfig(config, using_f1):
    group = []
    with codecs.open(config, "r", "utf-8") as in_stream:
        tmp = []
        for line in in_stream.readlines():
            if using_f1:
                if len(line.strip()) != 0:
                    units = line.strip().split("=")
                    obj = constructObject(units[0], float(units[1]))
                    tmp.append(obj)
                else:
                    group.append(tmp)
                    tmp = []
            else:
                if len(line.strip()) != 0:
                    units = line.strip().split("=")
                    obj = constructObject(units[0], 0.0)
                    tmp.append(obj)
                else:
                    group.append(tmp)
                    tmp = []
    return group



if __name__ == "__main__":
    config_file = "compare/config_vote_ensamble"
    using_f1 = False
    if using_f1 == True:
        groups = constructCompareGroupUsingConfig(config_file, using_f1)
        variable_weights = {}
        for group in groups:
            new_group = sorted(group, key=lambda x : x.score, reverse=True)
            for i in range(0, len(new_group)):
                for j in range(i + 1, len(new_group)):
                    comparePredictObject(new_group[i], new_group[j], variable_weights)

        sorted_score = sorted(variable_weights.items(), key=lambda x:x[1], reverse=True)

        out_file = "./compare/{}.txt".format(config_file.split("/")[1])
        with codecs.open(out_file, "w", "utf-8") as out_stream:
            for x in sorted_score:
                out_stream.write("{}%%{}".format(x[0], x[1]) + "\n")
    else:
        groups = constructCompareGroupUsingConfig(config_file, using_f1)
        variable_change_counts = {}
        for group in groups:
            new_group = sorted(group, key=lambda x : x.score, reverse=True)
            for i in range(0, len(new_group)):
                for j in range(i + 1, len(new_group)):
                    comparePredictObjectCounts(new_group[i], new_group[j], variable_change_counts)

        sorted_score = sorted(variable_change_counts.items(), key=lambda x:x[1], reverse=True)
        out_file = "./compare/{}_counts.txt".format(config_file.split("/")[1])
        candidate_set = {}
        entity_sets = set([])

        with codecs.open(out_file, "w", "utf-8") as out_stream:
            for x in sorted_score:
                modify_area = x[0].split(":=")[1].split("=>")[0]
                sentence = modify_area.split("^#^")[0]
                entity, index, label = modify_area.split("^#^")[1].split("$#$")
                key = "{}^#^{}^#^{}".format(sentence, entity, index)
                value = label
                entity_sets.add(entity)
                if key not in candidate_set:
                    candidate_set[key] = []
                candidate_set[key].append(value)
            for k,v in candidate_set.items():
                out_stream.write("{}=>{}\n".format(k,v))
