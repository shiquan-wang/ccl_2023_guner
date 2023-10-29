#!coding-utf-8
import os
import json
import logging
from collections import defaultdict, deque, Counter
from utils import convert_index_to_text, cal_f1, vote_token_BIO, vote_entity, \
    vote_token_BIO_analysis

class Ensemble():
    def __init__(self, multi_model_predict_files, target_file):
        """
        计算各label对应的最优model ranking
        """
        self.label_set = set(['per', 'ofi', 'book']) 
        return
    
        # vote_entity需要依赖dev集上结果时打开
        all_model_label_score = {}
        all_model_overall_score = {}
        for model_name, model_directory, dev_file, test_file in multi_model_predict_files:
            label_f1, overall_score = self.get_single_model_score(
                'output/' + model_directory + '/' + dev_file, target_file)
            all_model_label_score[model_name] = label_f1
            all_model_overall_score[model_name] = overall_score

        # 手动修改f1
        # all_model_label_score['bert-base-chinese'] = {"per": 0.9158, "ofi": 0.8453, 'book': 0.8837}
        # all_model_label_score['jian-bert-wwm-ext'] = {"per": 0.9512, "ofi": 0.8698, 'book': 0.8}
        # all_model_label_score['jian-chinese_wwm_ext_pytorch'] = {"per": 0.9293, "ofi": 0.8432, 'book': 0.8696}
        # all_model_label_score['jian-new-macbert-large'] = {"per": 0.9283, "ofi": 0.8536, 'book': 0.8837}
        # all_model_label_score['yuan-plw'] = {"per": 0.9353, "ofi": 0.862, 'book': 0.8444}

        # all_model_overall_score['bert-base-chinese']['f1'] = 0.8891
        # all_model_overall_score['jian-bert-wwm-ext']['f1'] = 0.922
        # all_model_overall_score['jian-chinese_wwm_ext_pytorch']['f1'] = 0.9003
        # all_model_overall_score['jian-new-macbert-large']['f1'] = 0.9037
        # all_model_overall_score['yuan-plw']['f1'] = 0.9101

        # 计算每种实体score最高的model
        self.label_set = set() 
        self.label_choose_model = {}
        self.get_label_best_model(all_model_label_score)

        # logging
        label_list = list(self.label_set)
        table = pt.PrettyTable(['model score on dev', 'overall_f1'] + label_list)
        for model_name, _, _, _ in multi_model_predict_files:
            table.add_row([model_name] +
                          ["{:3.6f}".format(all_model_overall_score[model_name]['f1'])] +
                          ["{:3.6f}".format(all_model_label_score[model_name][x]) for x in label_list])
        logging.info('\n{}'.format(table))
        table = pt.PrettyTable(['label choose model', 'model', 'f1'])
        for label in label_list:
            table.add_row([label,
                           self.label_choose_model[label][0][0],
                           "{:3.6f}".format(self.label_choose_model[label][0][1])])
            for i in range(1, len(self.label_choose_model[label])):
                table.add_row(['',
                               self.label_choose_model[label][i][0],
                               "{:3.6f}".format(self.label_choose_model[label][i][1])])
        logging.info('\n{}'.format(table))
        
    def load_predict_file(self, predict_file):
        with open(predict_file, 'r', encoding='utf-8') as fr:
            data = json.load(fr)
        all_row_entity_text = []
        for row in data:
            row_entity_text = [convert_index_to_text(entity['index'], entity['type'].lower()) \
                               for entity in row['entity']]
            all_row_entity_text.append(set(row_entity_text))
        return all_row_entity_text
    
    def load_target_file(self, predict_file):
        """dev集实体格式有点不一样"""
        with open(predict_file, 'r', encoding='utf-8') as fr:
            data = json.load(fr)
        all_row_entity_text = []
        for row in data:
            row_entity_text = [convert_index_to_text(entity['index'], entity['type'].lower()) \
                               for entity in row['ner']]
            all_row_entity_text.append(set(row_entity_text))
        return all_row_entity_text
    
    def get_single_model_score(self, predict_file, target_file):
        """label score计算方式保持和训练时一致"""
        predict_entities = self.load_predict_file(predict_file)
        target_entities = self.load_target_file(target_file)

        origin_counter = Counter()
        pred_counter = Counter()
        right_counter = Counter()

        for predict_row, target_row in zip(predict_entities, target_entities):
            pred_counter.update([x.split('-#-')[1] for x in predict_row])
            origin_counter.update([x.split('-#-')[1] for x in target_row])
            right_counter.update([x.split('-#-')[1] for x in predict_row.intersection(target_row)])

        # 计算每种类型的指标
        label_score = {}
        for label, count in origin_counter.items():
            origin = count
            found = pred_counter.get(label, 0)
            right = right_counter.get(label, 0)
            f1, precision, recall = cal_f1(right, found, origin)
            label_score[label] = f1
        f1, precision, recall = cal_f1(sum(right_counter.values()), sum(pred_counter.values()), sum(origin_counter.values()))
        overall_score = {'f1': f1, 'acc': precision, 'recall': recall}
        return label_score, overall_score
    
    def get_single_label_sorted_model(self, all_model_label_score, target_label):
        all_model = []
        for model_file, label_score in all_model_label_score.items():
            if target_label in label_score:
                value = label_score[target_label]
                all_model.append([model_file, value])
        sorted_model = sorted(all_model, key=lambda x: x[1], reverse=True)
        return sorted_model
        
    def get_label_best_model(self, all_model_label_score):
        for model_file, label_score in all_model_label_score.items():  # 先收集所有label
            for label, score in label_score.items():
                self.label_set.add(label)

        for label in self.label_set:
            sorted_model = self.get_single_label_sorted_model(all_model_label_score, label)
            self.label_choose_model[label] = sorted_model

    def load_predict_entity(self, predict_file):
        with open(predict_file, 'r', encoding='utf-8') as fr:
            data = json.load(fr)
        all_row_entity = []
        all_row_sentences = []
        for row in data:
            # 转换为{entity_type: entity_list}形式
            row_entity = {}
            for entity in row['entity']:
                row_entity.setdefault(entity['type'], [])
                row_entity[entity['type']].append(entity)
            all_row_entity.append(row_entity)
            all_row_sentences.append(row['sentence'])
        return all_row_entity, all_row_sentences
    
    def ensemble_single_row_predict_res(self, 
                                        multi_model_predict, 
                                        i, 
                                        sentence, 
                                        vote_model, 
                                        model_index_mapping,
                                        vote_token_threshold, 
                                        vote_entity_model_num):
        cand_ners = [(k, v[i]) for k, v in multi_model_predict.items()]
        if vote_model == 'vote_token':
            new_ners = vote_token_BIO(cand_ners, sentence, model_index_mapping,  vote_token_threshold)
        elif vote_model == 'vote_entity':
        #         cand_ners = [(k, v[i][label]) for k, v in multi_model_predict.items() \
        #                       if label in v[i]]
            new_ners = vote_entity(cand_ners)
        else:
            raise Exception('vote model: {} not implemented'.format(vote_model))
        return new_ners
    
    def ensemble_test_predict(self, 
                              multi_model_predict_files, 
                              dataset, 
                              vote_model, 
                              vote_token_threshold=2, 
                              vote_entity_model_num=3):
        res = []
        if len(multi_model_predict_files) == 0:
            return []
        multi_model_predict = {}
        data_num = 0
        all_row_sentence = []
        # 集成test集结果or dev集结果
        model_index_mapping = {}
        for i, (model_name, model_directory, dev_file, test_file) in enumerate(multi_model_predict_files):
            model_index_mapping[model_name] = i
            # load file
            ensemble_file = test_file if dataset == 'test' else dev_file
            all_row_entity, all_row_sentence = self.load_predict_entity(
                'output/' + model_directory + '/' + ensemble_file)
            multi_model_predict[model_name] = all_row_entity
            data_num = len(multi_model_predict[model_name])
        
        for i in range(data_num):
            new_predict = self.ensemble_single_row_predict_res(
                multi_model_predict, i, all_row_sentence[i], vote_model, model_index_mapping, 
                vote_token_threshold, vote_entity_model_num)
            res.append({
                'sentence': all_row_sentence[i],
                'entity': new_predict,
            })
        return res

    def ensemble_single_row_predict_res_analysis(self, 
                                                 multi_model_predict, 
                                                 i, 
                                                 sentence, 
                                                 vote_model, 
                                                 model_index_mapping,
                                                 vote_token_threshold, 
                                                 vote_entity_model_num):
        cand_ners = [(k, v[i]) for k, v in multi_model_predict.items()]
        if vote_model == 'vote_token':
            sentence_output, equal_sign = vote_token_BIO_analysis(
                cand_ners, sentence, model_index_mapping, vote_token_threshold)
        # elif vote_model == 'vote_entity':
        #     label_new_ner = vote_entity(cand_ners, 
        #                                 self.label_choose_model[label][: vote_entity_model_num])
        else:
            raise Exception('vote model: {} not implemented'.format(vote_model))
        return sentence_output, equal_sign
    
    def ensemble_test_predict_analysis(self,
                                       multi_model_predict_files, 
                                       dataset, 
                                       vote_model, 
                                       vote_token_threshold=2, 
                                       vote_entity_model_num=3):
        res = []
        if len(multi_model_predict_files) == 0:
            return []
        multi_model_predict = {}
        data_num = 0
        all_row_sentence = []
        # 集成test集结果or dev集结果
        model_index_mapping = {}  # 方便保证输出事model顺序一致
        title_row = ['原文', 'token']  # 输出的第一列标题栏
        for i, (model_name, model_directory, dev_file, test_file) in enumerate(multi_model_predict_files):
            model_index_mapping[model_name] = i
            title_row.append(model_name)
            # load file
            ensemble_file = test_file if dataset == 'test' else dev_file
            all_row_entity, all_row_sentence = self.load_predict_entity(
                'output/' + model_directory + '/' + ensemble_file)
            multi_model_predict[model_name] = all_row_entity
            data_num = len(multi_model_predict[model_name])
        
        title_row.extend(['不一致率(熵)', '投票结果'])
        res.append(title_row)
        for i in range(data_num):
            sentence_output, equal_sign = self.ensemble_single_row_predict_res_analysis(
                multi_model_predict, i, all_row_sentence[i], vote_model, model_index_mapping, 
                vote_token_threshold, vote_entity_model_num)
            if not equal_sign:
                res.extend(sentence_output)
        return res    


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    # 选取最好的几个
    # 各个模型格式：模型名，模型目录名，目录下dev结果文件名，目录下test集结果文件名
    all_models = [
        # # top model
        # ['jian_bert_wwm_ext_2/199', 'jian_bert_wwm_ext_2', '199_dev_result.txt', '199_predict_result.txt'],
        # ['jian_bert_wwm_ext_1/149', 'jian_bert_wwm_ext_1', '149_dev_result.txt', '149_predict_result.txt'],
        # ['bert_wwm_ext_1/249', 'bert_wwm_ext_1', '249_dev_result.txt', '249_predict_result.txt'],
        # ['jian-new-bert-wwm-ext', 'jian-new-bert-wwm-ext', 'dev_result.txt', 'predict_result.txt'],
        # ['jian-bert-wwm-ext/249', 'jian-bert-wwm-ext', '249_dev_result.txt', '249_predict_result.txt'],
        # ['jian-new-macbert-large', 'jian-new-macbert-large', 'dev_result.txt', 'predict_result.txt'],
        # ['macbert-large', 'macbert-large', 'dev_result.txt', 'predict_result.txt'],
        # # 追加更多model
        # ['jian_bert_wwm_ext_2/149', 'jian_bert_wwm_ext_2', '149_dev_result.txt', '149_predict_result.txt'],
        # ['jian_bert_wwm_ext_2', 'jian_bert_wwm_ext_2', 'dev_result.txt', 'predict_result.txt'],
        # ['jian_bert_wwm_ext_1/249', 'jian_bert_wwm_ext_1', '249_dev_result.txt', '249_predict_result.txt'],
        # ['jian_bert_wwm_ext_1', 'jian_bert_wwm_ext_1', 'dev_result.txt', 'predict_result.txt'],
        # ['jian-chinese_wwm_ext_pytorch', 'jian-chinese_wwm_ext_pytorch', 'dev_result.txt', 'predict_result.txt'],
        # ['jian-bert-wwm-ext/199', 'jian-bert-wwm-ext', '199_dev_result.txt', '199_predict_result.txt'],
        # ['jian-bert-wwm-ext', 'jian-bert-wwm-ext', 'dev_result.txt', 'predict_result.txt'],
        # ['bert_wwm_ext_1', 'bert_wwm_ext_1', 'dev_result.txt', 'predict_result.txt'],
        # f1在93以上的模型进行token集成
        # ['all-jian_bert_wwm_ext_1/70', 'no_dev/all-jian_bert_wwm_ext_1', '', '70_predict_result.txt'],
        # ['all-jian_bert_wwm_ext_1/99', 'no_dev/all-jian_bert_wwm_ext_1', '', '99_predict_result.txt'],
        # ['all-jian_bert_wwm_ext_1/149', 'no_dev/all-jian_bert_wwm_ext_1', '', '149_predict_result.txt'],
        # ['all-jian_bert_wwm_ext_1/199', 'no_dev/all-jian_bert_wwm_ext_1', '', '199_predict_result.txt'],
        # ['all-jian_bert_wwm_ext_1/249', 'no_dev/all-jian_bert_wwm_ext_1', '', '249_predict_result.txt'],
        # ['all-jian_bert_wwm_ext_2/99', 'no_dev/all-jian_bert_wwm_ext_2', '', '99_predict_result.txt'],
        # ['all-jian_bert_wwm_ext_2/149', 'no_dev/all-jian_bert_wwm_ext_2', '', '149_predict_result.txt'],
        # ['all-jian_bert_wwm_ext_2/449', 'no_dev/all-jian_bert_wwm_ext_2', '', '449_predict_result.txt'],

        # ['all_jian_bert_wwm_ext_1/no_trick/249', 'all_jian_bert_wwm_ext_1/no_trick', '', '249_predict_result.txt'],
        # ['all_jian_bert_wwm_ext_1/no_trick/299', 'all_jian_bert_wwm_ext_1/no_trick', '', '299_predict_result.txt'],
        # ['to_submit/24_similar_test/200', 'to_submit/24_similar_test/', '', '200predict_result.txt'],
        # f1在94以上的模型
        # 格式：模型名，模型输出output下目录名，dev集结果文件名，test集结果文件名
        ['all_jian_bert_wwm_ext_1/no_trick/249', 'all_jian_bert_wwm_ext_1/no_trick', '', '249_predict_result.txt'],
        ['all_jian_bert_wwm_ext_1/no_trick/299', 'all_jian_bert_wwm_ext_1/no_trick', '', '299_predict_result.txt'],
        ['all_jian_bert_wwm_ext_1_fry_hyperp/200', 'all_jian_bert_wwm_ext_1_fry_hyperp', '', '200_predict_result.txt'],
        ['all_jian_bert_wwm_ext_1_fry_hyperp/300', 'all_jian_bert_wwm_ext_1_fry_hyperp', '', '300_predict_result.txt'],
        ['all_jian_bert_wwm_ext_1_fry_hyperp/400', 'all_jian_bert_wwm_ext_1_fry_hyperp', '', '400_predict_result.txt'],
        ['all_jian_bert_wwm_ext_1_fry_hyperp/500', 'all_jian_bert_wwm_ext_1_fry_hyperp', '', '500_predict_result.txt'],

    ]
    dev_target_file = 'data/bisai/dev.json'
    vote_model = 'vote_token'
    # vote_model = 'vote_entity'
    # vote_token_threshold = 0
    # vote_entity_model_num = 15
    # output_directory = 'ensemble/{}_05-19-12-13'.format(vote_model)

    vote_token_threshold = 2  # vote_token模式参数，投票时非'O'结果最低票数
    vote_entity_model_num = 15  # vote_entity模式参数，让每个lable对应ranking top几的模型纳入投票
    output_directory = 'ensemble/{}_05-18-16-00'.format(vote_model)

    if not os.path.exists('output/' + output_directory):
        os.makedirs('output/' + output_directory)

    ensembler = Ensemble(all_models, dev_target_file)

    # 集成dev集，方便获得集成后dev上指标
    # ensembled_predict = ensembler.ensemble_test_predict(all_models, 'dev', vote_model, 
    #                                                     vote_token_threshold, vote_entity_model_num)  
    # with open('output/{}/dev_result.txt'.format(output_directory), 'w', encoding='utf-8') as fw:
    #     json.dump(ensembled_predict, fw, ensure_ascii=False)
    # # 计算集成后在dev上的f1
    # label_f1, overall_score = ensembler.get_single_model_score(
    #     'output/{}/dev_result.txt'.format(output_directory), dev_target_file)
    
    # logging.info('ensembled dev predict label score: {}'.format(json.dumps(label_f1, ensure_ascii=False)))
    # logging.info('ensembled dev predict overall score: {}'.format(json.dumps(overall_score, ensure_ascii=False)))

    # 集成test集
    ensembled_predict = ensembler.ensemble_test_predict(
        all_models, 'test', vote_model, vote_token_threshold, vote_entity_model_num)
    with open('output/{}/predict_result.txt'.format(output_directory), 'w', encoding='utf-8') as fw:
        json.dump(ensembled_predict, fw, ensure_ascii=False)
    # case分析输出
    analysys_output = ensembler.ensemble_test_predict_analysis(
        all_models, 'test', vote_model, vote_token_threshold, vote_entity_model_num)
    with open('output/{}/analysis_result.txt'.format(output_directory), 'w', encoding='utf-8') as fw:
        for row in analysys_output:
            fw.write('{}\n'.format('\t'.join(row)))