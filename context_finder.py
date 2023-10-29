#!coding=utf-8
import sys
import os
import codecs
import json
import codecs
def getFantiToJiantiMap(fanti_file, jianti_file):
    f = codecs.open(fanti_file, "r", "utf-8")
    fanti_obj = json.loads(f.read())

    f = codecs.open(jianti_file, "r", "utf-8")
    jianti_obj = json.loads(f.read())

    result = {}
    for fanti_o, jianti_o in zip(fanti_obj, jianti_obj):
        fanti_text = "".join(fanti_o["sentence"])
        jianti_text = "".join(jianti_o["sentence"])
        result[fanti_text] = jianti_text
    return result


def getJiantiFile(target_sentence, jianti_out_file):
    fanti_file = "./data/bisai/predict.json"
    jianti_file = "./data/jian_all/predict.json"
    fanti_to_jianti_map = getFantiToJiantiMap(fanti_file, jianti_file)
    with codecs.open(jianti_out_file, "w", "utf-8") as out_stream:
        for line in  codecs.open(target_sentence, "r", "utf-8").readlines():
            units = line.strip().split("==")
            sentence = units[0]
            out_stream.write(fanti_to_jianti_map[sentence] +"\n")

def getPredictContextJson(jianti_out_file, out_json_file):
    result = []
    with codecs.open(jianti_out_file, "r", "utf-8") as in_stream:
        for line in in_stream.readlines():
            result.append({
                    'sentence': list(line.strip()),
                    'ner': list(),
                })

    with open(out_json_file, 'w', encoding='utf-8') as fw:
        json.dump(result, fw, ensure_ascii=False)

    return result


if __name__ == "__main__":
    None