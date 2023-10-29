"""
{"sentence": ["帝", "悅", "，", "以", "資", "博", "練", "，", "帝", "敕", "東", "宮", "儀", "典", "簿", "最", "悉", "聽", "淹", "裁", "訂", "。"],
 "entity": [{"text": ["淹"], "index": [18], "type": "per"}, {"text": ["淹"], "index": [18], "type": "per"}]
 }
 帝悅，以資博練，{帝|OFI}敕東宮儀典簿最悉聽{淹|PER}裁訂。

 """

import json

d = {'per':"PER",'ofi': 'OFI','book': "BOOK"}

f = open("jicheng_plw/5-16/vote_token_05-16-16-33/predict_result.txt", 'r',encoding="utf-8")
datas = json.load(f)
f.close()
write = open("jicheng_plw/",'w',encoding='utf-8')


for data in datas:
    sentence = data['sentence']
    entity = data['entity']
    ids = []
    tags = []
    for en in entity:
        id = en['index']
        type = en['type']
        st = id[0]
        end = id[-1]
        flag = 0
        for span in ids:
            l = span[0]
            r = span[1]-1
            if st<l and end<l or st>r and end>r:
                flag+=1
        if flag == len(ids):
            ids.append([st,end+1])
            tag = '{' + ''.join(sentence[st:end+1]) + '|' + d[type] + '}'
            tags.append(tag)
    newline = sentence.copy()
    for i in range(len(ids)):
        start = ids[i][0]
        end = ids[i][1]
        if start == end-1:
            newline.insert(start,tags[i])
            del newline[start+1]
        else:
            newline.insert(start,tags[i])
            del newline[start+1:end+1]
            changdu = end-start-1
            j = start+1
            for _ in range(changdu):
                newline.insert(j,'#')
                j+=1
    assert len(newline)==len(sentence)
    for i in range(len(newline) - 1, -1,-1):
        if newline[i] == '#':
            newline.remove('#')
    newline=''.join(newline)
    write.writelines(newline + '\n')
    print(newline)
write.close()