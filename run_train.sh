#!/bin/bash

BERT_NAME[0]=personal_models/jian_bert_wwm_ext_1

SUB_OUTPATH[0]=output/all_jian_bert_wwm_ext_1/no_trick_test/

DEVICE=0

python3 main.py --bert_name=${BERT_NAME[0]} --sub_outpath=${SUB_OUTPATH[0]} --device=${DEVICE}
