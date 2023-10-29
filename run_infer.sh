#!/bin/bash

BERT_NAME[0]=personal_models/jian_bert_wwm_ext_1

SUB_OUTPATH[0]=output/all_jian_bert_wwm_ext_1/no_trick/
SUB_OUTPATH[1]=output/all_jian_bert_wwm_ext_1_fry_hyperp/
SUB_OUTPATH[2]=output/all_jian_bert_wwm_ext_1_fry_hyperp_dropout03/
SUB_OUTPATH[3]=output/all_24_similar_official_train_test_jianti_outputs/

ENSAMBLE_OUTPATH=output/ensamble_output/

DEVICE=0

#python3 main.py --bert_name=${BERT_NAME} --sub_outpath=${SUB_OUTPATH} --device=${DEVICE}

python3 inferent.py --bert_name=${BERT_NAME[0]} --sub_outpath=${SUB_OUTPATH[0]} --model_version=61 --outpath=${ENSAMBLE_OUTPATH};

python3 inferent.py --bert_name=${BERT_NAME[0]} --sub_outpath=${SUB_OUTPATH[0]} --model_version=65 --outpath=${ENSAMBLE_OUTPATH};

python3 inferent.py --bert_name=${BERT_NAME[0]} --sub_outpath=${SUB_OUTPATH[0]} --model_version=249 --outpath=${ENSAMBLE_OUTPATH};

python3 inferent.py --bert_name=${BERT_NAME[0]} --sub_outpath=${SUB_OUTPATH[0]} --model_version=299 --outpath=${ENSAMBLE_OUTPATH};

python3 inferent.py --bert_name=${BERT_NAME[0]} --sub_outpath=${SUB_OUTPATH[1]} --model_version=300 --outpath=${ENSAMBLE_OUTPATH};

python3 inferent.py --bert_name=${BERT_NAME[0]} --sub_outpath=${SUB_OUTPATH[1]} --model_version=500 --outpath=${ENSAMBLE_OUTPATH};

python3 inferent.py --bert_name=${BERT_NAME[0]} --sub_outpath=${SUB_OUTPATH[2]} --model_version=300 --outpath=${ENSAMBLE_OUTPATH};

python3 inferent.py --bert_name=${BERT_NAME[0]} --sub_outpath=${SUB_OUTPATH[3]} --model_version=300 --outpath=${ENSAMBLE_OUTPATH};
