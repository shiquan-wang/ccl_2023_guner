#!/bin/bash

python pair_wise_model_vote.py

python do_vote.py

python ./data/context_correction/context_wsq.py