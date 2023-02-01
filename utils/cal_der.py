import os
import sys
import torch
from collections import Counter
import numpy as np

from hyperpyyaml import load_hyperpyyaml

sys.path.insert(0,os.getcwd())
from utils.diarization import DER, get_oracle_num_spkrs_from_uttrttm

def load_rttm(file_name):
    rttm_content =[]
    with open(file_name, 'r') as fin:
        for line in fin:
            rttm_content.append(line.rstrip())
    return rttm_content

def cal_der(pre_rttm_dir, ref_rttm_dir):

    # ref_rttm_dir = "/exhome1/weiguang/data/M2MET/Eval_Ali_far/rttm_dir"
    # pre_rttm_dir = "/exhome1/weiguang/code/TSVAD-pytorch/exp/hyp_rttm"

    list_result_dict = []
    num_spkrs_list = []
    for hyp_rttm_name in os.listdir(pre_rttm_dir):
        if hyp_rttm_name.endswith('.rttm'):
            rec_id = hyp_rttm_name.split('.rttm')[0]

            ref_rttm_content = load_rttm(os.path.join(ref_rttm_dir, rec_id+'.rttm'))
            hyp_rttm_content = load_rttm(os.path.join(pre_rttm_dir, rec_id+'.rttm'))

            ref_num_spkrs = get_oracle_num_spkrs_from_uttrttm(ref_rttm_content)

            der_result = DER(ref_rttm_content, hyp_rttm_content)

            list_result_dict.append(der_result)
            num_spkrs_list.append(ref_num_spkrs)

    # index 0: avg results
    # index 1: info of 2 speakers
    # index 2: info of 3 speakers
    # ...
    avg_result_dict = [{}] * 4  
    num_spkrs_utts = [0]*4
    for utt_dict, num_spkrs in zip(list_result_dict, num_spkrs_list):
        avg_result_dict[0] = dict(Counter(avg_result_dict[0]) + Counter(utt_dict))
        avg_result_dict[num_spkrs - 1] = dict(Counter(avg_result_dict[num_spkrs - 1]) + Counter(utt_dict))

        num_spkrs_utts[0] += 1
        num_spkrs_utts[num_spkrs - 1] += 1

    for utt_dict, num_utts in zip(avg_result_dict, num_spkrs_utts):
        for key in utt_dict.keys():
            if 'rate' in key:
                utt_dict[key] /= num_utts
            elif 'total' in key:
                continue
            else:
                utt_dict[key] /= utt_dict['total']

        print(utt_dict)

