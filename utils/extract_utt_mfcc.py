import os
import numpy as np
import torch
from pyannote.core import Annotation, Segment

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestCentroid

import soundfile as sf

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from asvtorch.src.ivector.ivector_extractor import IVectorExtractor
from asvtorch.src.ivector.gmm import DiagGmm, Gmm
from asvtorch.src.backend.vector_processing import VectorProcessor
from asvtorch.src.backend.plda import Plda

sys.path.append(os.path.dirname(__file__))
from kaldi_data import KaldiData
from feature import extract_mfcc


def extract_utt_mfcc(config):
    mfcc_config = config['mfcc_config']
    
    kaldi_obj = KaldiData(config["path_dataset"])
    
    for rec_id, wav_path in kaldi_obj.wavs.items():
        signal_data, sr = sf.read(wav_path)
        if len(signal_data.shape) > 1:
            signal_data = signal_data[:, 0]
        
        mfcc_feat = extract_mfcc(signal_data, mfcc_config['sampling_rate'], num_ceps=mfcc_config['num_ceps'], low_freq=mfcc_config['low_freq'], high_freq=mfcc_config['high_freq'], cmn=mfcc_config['cmn'], delta=mfcc_config['delta'])
        
        output_file = os.path.join(config["output_directory"], rec_id)
        np.save(output_file, mfcc_feat)
        

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/exhome1/weiguang/code/TSVAD-pytorch/config/ivector_extractor.json', help='JSON file for configuration')
    parser.add_argument('-p', '--path_dataset', type=str, default='/exhome1/weiguang/data/M2MET/Test_Ali_far', help='Directory for datasets')
    parser.add_argument('-o', '--output_directory', type=str, default='/exhome1/weiguang/data/M2MET/Test_Ali_far/mfcc', help='Directory for output ivectors')
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    
    if args.path_dataset is None:
        print("Please enter the dataset path!")
        sys.exit()
    if args.output_directory is None:
        print("Plase enter the directory to store ivectors!")
        sys.exit()
    
    config["path_dataset"] = args.path_dataset
    config["output_directory"] = args.output_directory
    
    extract_utt_mfcc(config)