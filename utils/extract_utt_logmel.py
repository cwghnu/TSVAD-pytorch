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
from feature import extract_fbank


def extract_utt_mfcc(config):
    logmel_config = config['logmel_config']
    
    kaldi_obj = KaldiData(config["path_dataset"])
    
    for rec_id, wav_path in kaldi_obj.wavs.items():
        signal_data, sr = sf.read(wav_path)
        if len(signal_data.shape) > 1:
            signal_data = signal_data[:, 0]
        
        logmel_feat = extract_fbank(signal_data, logmel_config['sampling_rate'], num_mel_bins=logmel_config['num_mel_bins'], low_freq=logmel_config['low_freq'], high_freq=logmel_config['high_freq'])
        
        output_file = os.path.join(config["output_directory"], rec_id)
        np.save(output_file, logmel_feat)
        print("Finished: {}".format(rec_id))
        

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/exhome1/weiguang/code/TSVAD-pytorch/config/logmel_extractor.json', help='JSON file for configuration')
    parser.add_argument('-p', '--path_dataset', type=str, default='/exhome1/weiguang/data/Alimeeting/Train_Ali_far', help='Directory for datasets')
    parser.add_argument('-o', '--output_directory', type=str, default='/exhome1/weiguang/data/Alimeeting/Train_Ali_far/logmel', help='Directory for output logmel features')
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    
    if args.path_dataset is None:
        print("Please enter the dataset path!")
        sys.exit()
    if args.output_directory is None:
        print("Plase enter the directory to store logmel features!")
        sys.exit()
    
    config["path_dataset"] = args.path_dataset
    config["output_directory"] = args.output_directory

    if not os.path.exists(config["output_directory"]):
        os.makedirs(config["output_directory"])
    
    # extract_utt_mfcc(config)

    logmel_file = "/exhome1/weiguang/data/Alimeeting/Eval_Ali/Eval_Ali_far/logmel/R8001_M8004_MS801.npy"

    data = np.load(logmel_file)
    print(data.shape)