import os
import numpy as np
import torch
from pyannote.core import Annotation, Segment

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestCentroid
from torchmetrics.functional import pairwise_cosine_similarity

import soundfile as sf

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from asvtorch.src.networks.network_io import initialize_net

sys.path.append(os.path.dirname(__file__))
from kaldi_data import KaldiData
from feature import extract_mfcc, exclude_overlaping

def load_model(model_filepath, device):
    loaded_states = torch.load(model_filepath, map_location=device)
    state_dict = loaded_states['model_state_dict']
    key1 = 'feat_dim_param'
    key2 = 'n_speakers_param'
    feat_dim = state_dict[key1].item()
    n_speakers = state_dict[key2].item()
    net = initialize_net(feat_dim, n_speakers)
    net.to(device)
    net.load_state_dict(state_dict)
    net.eval()

    return net

def extract_utt_ivector_by_rttm(config):
    mfcc_config = config['mfcc_config']
    device = torch.device(config["device"])
    min_seg_length = 1.0

    model_filepath = config['model_filepath']
    model = load_model(model_filepath, device)
    
    kaldi_obj = KaldiData(config["path_dataset"])
    
    for rec_id, wav_path in kaldi_obj.wavs.items():
        signal_data, sr = sf.read(wav_path)
        rate = sr
        if len(signal_data.shape) > 1:
            signal_data = signal_data[:, 0]
            
        filtered_segments = kaldi_obj.segments[rec_id]
        speakers = np.unique(
            [kaldi_obj.utt2spk[seg['utt']] for seg
                in filtered_segments]).tolist()
        n_speaker = len(speakers)
        
        ref_label = Annotation()
        for seg in filtered_segments:
            speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
            ref_label[Segment(seg['st'], seg['et'])] = speaker_index

        ref_label_no_overlap = exclude_overlaping(ref_label)
        
        mfcc_feat = extract_mfcc(signal_data, mfcc_config['sampling_rate'], num_ceps=mfcc_config['num_ceps'], low_freq=mfcc_config['low_freq'], high_freq=mfcc_config['high_freq'], cmn=mfcc_config['cmn'], delta=mfcc_config['delta'])
        mfcc_feat = mfcc_feat.to(device)
        
        start = 0
        end = mfcc_feat.shape[0]
        mfcc_feat = mfcc_feat.permute(1, 0)
        xvector_emb = []
        frame_shift = 0.01 * sr
        for segment, track, label in ref_label_no_overlap.itertracks(yield_label=True):
            st, et = segment.start, segment.end
            if et - st < min_seg_length:   # remove short segments
                continue
            speaker_index = label
            start_frame = np.rint(
                    st * rate / frame_shift).astype(int)
            end_frame = np.rint(
                    et * rate / frame_shift).astype(int)
            rel_start = rel_end = None
            if start <= start_frame and start_frame < end:
                rel_start = start_frame - start
            if start < end_frame and end_frame <= end:
                rel_end = end_frame - start
            if rel_start is not None or rel_end is not None:
                with torch.no_grad():
                    xvector_emb.append(model(mfcc_feat[None, :, rel_start:rel_end], 'extract_embeddings'))
        xvector_emb = torch.cat(xvector_emb, dim=0)
        
        score_matrix = pairwise_cosine_similarity(xvector_emb, xvector_emb)
        score_matrix = score_matrix.detach().cpu().numpy()
        score_matrix = score_matrix - np.min(score_matrix)
        score_matrix = score_matrix / np.max(score_matrix)
        
        clustering_label = SpectralClustering(affinity="precomputed", random_state=777, n_clusters=n_speaker).fit_predict(score_matrix)
        clf = NearestCentroid()
        clf.fit(xvector_emb, clustering_label)
        
        print(clf.centroids_.shape)
        
        output_file = os.path.join(config["output_directory"], rec_id)
        np.save(output_file, clf.centroids_)
        

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/exhome1/weiguang/code/TSVAD-pytorch/config/ivector_extractor.json', help='JSON file for configuration')
    parser.add_argument('-p', '--path_dataset', type=str, default='/exhome1/weiguang/data/M2MET/Test_Ali_far', help='Directory for datasets')
    parser.add_argument('-o', '--output_directory', type=str, default='/exhome1/weiguang/data/M2MET/Test_Ali_far/ivec', help='Directory for output ivectors')
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
    
    extract_utt_ivector_by_rttm(config)