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
from feature import extract_mfcc, exclude_overlaping, index_aligned_labels

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
        
        label_array = np.zeros((len(mfcc_feat), n_speaker), dtype=np.int32)

        output_file = os.path.join(config["mfcc_output_directory"], rec_id)
        np.save(output_file, mfcc_feat)
        
        mfcc_feat = mfcc_feat.to(device)
        
        start = 0
        end = mfcc_feat.shape[0]
        mfcc_feat = mfcc_feat.permute(1, 0)
        xvector_emb = []
        label_ref = []
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
                    label_array[rel_start:rel_end, speaker_index] = 1
                    label_ref.append(speaker_index)
        xvector_emb = torch.cat(xvector_emb, dim=0)
        
        score_matrix = pairwise_cosine_similarity(xvector_emb, xvector_emb)
        score_matrix = score_matrix.detach().cpu().numpy()
        score_matrix = score_matrix - np.min(score_matrix)
        score_matrix = score_matrix / np.max(score_matrix)
        
        clustering_label = SpectralClustering(affinity="precomputed", random_state=777, n_clusters=n_speaker).fit_predict(score_matrix)
        clf = NearestCentroid()
        xvector_emb = xvector_emb.detach().cpu().numpy()
        clf.fit(xvector_emb, clustering_label)
        
        print(clf.centroids_.shape)
        print(clf.classes_)
        
        index_aligned = index_aligned_labels(clustering_label, label_ref, n_classes=n_speaker)
        assert len(np.unique(index_aligned)) == n_speaker
        print(index_aligned)
        centroids = clf.centroids_[index_aligned]
        
        output_file = os.path.join(config["output_directory"], rec_id)
        np.save(output_file, centroids)
        

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/exhome1/weiguang/code/TSVAD-pytorch/config/xvector_extractor.json', help='JSON file for configuration')
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    
    extract_utt_ivector_by_rttm(config)