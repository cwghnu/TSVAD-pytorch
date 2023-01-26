import os
import numpy as np
import torch
from pyannote.core import Annotation, Segment

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestCentroid

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from asvtorch.src.ivector.ivector_extractor import IVectorExtractor
from asvtorch.src.ivector.gmm import DiagGmm, Gmm
from asvtorch.src.backend.vector_processing import VectorProcessor
from asvtorch.src.backend.plda import Plda

sys.path.append(os.path.dirname(__file__))
from kaldi_data import KaldiData
from feature import extract_mfcc, exclude_overlaping
from ivector_feat import extract_ivector_from_mfcc

def load_models(config):
    
    device = torch.device(config["device"])
    
    ubm = Gmm.from_kaldi(config["ubm_path"], device)
    diag_ubm = DiagGmm.from_full_gmm(ubm, device)
    
    ivector_extractor = IVectorExtractor.from_npz_file(config["ivec_path"], device)
    
    vector_processor = VectorProcessor.load(config["vec_processor_path"], device)
    
    plda = Plda.load(config["plda_path"], device)
    
    return ubm, diag_ubm, ivec_extractor, vec_processor, plda

def extract_utt_ivector_by_rttm(config):
    mfcc_config = config['mfcc_config']
    device = torch.device(config["device"])
    ubm, diag_ubm, ivec_extractor, vec_processor, plda = load_models(config)
    
    kaldi_obj = KaldiData(config["path_dataset"])
    
    for rec_id, wav_path in kaldi_obj.items():
        signal_data, sr = sf.read(wav_path)
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
            if seg['st'] - seg['et'] > 0.2:
                ref_label[Segment(seg['st'], seg['et'])] = speaker_index

        ref_label_no_overlap = exclude_overlaping(ref_label)
        
        mfcc_feat = extract_mfcc(signal_data, mfcc_config['sampling_rate'], num_ceps=mfcc_config['num_ceps'], low_freq=mfcc_config['low_freq'], high_freq=mfcc_config['high_freq'], cmn=mfcc_config['cmn'], delta=mfcc_config['delta'])
        
        ivector_emb = []
        for segment, track, label in ref_label_no_overlap.itertracks(yield_label=True):
            st, et = segment.start, segment.end
            if st - et < 0.2:   # remove short segments
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
                T[rel_start:rel_end, speaker_index] = 1
                lebel_emb.append(speaker_index)
                with torch.no_grad():
                    sub_len = len(mfcc_feat[rel_start:rel_end, :])
                    ivector_chunk = extract_ivector_from_mfcc(mfcc_feat[rel_start:rel_end, :], ivector_extractor, ubm, diag_ubm, sub_sampling=sub_len)
                    ivector_emb.append(ivector_chunk)
        print("ivector shape: ", ivector_chunk.shape)
        ivector_emb = torch.cat(ivector_emb, dim=0)
        ivector_emb_norm = vector_processor.process(ivector_emb)
        
        score_matrix = plda.score_all_vs_all(ivector_emb_norm, ivector_emb_norm, 200)
        score_matrix = score_matrix - np.min(score_matrix)
        score_matrix = score_matrix / np.max(score_matrix)
        
        clustering_label = SpectralClustering(affinity="precomputed", random_state=777, n_clusters=n_speaker).fit_predict(score_matrix)
        clf = NearestCentroid()
        clf.fit(ivector_emb_norm, clustering_label)
        
        print(clf.centroids_)
        
        output_file = os.path.join(config["output_directory"], rec_id+".ivector")
        with open(output_file, 'w') as f_save:
            np.save(f_save, clf.centroids_)
        

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../config/ivector_extractor.json', help='JSON file for configuration')
    parser.add_argument('-p', '--path_dataset', type=str, default=None,
                        help='Directory for datasets')
    parser.add_argument('-o', '--output_directory', type=str, default=None,
                        help='Directory for output ivectors')
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