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
from feature import extract_mfcc, exclude_overlaping, index_aligned_labels
from ivector_feat import extract_ivector_from_mfcc

from multiprocessing import Pool

def load_models(config):
    
    device = torch.device(config["device"])
    
    ubm = Gmm.from_kaldi(config["ubm_path"], device)
    diag_ubm = DiagGmm.from_full_gmm(ubm, device)
    
    ivec_extractor = IVectorExtractor.from_npz_file(config["ivec_path"], device)
    
    vec_processor = VectorProcessor.load(config["vec_processor_path"], device)
    
    plda = Plda.load(config["plda_path"], device)
    
    return ubm, diag_ubm, ivec_extractor, vec_processor, plda

def task(rec_id, wav_path, config):
    kaldi_obj = KaldiData(config["path_dataset"])
    mfcc_config = config['mfcc_config']
    device = torch.device(config["device"])
    min_seg_length = 1.0
    
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
        
    mfcc_feat = extract_mfcc(signal_data, mfcc_config['sampling_rate'], num_ceps=mfcc_config['num_ceps'], low_freq=mfcc_config['low_freq'], high_freq=mfcc_config['high_freq'], cmn=mfcc_config['cmn'], delta=mfcc_config['delta'])
    
    label_array = np.zeros((len(mfcc_feat), n_speaker), dtype=np.int32)
    
    start = 0
    end = mfcc_feat.shape[0]
    ivector_emb = []
    label_ref = []
    mfcc_spks = [torch.zeros((0,72)).to(device)]*n_speaker
    frame_shift = 0.01 * sr
    for segment, track, label in ref_label.itertracks(yield_label=True):
        st, et = segment.start, segment.end
        # if et - st < min_seg_length:   # remove short segments
        #     continue
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
                label_array[rel_start:rel_end, speaker_index] = 1
                label_ref.append(speaker_index)

    output_file = os.path.join(config["label_output_directory"], rec_id)
    print(rec_id)
    np.save(output_file, label_array)

def extract_utt_ivector_by_rttm(config):
    
    kaldi_obj = KaldiData(config["path_dataset"])
    
    p = Pool(5)
    
    for rec_id, wav_path in kaldi_obj.wavs.items():
        p.apply_async(task, args=(rec_id, wav_path, config, ))
        
        # task(rec_id, wav_path, config)
        
    p.close()
    p.join()
        

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/exhome1/weiguang/code/TSVAD-pytorch/config/ivector_extractor.json', help='JSON file for configuration')
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)

    if not os.path.exists(config['label_output_directory']):
        os.makedirs(config['label_output_directory'])
    
    extract_utt_ivector_by_rttm(config)