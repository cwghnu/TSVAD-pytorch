import os
import numpy as np
import torch
from pyannote.core import Annotation, Segment

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestCentroid
from torchmetrics.functional import pairwise_cosine_similarity

from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization

import soundfile as sf

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(os.path.dirname(__file__))
from kaldi_data import KaldiData
from model.xvector import Xvector
from feature import exclude_overlaping, index_aligned_labels, extract_fbank

def load_model(model_filepath, device):
    nnet = Xvector(in_channels=24)
    checkpoint = torch.load(model_filepath, map_location="cpu")
    nnet.load_state_dict(checkpoint)
    nnet.to(device)
    nnet.eval()

    return nnet

def extract_utt_xvector_by_rttm(config):
    mfcc_config = config['mfcc_config']
    logmel_config = config['logmel_config']
    device = torch.device(config["device"])
    min_seg_length = 0.3

    model_filepath = config['model_filepath']
    model = load_model(model_filepath, device)
    
    kaldi_obj = KaldiData(config["path_dataset"])
    
    for rec_id, wav_path in kaldi_obj.wavs.items():
        # output_file = os.path.join(config["xvec_output_directory"], rec_id + ".npy")
        # if os.path.exists(output_file):
        #     continue

        signal_data, sr = sf.read(wav_path)
        rate = sr
        
        # if len(signal_data.shape) > 1:
        #     signal_data = signal_data[:, 0]
        # signal_data = torch.from_numpy(signal_data[None, :]).float()

        signal_data = torch.from_numpy(signal_data).float() 
        signal_data = signal_data.permute(1,0).contiguous() # [num_channels, num_samples]
            
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

        logmel_feat = extract_fbank(signal_data[0, :], logmel_config['sampling_rate'], num_mel_bins=logmel_config['num_mel_bins'], low_freq=logmel_config['low_freq'], high_freq=logmel_config['high_freq'])
        # print("logmel shape: {}".format(logmel_feat.shape))
        # output_file = os.path.join(config["logmel_output_directory"], rec_id)
        # np.save(output_file, logmel_feat)
        label_array = np.zeros((len(logmel_feat), n_speaker), dtype=np.int32)
        start = 0
        end = logmel_feat.shape[0]
        frame_shift = 0.01 * sr
        for segment, track, label in ref_label.itertracks(yield_label=True):
            st, et = segment.start, segment.end
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
        output_file = os.path.join(config["label_output_directory"], rec_id)
        np.save(output_file, label_array)
        print("label shape: {}".format(label_array.shape))

        mfcc_feat = Fbank(n_mels=24)(signal_data)
        mfcc_feat = mfcc_feat.to(device)
        norm = InputNormalization(norm_type="sentence", std_norm=False)
        mfcc_feat = norm(mfcc_feat, torch.ones(mfcc_feat.shape[0]))  # [num_channels, num_frames, num_feat]

        mfcc_feat = mfcc_feat[:, :logmel_feat.shape[0], :]
        output_file = os.path.join(config["mfcc_output_directory"], rec_id)
        print("mfcc shape: {}".format(mfcc_feat.shape))
        mfcc_feat = mfcc_feat.cpu().numpy()
        assert mfcc_feat.shape[1] == logmel_feat.shape[0]
        np.save(output_file, mfcc_feat)
        mfcc_feat = torch.from_numpy(mfcc_feat).to(device)
        
        xvector_emb = []
        label_ref = []
        end = mfcc_feat.shape[1]
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
                    try:
                        xvector_seg = model(mfcc_feat[:, rel_start:rel_end, :])
                        xvector_seg = torch.mean(xvector_seg, dim=0)
                        xvector_emb.append(xvector_seg)
                        label_ref.append(speaker_index)
                    except:
                        print(mfcc_feat.shape)
                        print(start_frame, end_frame)
                        print(rel_start, rel_end)
                        print(mfcc_feat[:, rel_start:rel_end, :].shape)
                        continue
        xvector_emb = torch.cat(xvector_emb, dim=0)
        label_ref = np.array(label_ref)

        xvector_emb = xvector_emb.detach().cpu().numpy()

        centroids = np.zeros((n_speaker, xvector_emb.shape[-1]))

        for i in range(n_speaker):
            centroids[i, :] = np.mean(xvector_emb[label_ref==i], axis=0)
        
        output_file = os.path.join(config["xvec_output_directory"], rec_id)
        print("rec_id: {}, x-vector shape: {}".format(rec_id, centroids.shape))
        np.save(output_file, centroids)
        

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/export/home2/cwguang/code/TSVAD-pytorch/config/xvector_extractor.json', help='JSON file for configuration', required=False)
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)

    try:
        if not os.path.exists(config['mfcc_output_directory']):
            os.makedirs(config['mfcc_output_directory'])
        if not os.path.exists(config['logmel_output_directory']):
            os.makedirs(config['logmel_output_directory'])
        if not os.path.exists(config['xvec_output_directory']):
            os.makedirs(config['xvec_output_directory'])
        if not os.path.exists(config['label_output_directory']):
            os.makedirs(config['label_output_directory'])
    except:
        pass
    
    extract_utt_xvector_by_rttm(config)