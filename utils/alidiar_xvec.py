import numpy as np
import torch
import soundfile as sf

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from torchmetrics.functional import pairwise_cosine_similarity

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from asvtorch.src.networks.network_io import initialize_net

sys.path.append(os.path.dirname(__file__))
from kaldi_data import KaldiData
from xvector_feat import extract_mfcc

def dira_recording(kaldi_obj, model, device, recid):

    signal_data, sr = kaldi_obj.load_wav(recid)
    signal_data = signal_data[:, 0]

    utt_len = len(signal_data) / sr

    ref_label = Annotation()
    hyp_label = Annotation()

    mfcc_feat = extract_mfcc(signal_data, sr)
    print("MFCC feature: ", mfcc_feat.shape, type(mfcc_feat))

    mfcc_feat = mfcc_feat.to(device)

    start = 0
    end = mfcc_feat.shape[0]
    filtered_segments = kaldi_obj.segments[recid]
    frame_shift = 0.01 * sr
    rate = sr
    speakers = np.unique(
            [kaldi_obj.utt2spk[seg['utt']] for seg
                in filtered_segments]).tolist()
    n_speaker = len(speakers)
    T = np.zeros((mfcc_feat.shape[0], n_speaker), dtype=np.int32)
    mfcc_feat = mfcc_feat.permute(1, 0)
    xvector_emb = []
    lebel_emb = []
    for seg in filtered_segments:
        speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
        # print("segment : [{}, {}], spk: {}".format(seg['st'], seg['et'], speaker_index))
        ref_label[Segment(seg['st'], seg['et'])] = speaker_index
        start_frame = np.rint(
                seg['st'] * rate / frame_shift).astype(int)
        end_frame = np.rint(
                seg['et'] * rate / frame_shift).astype(int)
        rel_start = rel_end = None
        if start <= start_frame and start_frame < end:
            rel_start = start_frame - start
        if start < end_frame and end_frame <= end:
            rel_end = end_frame - start
        if rel_start is not None or rel_end is not None:
            T[rel_start:rel_end, speaker_index] = 1
            lebel_emb.append(speaker_index)
            with torch.no_grad():
                xvector_emb.append(model(mfcc_feat[None, :, rel_start:rel_end], 'extract_embeddings'))
    xvector_emb = torch.cat(xvector_emb, dim=0)
    lebel_emb = np.array(lebel_emb)
    
    score_matrix = pairwise_cosine_similarity(xvector_emb, xvector_emb)
    score_matrix = score_matrix.detach().cpu().numpy()
    clustering = SpectralClustering(affinity="precomputed", random_state=777, n_clusters=n_speaker).fit_predict(score_matrix)
    print("clustering label:", clustering.shape)

    for idx, seg in enumerate(filtered_segments):
        speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
        hyp_label[Segment(seg['st'], seg['et'])] = clustering[idx]

    diarizationErrorRate = DiarizationErrorRate(collar=0.25)
    diar_info = diarizationErrorRate(ref_label, hyp_label, detailed=True, uem=Segment(0, utt_len))
    print(diar_info)
    return diar_info['diarization error rate']

def dira_alimeeting():
    kaldi_obj = KaldiData("/exhome1/weiguang/data/M2MET/Eval_Ali_far")

    device = torch.device("cuda:0")
    model_filepath = "/exhome1/weiguang/data/voxceleb/voxceleb_xvector_outputs/full_system_default/networks/epoch.40.pt"
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

    der_utts = []

    for utt_id in kaldi_obj.wavs.keys():
        der_tmp = dira_recording(kaldi_obj, net, device, utt_id)
        der_utts.append(der_tmp)

    print("Average DER: {:.3f}".format(np.mean(der_utts)))

if __name__ == "__main__":
    dira_alimeeting()