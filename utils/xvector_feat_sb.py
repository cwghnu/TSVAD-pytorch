import numpy as np
import soundfile as sf
import librosa
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import accuracy_score
from torchmetrics.functional import pairwise_cosine_similarity
from torchaudio.compliance.kaldi import mfcc as kaldi_mfcc

from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization

from sklearn.neighbors import NearestCentroid

from openTSNE import TSNE
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append(os.path.dirname(__file__))

from model.xvector import Xvector

sys.path.append(os.path.dirname(__file__))
from kaldi_data import KaldiData
from feature import exclude_overlaping, index_aligned_labels

def test_mfcc():
    wav_path = "/exhome1/weiguang/data/Alimeeting/Test_Ali/Test_Ali_far/audio_dir/R8009_M8026_MS812.wav"  # 2 speaker
    # wav_path = "/exhome1/weiguang/data/Alimeeting/Test_Ali/Test_Ali_far/audio_dir/R8002_M8002_MS802.wav"  # 4 spks
    # wav_path = "/exhome1/weiguang/data/Alimeeting/Test_Ali/Test_Ali_far/audio_dir/R8008_M8015_MS808.wav"    # 3 spks
    wav_path = "/exhome1/weiguang/data/Alimeeting/Eval_Ali/Eval_Ali_far/audio_dir/R8008_M8013_MS807.wav"    # 3 spks
    device = torch.device("cpu")
    signal_data, sr = sf.read(wav_path)
    signal_data = signal_data[:, 0]
    signal_data = torch.from_numpy(signal_data[None, :]).float()

    utt_len = len(signal_data) / sr

    ref_label = Annotation()
    hyp_label = Annotation()

    mfcc_feat = Fbank(n_mels=24)(signal_data)
    mfcc_feat = mfcc_feat.to(device)
    norm = InputNormalization(norm_type="sentence", std_norm=False)
    mfcc_feat = norm(mfcc_feat, torch.ones(1))

    pretrain_model_path = "/exhome1/weiguang/code/TSVAD-pytorch/checkpoints/speechbrain_xvector/embedding_model.ckpt"
    nnet = Xvector(in_channels=24)
    checkpoint = torch.load(pretrain_model_path, map_location="cpu")
    nnet.load_state_dict(checkpoint)
    nnet.to(device)
    nnet.eval()

    start = 0
    end = mfcc_feat.shape[1]
    kaldi_obj = KaldiData("/exhome1/weiguang/data/Alimeeting/Eval_Ali/Eval_Ali_far")
    recid = "R8008_M8013_MS807"
    filtered_segments = kaldi_obj.segments[recid]
    frame_shift = 0.01 * sr
    rate = sr
    speakers = np.unique(
            [kaldi_obj.utt2spk[seg['utt']] for seg
                in filtered_segments]).tolist()
    n_speaker = len(speakers)
    T = np.zeros((mfcc_feat.shape[0], n_speaker), dtype=np.int32)
    xvector_emb = []
    lebel_emb = []
    for seg in filtered_segments:
        speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
        ref_label[Segment(seg['st'], seg['et'])] = speaker_index

    ref_label_no_overlap = exclude_overlaping(ref_label)
    min_seg_length = 0.3
    mfcc_spks = [torch.zeros((0,24)).to(device) for i in range(n_speaker)]

    for segment, track, label in ref_label_no_overlap.itertracks(yield_label=True):
        st, et = segment.start, segment.end
        if et - st < min_seg_length:
            continue
        speaker_index = label
        ref_label[Segment(st, et)] = speaker_index
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
                xvector_emb.append(nnet(mfcc_feat[:, rel_start:rel_end, :])[0, ...])
                mfcc_spks[speaker_index] = torch.cat((mfcc_spks[speaker_index], mfcc_feat[0, rel_start:rel_end, :]), dim=0)
    xvector_emb = torch.cat(xvector_emb, dim=0)
    lebel_emb = np.array(lebel_emb)
    xvector_emb = xvector_emb.cpu().numpy()

    print("lebel_emb", np.unique(lebel_emb))

    centroids = np.zeros((n_speaker, xvector_emb.shape[-1]))
    
    # for i in range(n_speaker):
    #     centroids[i, :] = np.mean(xvector_emb[lebel_emb==i], axis=0)

    xvector_spks = []
    for mfcc_spk in mfcc_spks:
        with torch.no_grad():
            xvector_spk = nnet(mfcc_spk[None, ...])[0, ...]
            xvector_spks.append(xvector_spk)
    xvector_spks = torch.cat(xvector_spks, dim=0)
    centroids = xvector_spks.cpu().numpy()

    xvector_emb_centroids = np.concatenate((xvector_emb, centroids), axis=0)

    tsne = TSNE(
        perplexity=30,
        metric="cosine",    # euclidean
        # callbacks=ErrorLogger(),
        n_jobs=32,
        random_state=42,
    )
    xembeddings = tsne.fit(xvector_emb_centroids)
    vis_x = xembeddings[:-n_speaker, 0]
    vis_y = xembeddings[:-n_speaker, 1]
    plt.scatter(vis_x, vis_y, c=lebel_emb, cmap=plt.cm.get_cmap("Set3", n_speaker), marker='.')

    vis_x = xembeddings[-n_speaker:, 0]
    vis_y = xembeddings[-n_speaker:, 1]
    plt.scatter(vis_x, vis_y, c=np.arange(n_speaker).tolist(), cmap=plt.cm.get_cmap("Set1", n_speaker), marker='.')

    plt.colorbar(ticks=range(n_speaker))
    plt.clim(-0.5, n_speaker-0.5)
    plt.savefig(os.path.join(os.path.dirname(__file__), "tsne-xvector-pretrain-3spks-eval-norm.png"), dpi=600)
    

if __name__ == "__main__":
    test_mfcc()