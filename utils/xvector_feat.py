import numpy as np
import kaldi
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.feat.mel import MelBanksOptions
from kaldi.feat.window import FrameExtractionOptions
from kaldi.feat.functions import compute_deltas, DeltaFeaturesOptions, SlidingWindowCmnOptions, sliding_window_cmn
from kaldi.matrix import SubVector, SubMatrix
import soundfile as sf
import librosa
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import accuracy_score
from torchmetrics.functional import pairwise_cosine_similarity
from torchaudio.compliance.kaldi import mfcc as kaldi_mfcc

from openTSNE import TSNE
# from openTSNE.callbacks import ErrorLogger
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append(os.path.dirname(__file__))

from asvtorch.src.ivector.ivector_extractor import IVectorExtractor
from asvtorch.src.ivector.gmm import DiagGmm, Gmm
from asvtorch.src.backend.vector_processing import VectorProcessor
from asvtorch.src.backend.plda import Plda

from asvtorch.src.networks.architectures import StandardNet as TDNN
from asvtorch.src.networks.network_io import initialize_net

# from hyperion.clustering.ahc import AHC

sys.path.append(os.path.dirname(__file__))
from kaldi_data import KaldiData
from feature import extract_mfcc, exclude_overlaping

def test_mfcc():
    wav_path = "/exhome1/weiguang/data/M2MET/Test_Ali_far/audio_dir/R8009_M8026_MS812.wav"  # 2 speaker
    # wav_path = "/exhome1/weiguang/data/M2MET/Test_Ali_far/audio_dir/R8002_M8002_MS802.wav"
    signal_data, sr = sf.read(wav_path)
    signal_data = signal_data[:, 0]

    utt_len = len(signal_data) / sr

    ref_label = Annotation()
    hyp_label = Annotation()

    mfcc_feat = extract_mfcc(signal_data, sr, high_freq=7900, cmn=True, delta=False)
    print("MFCC feature: ", mfcc_feat.shape, type(mfcc_feat))

    device = torch.device("cuda:1")
    mfcc_feat = mfcc_feat.to(device)
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

    start = 0
    end = mfcc_feat.shape[0]
    kaldi_obj = KaldiData("/exhome1/weiguang/data/M2MET/Test_Ali_far")
    recid = "R8009_M8026_MS812"
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

    ref_label_no_overlap = exclude_overlaping(ref_label)
    # ref_label_no_overlap = ref_label

    for segment, track, label in ref_label_no_overlap.itertracks(yield_label=True):
        st, et = segment.start, segment.end
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
                xvector_emb.append(net(mfcc_feat[None, :, rel_start:rel_end], 'extract_embeddings'))
    xvector_emb = torch.cat(xvector_emb, dim=0)
    lebel_emb = np.array(lebel_emb)

    # Test ivectors with t-SNE
    # ivector_chunk_norm = ivector_chunk_norm.numpy()
    tsne = TSNE(
        perplexity=30,
        metric="cosine",    # euclidean
        # callbacks=ErrorLogger(),
        n_jobs=32,
        random_state=42,
    )
    xembeddings = tsne.fit(xvector_emb.detach().cpu().numpy())
    vis_x = xembeddings[:, 0]
    vis_y = xembeddings[:, 1]
    plt.scatter(vis_x, vis_y, c=lebel_emb, cmap=plt.cm.get_cmap("Set3", n_speaker), marker='.')
    plt.colorbar(ticks=range(n_speaker))
    plt.clim(-0.5, n_speaker-0.5)
    plt.savefig(os.path.join(os.path.dirname(__file__), "tsne-xvector.png"), dpi=600)

    
    # file_plda = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs/PLDA/plda.npz"
    # plda = Plda.load(file_plda, device)
    # score_matrix = plda.score_all_vs_all(ivector_chunk_norm, ivector_chunk_norm, 200)
    # score_matrix = score_matrix - np.min(score_matrix)
    # score_matrix = score_matrix / np.max(score_matrix)
    # print("score shape: ", score_matrix.shape, score_matrix.max(), score_matrix.min())
    
    score_matrix = pairwise_cosine_similarity(xvector_emb, xvector_emb)
    score_matrix = score_matrix.detach().cpu().numpy()
    score_matrix = score_matrix - np.min(score_matrix)
    score_matrix = score_matrix / np.max(score_matrix)
    # # clustering = AgglomerativeClustering(affinity="precomputed", linkage='single').fit_predict(score_matrix)
    clustering = SpectralClustering(affinity="precomputed", random_state=777, n_clusters=n_speaker).fit_predict(score_matrix)
    print("clustering label:", clustering.shape)

    # acc_1 = accuracy_score(lebel_emb, clustering)
    # acc_2 = accuracy_score(1-lebel_emb, clustering)
    # print("Accuracy: {}, {}".format(acc_1, acc_2))

    # plt.figure()
    # time_tick = np.arange(len(lebel_emb))
    # plt.subplot(2,1,1)
    # plt.plot(time_tick, lebel_emb, linewidth=2.0)
    # plt.subplot(2,1,2)
    # plt.plot(time_tick, clustering, linewidth=2.0)
    # plt.savefig(os.path.join(os.path.dirname(__file__), "diar_clustering_xvec.png"), dpi=600)

    idx = 0
    for segment, track, label in ref_label_no_overlap.itertracks(yield_label=True):
        st, et = segment.start, segment.end
        hyp_label[Segment(st, et)] = clustering[idx]
        idx += 1


    # for idx, seg in enumerate(filtered_segments):
    #     speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
    #     hyp_label[Segment(seg['st'], seg['et'])] = clustering[idx]

    diarizationErrorRate = DiarizationErrorRate(skip_overlap=False, collar=0.25)
    diar_info = diarizationErrorRate(ref_label_no_overlap, hyp_label, detailed=True, uem=Segment(0, utt_len))
    print(diar_info)
    return diar_info['diarization error rate']

    # total_frames = clustering.shape[0] * sub_sampling
    # time_tick = librosa.frames_to_time(np.arange(total_frames), sr=sr, n_fft=0.025*sr, hop_length=0.01*sr)
    # predicted_label = np.repeat(clustering, sub_sampling)

    # plt.figure()
    # plt.plot(time_tick, predicted_label, linewidth=2.0)
    # plt.savefig(os.path.join(os.path.dirname(__file__), "diar_clustering.png"), dpi=600)
    

def test_ivector_plda_ahc_with_clean_data():
    wav_path = "/exhome1/weiguang/data/voxceleb/voxceleb1/wav/id10001/1zcIwhmdeo4/00001.wav"
    signal_data_spk1, sr = sf.read(wav_path)

    wav_path = "/exhome1/weiguang/data/voxceleb/voxceleb1/wav/id10002/0_laIeN-Q44/00001.wav"
    signal_data_spk2, sr = sf.read(wav_path)

    sub_sampling = 10

    mfcc_feat_spk1 = extract_mfcc(signal_data_spk1, sr)
    print("SPK1 MFCC feature: ", mfcc_feat_spk1.shape, type(mfcc_feat_spk1))
    mfcc_feat_spk2 = extract_mfcc(signal_data_spk2, sr)
    print("SPK2 MFCC feature: ", mfcc_feat_spk2.shape, type(mfcc_feat_spk2))

    # device = torch.device("cpu")
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

    with torch.no_grad():
        mfcc_feat_spk1 = mfcc_feat_spk1.to(device)
        mfcc_feat_spk2 = mfcc_feat_spk2.to(device)
        mfcc_feat_spk1 = mfcc_feat_spk1.permute(1,0)    # [F, n_frames]
        mfcc_feat_spk2 = mfcc_feat_spk2.permute(1,0)

        # xvector_spk1 = net(mfcc_feat_spk1[None, :], 'extract_features')
        # xvector_spk2 = net(mfcc_feat_spk2[None, :], 'extract_features')

        xvector_spk1 = []
        xvector_spk2 = []
        for idx_start in range(0, mfcc_feat_spk1.shape[1], sub_sampling):
            input_tmp = mfcc_feat_spk1[None, :, idx_start:idx_start+sub_sampling]
            if input_tmp.shape[-1] < 10:
                break
            xvector_spk1.append(net(input_tmp, 'extract_embeddings'))
        for idx_start in range(0, mfcc_feat_spk2.shape[1], sub_sampling):
            input_tmp = mfcc_feat_spk2[None, :, idx_start:idx_start+sub_sampling]
            if input_tmp.shape[-1] < 10:
                break
            xvector_spk2.append(net(input_tmp, 'extract_embeddings'))
        xvector_spk1 = torch.cat(xvector_spk1, dim=0)
        xvector_spk2 = torch.cat(xvector_spk2, dim=0)

    xvector_spk = torch.cat([xvector_spk1, xvector_spk2], dim=0)
    cls_label = np.zeros(len(xvector_spk))
    cls_label[len(xvector_spk1):] = 1

    # Test ivectors with t-SNE
    n_speaker = 2
    tsne = TSNE(
        perplexity=10,
        metric="cosine",    # euclidean
        # callbacks=ErrorLogger(),
        n_jobs=32,
        random_state=42,
    )
    xembeddings = tsne.fit(xvector_spk.detach().cpu().numpy())
    vis_x = xembeddings[:, 0]
    vis_y = xembeddings[:, 1]
    plt.scatter(vis_x, vis_y, c=cls_label, cmap=plt.cm.get_cmap("Set3", n_speaker), marker='.')
    plt.colorbar(ticks=range(n_speaker))
    plt.clim(-0.5, n_speaker-0.5)
    plt.savefig(os.path.join(os.path.dirname(__file__), "tsne_xvector_clean.png"), dpi=600)

    # file_plda = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs/PLDA/plda.npz"
    # plda = Plda.load(file_plda, device)
    # score_matrix = plda.score_all_vs_all(ivector_chunk_norm, ivector_chunk_norm, 200)
    # score_matrix = score_matrix - np.min(score_matrix)
    # score_matrix = score_matrix / np.max(score_matrix)
    # print("score shape: ", score_matrix.shape, score_matrix.max(), score_matrix.min())
    
    # score_matrix = score_matrix.numpy()
    # clustering = AgglomerativeClustering(affinity="precomputed", linkage='average').fit_predict(score_matrix)
    # score_matrix = torch.nn.functional.cosine_similarity(xvector_spk, xvector_spk)
    score_matrix = pairwise_cosine_similarity(xvector_spk, xvector_spk)
    score_matrix = score_matrix.detach().cpu().numpy()
    clustering = SpectralClustering(affinity="precomputed", random_state=777, n_clusters=2).fit_predict(score_matrix)
    print("clustering label:", clustering.shape)

    total_frames = clustering.shape[0] * sub_sampling
    time_tick = librosa.frames_to_time(np.arange(total_frames), sr=sr, n_fft=0.025*sr, hop_length=0.01*sr)
    predicted_label = np.repeat(clustering, sub_sampling)

    plt.figure()
    plt.plot(time_tick, predicted_label, linewidth=2.0)
    plt.savefig(os.path.join(os.path.dirname(__file__), "diar_x_vec_clustering_clean.png"), dpi=600)

if __name__ == "__main__":
    test_mfcc()
    # test_ivector_plda_ahc_with_clean_data()