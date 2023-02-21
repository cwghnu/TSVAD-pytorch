import numpy as np
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.feat.mel import MelBanksOptions
from kaldi.feat.window import FrameExtractionOptions
from kaldi.feat.functions import compute_deltas, DeltaFeaturesOptions
from kaldi.matrix import SubVector
import soundfile as sf
import librosa
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from torchmetrics.functional import pairwise_cosine_similarity
from sklearn.neighbors import NearestCentroid

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

# from hyperion.clustering.ahc import AHC

sys.path.append(os.path.dirname(__file__))
from kaldi_data import KaldiData
from feature import extract_mfcc, exclude_overlaping, index_aligned_labels

def topk_(matrix, K, axis=1):
    topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
    topk_data = matrix.take(topk_index, axis=axis)
    return topk_data

def accumulate_stats(feats, counts, posteriors, indices, n_gaussians=2048):
    """Computes 0th and 1st order statistics from the selected posteriors.
    
    Arguments:
        feats {ndarray} -- Feature array (feature vectors as rows).
        counts {ndarray} -- Array containing the numbers of selected posteriors for each frame.
        posteriors {ndarray} -- Array containing posteriors (flattened).
        indices {ndarray} -- Array containing Gaussian indices (flattened).
    
    Returns:
        ndarray -- 0th order statistics (row vector).
        ndarray -- 1st order statistics (row index = component index).
    """

    data_dims = [n_gaussians, feats.shape[-1]]

    n = torch.zeros(data_dims[0], dtype=torch.float32, device=feats.device)
    f = torch.zeros(data_dims, dtype=torch.float32, device=feats.device)
    for frame_index in range(counts.shape[0]):
        gaussian_indices = indices[:, frame_index]
        frame_posteriors = posteriors[:, frame_index]
        n[gaussian_indices] += frame_posteriors
        f[gaussian_indices, :] += torch.matmul(frame_posteriors[:, None], feats[frame_index:frame_index+1, :]) # torch.outer(frame_posteriors, feats[frame_index, :])         
    return n, f

def compute_posterior(frames_mfcc, ubm, diag_ubm, sub_sampling=100, n_top_gaussians=20, posterior_threshold = 0.025):

    sub_batch_count = int(np.ceil(ubm.means.size()[0] / ubm.means.size()[1]))
    chunks = torch.chunk(frames_mfcc, sub_batch_count, dim=0)
    top_gaussians = []
    for chunk in chunks:
        posteriors = diag_ubm.compute_posteriors(chunk)
        top_gaussians.append(torch.topk(posteriors, n_top_gaussians, dim=0, largest=True, sorted=False)[1])

    top_gaussians = torch.cat(top_gaussians, dim=1)     # [top_gaussians=20, N]

    # posteriors = diag_ubm.compute_posteriors(frames_mfcc)
    # top_gaussians = torch.topk(posteriors, n_top_gaussians, dim=0, largest=True, sorted=False)[1]

    posteriors = ubm.compute_posteriors_top_select(frames_mfcc, top_gaussians)       # [top_gaussians=20, N]

    # Posterior thresholding:
    max_indices = torch.argmax(posteriors, dim=0)
    mask = posteriors.ge(posterior_threshold)
    top_counts = torch.sum(mask, dim=0)     # [N]
    posteriors[~mask] = 0
    divider = torch.sum(posteriors, dim=0)
    mask2 = divider.eq(0) # For detecting special cases
    posteriors[:, ~mask2] = posteriors[:, ~mask2] / divider[~mask2]
    # Special case that all the posteriors are discarded (force to use 1):
    # Special case: the probability of each component is small, in this case, choose the one with biggest probability as the selected component
    posteriors[max_indices[mask2], mask2] = 1
    mask[max_indices[mask2], mask2] = 1
    top_counts[mask2] = 1

    p_save = posteriors[:torch.sum(top_counts)]
    t_save = top_gaussians[:torch.sum(top_counts)]
    c_save = top_counts
    
    if sub_sampling >= p_save.shape[1]:
        p_save_chunk = [p_save]
        t_save_chunk = [t_save]
        c_save_chunk = [c_save]
    else:
        p_save_chunk, t_save_chunk, c_save_chunk = torch.split(p_save, sub_sampling, dim=1), torch.split(t_save, sub_sampling, dim=1), torch.split(c_save, sub_sampling, dim=0)

    n_all = []
    f_all = []
    for p_chunk, t_chunk, c_chunk in zip(p_save_chunk, t_save_chunk, c_save_chunk):
        n, f = accumulate_stats(frames_mfcc, c_chunk, p_chunk, t_chunk)
        n_all.append(n)
        f_all.append(f)
    n_all = torch.stack(n_all, dim=0) # [B, N_gaussian]
    f_all = torch.stack(f_all, dim=2) # [N_gaussian, feat, B]

    return n_all, f_all

def extract_ivector_from_mfcc(frames_mfcc, ivector_extractor, ubm, diag_ubm, sub_sampling=100, n_top_gaussians=20, posterior_threshold = 0.025):

    n_all, f_all = compute_posterior(frames_mfcc, ubm, diag_ubm, sub_sampling=sub_sampling, n_top_gaussians=n_top_gaussians, posterior_threshold=posterior_threshold)
    batch_size = n_all.size()[0]
    component_batches = ivector_extractor._get_component_batches(16)
    means = ivector_extractor._compute_posterior_means_and_covariances(n_all, f_all, batch_size, component_batches)[0]
    means[:, 0] -= ivector_extractor.prior_offset

    return means

# def extract_mfcc(signal_data, sr, mfcc_opts=None):
#     if type(signal_data) is not SubVector:
#         signal_data = SubVector(signal_data)

#     if mfcc_opts is None:
#         mfcc_opts = MfccOptions()
#         mfcc_opts.frame_opts.samp_freq = sr
#         mfcc_opts.frame_opts.frame_length_ms = 25
#         mfcc_opts.frame_opts.frame_shift_ms = 10

#         mel_opts = MelBanksOptions()
#         mel_opts.low_freq = 20
#         mel_opts.high_freq = 7600
#         mel_opts.num_bins = 30
#         mfcc_opts.mel_opts = mel_opts

#         mfcc_opts.num_ceps = 24

#     mfcc = Mfcc(mfcc_opts)
#     mfcc_feat = mfcc.compute_features(signal_data, sr, 1.0)

#     delta_opts = DeltaFeaturesOptions()
#     mfcc_feat_delta = compute_deltas(delta_opts, mfcc_feat)

#     return torch.Tensor(mfcc_feat_delta)

def test_mfcc():
    wav_path = "/exhome1/weiguang/data/M2MET/Test_Ali_far/audio_dir/R8002_M8002_MS802.wav"
    signal_data, sr = sf.read(wav_path)
    signal_data = signal_data[:, 0]
    
    utt_len = len(signal_data) / sr

    ref_label = Annotation()
    hyp_label = Annotation()

    mfcc_feat = extract_mfcc(signal_data, sr, num_ceps=24, cmn=True, delta=True)
    print("MFCC feature: ", mfcc_feat.shape, type(mfcc_feat))
    
    filename_ubm = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs_clean/ubms/full_ubm_2048/final.ubm"
    filename_ive = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs_clean/ivector_400/ivector_extractors/iter.20.npz"
    file_plda = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs_clean/PLDA/plda.npz"
    file_vec_processor = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs_clean/vec_norm/vec_processor"
    
    # filename_ubm = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs/ubms/full_ubm_2048/final.ubm"
    # filename_ive = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs/ivector_400/ivector_extractors/iter.20.npz"
    # file_plda = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs/PLDA/plda.npz"
    # file_vec_processor = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs/vec_norm/vec_processor"

    device = torch.device("cuda:1")
    mfcc_feat = mfcc_feat.to(device)
    ivector_extractor = IVectorExtractor.from_npz_file(filename_ive, device)
    ubm = Gmm.from_kaldi(filename_ubm, device)
    diag_ubm = DiagGmm.from_full_gmm(ubm, device)
    vector_processor = VectorProcessor.load(file_vec_processor, device)
    

    start = 0
    end = mfcc_feat.shape[0]
    kaldi_obj = KaldiData("/exhome1/weiguang/data/M2MET/Test_Ali_far")
    recid = "R8002_M8002_MS802"
    filtered_segments = kaldi_obj.segments[recid]
    frame_shift = 0.01 * sr
    rate = sr
    speakers = np.unique(
            [kaldi_obj.utt2spk[seg['utt']] for seg
                in filtered_segments]).tolist()
    n_speaker = len(speakers)
    T = np.zeros((mfcc_feat.shape[0], n_speaker), dtype=np.int32)
    ivector_emb = []
    lebel_emb = []
    for seg in filtered_segments:
        speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
        # print("segment : [{}, {}], spk: {}".format(seg['st'], seg['et'], speaker_index))
        ref_label[Segment(seg['st'], seg['et'])] = speaker_index

    ref_label_no_overlap = exclude_overlaping(ref_label)
    
    mfcc_spks = [torch.zeros((0,72)).to(device)]*n_speaker

    for segment, track, label in ref_label_no_overlap.itertracks(yield_label=True):
        st, et = segment.start, segment.end
        if et - st < 1.0:
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
                
                mfcc_spks[speaker_index] = torch.cat((mfcc_spks[speaker_index], mfcc_feat[rel_start:rel_end, :]), dim=0)
                
                ivector_chunk = extract_ivector_from_mfcc(mfcc_feat[rel_start:rel_end, :], ivector_extractor, ubm, diag_ubm, sub_sampling=sub_len)
                ivector_emb.append(ivector_chunk)
                
    print("ivector shape: ", ivector_chunk.shape)
    ivector_emb = torch.cat(ivector_emb, dim=0)
    # ivector_emb_norm = vector_processor.process(ivector_emb)
    ivector_emb_norm = ivector_emb
    lebel_emb = np.array(lebel_emb)
    
    ivector_spks = []
    for mfcc_spk in mfcc_spks:
        sub_len = len(mfcc_spk)
        ivector_spk = extract_ivector_from_mfcc(mfcc_spk, ivector_extractor, ubm, diag_ubm, sub_sampling=sub_len)
        ivector_spks.append(ivector_spk)
    ivector_spks = torch.cat(ivector_spks, dim=0)
    # ivector_spks = vector_processor.process(ivector_spks)
    ivector_spks = ivector_spks.cpu().numpy()

    plda = Plda.load(file_plda, device)
    score_matrix = plda.score_all_vs_all(ivector_emb_norm, ivector_emb_norm, 200)
    
    # score_matrix = pairwise_cosine_similarity(ivector_emb_norm, ivector_emb_norm)
    # score_matrix = score_matrix.detach().cpu().numpy()
    
    score_matrix = score_matrix - np.min(score_matrix)
    score_matrix = score_matrix / np.max(score_matrix)
    print("score shape: ", score_matrix.shape, score_matrix.max(), score_matrix.min())
    
    # # score_matrix = score_matrix.detach().cpu().numpy()
    # clustering = AgglomerativeClustering(affinity="precomputed", linkage='single').fit_predict(score_matrix)
    clustering = SpectralClustering(affinity="precomputed", random_state=777, n_clusters=n_speaker).fit_predict(score_matrix)
    # print("clustering label:", clustering.shape)

    # total_frames = clustering.shape[0] * sub_sampling
    # time_tick = librosa.frames_to_time(np.arange(total_frames), sr=sr, n_fft=0.025*sr, hop_length=0.01*sr)
    # predicted_label = np.repeat(clustering, sub_sampling)

    # plt.figure()
    # plt.plot(time_tick, predicted_label, linewidth=2.0)
    # plt.savefig(os.path.join(os.path.dirname(__file__), "diar_clustering.png"), dpi=600)
    
    # clf = NearestCentroid()
    ivector_emb_norm = ivector_emb_norm.cpu().numpy()
    # clf.fit(ivector_emb_norm, clustering)

    # index_aligned = index_aligned_labels(clustering, lebel_emb, n_classes=n_speaker)
    # assert len(np.unique(index_aligned)) == n_speaker
    # print(index_aligned)
    # centroids = clf.centroids_[list(index_aligned)]
    
    # for i in range(n_speaker):
    #     centroids[i, :] = np.mean(ivector_emb_norm[lebel_emb==i], axis=0)

    ivector_emb_centroids = np.concatenate((ivector_emb_norm, ivector_spks), axis=0)
    
    # Test ivectors with t-SNE
    tsne = TSNE(
        perplexity=30,
        metric="euclidean",    # euclidean
        # callbacks=ErrorLogger(),
        n_jobs=32,
        random_state=42,
    )
    xembeddings = tsne.fit(ivector_emb_centroids)
    vis_x = xembeddings[:-n_speaker, 0]
    vis_y = xembeddings[:-n_speaker, 1]
    plt.scatter(vis_x, vis_y, c=lebel_emb, cmap=plt.cm.get_cmap("Set3", n_speaker), marker='.')

    vis_x = xembeddings[-n_speaker:, 0]
    vis_y = xembeddings[-n_speaker:, 1]
    plt.scatter(vis_x, vis_y, c=[0,1,2,3], cmap=plt.cm.get_cmap("Set1", n_speaker), marker='.')
    plt.colorbar(ticks=range(n_speaker))
    plt.clim(-0.5, n_speaker-0.5)
    plt.savefig(os.path.join(os.path.dirname(__file__), "tsne-ivector-4spk-nonorm-filtered.png"), dpi=600)

    # idx = 0
    # for segment, track, label in ref_label_no_overlap.itertracks(yield_label=True):
    #     st, et = segment.start, segment.end
    #     hyp_label[Segment(st, et)] = clustering[idx]
    #     idx += 1
    
    # # for idx, seg in enumerate(filtered_segments):
    # #     speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
    # #     hyp_label[Segment(seg['st'], seg['et'])] = clustering[idx]

    # diarizationErrorRate = DiarizationErrorRate(skip_overlap=False, collar=0.25)
    # diar_info = diarizationErrorRate(ref_label_no_overlap, hyp_label, detailed=True, uem=Segment(0, utt_len))
    # print(diar_info)
    # return diar_info['diarization error rate']
    

def test_ivector_plda_ahc_with_clean_data():
    wav_path = "/exhome1/weiguang/data/voxceleb/voxceleb1/wav/id10001/1zcIwhmdeo4/00001.wav"
    signal_data_spk1, sr = sf.read(wav_path)

    wav_path = "/exhome1/weiguang/data/voxceleb/voxceleb1/wav/id10002/0_laIeN-Q44/00001.wav"
    signal_data_spk2, sr = sf.read(wav_path)

    sub_sampling = 10

    mfcc_feat_spk1 = extract_mfcc(signal_data_spk1, sr, num_ceps=24, cmn=True, delta=True)
    print("SPK1 MFCC feature: ", mfcc_feat_spk1.shape, type(mfcc_feat_spk1))
    mfcc_feat_spk2 = extract_mfcc(signal_data_spk2, sr, num_ceps=24, cmn=True, delta=True)
    print("SPK2 MFCC feature: ", mfcc_feat_spk2.shape, type(mfcc_feat_spk2))

    # device = torch.device("cpu")
    device = torch.device("cuda:0")
    mfcc_feat_spk1 = mfcc_feat_spk1.to(device)
    mfcc_feat_spk2 = mfcc_feat_spk2.to(device)
    filename_ive = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs_clean/ivector_400/ivector_extractors/iter.20.npz"
    ivector_extractor = IVectorExtractor.from_npz_file(filename_ive, device)
    filename_ubm = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs_clean/ubms/full_ubm_2048/final.ubm"
    ubm = Gmm.from_kaldi(filename_ubm, device)
    diag_ubm = DiagGmm.from_full_gmm(ubm, device)
    file_vec_processor = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs_clean/vec_norm/vec_processor"
    vector_processor = VectorProcessor.load(file_vec_processor, device)

    ivector_chunk_spk1 = extract_ivector_from_mfcc(mfcc_feat_spk1, ivector_extractor, ubm, diag_ubm, sub_sampling=sub_sampling)
    print("SPK1 ivector shape: ", ivector_chunk_spk1.shape)
    ivector_chunk_norm_spk1 = vector_processor.process(ivector_chunk_spk1)
    ivector_chunk_spk2 = extract_ivector_from_mfcc(mfcc_feat_spk2, ivector_extractor, ubm, diag_ubm, sub_sampling=sub_sampling)
    print("SPK2 ivector shape: ", ivector_chunk_spk2.shape)
    ivector_chunk_norm_spk2 = vector_processor.process(ivector_chunk_spk2)

    cls_label = np.zeros(len(ivector_chunk_norm_spk1) + len(ivector_chunk_norm_spk2))
    cls_label[len(ivector_chunk_norm_spk1):] = 1
    ivector_chunk_norm = torch.cat([ivector_chunk_norm_spk1, ivector_chunk_norm_spk2])

    # Test ivectors with t-SNE
    n_speaker = 2
    tsne = TSNE(
        perplexity=10,
        metric="cosine",    # euclidean
        # callbacks=ErrorLogger(),
        n_jobs=32,
        random_state=42,
    )
    xembeddings = tsne.fit(ivector_chunk_norm.detach().cpu().numpy())
    vis_x = xembeddings[:, 0]
    vis_y = xembeddings[:, 1]
    plt.scatter(vis_x, vis_y, c=cls_label, cmap=plt.cm.get_cmap("Set3", n_speaker), marker='.')
    plt.colorbar(ticks=range(n_speaker))
    plt.clim(-0.5, n_speaker-0.5)
    plt.savefig(os.path.join(os.path.dirname(__file__), "tsne-ivector_clean.png"), dpi=600)

    # file_plda = "/exhome1/weiguang/data/voxceleb/voxceleb_ivector_outputs/PLDA/plda.npz"
    # plda = Plda.load(file_plda, device)
    # score_matrix = plda.score_all_vs_all(ivector_chunk_norm, ivector_chunk_norm, 200)
    # score_matrix = score_matrix - np.min(score_matrix)
    # score_matrix = score_matrix / np.max(score_matrix)
    # print("score shape: ", score_matrix.shape, score_matrix.max(), score_matrix.min())
    
    # score_matrix = score_matrix.numpy()
    # clustering = AgglomerativeClustering(affinity="precomputed", linkage='average').fit_predict(score_matrix)
    # score_matrix = torch.nn.functional.cosine_similarity(ivector_chunk_norm, ivector_chunk_norm)
    score_matrix = pairwise_cosine_similarity(ivector_chunk_norm, ivector_chunk_norm)
    score_matrix = score_matrix.detach().cpu().numpy()
    score_matrix = score_matrix - np.min(score_matrix)
    score_matrix = score_matrix / np.max(score_matrix)
    clustering = SpectralClustering(affinity="precomputed", random_state=777, n_clusters=2).fit_predict(score_matrix)
    print("clustering label:", clustering.shape)

    total_frames = clustering.shape[0] * sub_sampling
    time_tick = librosa.frames_to_time(np.arange(total_frames), sr=sr, n_fft=0.025*sr, hop_length=0.01*sr)
    predicted_label = np.repeat(clustering, sub_sampling)

    plt.figure()
    plt.plot(time_tick, predicted_label, linewidth=2.0)
    plt.savefig(os.path.join(os.path.dirname(__file__), "diar_clustering_clean.png"), dpi=600)

if __name__ == "__main__":
    test_mfcc()
    # test_ivector_plda_ahc_with_clean_data()