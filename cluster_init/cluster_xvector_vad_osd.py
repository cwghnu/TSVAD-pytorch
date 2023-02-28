import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import accuracy_score
from torchmetrics.functional import pairwise_cosine_similarity
import soundfile as sf
import pickle

from openTSNE import TSNE
import matplotlib.pyplot as plt

from pyannote.core import Annotation, Segment, SlidingWindowFeature, Timeline
from pyannote.audio import Model, Inference
from pyannote.audio.utils.signal import binarize, Binarize
from pyannote.audio.pipelines.segmentation import SpeakerSegmentation
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils import SpeakerDiarizationMixin
from pyannote.audio.pipelines import VoiceActivityDetection as VoiceActivityDetectionPipeline
from pyannote.audio.pipelines import OverlappedSpeechDetection as OverlappedSpeechDetectionPipeline

from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.processing import diarization as diar
import speechbrain as sb
from speechbrain.processing.PLDA_LDA import StatObject_SB
from speechbrain.processing.PLDA_LDA import Ndx
from speechbrain.processing.PLDA_LDA import fast_PLDA_scoring

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.diarization import read_rttm, get_oracle_num_spkrs_from_uttrttm
from model.xvector import Xvector

def compute_affinity_matrix(X):
    """Compute the affinity matrix from data.
    Note that the range of affinity is [0,1].
    Args:
        X: numpy array of shape (n_samples, n_features)
    Returns:
        affinity: numpy array of shape (n_samples, n_samples)
    """
    # Normalize the data.
    l2_norms = np.linalg.norm(X, axis=1)
    X_normalized = X / l2_norms[:, None]
    # Compute cosine similarities. Range is [-1,1].
    cosine_similarities = np.matmul(X_normalized, np.transpose(X_normalized))
    # Compute the affinity. Range is [0,1].
    # Note that this step is not mentioned in the paper!
    affinity = (cosine_similarities + 1.0) / 2.0
    return affinity

def compute_sorted_eigenvectors(A):
    """Sort eigenvectors by the real part of eigenvalues.
    Args:
        A: the matrix to perform eigen analysis with shape (M, M)
    Returns:
        w: sorted eigenvalues of shape (M,)
        v: sorted eigenvectors, where v[;, i] corresponds to ith largest
           eigenvalue
    """
    # Eigen decomposition.
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    # Sort from largest to smallest.
    index_array = np.argsort(-eigenvalues)
    # Re-order.
    w = eigenvalues[index_array]
    v = eigenvectors[:, index_array]
    return w, v

def compute_number_of_clusters(
        eigenvalues, max_clusters=None, stop_eigenvalue=1e-2):
    """Compute number of clusters using EigenGap principle.
    Args:
        eigenvalues: sorted eigenvalues of the affinity matrix
        max_clusters: max number of clusters allowed
        stop_eigenvalue: we do not look at eigen values smaller than this
    Returns:
        number of clusters as an integer
    """
    max_delta = 0
    max_delta_index = 0
    range_end = len(eigenvalues)
    if max_clusters and max_clusters + 1 < range_end:
        range_end = max_clusters + 1
    for i in range(1, range_end):
        if eigenvalues[i - 1] < stop_eigenvalue:
            break
        delta = eigenvalues[i - 1] / eigenvalues[i]
        if delta > max_delta:
            max_delta = delta
            max_delta_index = i
    return max_delta_index


def init_xvector(wav_file_path, xvector_file_path, ref_rttm_file_path, min_seg_length=0.2, use_cluster_number=False, min_cluster_number=2, max_cluster_number=4, plot=False):
    device = torch.device("cuda:0")
    signal_data, sr = sf.read(wav_file_path)
    if len(signal_data.shape) > 1:
        signal_data = signal_data[:, 0]

    signal_data = torch.from_numpy(signal_data[None, :]).float()

    utt_len = len(signal_data) / sr

    ref_label = Annotation()
    hyp_label = Annotation()

    rate = sr
    frame_shift = 0.01 * rate

    mfcc_feat = Fbank(n_mels=24)(signal_data)
    mfcc_feat = mfcc_feat.to(device)
    norm = InputNormalization(norm_type="sentence", std_norm=False)
    mfcc_feat = norm(mfcc_feat, torch.ones(1))

    pretrain_model_path = "/export/home2/cwguang/code/TSVAD-pytorch/checkpoints/speechbrain_xvector/embedding_model.ckpt"
    nnet = Xvector(in_channels=24)
    checkpoint = torch.load(pretrain_model_path, map_location="cpu")
    nnet.load_state_dict(checkpoint)
    nnet.to(device)
    nnet.eval()

    model = Model.from_pretrained("/export/home2/cwguang/code/TSVAD-pytorch/checkpoints/segment_sincnet/segmentation_model.ckpt")
    audio_in_memory = {
        "waveform": signal_data,
        "sample_rate": sr,
    }

    vad_pipeline = VoiceActivityDetectionPipeline(segmentation=model)
    initial_params = {"onset": 0.5, "offset": 0.5, "min_duration_on": 0.0, "min_duration_off": 0.0}
    vad_pipeline.instantiate(initial_params)
    vad_timeline = vad_pipeline(audio_in_memory).get_timeline()

    osd_pipeline = OverlappedSpeechDetectionPipeline(segmentation=model)
    initial_params = {"onset": 0.5, "offset": 0.5, "min_duration_on": 0.1, "min_duration_off": 0.1}
    osd_pipeline.instantiate(initial_params)
    osd_timeline = osd_pipeline(audio_in_memory).get_timeline()

    nonoverlap_timeline = vad_timeline.extrude(osd_timeline).to_annotation()

    start = 0
    end = mfcc_feat.shape[1]
    xvector_emb = []
    for segment, track, label in nonoverlap_timeline.itertracks(yield_label=True):
        st, et = segment.start, segment.end
        # remove short segments
        if et - st < min_seg_length:
            continue

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
                xvector_emb.append(nnet(mfcc_feat[:, rel_start:rel_end, :])[0, ...])
    xvector_emb = torch.cat(xvector_emb, dim=0)

    xvector_emb = xvector_emb.detach().cpu().numpy()

    ref_rttm_content = read_rttm(ref_rttm_file_path)
    k_gt = get_oracle_num_spkrs_from_uttrttm(ref_rttm_content)

    # plda_file = "/export/home2/cwguang/code/TSVAD-pytorch/checkpoints/plda/plda_sb"
    # with open(plda_file, "rb") as input:
    #     plda = pickle.load(input)

    # en_N = xvector_emb.shape[0]
    # dim = 512
    # en_xv = xvector_emb
    # en_sgs = ['en'+str(i) for i in range(en_N)]
    # en_sets = np.array(en_sgs, dtype="|O")
    # en_s = np.array([None] * en_N)
    # en_stat0 = np.array([[1.0]]* en_N)
    # en_stat = StatObject_SB(modelset=en_sets, segset=en_sets, start=en_s, stop=en_s, stat0=en_stat0, stat1=en_xv)
    # ndx = Ndx(models=en_sets, testsegs=en_sets)
    # scores_plda = fast_PLDA_scoring(en_stat, en_stat, ndx, plda.mean, plda.F, plda.Sigma)
    # score_matrix = scores_plda.scoremat
    # score_matrix = 0.5 * (score_matrix + score_matrix.T)
    # score_matrix = score_matrix - np.min(score_matrix)
    # score_matrix = score_matrix / np.max(score_matrix)
    # score_matrix = np.nan_to_num(score_matrix, nan=0.0)
    
    if use_cluster_number:

        k = k_gt
        clust = diar.Spec_Cluster(affinity="precomputed", random_state=777, n_clusters=k)
        clust.perform_sc(xvector_emb)
        clustering_label = clust.labels_

        # clustering_label = diar.spectral_clustering_sb(
        #     score_matrix, n_clusters=k,
        # )

        # clust = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
        # clustering_label = clust.fit_predict(score_matrix)
    else:
        clust = diar.Spec_Clust_unorm(min_num_spkrs=2, max_num_spkrs=10)
        score_matrix = clust.get_sim_mat(xvector_emb)
        pruned_sim_mat = clust.p_pruning(score_matrix, 0.2)
        sym_pruned_sim_mat = 0.5 * (pruned_sim_mat + pruned_sim_mat.T)
        laplacian = clust.get_laplacian(sym_pruned_sim_mat)
        spec_emb, k = clust.get_spec_embs(laplacian, None)
        if k < min_cluster_number:
            k = min_cluster_number
        if k > max_cluster_number:
            k = max_cluster_number

        clust.cluster_embs(spec_emb, k)
        clustering_label = clust.labels_

    # (eigenvalues, eigenvectors) = compute_sorted_eigenvectors(score_matrix)
    # # Get number of clusters.
    # k = compute_number_of_clusters(eigenvalues, 100, 0.7)

    print("Cluster number: {}, ground truth: {}".format(k, k_gt))

    # clustering_label = SpectralClustering(affinity="precomputed", random_state=777, n_clusters=k).fit_predict(score_matrix)
    # print("clustering label:", clustering_label.shape)  # [num_samples]

    xvector_spks = []
    for idx_spk in range(k):
        xvector_mean = np.mean(xvector_emb[clustering_label==idx_spk], axis=0)
        xvector_spks.append(xvector_mean)
    xvector_spks = np.stack(xvector_spks, axis=0)

    print("x-vector shape: {}".format(xvector_spks.shape))
    np.save(xvector_file_path, xvector_spks)

    if plot:

        plt.figure()
        tsne = TSNE(
            perplexity=30,
            metric="cosine",    # euclidean
            # callbacks=ErrorLogger(),
            n_jobs=32,
            random_state=42,
        )
        xvector_emb_centroids = np.concatenate((xvector_emb, xvector_spks), axis=0)
        xembeddings = tsne.fit(xvector_emb_centroids)
        vis_x = xembeddings[:-k, 0]
        vis_y = xembeddings[:-k, 1]
        plt.scatter(vis_x, vis_y, c=clustering_label, cmap=plt.cm.get_cmap("Set3", k), marker='.')

        vis_x = xembeddings[-k:, 0]
        vis_y = xembeddings[-k:, 1]
        plt.scatter(vis_x, vis_y, c=np.arange(k).tolist(), cmap=plt.cm.get_cmap("Set1", k), marker='.')

        plt.colorbar(ticks=range(k))
        plt.clim(-0.5, k-0.5)
        plt.savefig(os.path.join(os.path.dirname(__file__), "{}.png".format(os.path.basename(wav_file_path).split(".")[0])), dpi=600)

    return k_gt, (k - k_gt)**2
    

def gen_init_xvector(wav_file_dir, output_file_dir, rttm_file_dir):
    rmses = [[], [], []]
    for wav_file in os.listdir(wav_file_dir):
        if wav_file.endswith(".wav"):
            print("Processing {}".format(wav_file))
            wav_id = wav_file.split(".wav")[0]

            wav_file_path = os.path.join(wav_file_dir, wav_file)
            xvector_file_path = os.path.join(output_file_dir, wav_id)
            ref_rttm_file_path = os.path.join(rttm_file_dir, wav_id+".rttm")
            num_spks, rmse_cluster = init_xvector(wav_file_path, xvector_file_path, ref_rttm_file_path, use_cluster_number=True)
            rmses[num_spks-2].append(rmse_cluster)

    for i in range(len(rmses)):
        print("Num spks: {}, RMSE: {}".format(i+2, np.sqrt(np.mean(rmses[i]))))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    wav_file_dir = "/export/home2/cwguang/datasets/Test_Ali/Test_Ali_far/audio_dir"
    output_file_dir = "/export/home2/cwguang/code/TSVAD-pytorch/exp/Test/cluster/xvec"
    rttm_file_dir = "/export/home2/cwguang/datasets/Test_Ali/Test_Ali_far/rttm_groundtruth"

    dataset_name = os.path.dirname(wav_file_dir).split(os.sep)[-1]
    
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    gen_init_xvector(wav_file_dir, output_file_dir, rttm_file_dir)