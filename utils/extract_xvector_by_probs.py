import argparse
import numpy as np
import os
import sys
from scipy.signal import medfilt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.diarization import read_rttm, rttm2annotation
from pyannote.core import Annotation, Segment, Timeline
import torch
import soundfile as sf

from model.xvector import Xvector

def load_model(model_filepath, device):
    nnet = Xvector(in_channels=24)
    checkpoint = torch.load(model_filepath, map_location="cpu")
    nnet.load_state_dict(checkpoint)
    nnet.to(device)
    nnet.eval()

    return nnet

def extract_xvec_with_tsvad(hyp_probs_dir, old_xvector_dir, xvector_dir, feat_dir):

    threshold = 0.4
    median = 51
    sampling_rate = 16000
    frame_shift = sampling_rate * 0.01
    subsampling = 1

    device = torch.device("cuda:0")
    model_filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints/speechbrain_xvector/embedding_model.ckpt")
    model = load_model(model_filepath, device)

    filepaths = os.listdir(hyp_probs_dir)

    for filepath in filepaths:
        session, _ = os.path.splitext(os.path.basename(filepath))
        file_full_path = os.path.join(hyp_probs_dir, filepath)
        data = np.load(file_full_path)
        num_speaker = data.shape[1]

        data_high_probs = data / np.sum(data, axis=1, keepdims=True)
        # data = np.where(data > threshold, 1, 0)
        data = np.where((data > threshold) & (data_high_probs > 0.8), 1, 0)

        if median > 1:
            data = medfilt(data, (median, 1))

        hyp_annot = Annotation()

        for spkid, frames in enumerate(data.T):
            frames = np.pad(frames, (1, 1), 'constant')
            changes, = np.where(np.diff(frames, axis=0) != 0)
            for s, e in zip(changes[::2], changes[1::2]):
                start_time = s * frame_shift * subsampling / sampling_rate
                end_time = e * frame_shift * subsampling / sampling_rate
                hyp_annot[Segment(start_time, end_time)] = spkid

        hyp_annot.support(collar=0.5)
        hyp_annot.extrude(hyp_annot.get_overlap())

        mfcc_feat = np.load(os.path.join(feat_dir, session+".npy"))
        if len(mfcc_feat.shape) == 3:
            mfcc_feat = mfcc_feat[0]
        mfcc_feat = torch.from_numpy(mfcc_feat)[None, ...]  # [1, num_frames, num_feat]

        speaker_emb = dict()
        start = 0
        end = mfcc_feat.shape[1]
        min_seg_length = 0.2
        rate = sampling_rate
        for segment, track, label in hyp_annot.itertracks(yield_label=True):
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
                        xvector_tmp = model(mfcc_feat[:, rel_start:rel_end, :].to(device))[0, ...]
                        if label in list(speaker_emb.keys()):
                            speaker_emb[label].append(xvector_tmp.cpu().numpy())
                        else:
                            speaker_emb[label] = [xvector_tmp.cpu().numpy()]
                    except:
                        print(mfcc_feat.shape)
                        print(start_frame, end_frame)
                        print(rel_start, rel_end)
                        print(mfcc_feat[:, rel_start:rel_end, :].shape)
                        continue

        centroids = np.load(os.path.join(old_xvector_dir, session+".npy"))

        for idx, (k, v) in enumerate(speaker_emb.items()):
            array_tmp = np.concatenate(v, axis=0)
            centroids[int(k)] = np.mean(array_tmp, axis=0)

        output_file = os.path.join(xvector_dir, session)
        print("rec_id: {}, x-vector shape: {}".format(session, centroids.shape))
        np.save(output_file, centroids)
        

if __name__ == "__main__":
    hyp_probs_dir = "/export/home2/cwguang/code/TSVAD-pytorch/exp/hyp_probs"
    xvector_dir = "/export/home2/cwguang/code/TSVAD-pytorch/exp/Test/tsvad_1it/xvec"
    old_xvector_dir = "/export/home2/cwguang/code/TSVAD-pytorch/exp/Test/cluster/xvec"
    feat_dir = "/export/home2/cwguang/datasets/Test_Ali/Test_Ali_far/mfcc_xvec"

    if not os.path.exists(xvector_dir):
        os.makedirs(xvector_dir)

    extract_xvec_with_tsvad(hyp_probs_dir, old_xvector_dir, xvector_dir, feat_dir)