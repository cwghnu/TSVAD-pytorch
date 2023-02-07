import numpy as np
import torch
from itertools import permutations
import copy

from torchaudio.compliance.kaldi import mfcc as kaldi_mfcc
from torchaudio.compliance.kaldi import fbank as kaldi_fbank
from kaldi.matrix import SubVector, SubMatrix
from kaldi.feat.functions import compute_deltas, DeltaFeaturesOptions, SlidingWindowCmnOptions, sliding_window_cmn
from pyannote.core import Annotation, Segment, Timeline

def extract_mfcc(signal_data, sr, num_mel_bins=30, num_ceps=30, low_freq=20, high_freq=7600, cmn=True, delta=True):
    
    mfcc_feat = kaldi_mfcc(torch.Tensor(signal_data)[None, :], sample_frequency=sr, frame_length=25, frame_shift=10, low_freq=low_freq, high_freq=high_freq, num_mel_bins=num_mel_bins, num_ceps=num_ceps, snip_edges=False)
    mfcc_feat = mfcc_feat.numpy()

    if cmn:
        cmn_opts = SlidingWindowCmnOptions()
        cmn_opts.cmn_window = 300
        cmn_opts.center = True
        mfcc_feat = SubMatrix(mfcc_feat)
        sliding_window_cmn(cmn_opts, mfcc_feat, mfcc_feat)
        mfcc_feat = mfcc_feat.numpy()

    if delta:
        delta_opts = DeltaFeaturesOptions()
        mfcc_feat = SubMatrix(mfcc_feat)
        mfcc_feat = compute_deltas(delta_opts, mfcc_feat)
        mfcc_feat = mfcc_feat.numpy()

    return torch.Tensor(mfcc_feat)

def extract_fbank(signal_data, sr, num_mel_bins=80, low_freq=20, high_freq=0.0):
    
    logmel_feat = kaldi_fbank(torch.Tensor(signal_data)[None, :], sample_frequency=sr, frame_length=25, frame_shift=10, low_freq=low_freq, high_freq=high_freq, num_mel_bins=num_mel_bins, snip_edges=False)

    return logmel_feat

def exclude_overlaping(annotation: 'Annotation') -> Annotation:
    overlap_timeline = Timeline()

    # for (seg1, track1), (seg2, track2) in annotation.co_iter(annotation):
    #     if seg1 == seg2 and track1 == track2:
    #         continue
    #     overlap_timeline.add(seg1 & seg2)
    
    annotation_no_overlap = annotation.extrude(annotation.get_overlap())
    return annotation_no_overlap

def index_aligned_labels(hyp, ref, n_classes=4):
    hyp = np.array(hyp)
    ref = np.array(ref)
    
    # hyp_index = []
    # for i in range(n_classes):
    #     tmp = hyp[ref == i]
    #     c = np.bincount(tmp)
    #     most_index = np.argmax(c)
    #     hyp_index.append(most_index)
    # print(hyp_index)

    err_min = 1.0
    final_permute = None
    for permute_array in permutations(np.arange(n_classes), n_classes):
        hyp_tmp = copy.deepcopy(hyp)

        # replace numbers
        for idx_class in range(n_classes):
            hyp_tmp[hyp_tmp == permute_array[idx_class]] = idx_class

        diff_array = ref - hyp_tmp
        diff_array[diff_array != 0] = 1

        err_tmp = np.mean(diff_array)
        if err_tmp < err_min:
            err_min = err_tmp
            final_permute = permute_array

    # print(final_permute)
        
    return final_permute