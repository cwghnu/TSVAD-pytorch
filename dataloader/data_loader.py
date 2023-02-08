import os
import sys
import torch
import random
import numpy as np
from utils.kaldi_data import KaldiData
from itertools import permutations

def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)

class Dataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, path_dataset, mfcc_config, chunk_size=2000, max_speakers=4, permute_spk=True, vec_type="ivec", feat_type="mfcc"):

        self.vec_type = vec_type
        self.feat_type = feat_type
        if self.feat_type == "mfcc":
            self.feat_type = self.feat_type + "_" + self.vec_type
        self.kaldi_obj = KaldiData(path_dataset)
        self.feat_dir = os.path.join(path_dataset, self.feat_type)
        self.vec_dir = os.path.join(path_dataset, self.vec_type)
        self.label_dir = os.path.join(path_dataset, "label" + "_" + self.vec_type)
        
        self.rate = mfcc_config["sampling_rate"]
        self.frame_len = mfcc_config["frame_length"] * self.rate / 1000
        self.frame_shift = mfcc_config["frame_shift"] * self.rate  / 1000
        self.subsampling = 1
        self.chunk_size = chunk_size
        self.max_speakers = max_speakers
        
        self.total_chunk = 0
        for rec in self.kaldi_obj.wavs:
            num_frames = _count_frames(self.kaldi_obj.reco2dur[rec]*self.rate, self.frame_len, self.frame_shift)
            num_chunks = num_frames // self.chunk_size
            self.total_chunk += num_chunks
        print("[Dataset Msg] total number of chunks: {}".format(self.total_chunk))

        self.utt_ids = list(self.kaldi_obj.wavs.keys())
        self.permute_spk = permute_spk
        self.all_permutations = list(permutations(np.arange(self.max_speakers), self.max_speakers))

    # @lru_cache
    def __getitem__(self, index):
        
        idx_utt = index % len(self.utt_ids)
        utt_id = self.utt_ids[idx_utt]
        
        feat_utt = np.load(os.path.join(self.feat_dir, utt_id + '.npy'))    # [num_frames, 72]
        vec_utt = np.load(os.path.join(self.vec_dir, utt_id + '.npy'))    # [num_speakers, 400]
        label_utt = np.load(os.path.join(self.label_dir, utt_id + '.npy'))  # [num_frames, num_speakers]
        
        assert len(feat_utt) == len(label_utt)
        assert len(vec_utt) == label_utt.shape[1]
        
        max_start  = len(feat_utt) - self.chunk_size
        idx_start = random.randint(0, max_start)
        
        feat_utt = feat_utt[idx_start:(idx_start+self.chunk_size)]      # [num_frames, 72]
        label_utt = label_utt[idx_start:(idx_start+self.chunk_size)]    # [num_frames, num_speakers]

        num_frames, num_speakers = label_utt.shape
        ivec_dim = vec_utt.shape[-1]

        if num_speakers < self.max_speakers:
            # label_utt = np.concatenate((label_utt, np.zeros((num_frames, self.max_speakers - num_speakers))), axis=-1)
            # vec_utt = np.concatenate((vec_utt, np.zeros((self.max_speakers - num_speakers, ivec_dim))), axis=0)

            label_utt = np.concatenate((label_utt, label_utt[:, :self.max_speakers - num_speakers]), axis=-1)
            vec_utt = np.concatenate((vec_utt, vec_utt[:self.max_speakers - num_speakers]), axis=0)

        if self.permute_spk:
            idx_permutation = random.randint(0, len(self.all_permutations)-1)
            array_permutation = list(self.all_permutations[idx_permutation])
            vec_utt = vec_utt[array_permutation]
            label_utt = label_utt[..., array_permutation]


        return {
            "feat": torch.from_numpy(feat_utt).float(),
            "label": torch.from_numpy(label_utt).float(),
            "spk_vector": torch.from_numpy(vec_utt).float(),
        }
    
    def __len__(self):
        return self.total_chunk


class EvalDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, feats_dir, ivectors_dir):
        
        self.utt2feat   = files_to_dict(os.path.join(feats_dir,'feats.scp'))
        self.utt2iv     = files_to_dict(os.path.join(ivectors_dir,'ivector_online.scp'))
        self.utt_list   = [k for k in self.utt2feat.keys()]
        
    def __getitem__(self, index):
        utt         = self.utt_list[index]

        feat        = load_scp_to_torch(self.utt2feat[utt]).unsqueeze(0).cuda()
        ivectors    = load_scp_to_torch(self.utt2iv[utt]).mean(dim=0).cuda()
        
        return utt, feat, ivectors
    
    def __len__(self):
        return len(self.utt_list)
