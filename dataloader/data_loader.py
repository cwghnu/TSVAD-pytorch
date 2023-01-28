import os
import sys
import torch
import random
import numpy as np
from utils.kaldi_data import KaldiData

def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)

class Dataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, path_dataset, mfcc_config, chunk_size=2000, max_speakers=4):

        self.kaldi_obj = KaldiData(path_dataset)
        self.mfcc_dir = os.path.join(path_dataset, "mfcc")
        self.ivec_dir = os.path.join(path_dataset, "ivec")
        self.label_dir = os.path.join(path_dataset, "label")
        
        self.nframes = nframes
        self.rate = mfcc_config["sampling_rate"]
        self.frame_len = mfcc_config["frame_length"] * self.rate
        self.frame_shift = mfcc_config["frame_shift"] * self.rate
        self.subsampling = 1
        self.chunk_size = chunk_size
        self.max_speakers = 4
        
        self.total_chunk = 0
        for rec in self.kaldi_obj.wavs:
            num_frames = _count_frames(self.kaldi_obj.reco2dur[rec]*self.rate, self.frame_len, self.frame_shift)
            num_chunks = num_frames // self.frame_lenchunk_size
            self.total_chunk += num_chunks
        self.total_chunk *= 2
        print("[Dataset Msg] total number of chunks: {}".format(self.total_chunk))

    @lru_cache
    def __getitem__(self, index):
        
        idx_utt = index % len(self.kaldi_obj.wavs)
        utt_id = self.kaldi_obj.wavs[idx_utt]
        
        mfcc_utt = np.load(os.path.join(self.mfcc_dir, utt_id + '.npy'))    # [num_frames, 72]
        ivec_utt = np.load(os.path.join(self.ivec_dir, utt_id + '.npy'))    # [num_speakers, 400]
        label_utt = np.load(os.path.join(self.label_dir, utt_id + '.npy'))  # [num_frames, num_speakers]
        
        assert len(mfcc_utt) == len(label_utt)
        assert len(ivec_utt) == label_utt.shape[1]
        
        max_start  = len(mfcc_utt) - self.chunk_size
        idx_start = random.randint(0, max_start)
        
        mfcc_utt = mfcc_utt[idx_start:(idx_start+self.chunk_size)]
        label_utt = label_utt[idx_start:(idx_start+self.chunk_size)]

        return {
            "mfcc": mfcc_utt,
            "label": label_utt,
            "ivector": ivec_utt,
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
