import os
import sys
import torch
import random
import numpy as np

sys.path.insert(0,os.getcwd())
from utils.kaldi_data import KaldiData
from itertools import permutations


def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)

def random_dict(dict_in):
    dict_in_keys = list(dict_in.keys())
    random.shuffle(dict_in_keys)
    new_dict = {}
    for key in dict_in_keys:
        new_dict[key] = dict_in.get(key)
    return new_dict

class Dataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, path_dataset, mfcc_config, chunk_size=2000, chunk_step=1000, max_speakers=4, permute_spk=True, vec_type="ivec", feat_type="mfcc"):

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
        self.chunk_step = chunk_step
        self.max_speakers = max_speakers
        
        self.total_chunk = 0
        self.chunk_indices = []
        for rec in self.kaldi_obj.wavs:
            num_frames = _count_frames(self.kaldi_obj.reco2dur[rec]*self.rate, self.frame_len, self.frame_shift)
            num_chunks = (num_frames - self.chunk_size + self.chunk_step) // self.chunk_step
            self.total_chunk += num_chunks

            for idx_chunk in range(num_chunks):
                if idx_chunk*self.chunk_step + self.chunk_size < num_frames:
                    self.chunk_indices.append(
                        (rec, idx_chunk*self.chunk_step, idx_chunk*self.chunk_step + self.chunk_size)
                    )

        print("[Dataset Msg] total number of chunks: {}".format(self.total_chunk))

        self.utt_ids = list(self.kaldi_obj.wavs.keys())
        self.permute_spk = permute_spk
        self.all_permutations = list(permutations(np.arange(self.max_speakers), self.max_speakers))

    def find_reco_with_diff_spks(self, reco_id):

        current_spks = self.kaldi_obj.reco2spk[reco_id]
        return_reco = ""

        shuffled_reco2spk = random_dict(self.kaldi_obj.reco2spk)

        for key, value in shuffled_reco2spk.items():
            if len(value & current_spks) == 0 and len(value) + len(current_spks) >= self.max_speakers:
                return_reco = key
                # print(reco_id, current_spks, "-->", return_reco, value)
                break

        return return_reco

    # @lru_cache
    def __getitem__(self, index):

        utt_id, idx_start, idx_end = self.chunk_indices[index]
        
        # idx_utt = index % len(self.utt_ids)
        # utt_id = self.utt_ids[idx_utt]
        
        feat_utt = np.load(os.path.join(self.feat_dir, utt_id + '.npy'))    # [num_frames, 72]
        vec_utt = np.load(os.path.join(self.vec_dir, utt_id + '.npy'))    # [num_speakers, 400]
        label_utt = np.load(os.path.join(self.label_dir, utt_id + '.npy'))  # [num_frames, num_speakers]
        
        assert len(feat_utt) == len(label_utt)
        assert len(vec_utt) == label_utt.shape[1]

        # max_start  = len(feat_utt) - self.chunk_size
        # idx_start = random.randint(0, max_start)
        # idx_end = idx_start + self.chunk_size

        # print(utt_id, idx_start, idx_end)
        
        feat_utt = feat_utt[idx_start:idx_end]      # [num_frames, 72]
        label_utt = label_utt[idx_start:idx_end]    # [num_frames, num_speakers]

        num_frames, num_speakers = label_utt.shape
        ivec_dim = vec_utt.shape[-1]
        
        index_array = list(np.arange(num_speakers))

        if num_speakers < self.max_speakers:
            # pad zero vectors
            # label_utt = np.concatenate((label_utt, np.zeros((num_frames, self.max_speakers - num_speakers))), axis=-1)
            # vec_utt = np.concatenate((vec_utt, np.zeros((self.max_speakers - num_speakers, ivec_dim))), axis=0)

            # pad the same speakers
            # idx_pad_spks = random.choices(np.arange(num_speakers), k=self.max_speakers-num_speakers)
            # label_utt = np.concatenate((label_utt, label_utt[:, idx_pad_spks]), axis=-1)
            # vec_utt = np.concatenate((vec_utt, vec_utt[idx_pad_spks]), axis=0)
            
            # print("before padding spks, vec_utt shape: {}".format(vec_utt.shape))

            # pad speakers from different sessions
            reco_id = self.find_reco_with_diff_spks(utt_id)
            vec_tmp = np.load(os.path.join(self.vec_dir, reco_id + '.npy'))
            label_utt = np.concatenate((label_utt, np.zeros((num_frames, self.max_speakers - num_speakers))), axis=-1)
            vec_utt = np.concatenate((vec_utt, vec_tmp[:self.max_speakers - num_speakers]), axis=0)
            
            # print("after padding spks, vec_utt shape: {}".format(vec_utt.shape))

        index_array = np.arange(num_speakers)

        if self.permute_spk:
            idx_permutation = random.randint(0, len(self.all_permutations)-1)
            array_permutation = list(self.all_permutations[idx_permutation])
            vec_utt = vec_utt[array_permutation]
            label_utt = label_utt[..., array_permutation]
            
            index_array = np.argwhere(np.array(array_permutation) < num_speakers)[:, 0]

            # print("{} spks, permutation: {}, index_array: {}".format(num_speakers, array_permutation, index_array))


        return {
            "feat": torch.from_numpy(feat_utt).float(),
            "label": torch.from_numpy(label_utt).float(),
            "spk_vector": torch.from_numpy(vec_utt).float(),
            "index_spks": torch.from_numpy(index_array).long(),
        }
    
    def __len__(self):
        return self.total_chunk


def test_dataset():
    import json
    with open("config/tsvad_config_xvec.json") as f:
        data = f.read()
    config = json.loads(data)
    infer_config = config["infer_config"]
    train_config = config["train_config"]
    global model_config
    model_config = config['model_config']

    with open(infer_config['mfcc_config']) as f:
        data = f.read()
    mfcc_config = json.loads(data)
    mfcc_config = mfcc_config["mfcc_config"]

    nframes = infer_config.get('nframes', 40)
    chunk_step = infer_config.get('chunk_step', 20)

    evalset = Dataset(
        # infer_config['eval_dir'],
        train_config['training_dir'],
        mfcc_config, 
        chunk_size=nframes, 
        chunk_step=chunk_step,
        permute_spk=True,
        vec_type=infer_config['vec_type'], 
        feat_type=infer_config['feat_type']
    )

    # for i in range(evalset.total_chunk):
    #     evalset.__getitem__(i)

if __name__ == "__main__":
    test_dataset()