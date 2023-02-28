import torch 
import numpy as np
from itertools import permutations, combinations
from torch.nn import CosineEmbeddingLoss

def spk_emb_loss(spk_emb):
    # spk_emb: [num_spks, num_feat]

    num_spks = spk_emb.shape[0]
    idx_combs = combinations(np.arange(num_spks), 2)
    idx_combs = np.array(list(idx_combs))   # [num_combines, 2]
    idx_combs = torch.from_numpy(idx_combs)

    pair_emb_1 = spk_emb[idx_combs[:, 0]]
    pair_emb_2 = spk_emb[idx_combs[:, 1]]

    triplet_loss = CosineEmbeddingLoss(margin=0.2, reduction="sum")

    target = - torch.ones(len(pair_emb_1)).to(spk_emb.device)

    loss = triplet_loss(pair_emb_1, pair_emb_2, target)

    return loss