import sys
import torch
import numpy as np
import torch.nn as nn
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
import psutil
import os
import math

class CNN_ReLU_BatchNorm(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3, stride=(1, 1), padding=1):
        super(CNN_ReLU_BatchNorm, self).__init__()
        self.cnn = nn.Sequential(
                      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                      nn.ReLU(),
                      nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
                   )

    def forward(self, feature):
        feature = self.cnn(feature)
        return feature

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    pe.require_grad = False

    return pe
     

class Model(nn.Module):
    def __init__(self, feat_dim=80, vec_dim=512, out_channels=[64, 64, 128, 128], rproj=512, nproj=512, cell=1024, max_speaker=4):
        super(Model, self).__init__()

        self.feat_dim = feat_dim
        self.vec_dim = vec_dim
        self.max_speaker = max_speaker
        self.rproj = rproj
        self.nproj = nproj

        self.pos_emb = positionalencoding1d(self.rproj, 2000)[None, ...]

        batchnorm = nn.BatchNorm2d(1, eps=0.001, momentum=0.99)
        
        cnn_relu_batchnorm1 = CNN_ReLU_BatchNorm(in_channels=1, out_channels=out_channels[0])
        cnn_relu_batchnorm2 = CNN_ReLU_BatchNorm(in_channels=out_channels[0], out_channels=out_channels[1])
        cnn_relu_batchnorm3 = CNN_ReLU_BatchNorm(in_channels=out_channels[1], out_channels=out_channels[2], stride=(1, 2))
        cnn_relu_batchnorm4 = CNN_ReLU_BatchNorm(in_channels=out_channels[2], out_channels=out_channels[3])
        
        self.cnn = nn.Sequential(
                      batchnorm,
                      cnn_relu_batchnorm1,
                      cnn_relu_batchnorm2,
                      cnn_relu_batchnorm3,
                      cnn_relu_batchnorm4
                   )
        
        self.linear_cat = nn.Linear(int(out_channels[-1]*self.feat_dim//2 + vec_dim), rproj)

        encoder_layer = nn.TransformerEncoderLayer(d_model=rproj, nhead=8, dim_feedforward=cell, batch_first=True, norm_first=True)
        self.transformer_speaker = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.linear_combime = nn.Linear(4*rproj, nproj)

        self.transformer_combine = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.output_layer = nn.Linear(nproj, self.max_speaker)

    def forward(self, batch):
        feats, targets, vectors = batch["feat"], batch["label"], batch["spk_vector"]

        pos_emb = self.pos_emb.to(feats.device)

        feats = feats[:, None, ...] # [B, 1， num_frames, 30]
        feats = self.cnn(feats)
        bs, chan, tframe, dim = feats.size()    # [B, 128, 2000, 15]
        
        feats    = feats.permute(0, 2, 1, 3)
        feats    = feats.contiguous().view(bs, tframe, chan*dim) # B x 1 x T x 1920
        feats    = feats.unsqueeze(1).repeat(1, 4, 1, 1)         # B x 4 x T x 1920
        vectors = vectors.view(bs, 4, self.vec_dim).unsqueeze(2)        # B x 4 x 1 x 512
        vectors = vectors.repeat(1, 1, tframe, 1)              # B x 4 x T x 512
        
        sd_in  = torch.cat((feats, vectors), dim=-1)            #  B x 4 x T x 2432
        sd_in  = self.linear_cat(sd_in).view(4*bs, tframe, -1)       # 4B x T x 384

        sd_in = sd_in + pos_emb[:, :tframe, :]

        sd_out = self.transformer_speaker(sd_in)               # 4B x T x 320
        sd_out = sd_out.contiguous().view(bs, 4, tframe, -1)     #  B x 4 x T x 320
        sd_out = sd_out.permute(0, 2, 1, 3)                      #  B x T x 4 x 320
        sd_out = sd_out.contiguous().view(bs, tframe, -1)        #  B x T x 1280

        sd_out = self.linear_combime(sd_out)

        outputs = self.transformer_combine(sd_out)                       #  B x T x 320

        preds   = self.output_layer(outputs)
        # preds   = nn.Sigmoid()(preds)
        
        return preds

if __name__ == "__main__":
    model = Model(feat_dim=24, rproj=512, nproj=512, cell=1024)
    model.eval()

    sampling_rate = 16000
    data_len = 30*60 * sampling_rate
    num_repeats = 1
    num_frames = int(data_len // (0.01 * sampling_rate))
    chunk_frames = 800

    time_list = []

    for i in range(num_repeats):
        time_total = 0
        for idx_start in range(0, num_frames, 600):
            batch = {
                "feat": torch.rand((1, chunk_frames, 24)),
                "label": torch.rand((1, chunk_frames, 4)),
                "spk_vector": torch.rand((1, 4, 512)),
            }
            time_st = time.time()
            with torch.no_grad():
                _ = model(batch)
            time_ed = time.time()
            time_total += time_ed - time_st
        
        time_list.append(time_total)
        
        flops = FlopCountAnalysis(model, batch)
        print("FLOPs: ", flops.total())
        print(flop_count_table(flops))
        print(parameter_count_table(model))

    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )

    print(np.mean(time_list))
        