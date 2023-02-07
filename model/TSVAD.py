import sys
import torch
import numpy as np
import torch.nn as nn
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table

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
     

class BLSTMP(nn.Module):
    def __init__(self, n_in, n_hidden, nproj=160, dropout=0, num_layers=1):
        super(BLSTMP, self).__init__()

        self.num_layers = num_layers

        self.rnns = nn.ModuleList([nn.LSTM(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True)])
        self.linears = nn.ModuleList([nn.Linear(2*n_hidden, 2*nproj)])

        for i in range(num_layers-1):
            self.rnns.append(nn.LSTM(2*nproj, n_hidden, bidirectional=True, dropout=dropout, batch_first=True))
            self.linears.append(nn.Linear(2*n_hidden, 2*nproj))
    
    def forward(self, feature):
        recurrent, _ = self.rnns[0](feature)
        output = self.linears[0](recurrent)

        for i in range(self.num_layers-1):
            output, _ = self.rnns[i+1](output)
            output = self.linears[i+1](output)
        
        return output
     

class Model(nn.Module):
    def __init__(self, feat_dim=80, vec_dim=400, out_channels=[64, 64, 128, 128], rproj=256, nproj=256, cell=896):
        super(Model, self).__init__()

        self.feat_dim = feat_dim
        self.vec_dim = vec_dim
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

        # self.linear_speech = nn.Linear(int(out_channels[-1]*self.feat_dim//2), vec_dim)
        
        self.linear_cat = nn.Linear(int(out_channels[-1]*self.feat_dim//2 + vec_dim), 3*rproj)

        # self.rnn_speaker_detection = BLSTMP(3*rproj, cell, num_layers=2)
        # self.rnn_combine = BLSTMP(8*nproj, cell)

        self.rnn_speaker_detection = nn.LSTM(3*rproj, cell, bidirectional=True, batch_first=True, proj_size=nproj, num_layers=2)
        self.rnn_combine = nn.LSTM(8*nproj, cell, bidirectional=True, batch_first=True, proj_size=nproj, num_layers=1)

        self.output_layer = nn.Linear(nproj*2, 4)

    def forward(self, batch):
        feats, targets, vectors = batch["feat"], batch["label"], batch["spk_vector"]

        feats = feats[:, None, ...] # [B, 1， num_frames, 30]
        feats = self.cnn(feats)
        bs, chan, tframe, dim = feats.size()    # [B, 128, 2000, 15]
        
        feats    = feats.permute(0, 2, 1, 3)
        feats    = feats.contiguous().view(bs, tframe, chan*dim) # B x 1 x T x 1920
        feats    = feats.unsqueeze(1).repeat(1, 4, 1, 1)         # B x 4 x T x 1920
        vectors = vectors.view(bs, 4, self.vec_dim).unsqueeze(2)        # B x 4 x 1 x 512
        vectors = vectors.repeat(1, 1, tframe, 1)              # B x 4 x T x 512
        
        # feats = self.linear_speech(feats)
        sd_in  = torch.cat((feats, vectors), dim=-1)            #  B x 4 x T x 2432
        sd_in  = self.linear_cat(sd_in).view(4*bs, tframe, -1)       # 4B x T x 384

        sd_out, _ = self.rnn_speaker_detection(sd_in)               # 4B x T x 320
        sd_out = sd_out.contiguous().view(bs, 4, tframe, -1)     #  B x 4 x T x 320
        sd_out = sd_out.permute(0, 2, 1, 3)                      #  B x T x 4 x 320
        sd_out = sd_out.contiguous().view(bs, tframe, -1)        #  B x T x 1280

        outputs, _ = self.rnn_combine(sd_out)                       #  B x T x 320

        # outputs = outputs.contiguous().view(bs, tframe, 4, -1)   #  B x T x 4 x 80
        # preds   = self.output_layer(outputs).squeeze(-1)         #  B x T x 4
        # preds   = nn.Sigmoid()(preds)

        preds   = self.output_layer(outputs)
        preds   = nn.Sigmoid()(preds)

        
        # loss = nn.BCELoss(reduction='sum')(preds, targets) / tframe / bs
        # loss_detail = {"diarization loss": loss.item()}
        
        return preds

if __name__ == "__main__":
    model = Model()

    data_len = 30*60
    sampling_rate = 16000
    num_repeats = 1
    num_frames = int(data_len // (0.01 * sampling_rate))

    time_list = []

    for i in range(num_repeats):
        batch = {
            "feat": torch.rand((1, num_frames, 80)),
            "label": torch.rand((1, num_frames, 4)),
            "spk_vector": torch.rand((1, 4, 400)),
        }
        
        time_st = time.time()
        _ = model(batch)
        time_ed = time.time()
        time_list.append(time_ed - time_st)
        
        flops = FlopCountAnalysis(model, batch)
        print("FLOPs: ", flops.total())
        print(flop_count_table(flops))
        print(parameter_count_table(model))

    print(np.mean(time_list))
        