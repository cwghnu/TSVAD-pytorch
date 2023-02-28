import torch

# from .radam import RAdam 
from importlib import import_module

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.loss import spk_emb_loss


class Trainer(object):
    def __init__(self, train_config, model_config):
        learning_rate  = train_config.get('learning_rate', 1e-4)
        model_type     = train_config.get('model_type', 'tsvad')
        self.opt_param = train_config.get('optimize_param', {
                                'optim_type': 'RAdam',
                                'learning_rate': 1e-4,
                                'max_grad_norm': 10,
                                'lr_scheduler':{
                                    'step_size': 100000,
                                    'gamma': 0.5,
                                    'last_epoch': -1
                                }
                            })    

        self.gpus = train_config.get('gpus', [1,0])
        self.device = torch.device("cuda:{}".format(self.gpus[0]))

        module = import_module('model.{}'.format(model_type), package=None)
        MODEL = getattr(module, 'Model')
        model = MODEL(**model_config).to(self.device)

        print(model)

        self.model = model.to(self.device)
        self.learning_rate = learning_rate

        if self.opt_param['optim_type'].upper() == 'RADAM':
            # self.optimizer = RAdam( self.model.parameters(), 
            #                         lr=self.opt_param['learning_rate'],
            #                         betas=(0.5,0.999),
            #                         weight_decay=0.0)
            self.optimizer = torch.optim.RAdam( self.model.parameters(), 
                                    lr=self.opt_param['learning_rate'],
                                    betas=(0.5,0.999),
                                    weight_decay=0.0)
        else:
            self.optimizer = torch.optim.Adam( self.model.parameters(),
                                               lr=self.opt_param['learning_rate'],
                                               betas=(0.5,0.999),
                                               weight_decay=0.0)

        if 'lr_scheduler' in self.opt_param.keys():
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                                optimizer=self.optimizer,
                                **self.opt_param['lr_scheduler']
                            )
        else:
            self.scheduler = None


        self.iteration = 0
        self.model.train()

    def step(self, input, iteration=None):
        assert self.model.training
        self.model.zero_grad()

        # input = [x.to(self.device) for x in input]
        for key in input.keys():
            if key != "index_spks":
                input[key] = input[key].to(self.device)

        preds, spk_emb = torch.nn.parallel.data_parallel(
            self.model,
            (input),
            self.gpus,
            self.gpus[0],
        )
        # if isinstance(results, set):
        #     preds, spk_emb = results[0], results[1]
        # else:
        #     preds = results
        #     spk_emb = None
        bs, tframe = input["label"].shape[0:2]

        loss_batches = []
        loss_batches_spks = []
        for idx, idx_batch in enumerate(input["index_spks"]):
            # print(idx, idx_batch)
            # loss_batches.append(torch.nn.BCELoss(reduction='sum')(preds[idx, :, idx_batch], input["label"][idx, :, idx_batch]) / tframe) 
            loss_diar = torch.nn.BCEWithLogitsLoss(reduction='sum')(preds[idx, :, idx_batch], input["label"][idx, :, idx_batch]) / tframe
            if spk_emb is not None:
                loss_spk = spk_emb_loss(spk_emb[idx, idx_batch, :])
                loss_batches.append(loss_diar)
                loss_batches_spks.append(loss_spk)
            else:
                loss_batches.append(loss_diar)

        loss_batches = torch.stack(loss_batches).mean()
        if spk_emb is not None:
            loss_batches_spks = torch.stack(loss_batches_spks).mean()
            loss = loss_batches
            # loss = loss_batches + loss_batches_spks
            loss_detail = {"diarization loss": loss_batches.item(), "spks loss": loss_batches_spks.item()}
        else:
            loss = loss_batches
            loss_detail = {"diarization loss": loss_batches.item()}

        # loss = torch.nn.BCELoss(reduction='sum')(preds, input["label"]) / tframe / bs
        
        # loss, loss_detail = self.model(input)

        loss.backward()
        if self.opt_param['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.opt_param['max_grad_norm'])
        self.optimizer.step()
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']

        if self.scheduler is not None:
            self.scheduler.step()

        if iteration is not None:
            self.iteration = iteration + 1
        else:
            self.iteration += 1

        return self.iteration, loss_detail, learning_rate


    def save_checkpoint(self, checkpoint_path):
        torch.save( {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iteration': self.iteration,
            }, checkpoint_path)
        print("Saved state dict. to {}".format(checkpoint_path))


    def load_checkpoint(self, checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint_data['model'])
        # self.optimizer.load_state_dict(checkpoint_data['optimizer'])
        return checkpoint_data['iteration']
