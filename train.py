import os
import sys
sys.path.insert(0,os.getcwd())

import time
import torch
import logging
import numpy as np
from pathlib import Path
from importlib import import_module
import json
import torch.nn as nn

from torch.utils.data import DataLoader
from itertools import permutations, combinations

from dataloader.data_loader import Dataset
from utils.loss import spk_emb_loss

def collate_fn(batches):
    feat_batches = [item['feat'] for item in batches]
    label_batches = [item['label'] for item in batches]
    vector_batches = [item['spk_vector'] for item in batches]
    index_batches = [item['index_spks'] for item in batches]
    
    feat_batches = torch.stack(feat_batches)
    label_batches = torch.stack(label_batches)
    vector_batches = torch.stack(vector_batches)
    
    egs = {
        'feat': feat_batches,
        'label': label_batches,
        "spk_vector": vector_batches,
        "index_spks": index_batches,
    }
    
    return egs

def train(train_config): 
    # Initial
    output_directory     = train_config.get('output_directory', '')
    max_iter             = train_config.get('max_iter', 100000)
    max_epoch             = train_config.get('max_epoch', 100000)
    batch_size           = train_config.get('batch_size', 128)
    nframes              = train_config.get('nframes', 40)
    chunk_step           = train_config.get('chunk_step', 20)
    iters_per_checkpoint = train_config.get('iters_per_checkpoint', 10000)
    iters_per_log        = train_config.get('iters_per_log', 1000)
    seed                 = train_config.get('seed', 1234)
    checkpoint_path      = train_config.get('checkpoint_path', '')
    trainer_type         = train_config.get('trainer_type', 'basic')
    epochs_per_eval      = train_config.get('epochs_per_eval', 5)

    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)   

    # Initial trainer
    module = import_module('trainer.{}'.format(trainer_type), package=None)
    TRAINER = getattr( module, 'Trainer')
    trainer = TRAINER( train_config, model_config)

    # Load checkpoint if the path is given 
    iteration = 1
    epoch = 0
    if checkpoint_path != "":
        iteration = trainer.load_checkpoint( checkpoint_path)
        iteration += 1  # next iteration is iteration + 1
        
    with open(train_config['mfcc_config']) as f:
        data = f.read()
    mfcc_config = json.loads(data)
    mfcc_config = mfcc_config["mfcc_config"]

    # Load training data
    trainset = Dataset(
        train_config['training_dir'], 
        mfcc_config, 
        chunk_size=nframes,
        chunk_step=chunk_step,
        vec_type=train_config['vec_type'], 
        feat_type=train_config['feat_type'],
        # use_mix_up=False,
    )    
    train_loader = DataLoader(
        trainset, 
        num_workers=train_config['num_workers'], 
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    # Load evaluation data
    evalset = Dataset(
        train_config['eval_dir'], 
        mfcc_config, 
        chunk_size=nframes,
        chunk_step=chunk_step,
        vec_type=train_config['vec_type'], 
        feat_type=train_config['feat_type'],
        use_mix_up=False,
        random_channel=False
    )    
    eval_loader = DataLoader(
        evalset, 
        num_workers=train_config['num_workers'],
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Get shared output_directory ready
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Prepare logger
    logger = logging.getLogger("logger")
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=str(output_directory/'Stat'))
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s",
                                  datefmt="%m-%d %H:%M:%S")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    logger.info("Output directory: {}".format(output_directory))
    logger.info("Training utterances: {}".format(len(trainset)))
    logger.info("Batch size: {}".format(batch_size))
    logger.info("# of frames per sample: {}".format(nframes))

    # ================ MAIN TRAINNIG LOOP! ===================
    
    logger.info("Start traininig...")

    loss_log = dict()
    # while iteration <= max_iter:
    while epoch < max_epoch:
        trainer.model.train()
        for i, batch in enumerate(train_loader):
            
            iteration, loss_detail, lr = trainer.step(batch, iteration=iteration)

            # Keep Loss detail
            for key,val in loss_detail.items():
                if key not in loss_log.keys():
                    loss_log[key] = list()
                loss_log[key].append(val)
            
            # Save model per N iterations
            # if iteration % iters_per_checkpoint == 0:
            #     checkpoint_path =  output_directory / "{}_{}".format(time.strftime("%m-%d_%H-%M", time.localtime()),iteration)
            #     trainer.save_checkpoint( checkpoint_path)

            # Show log per M iterations
            if iteration % iters_per_log == 0 and len(loss_log.keys()) > 0:
                mseg = 'Iter {}:'.format( iteration)
                for key,val in loss_log.items():
                    mseg += '  {}: {:.6f}'.format(key,np.mean(val))
                mseg += '  lr: {:.6f}'.format(lr)
                logger.info(mseg)
                loss_log = dict()

            # if iteration > max_iter:
            #     break
            
        epoch += 1

        if epoch % epochs_per_eval == 0:
            eval_loss = []
            trainer.model.eval()
            for i, batch in enumerate(eval_loader):
                with torch.no_grad():
                    for key in batch.keys():
                        if key != "index_spks":
                            batch[key] = batch[key].to("cuda:1")
                    preds, spk_emb = trainer.model(batch)
                    # if isinstance(results, set):
                    #     preds, spk_emb = results[0], results[1]
                    # else:
                    #     preds = results
                    #     spk_emb = None
                    targets = batch["label"]
                    bs, num_frames = targets.shape[0:2]
                    loss_batches = []
                    for idx, idx_batch in enumerate(batch["index_spks"]):
                        # loss_batches.append(torch.nn.BCELoss(reduction='sum')(preds[idx, :, idx_batch], batch["label"][idx, :, idx_batch]) / num_frames)
                        loss_diar = torch.nn.BCEWithLogitsLoss(reduction='sum')(preds[idx, :, idx_batch], batch["label"][idx, :, idx_batch]) / num_frames
                        if spk_emb is not None:
                            # loss_spk = spk_emb_loss(spk_emb[idx, idx_batch, :])
                            loss_batches.append(loss_diar)
                        else:
                            loss_batches.append(loss_diar)
                    loss = torch.stack(loss_batches).mean()
                    # loss = nn.BCELoss(reduction='sum')(preds, targets) / num_frames / bs
                    eval_loss.append(loss.item())
            mseg = 'Epoch {}:'.format( epoch)
            mseg += "Eval loss: {}".format(np.mean(eval_loss))
            logger.info(mseg)

        checkpoint_path =  output_directory / "{}_{}".format(time.strftime("%m-%d_%H-%M", time.localtime()),epoch)
        trainer.save_checkpoint( checkpoint_path)

        if epoch > max_epoch:
            break
        

    print('Finished')
        

if __name__ == "__main__":

    import argparse
    import json

    # import psutil
    # process = psutil.Process(os.getpid())
    # process.nice(psutil.IOPRIO_CLASS_RT)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/tsvad_config_xvec.json',
                        help='JSON file for configuration')

    parser.add_argument('-g', '--gpu', type=str, default='1,2',
                        help='Using gpu #')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global model_config
    model_config = config["model_config"]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    # torch.multiprocessing.set_start_method('spawn')

    train(train_config)
