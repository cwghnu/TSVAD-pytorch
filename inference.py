import os
import sys
sys.path.insert(0,os.getcwd())

import numpy as np
import torch
import logging
from tqdm import tqdm
from importlib import import_module
import json

from torch.utils.data import DataLoader
from dataloader.data_loader import Dataset

from utils.make_rttm import make_rttm
from utils.cal_der import cal_der

def inference(infer_config, model_config):
    # Initial
    model_type = infer_config.get('model_type', 'tsvad')
    model_path = infer_config.get('model_path', '')
    output_dir = infer_config.get('output_dir', '')
    nframes    = infer_config.get('nframes', 40)
    chunk_step = infer_config.get('chunk_step', 20)
    max_speakers    = infer_config.get('max_speakers', 4)
    hyp_rttm_dir = infer_config.get('hyp_rttm_dir', '')
    ref_rttm_dir = infer_config.get('ref_rttm_dir', '')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load Model
    module = import_module('model.{}'.format(model_type))
    MODEL = getattr(module, 'Model')
    model = MODEL(**model_config)
    model.load_state_dict(torch.load(model_path, map_location="cpu")['model'])
    
    print (model)

    device = torch.device("cuda:0")
    model = model.to(device)

    with open(infer_config['mfcc_config']) as f:
        data = f.read()
    mfcc_config = json.loads(data)
    mfcc_config = mfcc_config["mfcc_config"]

    # Load evaluation data
    evalset = Dataset(
        infer_config['eval_dir'], 
        mfcc_config, 
        chunk_size=nframes, 
        chunk_step=chunk_step,
        permute_spk=False,
        vec_type=infer_config['vec_type'], 
        feat_type=infer_config['feat_type']
    )

    # Prepare logger
    logger = logging.getLogger("logger")
    handler1 = logging.StreamHandler()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s",
                                  datefmt="%m-%d %H:%M:%S")
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    logger.info("Evaluation utterances: {}".format(len(evalset)))

    # ================ MAIN EVALUATION LOOP! ===================

    logger.info("Start evaluation...")

    utt_ids = list(evalset.kaldi_obj.wavs.keys())
    
    model.eval()
    for i, utt_id in enumerate(tqdm(utt_ids)):
        feat_utt = np.load(os.path.join(evalset.feat_dir, utt_id + '.npy'))    # [num_frames, 72]
        vec_utt = np.load(os.path.join(evalset.vec_dir, utt_id + '.npy'))    # [num_speakers, 400]
        label_utt = np.load(os.path.join(evalset.label_dir, utt_id + '.npy'))  # [num_frames, num_speakers]

        num_frames, num_speakers = label_utt.shape
        vec_dim = vec_utt.shape[-1]

        if num_speakers < max_speakers:
            # label_utt = np.concatenate((label_utt, np.zeros((num_frames, max_speakers - num_speakers))), axis=-1)
            # vec_utt = np.concatenate((vec_utt, np.zeros((max_speakers - num_speakers, vec_dim))), axis=0)

            label_utt = np.concatenate((label_utt, label_utt[:, :max_speakers - num_speakers]), axis=-1)
            vec_utt = np.concatenate((vec_utt, vec_utt[:max_speakers - num_speakers]), axis=0)

        out_chunks = []

        idx_start = 0
        num_chunks = int(feat_utt.shape[0] // nframes)
        for idx_chunk in range(num_chunks):
            if idx_chunk == num_chunks - 1:
                feat_utt_chunk = feat_utt[idx_start:]      # [num_frames, 72]
                label_utt_chunk = label_utt[idx_start:]    # [num_frames, num_speakers]
            else:
                feat_utt_chunk = feat_utt[idx_start:(idx_start+nframes)]      # [num_frames, 72]
                label_utt_chunk = label_utt[idx_start:(idx_start+nframes)]    # [num_frames, num_speakers]

            input_chunk = {
                "feat": torch.from_numpy(feat_utt_chunk).float().unsqueeze(0).to(device),
                "label": torch.from_numpy(label_utt_chunk).float().unsqueeze(0).to(device),
                "spk_vector": torch.from_numpy(vec_utt).float().unsqueeze(0).to(device),
            }

            with torch.no_grad():
                preds = model(input_chunk).squeeze(0).cpu().numpy()

            out_chunks.append(preds)

            idx_start += nframes

        outdata = np.vstack(out_chunks)

        outdata = outdata[:, :num_speakers]
        
        output_file = os.path.join(output_dir, utt_id)
        np.save(output_file, outdata)

    make_rttm(output_dir, hyp_rttm_dir)
    cal_der(hyp_rttm_dir, ref_rttm_dir)
                
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/tsvad_config_xvec.json',
                        help='JSON file for configuration')
    parser.add_argument('-g', '--gpu', type=str, default='1',
                        help='Using gpu #')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    infer_config = config["infer_config"]
    global model_config
    model_config = config['model_config']

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    inference(infer_config, model_config)
