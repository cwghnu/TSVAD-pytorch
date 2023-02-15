import numpy as np
import torch
from collections import OrderedDict
import argparse
import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def average_model(ifiles, ofile):
    omodel = OrderedDict()

    for ifile in ifiles:
        tmp_model = torch.load(ifile, map_location="cpu")['model']
        for k, v in tmp_model.items():
            omodel[k] = omodel.get(k, 0) + v

    for k, v in omodel.items():
        omodel[k] = v / len(ifiles)

    torch.save( {
        'model': omodel,
    }, ofile)

def average_model_from_dir(file_dir, num_avg=5):

    log_file = os.path.join(file_dir, "Stat")
    loss_dict = {}
    pattern_epoch = r".*Epoch ([\d]+?):.*"
    pattern_loss = r".*Eval loss: ([\d]\.?[\d]+?)\n"
    with open(log_file, "r") as f_log:
        for line in f_log:
            if "Epoch" in line:
                num_epoch = re.findall(pattern_epoch, line)[0]
                loss_epoch = float(re.findall(pattern_loss, line)[0])
                loss_dict[num_epoch] = loss_epoch

    sorted_loss_dict = sorted(loss_dict.items(), key=lambda x: x[1])
    selected_epoch = [num_epoch for num_epoch, loss in sorted_loss_dict[:num_avg]]

    model_files = []
    for tmp_file in os.listdir(file_dir):
        if "Stat" in tmp_file or "avg" in tmp_file:
            continue
        if tmp_file.split("_")[-1] in selected_epoch:
            model_files.append(tmp_file)

    # model_files.sort()
    # model_files = model_files[-num_avg:]

    selected_files = [os.path.join(file_dir, file_name) for file_name in model_files]
    output_file = os.path.join(file_dir, "avg")
    average_model(selected_files, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints/tsvad_xvec_logmel_12"),
                        help='model file path')
    args = parser.parse_args()

    average_model_from_dir(args.dir)