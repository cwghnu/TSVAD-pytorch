import argparse
import numpy as np
import os
import sys
from scipy.signal import medfilt
sys.path.insert(0,os.getcwd())
from utils.diarization import read_rttm, rttm2annotation
from pyannote.core import Annotation, Segment, Timeline

def make_rttm(hyp_probs_path, hyp_rttm_path):

    # hyp_probs_path = "/exhome1/weiguang/code/TSVAD-pytorch/exp/hyp_probs"
    # hyp_rttm_path = "/exhome1/weiguang/code/TSVAD-pytorch/exp/hyp_rttm"

    threshold = 0.5
    median = 51
    sampling_rate = 16000
    frame_shift = sampling_rate * 0.01
    subsampling = 1

    if not os.path.exists(hyp_rttm_path):
        os.makedirs(hyp_rttm_path)

    filepaths = os.listdir(hyp_probs_path)

    for filepath in filepaths:
        session, _ = os.path.splitext(os.path.basename(filepath))
        file_full_path = os.path.join(hyp_probs_path, filepath)
        rttm_full_path = os.path.join(hyp_rttm_path, session+".rttm")
        data = np.load(file_full_path)

        data = np.where(data > threshold, 1, 0)

        if median > 1:
            data = medfilt(data, (median, 1))

        with open(rttm_full_path, 'w') as wf:

            for spkid, frames in enumerate(data.T):
                frames = np.pad(frames, (1, 1), 'constant')
                changes, = np.where(np.diff(frames, axis=0) != 0)
                fmt = "SPEAKER {:s} 1 {:.2f} {:.2f} <NA> <NA> {:s} <NA> <NA>"
                for s, e in zip(changes[::2], changes[1::2]):
                    dur = (e - s) * frame_shift * subsampling / sampling_rate
                    print(fmt.format(
                            session,
                            s * frame_shift * subsampling / sampling_rate,
                            dur,
                            str(spkid)), file=wf)

        rttm_content = read_rttm(rttm_full_path)
        annot = rttm2annotation(rttm_content)
        annot.support(collar=0.3)

        filtered_annot = Annotation()
        for segment, track, label in annot.itertracks(yield_label=True):
            if segment.end - segment.start > 0.1:
                filtered_annot[Segment(segment.start, segment.end)] = label
        with open(rttm_full_path, 'w') as wf:
            filtered_annot.write_rttm(wf)

if __name__ == "__main__":
    hyp_probs_path = "/exhome1/weiguang/code/TSVAD-pytorch/exp/hyp_probs"
    hyp_rttm_path = "/exhome1/weiguang/code/TSVAD-pytorch/exp/hyp_rttm"
    make_rttm(hyp_probs_path, hyp_rttm_path)