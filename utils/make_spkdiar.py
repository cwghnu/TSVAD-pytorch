import argparse
import os
import subprocess
import re
import soundfile as sf

import pandas as pd
import numpy as np

def find_audios(wav_path):

    command = 'find %s -name "*.wav"' % (wav_path)
    # command = 'find %s -name "S??_U??.wav"' % (wav_path)
    wavs = subprocess.check_output(command, shell=True).decode('utf-8').splitlines()
    keys = [ os.path.splitext(os.path.basename(wav))[0] for wav in wavs ]
    #remove CH1 from chime5 U0x
    keys = [re.sub(r'\.CH1$','', k) for k in keys]
    data = {'key': keys, 'file_path': wavs}
    df_wav = pd.DataFrame(data)
    return df_wav

def sort_rttm(rttm):
    return rttm.sort_values(by=['file_id','tbeg'])

def rttm_is_sorted_by_tbeg(rttm):
    tbeg=rttm['tbeg'].values
    file_id=rttm['file_id'].values
    return np.all(np.logical_or(tbeg[1:]-tbeg[:-1]>=0,
                                file_id[1:] != file_id[:-1]))

def read_rttm(list_path, sep=' ', rttm_suffix=''):

    rttm_file='%s/rttm' % (list_path)
    rttm = pd.read_csv(rttm_file, sep=sep, header=None,
                       names=['segment_type','file_id','chnl','tbeg','tdur',
                              'ortho','stype','name','conf','slat', 'noname'],
                       dtype={'name': np.str, 'chnl': np.int, 'tbeg': np.float, 'tdur': np.float})
    #remove empty lines:
    index = (rttm['tdur']>= 0.025)
    rttm = rttm[index]
    rttm['ortho'] = '<NA>'
    rttm['stype'] = '<NA>'
    if not rttm_is_sorted_by_tbeg(rttm):
        print('RTTM %s not properly sorted, sorting it' % (rttm_file))
        rttm = sort_rttm(rttm)

    return rttm

def filter_wavs(df_wav, file_names):
    df_wav = df_wav.loc[df_wav['key'].isin(file_names)].sort_values('key')
    return df_wav

def write_wav(df_wav, output_path, bin_wav=False):

    with open(output_path + '/wav.scp', 'w') as f:
        for key,file_path in zip(df_wav['key'], df_wav['file_path']):
            if bin_wav:
                f.write('%s sox %s -t wav - remix 1 | \n' % (key, file_path))
            else:
                f.write('%s %s\n' % (key, file_path))

def write_utt2spk(file_names, output_path, rttm):
    
    with open(output_path + '/utt2spk', 'w') as f:
        for row in rttm.itertuples():
            tbeg = row.tbeg
            tend = row.tbeg + row.tdur
            segment_id = '%s_%07d_%07d' % (row.file_id, int(tbeg*100), int(tend*100))
            f.write('%s %s\n' % (row.name+'_'+segment_id, row.name))

def write_spk2utt(file_names, output_path, rttm):
    
    with open(output_path + '/spk2utt', 'w') as f:
        for row in rttm.itertuples():
            tbeg = row.tbeg
            tend = row.tbeg + row.tdur
            segment_id = '%s_%07d_%07d' % (row.file_id, int(tbeg*100), int(tend*100))
            f.write('%s %s\n' % (row.name, row.name+'_'+segment_id))

def write_rttm_spk(df_vad, output_path):
    file_path = output_path + '/rttm'
    df_vad[['segment_type', 'file_id', 'chnl',
            'tbeg','tdur','ortho', 'stype',
            'name', 'conf', 'slat']].to_csv(
                file_path, sep=' ', float_format='%.3f',
                index=False, header=False)

def write_segm_fmt(rttm_vad, output_path):
    with open(output_path + '/segments', 'w') as f:
        for row in rttm_vad.itertuples():
            tbeg = row.tbeg
            tend = row.tbeg + row.tdur
            segment_id = '%s_%07d_%07d' % (row.file_id, int(tbeg*100), int(tend*100))
            f.write('%s %s %.2f %.2f\n' % (row.name+'_'+segment_id, row.file_id, tbeg, tend))

def write_reco2dur(df_wav, output_path):
    with open(output_path + '/reco2dur', 'w') as f:
        for key,file_path in zip(df_wav['key'], df_wav['file_path']):
            data_signal, sampling_rate = sf.read(file_path)
            data_len = data_signal.shape[0] / sampling_rate
            f.write('%s %f\n' % (key, data_len))

def make_spkdiar(list_path, wav_path, output_path, bin_wav):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print('read audios')
    df_wav = find_audios(wav_path)
    print('read rttm')
    rttm = read_rttm(list_path)
    
    print(rttm[:10])
    print(df_wav[:10])
    
    print('make wav.scp')
    file_names = rttm['file_id'].sort_values().unique()
    df_wav = filter_wavs(df_wav, file_names)
    write_wav(df_wav, output_path, bin_wav)

    print('write utt2spk')
    # write_dummy_utt2spk(file_names, output_path)
    write_utt2spk(file_names, output_path, rttm)

    print('write spk2utt')
    write_spk2utt(file_names, output_path, rttm)

    print('write diar rttm')
    write_rttm_spk(rttm, output_path)

    #write vad in segment format
    print('write vad segments')
    write_segm_fmt(rttm, output_path)

    print('write reco2dur')
    write_reco2dur(df_wav, output_path)

if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,                
        fromfile_prefix_chars='@',
        description='Make JSALT19 datasets for spk diarization')

    parser.add_argument('--list-path', dest='list_path', default="/export/home2/cwguang/datasets/Train_Ali_far", required=False)
    parser.add_argument('--wav-path', dest='wav_path', default="/export/home2/cwguang/datasets/Train_Ali_far/audio_dir", required=False)
    parser.add_argument('--output-path', dest='output_path', default="/export/home2/cwguang/datasets/Train_Ali_far",  required=False)
    parser.add_argument('--bin-wav', dest='bin_wav', default=False, action='store_true')
    args=parser.parse_args()
    
    make_spkdiar(**vars(args))