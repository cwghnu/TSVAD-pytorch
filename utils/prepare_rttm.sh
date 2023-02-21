# work_dir=/exhome1/weiguang/data/Alimeeting/Train_Ali_far
# textgrid_dir=$work_dir/textgrid_dir
# dia_rttm_dir=$work_dir/rttm_groundtruth
# wav_dir=$work_dir/audio_dir

work_dir=/exhome1/weiguang/data/AISHELL4/train_L
textgrid_dir=$work_dir/TextGrid
dia_rttm_dir=$work_dir/rttm_groundtruth
wav_dir=$work_dir/wav

stage=1
stop_stage=8
nj=4.

mkdir -p $dia_rttm_dir || exit 1;

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Prepare the AliMeeting data
    echo "Prepare Alimeeting data"
    find $wav_dir -name "*\.wav" > $work_dir/wavlist
    sort  $work_dir/wavlist > $work_dir/tmp
    cp $work_dir/tmp $work_dir/wavlist
    awk -F '/' '{print $NF}' $work_dir/wavlist | awk -F '.' '{print $1}' > $work_dir/uttid
    paste $work_dir/uttid $work_dir/wavlist > $work_dir/wav.scp 
    paste $work_dir/uttid $work_dir/uttid > $work_dir/utt2spk
    cp $work_dir/utt2spk $work_dir/spk2utt
    cp $work_dir/uttid $work_dir/text
fi

if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Process textgrid to obtain rttm label"
    find -L $textgrid_dir -iname "*.TextGrid" >  $work_dir/textgrid.flist
    sort  $work_dir/textgrid.flist  > $work_dir/tmp
    cp $work_dir/tmp $work_dir/textgrid.flist 
    paste $work_dir/uttid $work_dir/textgrid.flist > $work_dir/uttid_textgrid.flist
    while read text_file
    do
        text_grid=`echo $text_file | awk '{print $1}'`
        text_grid_path=`echo $text_file | awk '{print $2}'`
        python ./utils/make_textgrid_rttm.py --input_textgrid_file $text_grid_path \
                                           --uttid $text_grid \
                                           --output_rttm_file $dia_rttm_dir/${text_grid}.rttm
    done < $work_dir/uttid_textgrid.flist
fi