import mne
import os
import argparse
import sys
import pandas as pd
import shutil
import h5py
import natsort
from fx_bids import make_coordsys_ieeg_json, make_electrodes_ieeg_tsv, \
    make_coordsys_eeg_json, make_electrodes_eeg_tsv, export_data_bids, \
    make_events_json, make_mri_json


program_help = 'Converts data to BIDS format'

task = 'seegstim'
event_id = {'stim': 1}

parser = argparse.ArgumentParser(description=program_help)
parser.add_argument('-r', '--runs_table', help='Filepath to the runs table')
parser.add_argument('-d', '--dir_base', help='Path to base data directory')
parser.add_argument('-a', '--dir_anon', help='Path to anonymized MRIs')

args = parser.parse_args()
dir_out = os.path.join(args.dir_base, 'share', 'gnode', 'Localize-MI')
dir_spatial = os.path.join(args.dir_base, 'spatial')

if not os.path.isfile(args.runs_table):
    print('Runs table is required')
    sys.exit()

runs_table = pd.read_csv(args.runs_table, dtype={'subj_id': str, 'run_id': str, 'fname': str})
subjects = runs_table.subj_id.unique()

# create folder structure and populate it
folders_epo = ['eeg', 'ieeg', 'anat']
folders_sloc = ['anat', 'xfm', 'fwd']

for s in subjects:
    subj_code = 's%s' % s.strip('0')
    subj_id = 'sub-%s' % s

    # folders
    _ = [os.makedirs(os.path.join(dir_out, 'derivatives', 'epochs',
                                  subj_id, folder)) for folder in folders_epo]
    _ = [os.makedirs(os.path.join(dir_out, 'derivatives', 'sourcemodelling',
                                  subj_id, folder)) for folder in folders_sloc]

    os.makedirs(os.path.join(dir_out, subj_id, 'anat'))

    # MRIs
    shutil.copy(os.path.join(args.dir_anon, subj_code, 'T1_mf.nii'),
                os.path.join(dir_out, 'derivatives', 'epochs', subj_id, 'anat',
                             '%s_mod-T1w_maskface.nii' % subj_id))

    shutil.copy(os.path.join(args.dir_anon, subj_code, 'T1_pydefaced.nii'),
                os.path.join(dir_out, subj_id, 'anat',
                             '%s_T1w.nii' % subj_id))

    shutil.copy(os.path.join(dir_spatial, 'fwd', '%s-fwd.fif' % subj_code),
                os.path.join(dir_out, 'derivatives', 'sourcemodelling', subj_id,
                'fwd', '%s_fwd.fif' % subj_id))

    # transforms
    trans_in = os.path.join(dir_spatial, 'fwd', '%s-trans.fif' % subj_code)
    trans = mne.read_trans(trans_in)

    fname_trans_out = os.path.join(dir_out, 'derivatives', 'sourcemodelling',
                                   subj_id, 'xfm', '%s_from-head_to-surface.h5' % subj_id)

    hf = h5py.File(fname_trans_out, 'w')
    hf.create_dataset('trans', data=trans['trans'])
    hf.close()

    # seeg coordsystem
    dir_json_ieeg = os.path.join(dir_out, 'derivatives', 'epochs',
                                 subj_id, 'ieeg')
    make_coordsys_ieeg_json(dir_json_ieeg, subj_id, task)

    # ieeg electrodes
    subj_runs = runs_table.loc[runs_table.subj_code == subj_code]
    subj_chans = subj_runs.run_id.tolist()
    subj_chans = list(set([c.split('_')[0] for c in subj_chans]))
    subj_ch_mono = []
    for c in subj_chans:
        ch1 = c.split('-')[0]
        subj_ch_mono.append(ch1)
        if '\'' in ch1:
            subj_ch_mono.append(c[0] + '\'' + c.split('-')[-1])
        else:
            subj_ch_mono.append(c[0] + c.split('-')[-1])

    subj_ch_mono = natsort.natsorted(list(set(subj_ch_mono)))

    ch_info = pd.read_csv(os.path.join(dir_spatial, 'ch_info', '%s_seeg_ch_info.csv' % subj_code))
    ch_info = ch_info.loc[ch_info.name.isin(subj_ch_mono)]

    make_electrodes_ieeg_tsv(dir_json_ieeg, subj_id, ch_info, task)

    # eeg coordsystem
    dir_coordsys_eeg = os.path.join(dir_out, 'derivatives', 'epochs',
                                    subj_id, 'eeg')
    make_coordsys_eeg_json(dir_coordsys_eeg, subj_runs.fname.iloc[0], subj_id, task)

    # eeg electrodes
    make_electrodes_eeg_tsv(dir_coordsys_eeg, subj_runs.fname.iloc[0], subj_id, task)

    # events json
    fname_events_json = os.path.join(dir_out, 'derivatives', 'epochs',
                                     subj_id, 'eeg', '%s_task-%s_events.json' % (subj_id, task))
    # make_events_json(fname_events_json) # FIX, not needed

    # mri json
    make_mri_json(subj_id, dir_out)


for ix, r in runs_table.iterrows():
    export_data_bids(r.fname, dir_out, task, r.run_nr, r.subj_id,
                     event_id)

# scans
for s in subjects:
    subj_scans = os.listdir(os.path.join(dir_out, 'derivatives', 'epochs',
                                         'sub-%s' % s, 'eeg'))
    subj_scans = [os.path.join('eeg', scan) for scan in subj_scans if '.npy' in scan]
    subj_scans = pd.DataFrame({'filename': subj_scans})
    fname_scans = os.path.join(dir_out, 'derivatives', 'epochs',
                               'sub-%s' % s,  'sub-%s_scans.tsv' % s)
    subj_scans.to_csv(fname_scans, index=False, sep='\t')




