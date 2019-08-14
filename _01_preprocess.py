import sys
import mne
import glob
import json
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from info import dir_base, bad_chans, fname_bads_info
from fx_preprocess import import_egi_sdk, get_stim_info,  make_bip_coords, \
    merge_sessions, realign_epochs_by_peaks


def preprocess_subj(subject, load_bad_ch=False, load_bad_epo=False):
    print('Pre-processing subject: %s - Started' % subject)

    # SEEG chan info
    fname_ch_info = op.join(dir_base, 'spatial', 'ch_info',
                            '%s_seeg_ch_info.csv' % subject)

    ch_info = pd.read_csv(fname_ch_info)
    bip_ch_info = make_bip_coords(ch_info)
    bip_ch_info.to_csv(op.join(dir_base, 'spatial', 'ch_info',
                               '%s_bip_ch_info.csv' % subject), index=False)

    # Signal Processing
    files = glob.glob(op.join(dir_base, 'data', 'raw', 'stim_%s*ses1*' % subject))
    files.sort()

    # Bad channels and epochs
    if load_bad_ch or load_bad_epo:
        with open(fname_bads_info, 'r') as f:
            bads_info = json.load(f)

    for ix, fname_raw in enumerate(files):
        # get info
        print(fname_raw)
        stim_info = get_stim_info(fname_raw)
        info = '{subj}_{st_chan}_{st_int}_{st_dur}_{st_fq}'.format(**stim_info)

        print('\n\nSession: %s' % info)

        # IMPORT
        raw = import_egi_sdk(fname_raw)

        fname_more_ses = glob.glob(fname_raw[:fname_raw.find('ses1')] + '*')  # find all sessions
        fname_more_ses = [f for f in fname_more_ses if 'ses1' not in f]  # remove ses1

        if len(fname_more_ses) > 0:
            for f in fname_more_ses:
                raw_tmp = import_egi_sdk(f)
                raw = merge_sessions(raw, raw_tmp)
                del raw_tmp

        # add digitization
        fname_dig = op.join(dir_base, 'spatial', 'ch_info',
                            '%s_egi_dig.hpts' % subject)

        montage = mne.channels.read_montage(fname_dig, unit='mm')
        raw.set_montage(montage, set_dig=True)

        # mark bad channels
        if load_bad_ch:
            bads = bads_info[subject][info]['bad_ch']
        else:
            bads = bad_chans[subject]

        raw.info['bads'] = bads

        # filter
        raw = raw.filter(0.1, None, skip_by_annotation=[])
        if subject in ['s5', 's7']:
            raw = raw.notch_filter([50, 100, 150, 200])

        # find events
        events_trig = mne.find_events(raw, output='onset')

        # plot
        # raw.plot(n_channels=64, block=True, title=info, duration=15, events=events_trig)

        epo = mne.Epochs(raw, events_trig, event_id={'stim': 1}, tmin=-0.3,
                         tmax=0.05, baseline=(-0.3, -0.05),
                         reject_by_annotation=False, preload=True)

        if load_bad_epo:
            good_epo = bads_info[subject][info]['good_epo']
            bad_epochs = [i for i in range(len(epo)) if i not in good_epo]
            epo = epo.drop(bad_epochs)

        epo.plot(n_epochs=25, scalings={'eeg': 10e-5}, block=True, title=info)

        # epo.average().plot(spatial_colors=True)

        # interpolate
        epo.interpolate_bads(reset_bads=False)

        # fine alignment
        epo_alig = realign_epochs_by_peaks(epo, plot=True, how='max')
        plt.show(block=True)

        plt.figure(1)
        fname_alig_fig = op.join(dir_base, 'figures', 'prepro', info + '-alig.png')
        plt.savefig(fname_alig_fig)
        plt.close('all')

        print(info, '(%s/%s)' % (ix+1, len(files)))
        confirm = input("Press q to quit or any other key to continue")
        if confirm == 'q':
            break

        # add info
        epo_alig.info['description'] = info

        # save
        fname_epo = op.join(dir_base, 'data', 'fif', '%s-epo.fif' % info)
        epo_alig.save(fname_epo)

    print('Pre-processing - subject: %s - Done.' % subject)


if __name__ == '__main__':
    subj = sys.argv[1]
    preprocess_subj(subj)
