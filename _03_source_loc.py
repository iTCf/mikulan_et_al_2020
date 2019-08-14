import sys
import mne
import glob
import re
import os.path as op
import matplotlib.pyplot as plt
from datetime import datetime
from info import dir_base
from fx_source_loc import source_distributed
from itcfpy.spatial import get_egi_montage_subsampling, replace_subsampled_montage_bads


def source_loc_subj(subject):
    print('Source localization subject: %s - Started' % subject)
    mne.set_log_level('ERROR')  # only for final run

    distr_methods = ['MNE', 'dSPM', 'eLORETA']

    # get files
    files = glob.glob(op.join(dir_base, 'data', 'fif', '%s_*' % subject))

    fname_fwd = op.join(dir_base, 'spatial', 'fwd', '%s-fwd.fif' % subject)

    fwd = mne.read_forward_solution(fname_fwd)

    for ix_f, file in enumerate(files):
        # load
        print('\n\nProcessing session: %s (%s of %s) - %s' % (op.split(file)[-1].replace('-epo.fif', ''),
                                                              ix_f+1, len(files), datetime.now().strftime("%d-%m-%Y %H:%M:%S")))

        epo = mne.read_epochs(file, preload=True)

        epo.set_eeg_reference('average', projection=True)
        epo.apply_proj()

        # montage downsampling
        mont_subs = get_egi_montage_subsampling(plot=False)
        bad_chans = epo.info['bads']
        good_chans = [ch for ch in epo.ch_names if ch not in bad_chans+['STI']]

        montages = ['EGI-256'] + list(mont_subs.keys())

        for mont in montages:
            print('\nProcessing montage: ', mont)
            if mont == 'EGI-256':
                sub_epo = epo.copy()
            else:
                new_mont = replace_subsampled_montage_bads(good_chans, bad_chans, mont_subs[mont]['names'], plot=True)
                fname_sub_mont_fig = op.join(dir_base, 'figures', 'prepro', op.split(file)[-1].replace('-epo.fif', '-mont-%s.png' % mont))
                plt.savefig(fname_sub_mont_fig)
                plt.close('all')
                new_mont = [ch.lower() for ch in new_mont]
                sub_epo = epo.copy().pick_channels(new_mont)

            # covariance
            print('Computing Covariance')
            cov = mne.compute_covariance(sub_epo, method='auto', tmin=-0.25,
                                         tmax=-0.05)  # use method='auto' for final

            # evoked
            evo = sub_epo.average()
            evo_crop = evo.copy().crop(-0.002, 0.002)

            n_ch = re.search(r'-[0-9]+', mont)
            n_ch = n_ch.group().strip('-')

            info = evo_crop.info['description']
            info += '_%sch' % n_ch

            evo_crop.info['description'] = info  # used for filename when saving

            source_distributed(evo_crop, fwd, cov, distr_methods, dir_base)

    print('Finished - %s' % datetime.now().strftime("%d-%m-%Y %H:%M:%S"))


if __name__ == '__main__':
    subj = sys.argv[1]
    source_loc_subj(subj)
