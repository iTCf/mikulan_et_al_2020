import mne
import os
from info import dir_base
from itcfpy.spatial import get_egi_montage_subsampling, replace_subsampled_montage_bads
import os.path as op
import re


def compute_covs(fname_epo, dir_base):
    epo = mne.read_epochs(fname_epo)

    epo.set_eeg_reference('average', projection=True)
    epo.apply_proj()
    
    mont_subs = get_egi_montage_subsampling(plot=False)
    bad_chans = epo.info['bads']
    good_chans = [ch for ch in epo.ch_names if ch not in bad_chans+['STI']]

    montages = ['EGI-256'] + list(mont_subs.keys())

    for mont in montages:
        print('\nProcessing montage: ', mont)
        if mont == 'EGI-256':
            sub_epo = epo.copy()
        else:
            new_mont = replace_subsampled_montage_bads(good_chans, bad_chans, mont_subs[mont]['names'], plot=False)
            new_mont = [ch.lower() for ch in new_mont]
            sub_epo = epo.copy().pick_channels(new_mont)

        # covariance
        print('Computing Covariance')
        cov = mne.compute_covariance(sub_epo, method='auto', tmin=-0.25,
                                     tmax=-0.05)  # use method='auto' for final

        n_ch = re.search(r'-[0-9]+', mont)
        n_ch = n_ch.group().strip('-')
        info = epo.info['description'] + '_%sch' % n_ch

        fname_cov = op.join(dir_base, 'data', 'covs', info + '-cov.fif')
        cov.save(fname_cov)


if __name__ == '__main__':
    files = os.listdir(op.join(dir_base, 'data', 'fif'))
    files.sort()
    for f in files:
        compute_covs(op.join(dir_base, 'data', 'fif', f), dir_base)
