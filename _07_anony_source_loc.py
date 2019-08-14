import re
import sys
import mne
import glob
import numpy as np
import pandas as pd
import os.path as op
from info import dir_base
from fx_preprocess import realign_epochs_by_peaks
from fx_analysis import find_min_dist
from mne.minimum_norm import make_inverse_operator as m_inv_op
from mne.minimum_norm import apply_inverse as a_inv_op
from itcfpy.spatial import get_egi_montage_subsampling, replace_subsampled_montage_bads
from tqdm import tqdm


def anony_source_loc(subject, dir_base):
    results = pd.read_csv(op.join(dir_base, 'results', 'diff', '%s_results_distr_diff.csv' % subject))
    min_dists, all_min_dists = find_min_dist(results)

    fname_fwd_defaced = op.join(dir_base, 'spatial', 'bems', 'defaced', '%s_defaced-fwd.fif' % subject)
    fname_fwd_mf = op.join(dir_base, 'spatial', 'bems', 'mf', '%s_mf-fwd.fif' % subject)

    fwd_defaced = mne.read_forward_solution(fname_fwd_defaced)
    fwd_mf = mne.read_forward_solution(fname_fwd_mf)

    fwds = {'defaced': fwd_defaced, 'mf': fwd_mf}

    mont_subs = get_egi_montage_subsampling(plot=False)
    montages = ['EGI-256'] + list(mont_subs.keys())

    conds = all_min_dists.cond.unique()
    mne.set_log_level('ERROR')  # Only for final run
    np.warnings.filterwarnings('ignore')  # Only for final run

    print('\n' * 2)

    for c in tqdm(conds, position=0, desc='condition'):
        c_min_dists = all_min_dists.loc[all_min_dists.cond == c]

        fpath_epo = glob.glob(op.join(dir_base, 'data', 'fif', '%s_%s*' % (subject, c)))[0]
        epo = mne.read_epochs(fpath_epo)
        epo.set_eeg_reference('average', projection=True)
        epo.apply_proj()

        epo.interpolate_bads(reset_bads=False)

        bad_chans = epo.info['bads']
        good_chans = [ch for ch in epo.ch_names if ch not in bad_chans + ['STI']]

        monts_min_dist_names = c_min_dists.mont.unique()

        evos_min_dist = {k: [] for k in monts_min_dist_names}
        covs_min_dist = {k: [] for k in monts_min_dist_names}

        for n in monts_min_dist_names:
            mont = [m for m in montages if n.strip('ch') in m][0]
            if mont == 'EGI-256':
                sub_epo = epo.copy()
            else:
                new_mont = replace_subsampled_montage_bads(good_chans, bad_chans, mont_subs[mont]['names'], plot=False)
                new_mont = [ch.lower() for ch in new_mont]
                sub_epo = epo.copy().pick_channels(new_mont)

            # evoked
            evo = sub_epo.average()
            evo_crop = evo.copy().crop(-0.002, 0.002)

            n_ch = re.search(r'-[0-9]+', mont)
            n_ch = n_ch.group().strip('-')

            info = evo_crop.info['description']
            info += '_%sch' % n_ch

            evo_crop.info['description'] = info
            evos_min_dist[n] = evo_crop
            fname_cov = op.join(dir_base, 'data', 'covs', info + '-cov.fif')
            covs_min_dist[n] = mne.read_cov(fname_cov)

        for ix, r in tqdm(c_min_dists.iterrows(), position=1, desc='parameters', total=len(c_min_dists)):
            for f in tqdm(fwds.keys(), position=2, desc='anonymization'):
                inv = m_inv_op(evos_min_dist[r.mont].info, fwds[f], covs_min_dist[r.mont], loose=r.loose,
                               depth=r.depth)

                lambda2 = 1. / r.snr ** 2
                stc = a_inv_op(evos_min_dist[r.mont], inv, lambda2, method=r.method,
                               pick_ori=None)

                fname_stc = op.join(dir_base, 'results', 'anony',
                                    '%s__%s__%s_l%0.1f_d%0.1f_s%0.1f-stc'
                                    % (evos_min_dist[r.mont].info['description'], f,
                                       r.method, r.loose, r.depth, r.snr))

                stc.save(fname_stc)


if __name__ == '__main__':
    subj = sys.argv[1]
    anony_source_loc(subj, dir_base)
