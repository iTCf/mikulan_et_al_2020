import sys
import os.path as op
import pandas as pd
from info import dir_base, subjects_dir
from fx_analysis import calc_distr_results, calc_stim_skin_dist

pd.set_option('display.expand_frame_repr', False)


def analyze_subj(subject):
    fname_bip_ch_info = op.join(dir_base, 'spatial', 'ch_info',
                                '%s_bip_ch_info.csv' % subject)

    bip_ch_info = pd.read_csv(fname_bip_ch_info)

    # distance to skin
    dist_to_sk = calc_stim_skin_dist(bip_ch_info, subject, subjects_dir,
                                     dir_base)

    # distributed results
    distr_methods = ['MNE', 'dSPM', 'eLORETA']
    distr_results = calc_distr_results(subject, bip_ch_info, distr_methods,
                                       subjects_dir, dir_base)


if __name__ == '__main__':
    subj = sys.argv[1]
    analyze_subj(subj)
