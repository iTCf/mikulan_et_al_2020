import mne
import os.path as op
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import glob
import re
from fx_preprocess import get_stim_info
from tqdm import tqdm


def calc_distr_results(subject, bip_ch_info, distr_methods, subjects_dir,
                       dir_base):

    pre_results = {'ch': [], 'int': [], 'method': [], 'kind': [], 'mont': [],
                   'loose': [],  'depth': [], 'snr': [], 'dist': [],
                   'diff_x': [], 'diff_y': [], 'diff_z': []}

    for method in distr_methods:
        print('Calculating results - Method: ' + method)
        files_anat = glob.glob(op.join(dir_base, 'cluster', 'results', 'source_loc',
                                       method, '%s*rh.stc' % subject))

        files = files_anat

        for file in tqdm(files):
            # find source loc peak
            stim_info = get_stim_info(file)
            ch_stim_coords = bip_ch_info.loc[bip_ch_info.name ==
                                             stim_info['st_chan']]

            if 'rh' in file:  # avoid redundancy as stc files are 2 for each solution (one by hemishpere)
                stc = mne.read_source_estimate(file)

                dist, diffs = calc_dist_anat(stc, stim_info, ch_stim_coords,
                                             subject, subjects_dir)
                kind = 'anat'
            else:
                continue

            params = re.findall(r'[a-z]\d.\d', file)
            loose = [i for i in params if 'l' in i][0]
            depth = [i for i in params if 'd' in i][0]
            snr = [i for i in params if 's' in i][0]
            mont = re.findall(r'[0-9]+ch', file)[0]

            pre_results['ch'].append(stim_info['st_chan'])
            pre_results['int'].append(stim_info['st_int'])
            pre_results['method'].append(method)
            pre_results['mont'].append(mont)
            pre_results['kind'].append(kind)
            pre_results['loose'].append(float(re.findall(r'\d.\d', loose)[0]))
            pre_results['depth'].append(float(re.findall(r'\d.\d', depth)[0]))
            pre_results['snr'].append(float(re.findall(r'\d.\d', snr)[0]))
            pre_results['dist'].append(dist)
            pre_results['diff_x'].append(diffs[0])
            pre_results['diff_y'].append(diffs[1])
            pre_results['diff_z'].append(diffs[2])

    results = pd.DataFrame(pre_results)
    fn_results = op.join(dir_base, 'results', 'diff',
                         '%s_results_distr_diff.csv' % subject)
    results.to_csv(fn_results)
    return results


def calc_dist_anat(stc, stim_info, ch_stim_coords, subject, subjects_dir):
    hemi = 'lh' if '\'' in stim_info['st_chan'] else 'rh'
    vertno_max, time_max = stc.get_peak(hemi=hemi)

    fn_surf = op.join(subjects_dir, subject, 'surf', '%s.white' % hemi)
    surf = mne.read_surface(fn_surf)[0]

    vert_coords = surf[vertno_max, :]
    stim_surf_coords = ch_stim_coords[['x_surf', 'y_surf', 'z_surf']].\
        values.squeeze()

    dist = euclidean(vert_coords, stim_surf_coords)
    diffs = vert_coords - stim_surf_coords
    return np.round(dist, 2), np.round(diffs, 2)


def find_min_dist(results):
    chs = results.ch.unique()
    min_dists = {'cond': [], 'ch': [], 'int': [], 'd': [], 'method': [], 'mont': [], 'depth': [], 'loose': [],
                 'snr': [], 'n_min_dist': [], 'n_min_m': [], 'dif_x': [], 'dif_y': [], 'dif_z': [],
                  'p_mne': [], 'p_dspm': [], 'p_lor': [], 'n_mne': [], 'n_dspm': [], 'n_lor': [],
                 'n_256ch': [], 'n_128ch': [], 'n_64ch': [], 'n_32ch': []}

    all_min_dists = []
    for ch in chs:
        ch_dat = results.loc[results.ch == ch]
        ints = ch_dat.int.unique()
        for i in ints:
            int_dat = ch_dat.loc[ch_dat.int == i]
            min_dist = int_dat.dist.min()
            min_dist_dat = int_dat.loc[int_dat.dist.idxmin()]
            print('Ch: %s - Int: %s - Min dist: %s' % (ch, i, min_dist))
            min_dat = int_dat.loc[int_dat.dist == min_dist]
            min_dat['cond'] = '%s_%s' % (ch, i)
            all_min_dists.append(min_dat)
            n_min_dist = len(min_dat)
            n_min_m = len(min_dat.method.unique())
            min_dists['p_mne'].append(len(min_dat.loc[min_dat.method == 'MNE']) / len(min_dat))
            min_dists['p_dspm'].append(len(min_dat.loc[min_dat.method == 'dSPM']) / len(min_dat))
            min_dists['p_lor'].append(len(min_dat.loc[min_dat.method == 'eLORETA']) / len(min_dat))
            min_dists['n_mne'].append(len(min_dat.loc[min_dat.method == 'MNE']))
            min_dists['n_dspm'].append(len(min_dat.loc[min_dat.method == 'dSPM']))
            min_dists['n_lor'].append(len(min_dat.loc[min_dat.method == 'eLORETA']))
            for m in ['256ch', '128ch', '64ch', '32ch']:
                min_dists['n_%s' % m].append(len(min_dat.loc[min_dat.mont == m]) / len(min_dat))
            min_dists['cond'].append('%s_%s' % (ch, i))
            min_dists['ch'].append(ch)
            min_dists['int'].append(i)
            min_dists['d'].append(min_dist)
            min_dists['method'].append(min_dist_dat['method'])
            min_dists['mont'].append(min_dist_dat['mont'])
            min_dists['depth'].append(min_dist_dat['depth'])
            min_dists['loose'].append(min_dist_dat['loose'])
            min_dists['snr'].append(min_dist_dat['snr'])
            min_dists['n_min_dist'].append(n_min_dist)
            min_dists['n_min_m'].append(n_min_m)
            min_dists['dif_x'].append(min_dist_dat['diff_x'])
            min_dists['dif_y'].append(min_dist_dat['diff_y'])
            min_dists['dif_z'].append(min_dist_dat['diff_z'])

    min_dists = pd.DataFrame(min_dists)
    all_min_dists = pd.concat(all_min_dists)
    return min_dists, all_min_dists


def get_all_results(subjects, dir_base):
    min_dists_list = []
    results_list = []
    for subject in subjects:
        fn_results = op.join(dir_base, 'results', 'diff', '%s_results_distr_diff.csv'
                             % subject)
        results = pd.read_csv(fn_results)
        results['subj'] = subject
        min_dist, all_min_dists = find_min_dist(results)
        min_dist['subj'] = subject

        fname_skin_dist = op.join(dir_base, 'tables', '%s_skin_dist.csv' % subject)
        fname_mont_dist = op.join(dir_base, 'tables', '%s_mont_dist.csv' % subject)

        skin_dist = pd.read_csv(fname_skin_dist)
        mont_dist = pd.read_csv(fname_mont_dist)

        min_dist = pd.merge(min_dist, skin_dist, left_on='ch', right_on='name')
        min_dist = pd.merge(min_dist, mont_dist, on=['ch', 'subj', 'int'])

        min_dists_list.append(min_dist)
        results_list.append(results)

    min_dists = pd.concat(min_dists_list)
    min_dists = min_dists.sort_values('d', ascending=False)
    min_dists['subj'] = min_dists.subj.astype('category')
    min_dists['mont'] = min_dists['mont'].astype('category')
    min_dists['int'] = min_dists['int'].astype('category')
    min_dists['mont'].cat.reorder_categories(['256ch', '128ch', '64ch', '32ch'], inplace=True)
    min_dists['int'].cat.reorder_categories(['5ma', '1ma', '05ma',
                                             '03ma', '01ma'], inplace=True)

    all_results = pd.concat(results_list)

    return all_results, min_dists


def get_all_coords(subjects, dir_base):
    import pandas as pd
    import os.path as op

    coords_list = []
    for subject in subjects:
        fn_coords = op.join(dir_base, 'spatial', 'ch_info', '%s_bip_ch_info.csv'
                           % subject)
        coords = pd.read_csv(fn_coords)
        coords['subj'] = subject
        coords_list.append(coords)
    all_coords = pd.concat(coords_list)
    return all_coords


def calc_dist_to_mont(subject, fname_coords, fname_mont, fname_bads_info):
    import pandas as pd
    import json
    from itcfpy.spatial import get_egi_montage_subsampling, replace_subsampled_montage_bads
    import re
    import numpy as np

    coords = pd.read_csv(fname_coords)
    mont = pd.read_csv(fname_mont, delim_whitespace=True, names=['kind', 'name', 'x', 'y', 'z'])
    mont = mont.loc[mont.kind == 'eeg']

    with open(fname_bads_info, 'r') as f:
        bads_info = json.load(f)

    bads_info = bads_info[subject]

    sessions = list(bads_info.keys())
    mont_subs = get_egi_montage_subsampling(plot=False)
    montages = ['EGI-256'] + list(mont_subs.keys())
    all_eeg_ch = ['e%s' % (i + 1) for i in range(256)]

    all_dists = {'ses': [], 'dist256': [], 'dist128': [], 'dist64': [], 'dist32': []}

    for s in sessions:
        all_dists['ses'].append(s)
        stim_ch = s.split('_')[1]
        stim_coords = coords.loc[coords.name == stim_ch][['x_mri', 'y_mri', 'z_mri']].values
        bad_chans = bads_info[s]['bad_ch']
        good_chans = [ch for ch in all_eeg_ch if ch not in bad_chans]

        for m in montages:
            if m == 'EGI-256':
                new_mont_names = good_chans
            else:
                new_mont_names = replace_subsampled_montage_bads(good_chans, bad_chans,
                                                           mont_subs[m]['names'],
                                                           plot=False)
            new_mont_names = [n.lower() for n in new_mont_names]
            new_mont = mont.loc[mont.name.isin(new_mont_names)]
            new_mont_coords = new_mont[['x', 'y', 'z']].values
            dist = np.sqrt(np.sum((stim_coords - new_mont_coords) ** 2, axis=1))
            mean_dist = np.round(dist.mean(), decimals=2)
            n_ch = re.search(r'-[0-9]+', m)
            n_ch = n_ch.group().strip('-')

            all_dists['dist%s' % n_ch].append(mean_dist)

    all_dists = pd.DataFrame(all_dists)
    all_dists[['subj', 'ch', 'int', 'dur', 'fq']] = all_dists.ses.str.split('_', expand=True)
    return all_dists


def get_anony_results(dir_base, subjects, distr_methods, anon_methods, subjects_dir):
    import os
    import pandas as pd
    import glob
    dir_anony_results = os.path.join(dir_base, 'results', 'anony')

    pre_results = {'subj': [], 'ch': [], 'int': [], 'anon_m': [], 'method': [], 'kind': [], 'mont': [],
                   'loose': [],  'depth': [], 'snr': [], 'dist': [],
                   'diff_x': [], 'diff_y': [], 'diff_z': []}

    bip_ch_infos = {s: pd.read_csv(os.path.join(dir_base, 'spatial', 'ch_info',
                                                '%s_bip_ch_info.csv' % s)) for s in subjects}

    files = glob.glob(op.join(dir_anony_results, '*rh*'))

    for file in tqdm(files):
        # find source loc peak
        stim_info = get_stim_info(file)
        bip_ch_info = bip_ch_infos[stim_info['subj']]
        ch_stim_coords = bip_ch_info.loc[bip_ch_info.name ==
                                         stim_info['st_chan']]

        stc = mne.read_source_estimate(os.path.join(dir_anony_results, file))

        dist, diffs = calc_dist_anat(stc, stim_info, ch_stim_coords,
                                     stim_info['subj'], subjects_dir)

        kind = 'anat'
        params = re.findall(r'[a-z]\d.\d', file)
        loose = [i for i in params if 'l' in i][0]
        depth = [i for i in params if 'd' in i][0]
        snr = [i for i in params if 's' in i][0]
        mont = re.findall(r'[0-9]+ch', file)[0]
        method = [m for m in distr_methods if m in file][0]
        anon_m = [m for m in anon_methods if m in file][0]

        pre_results['subj'].append(stim_info['subj'])
        pre_results['ch'].append(stim_info['st_chan'])
        pre_results['int'].append(stim_info['st_int'])
        pre_results['anon_m'].append(anon_m)
        pre_results['method'].append(method)
        pre_results['mont'].append(mont)
        pre_results['kind'].append(kind)
        pre_results['loose'].append(float(re.findall(r'\d.\d', loose)[0]))
        pre_results['depth'].append(float(re.findall(r'\d.\d', depth)[0]))
        pre_results['snr'].append(float(re.findall(r'\d.\d', snr)[0]))
        pre_results['dist'].append(dist)
        pre_results['diff_x'].append(diffs[0])
        pre_results['diff_y'].append(diffs[1])
        pre_results['diff_z'].append(diffs[2])

    results = pd.DataFrame(pre_results)
    fn_results = op.join(dir_base, 'results',
                         'anony_results.csv')
    results.to_csv(fn_results)
    return results


def calc_stim_skin_dist(ch_info, subject, subjects_dir, dir_base):
    fname_head = op.join(subjects_dir, subject, 'bem', 'watershed',
                         '%s_outer_skin_surface' % subject)

    head = mne.read_surface(fname_head)

    all_names = list()
    all_dist = list()
    all_ch_coords = list()
    all_skin_coords = list()

    for ix_ch, ch in ch_info.iterrows():
        all_names.append(ch['name'])
        coords = ch[['x_surf', 'y_surf', 'z_surf']].tolist()
        all_ch_coords.append(coords)
        dist_all = np.sqrt(np.sum((head[0] - coords)**2, axis=1))
        min_dist = dist_all[np.argmin(dist_all)]
        all_dist.append(min_dist)
        all_skin_coords.append(head[0][np.argmin(dist_all)])

    all_ch_coords = np.array(all_ch_coords)
    all_dist = np.array(all_dist)
    all_skin_coords = np.array(all_skin_coords)

    dist_to_sk = pd.DataFrame({'name': all_names,
                            'sk_dist': np.round(all_dist, 2)})

    for ix_ax, ax in enumerate('xyz'):
        dist_to_sk['%s_surf_skin' % ax] = all_skin_coords[:, ix_ax]
        dist_to_sk['%s_surf' % ax] = all_ch_coords[:, ix_ax]

    dist_to_sk = dist_to_sk[['name', 'sk_dist', 'x_surf', 'y_surf', 'z_surf',
                       'x_surf_skin', 'y_surf_skin', 'z_surf_skin']]

    dist_to_sk.to_csv(op.join(dir_base, 'spatial', 'ch_info',
                           '%s_dist_to_skin.csv' % subject), index=False)
    return dist_to_sk

