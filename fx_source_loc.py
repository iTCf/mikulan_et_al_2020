import mne
import os.path as op
import numpy as np
from mne.minimum_norm import make_inverse_operator as m_inv_op
from mne.minimum_norm import apply_inverse as a_inv_op
from tqdm import tqdm


def make_bem_and_source_space(subject, subjects_dir, dir_base):
    # BEM
    mne.bem.make_watershed_bem(subject, subjects_dir, volume='T1', show=True)

    conductivity = (0.3, 0.006, 0.3)
    model = mne.make_bem_model(subject=subject, ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)

    bem = mne.make_bem_solution(model)
    mne.write_bem_solution(op.join(dir_base, 'spatial', 'fwd', '%s-bem.fif'
                                   % subject), bem)

    # Anatomical Source Space
    src = mne.setup_source_space(subject, spacing='oct6',
                                 subjects_dir=subjects_dir)

    mne.write_source_spaces(op.join(dir_base, 'spatial', 'fwd',
                                    '%s-src.fif' % subject), src)


def make_forward_solution(subject, subjects_dir, dir_base):
    import glob
    import mayavi.mlab as mlab

    files_epochs = glob.glob(op.join(dir_base, 'data', 'fif',
                                     '%s*' % subject.replace('anony', '')))
    epo = mne.read_epochs(files_epochs[0])

    trans_fname = op.join(dir_base, 'spatial', 'fwd', '%s-trans.fif' % subject)
    src_fname = op.join(dir_base, 'spatial', 'fwd', '%s-src.fif' % subject)
    bem_fname = op.join(dir_base, 'spatial', 'fwd', '%s-bem.fif' % subject)

    fwd = mne.make_forward_solution(epo.info, trans_fname, src_fname,
                                    bem_fname)

    trans = mne.read_trans(trans_fname)
    mne.viz.plot_alignment(epo.info, trans, subject=subject,
                           surfaces=['head', 'brain'],
                           subjects_dir=subjects_dir)
    mlab.show()

    mne.write_forward_solution(op.join(dir_base, 'spatial', 'fwd',
                                       '%s-fwd.fif' % subject), fwd)


def source_distributed(evo_crop, fwd, cov, methods, dir_base):
    param_loose = np.arange(0.1, 1.1, 0.1)
    param_depth = np.arange(0.1, 1.1, 0.1)
    param_snr = np.arange(1, 5, 1)
    mne.set_log_level('ERROR')  # Only for final run
    print('Computing source estimates - ', methods)
    for method in tqdm(methods, position=0, desc='methods'):
        for loose in tqdm(param_loose, position=1, desc='loose'):
            for depth in tqdm(param_depth, position=2, desc='depth'):
                # anatomical source space
                inv = m_inv_op(evo_crop.info, fwd, cov, loose=loose,
                               depth=depth)

                for snr in param_snr:
                    lambda2 = 1. / snr ** 2
                    stc = a_inv_op(evo_crop, inv, lambda2, method=method,
                                   pick_ori=None)

                    fname_stc = op.join(dir_base, 'results', 'source_loc',
                                        method,
                                        '%s__%s_l%0.1f_d%0.1f_s%0.1f-stc'
                                        % (evo_crop.info['description'],
                                           method, loose, depth, snr))
                    stc.save(fname_stc)


def make_anony_fwd(subject, dir_base, subjects_dir, conductivity=(0.3, 0.006, 0.3), ico=4):
    import os.path as op
    from mne.io.constants import FIFF
    from mne.bem import _surfaces_to_bem, _check_bem_size
    import glob
    import mayavi.mlab as mlab

    for m in ['anonymi', 'defaced', 'mf']:
        print('Preparing fwd - method: %s' % m)
        bem_dir = op.join(dir_base, 'spatial', 'bems', m)
        inner_skull = op.join(bem_dir, '%s_%s_inner_skull_surface' % (subject, m))
        outer_skull = op.join(bem_dir, '%s_%s_outer_skull_surface' % (subject, m))
        outer_skin = op.join(bem_dir, '%s_%s_outer_skin_surface' % (subject, m))
        surfaces = [inner_skull, outer_skull, outer_skin]
        ids = [FIFF.FIFFV_BEM_SURF_ID_BRAIN,
               FIFF.FIFFV_BEM_SURF_ID_SKULL,
               FIFF.FIFFV_BEM_SURF_ID_HEAD]

        surfaces = _surfaces_to_bem(surfaces, ids, conductivity, ico)
        _check_bem_size(surfaces)

        bem = mne.make_bem_solution(surfaces)
        bem_fname = op.join(dir_base, 'spatial', 'bems', m, '%s_%s-bem.fif'
                            % (subject, m))
        mne.write_bem_solution(bem_fname, bem)

        files_epochs = glob.glob(op.join(dir_base, 'data', 'fif',
                                         '%s*' % subject))
        epo = mne.read_epochs(files_epochs[0])

        trans_fname = op.join(dir_base, 'spatial', 'fwd', '%s-trans.fif' % subject)
        src_fname = op.join(dir_base, 'spatial', 'fwd', '%s-src.fif' % subject)

        fwd = mne.make_forward_solution(epo.info, trans_fname, src_fname,
                                        bem_fname)

        trans = mne.read_trans(trans_fname)
        mne.viz.plot_alignment(epo.info, trans, subject=subject,
                               surfaces=['head', 'brain'], bem=bem,
                               subjects_dir=subjects_dir)
        mlab.show()

        mne.write_forward_solution(op.join(dir_base, 'spatial', 'bems', m,
                                           '%s_%s-fwd.fif' % (subject, m)), fwd)



def get_egi_montage_subsampling(montages=('GSN-HydroCel-256', 'GSN-HydroCel-128',
                                        'GSN-HydroCel-64_1.0', 'GSN-HydroCel-32'),
                                plot=True):
    import numpy as np
    import mne
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # note: some commented lines represent differences between mne versions

    all_monts = {}
    all_pos = {}
    all_names = {}
    for m in montages:
        #mont = mne.channels.read_montage(m)
        mont = mne.channels.make_standard_montage(m)
        ch_names = mont.ch_names

        #pos = mont.pos
        ch_pos = mont._get_ch_pos()
        pos = np.array([ch_pos[k] for k in ch_pos.keys()])
        pos_names = [k for k in ch_pos.keys()]
        # print(list(zip(ch_names, pos_names)))

        # if m == 'GSN-HydroCel-32':
        #     all_names[m] = ch_names[3:-1]
        #     all_pos[m] = pos[3:-1]
        # else:
        #     all_names[m] = ch_names[3:]
        #     all_pos[m] = pos[3:]

        if m == 'GSN-HydroCel-32':
            ch_names = ch_names[:-1]
            pos = pos[:-1]

        all_names[m] = ch_names
        all_pos[m] = pos
        all_monts[m] = mont

    all_subsamp_names = {}
    all_subsamp_dists = {}
    for mont_name in montages[1:]:
        subsamp_chs = []
        subsamp_dists = []
        for c, p in zip(all_names[mont_name], all_pos[mont_name]):
            dist_all = np.sqrt(np.sum((all_pos[montages[0]] - p) ** 2, axis=1))
            if ((mont_name == 'GSN-HydroCel-128') and (c == 'E11')) or \
                ((mont_name == 'GSN-HydroCel-64_1.0') and (c == 'E8')):  # because two channels have E21 as closest electrode
                #  todo: fix to do it automatically
                subsamp_dists.append(dist_all[14])
                subsamp_chs.append('E15')
            else:
                subsamp_dists.append(dist_all[np.argmin(dist_all)])
                subsamp_chs.append(all_names[montages[0]][np.argmin(dist_all)])
            # print(c, subsamp_chs[-1])
        all_subsamp_names[mont_name] = subsamp_chs
        all_subsamp_dists[mont_name] = subsamp_dists

    # [len(x) for n, x in all_subsamp_names.items()]

    if plot:
        pos = all_pos[montages[0]]
        pos2d = all_monts[montages[0]].get_pos2d()[3:]

        fig = plt.figure(figsize=(20, 15))

        for ix, mont_name in enumerate(montages[1:]):
            ax = fig.add_subplot(2, 3, ix + 1, projection='3d')

            subs_ix = [all_names[montages[0]].index(ch) for ch in all_subsamp_names[mont_name]]
            subs_pos = pos[subs_ix, :]
            ax.scatter(subs_pos[:, 0], subs_pos[:, 1], subs_pos[:, 2], c='tab:blue', s=20, label=mont_name)

            out_ix = [i for i in range(256) if i not in subs_ix]
            out_chs = pos[out_ix, :]
            ax.scatter(out_chs[:, 0], out_chs[:, 1], out_chs[:, 2], c='tab:orange', s=20, label=montages[0])
            ax.legend()
            ax.view_init(0, 0)
            ax.axis('off')

            ax1 = fig.add_subplot(2, 3, ix + 4)

            subs_ix = [all_names[montages[0]].index(ch) for ch in all_subsamp_names[mont_name]]
            subs_pos = pos2d[subs_ix, :]

            ax1.scatter(subs_pos[:, 0], subs_pos[:, 1], c='tab:blue', s=20, label=mont_name)

            out_ix = [i for i in range(256) if i not in subs_ix]
            out_chs = pos2d[out_ix, :]
            ax1.scatter(out_chs[:, 0], out_chs[:, 1], c='tab:orange', s=20, label=montages[0])
            ax1.axis('off')
    ch_subsampled = {k: {'names': all_subsamp_names[k], 'dists': all_subsamp_dists[k]} for k in all_subsamp_names.keys()}
    return ch_subsampled


def replace_subsampled_montage_bads(good_chans, bad_chans,
                                    subsampled_chans, plot=True):
    # todo: add option to do it from digitalization coords

    import numpy as np
    import mne
    import matplotlib.pyplot as plt

    good_chans = [ch.upper() for ch in good_chans]
    bad_chans = [ch.upper() for ch in bad_chans]

    mont256 = mne.channels.make_standard_montage('GSN-HydroCel-256')
    names256 = mont256.ch_names
    ch_pos = mont256._get_ch_pos()
    pos256 = np.array([ch_pos[k] for k in ch_pos.keys()])

    # mont256 = mne.channels.read_montage('GSN-HydroCel-256')
    # names256 = mont256.ch_names[3:]
    # pos256 = mont256.pos[3:]

    names_to_choose = [ch for ch in names256 if (ch in good_chans) and (ch not in subsampled_chans)]
    ix_to_choose = [names256.index(ch) for ch in names_to_choose]
    pos_to_choose = pos256[ix_to_choose, :]

    names_to_change = [ch for ch in subsampled_chans if ch in bad_chans]

    new_subsampled_chs = subsampled_chans.copy()
    new_chs = []
    for ch in names_to_change:
        pos = pos256[names256.index(ch)]
        dist_all = np.sqrt(np.sum((pos_to_choose - pos) ** 2, axis=1))
        new_ch = names_to_choose[dist_all.argmin()]
        new_chs.append(new_ch)
        new_subsampled_chs[subsampled_chans.index(ch)] = new_ch
        names_to_choose.remove(new_ch)
        pos_to_choose = np.delete(pos_to_choose, dist_all.argmin(), 0)

    if plot:
        pos2d = mont256.get_pos2d()[3:, ]

        pos_changed = pos2d[[names256.index(ch) for ch in names_to_change]]
        pos_subs = pos2d[[names256.index(ch) for ch in subsampled_chans]]
        pos_bad = pos2d[[names256.index(ch) for ch in bad_chans]]

        pos_new = pos2d[[names256.index(ch) for ch in new_chs]]

        colors = ('tab:gray', 'tab:cyan', 'tab:red', 'tab:orange', 'tab:green')
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        for p, l, c in zip([pos2d, pos_subs, pos_bad, pos_changed, pos_new],
                           ['all', 'subsample', 'bad', 'changed', 'new'], colors):
            ax.scatter(p[:, 0], p[:, 1], label=l, c=c)
            ax.legend()
            ax.axis('off')
        fig.suptitle('Montage subsample - bad channels replacement\nChanges: %s' % len(names_to_change))

        for p_old, p_new in zip(pos_changed, pos_new):
            ax.arrow(p_old[0], p_old[1], p_new[0] - p_old[0], p_new[1] - p_old[1],
                     head_width=0.05, length_includes_head=True)
    return new_subsampled_chs
