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
