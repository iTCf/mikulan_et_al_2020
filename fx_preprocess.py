import mne
import os.path as op
import numpy as np
import matplotlib.pyplot as plt


def make_stim_table(files):
    """
    Create summary of stimulations

    Parameters
    ----------
    subject: string
    The subject's code

    Returns
    -------
    stim_table: pandas.DataFrame
    The summary table

    """
    import pandas as pd
    table = {'subj': [], 'st_chan': [], 'st_int': [], 'st_ses': []}

    for f in files:
        stim_info = get_stim_info(f)
        _ = [table[k].append(stim_info[k]) for k in table.keys()]

    table = pd.DataFrame(table)
    table = table.groupby(['st_chan', 'st_int']).size().reset_index()
    table = table.rename(columns={0: 'n_ses'})
    return table


def import_egi_sdk(fpath):
    """
    Import data from egi AmpServerPro SDK

    Parameters
    ----------
    fpath: string
    The file path

    Returns
    -------
    The mne.Raw object
    """

    from scipy.io import loadmat
    import numpy as np

    dat_base = loadmat(fpath)
    data = dat_base['data'].T[:-1, :]
    trig = dat_base['data'].T[-1, :]
    trig = trig.reshape(1, len(trig))

    # map trigger values
    marks, counts = np.unique(trig, return_counts=True)
    sort_counts = counts.copy()
    sort_counts.sort()

    trig[trig == marks[counts == sort_counts[-2]]] = 1  # sometimes there are more than 2 mark codes
    trig[trig != 1] = 0

    ch_names = ['e%s' % (ch + 1) for ch in range(256)]

    eeg_types = ['eeg'] * len(ch_names)
    eeg_info = mne.create_info(ch_names, 8000., eeg_types)

    raw = mne.io.RawArray(data * 1e-6, eeg_info)  # rescale to volts

    info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(trig, info)
    raw.add_channels([stim_raw], force_update_info=True)

    return raw


def make_dig_files(subject, fname_dig_polh, dir_base):
    import re
    import pandas as pd

    with open(fname_dig_polh) as f:
        lines = [line.rstrip('\n') for line in f]

    coords = {'id': [], 'x': [], 'y': [], 'z': []}
    for l in lines[3:-2]:
        try:
            id = re.findall(r'id="[A-Z]\d+?"', l)[0]
            id = re.findall(r'[A-Z]\d+', id)[0]

        except IndexError:
            id = re.findall(r'id="[A-Z][\d+]?"', l)[0]
            id = re.findall(r'[A-Z]', id)[0]
        coords['id'].append(id)

        for c in ['x', 'y', 'z']:
            coord = re.findall(r'%s="[-]?[0-9]+[.][0-9]+"' % c, l)[0]
            coord = re.findall(r'[-]?[0-9]+[.][0-9]+', coord)[0]
            coords[c].append(float(coord))

    coords = pd.DataFrame(coords)
    coords_vals = coords[['x', 'y', 'z']].values
    coords_vals[:, 0] = coords_vals[:, 0] * -1
    coords_vals[:, 1] = coords_vals[:, 1] * -1

    ch_names = ['1', '3'] + ['e%s' % (ix+1) for ix in range(256)] + ['2']
    ch_types = ['fid', 'fid'] + ['eeg']*256 + ['fid']

    # Save Slicer files (.fcsv)
    header = ['# Markups fiducial file version = 4.5', '# CoordinateSystem = 0', '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID']

    elec_info = ['vtkMRMLMarkupsFiducialNode_{},{},{},{},0,0,0,1,1,1,0,{},,' .format(ix+1, x, y, z, lab) for ix, (x, y, z, lab) in
                 enumerate(zip(coords_vals[:, 0], coords_vals[:, 1], coords_vals[:, 2], ch_names))]

    dat_to_write = header + elec_info

    with open(op.join(dir_base, 'spatial', 'ch_info', '%s_egi_dig.fcsv' % subject), 'w') as fid:
        fid.writelines('%s\n' % l for l in dat_to_write)

    # Save digitization file (.hpts)
    fname_dig_hpts = op.join(dir_base, 'spatial', 'ch_info', '%s_egi_dig.hpts' % subject)
    if not op.isfile(fname_dig_hpts):
        hpts = np.zeros(len(ch_names), dtype=[('ch_type', 'U6'), ('ch_name', 'U6'), ('x', float), ('y', float), ('z', float)])
        hpts['ch_type'] = ch_types
        hpts['ch_name'] = ch_names
        hpts['x'] = coords_vals[:, 0]
        hpts['y'] = coords_vals[:, 1]
        hpts['z'] = coords_vals[:, 2]

        np.savetxt(fname_dig_hpts, hpts, fmt="%3s %3s %10.3f %10.3f %10.3f")

    print('Done creating digitization files')


def find_stim_events(raw):
    """
    Find stimulation events by using peak detection after high-pass filtering
    the data.

    Parameters
    ----------
    raw : mne.Raw
        The dataset

    Returns
    -------
    events : np.array
        The events matrix in mne format
    """
    import peakutils
    import matplotlib.pyplot as plt
    import numpy as np

    good_ch = [ix for ix, ch in enumerate(raw.ch_names) if ch not in
               raw.info['bads']]
    dat = raw.get_data(picks=good_ch)

    hp = mne.filter.filter_data(dat, raw.info['sfreq'], 200, None,
                                method='iir', iir_params=None)
    gfp = hp.std(0)

    ev_ok = False
    raw_ok = True
    thres = 0.7
    while not ev_ok:
        indexes = peakutils.indexes(gfp, thres=thres, min_dist=100)
        plt.plot(raw.times, gfp)
        plt.plot(raw.times[indexes], gfp[indexes], 'o')
        plt.title('Find stimulations \nthreshold = %0.1f - found: %s' %
                  (thres, len(indexes)))
        plt.ylabel('gfp')
        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show(block=True)

        resp = input('ok? y (yes) / r (back to raw) / nr (new threshold): ')
        print(resp, type(resp))
        if resp == 'y':
            ev_ok = True
        elif resp == 'r':
            ev_ok = True
            raw_ok = False
        else:
            thres = float(resp)
    events = np.vstack((indexes, np.zeros(len(indexes)),
                        np.ones(len(indexes)))).T.astype(int)

    if raw_ok:
        return events
    else:
        print('Events not found')


def get_stim_info(string):
    import re
    try:
        subj = re.findall(r's[0-9]_', string)[0].replace('_', '')
        st_ch = re.findall(r'[A-Z](?:\')?[0-9]+-[0-9]+', string)[0]
        st_int = re.findall(r'[0-9]+ma', string)[0]
        st_dur = re.findall(r'[0-9]+ms', string)[0]
        st_fq = re.findall(r'[0-9]+hz', string)[0]
    except IndexError:
        print('Stimulation parameters not found')
        return

    stim_info = {'subj': subj, 'st_chan': st_ch, 'st_int': st_int, 'st_dur': st_dur, 'st_fq': st_fq}
    return stim_info


def rename_files(path, old_str, new_str):
    import os
    import glob
    for f in glob.glob(os.path.join(path, '*')):
        os.rename(f, f.replace(old_str, new_str))


def make_bip_coords(seeg_coords):
    import re
    import numpy as np
    import pandas as pd
    names = seeg_coords.name.values
    new_names = []
    new_coords = []
    new_areas = []

    for ix_ch, ch in enumerate(names):
        if ch is not names[-1]:
            next_name = names[ix_ch+1]

            match = re.match(r"([a-z]+)(?:\')?([0-9]+)", ch, re.I)
            if match:
                items_ch = match.groups()

            match = re.match(r"([a-z]+)(?:\')?([0-9]+)", next_name, re.I)
            if match:
                items_next_ch = match.groups()

            if (items_ch[0] == items_next_ch[0]) and (int(items_ch[1])
                                                      == int(items_next_ch[1])
                                                      - 1):
                new_names.append('%s-%s' %
                                 (ch, re.findall(r'\d+', next_name)[0]))
                cols_x_mean = seeg_coords.columns[1:-2]
                old_coord_ch1 = seeg_coords.loc[ix_ch][cols_x_mean].values
                old_coord_ch2 = seeg_coords.loc[ix_ch+1][cols_x_mean].values
                new_coord = np.mean([old_coord_ch1, old_coord_ch2], axis=0)
                new_coords.append(new_coord)
                new_areas.append(seeg_coords['area'].loc[ix_ch])
            del items_ch, items_next_ch

    new_coords = np.array(new_coords, dtype=float)

    bip_coords = pd.DataFrame(data=new_coords, columns=cols_x_mean)
    bip_coords['name'] = new_names
    bip_coords['area'] = new_areas
    bip_coords = bip_coords[['name', 'area']+cols_x_mean.values.tolist()]
    bip_coords = bip_coords.round(decimals=2)
    return bip_coords


def load_fcsv(fname_fcsv):
    import pandas as pd

    base_table = pd.read_csv(fname_fcsv, skiprows=2)

    desc_pre = base_table['desc']
    descs = {'area': [], 'ptd': [], 'gmpi': []}

    for r in desc_pre:
        spl = r.split(',')
        descs['gmpi'].append(float(spl[-1]))
        descs['area'].append(spl[0])
        descs['ptd'].append(float(spl[-3]))

    descs = pd.DataFrame(descs)
    ch_info = base_table[['label', 'x', 'y', 'z']]
    ch_info = ch_info.rename(columns={'label': 'name'})

    ch_info = pd.concat([ch_info, descs], axis=1)
    ch_info = ch_info[['name', 'area', 'gmpi', 'ptd', 'x', 'y', 'z']]

    # todo: add surf coords
    return ch_info


def calc_trigger_consitency_events(events_trig):
    tr = events_trig[:, 0]
    diff = [tr[ix+1] - tr[ix] for ix in range(len(tr)-1)]
    plt.figure()
    plt.plot(diff, '.')
    plt.ylim([16500, 16600])


def calc_trigger_consistency_epo(epo, plot=True):
    dat = epo.get_data()
    dat = dat[:, :-1, np.where((-0.002 < epo.times) & (epo.times < 0.002))]
    dat = dat.squeeze()

    times_crop = epo.times[np.where((-0.002 < epo.times) & (epo.times < 0.002))]

    ch_max_times = []
    ch_max_names = []
    for ix_ch, ch in enumerate(range(dat.shape[1])):
        if epo.info['ch_names'][ch] not in epo.info['bads']:
            d = dat[:, ch, :]
            ep_max_times = []
            for ep in d:
                sig_max_samp = np.argmax(ep)
                sig_max_time = times_crop[sig_max_samp]

                ep_max_times.append(sig_max_time)
            ch_max_times.append(ep_max_times)
            ch_max_names.append(epo.ch_names[ix_ch])

    if plot:
        plt.figure()
        plt.boxplot(ch_max_times)
        plt.xticks(np.arange(len(ch_max_names)), ch_max_names, rotation=90, fontsize=10)

    stds = [np.std(x) for x in ch_max_times]
    min_std_ch_name = ch_max_names[np.argmin(stds)]
    return min_std_ch_name


def find_ch_max_artifact(epo):
    epo_f = epo.copy().filter(200, None)
    f_dat = epo_f.get_data()[:, :-1, :]
    max_ix = np.median(f_dat.max(2), 0).argmax()
    ch_name = epo.ch_names[max_ix]
    return ch_name


def plot_single_epochs(ch_name, epo):
    ch_epo = epo.copy().pick_channels([ch_name])
    ch_dat = ch_epo.get_data().squeeze()
    y_max = ch_dat[:, np.where((-0.002 < epo.times) & (epo.times < 0.002))].max()*1.5

    fig, axes = plt.subplots(6, 10, sharex='all', sharey='all', figsize=(18, 10))
    for d, ax in zip(ch_dat, axes.flatten()):
        ax.plot(ch_epo.times * 1e3, d)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-y_max, y_max)
        ax.vlines(0, *ax.get_ylim(), linestyles='--', alpha=0.5)
    fig.suptitle('Single trials\nChannel: ' + ch_name)
    axes[-1, 0].set_xlabel('Time (ms)')
    axes[-1, 0].set_ylabel('Amplitude (V)')


def realign_epochs_by_peaks(epo, plot=False, how='max'):
    t_mask = (epo.times > -0.01) & (epo.times < 0.01)
    epo_dat = epo.get_data()

    if how == 'max':
        ch_name = find_ch_max_artifact(epo)
    elif how == 'consistency':
        ch_name = calc_trigger_consistency_epo(epo, plot=True)
    else:
        ch_name = how

    epo_dat_cut = epo_dat[:, epo.ch_names.index(ch_name), t_mask].squeeze()

    shifts = []

    ref_pk = np.argmax(epo_dat_cut[0, :])
    for e in epo_dat_cut[1:]:
        mov_pk = np.argmax(e)
        shift = ref_pk - mov_pk
        shifts.append(shift)

    t_mask_out = (epo.times > -0.25) & (epo.times < 0.045)
    t_mask_out_start = np.where(t_mask_out == True)[0][0]
    e0 = epo_dat[0, :, t_mask_out].T

    new_dat = [e0]
    for ix, e in enumerate(epo_dat[1:]):
        e_start = t_mask_out_start-shifts[ix]
        e_dat = e[:, e_start:e_start+sum(t_mask_out)]
        new_dat.append(e_dat)

    new_dat = np.stack(new_dat)

    new_epo = mne.EpochsArray(new_dat, epo.info, tmin=-0.25)
    new_epo.selection = epo.selection
    new_epo.drop_log = epo.drop_log
    new_epo.events = epo.events

    if plot:
        plot_single_epochs(ch_name, new_epo)

        plt.figure()
        plt.plot(epo.times[t_mask], epo_dat_cut.T)
        plt.title('Evoked - Pre-alignment')

        plt.figure()
        plt.plot(new_epo.times, new_dat[:, epo.ch_names.index(ch_name), :].T)
        plt.title('Evoked - Post-alignment')
    return new_epo


def merge_sessions(raw, raw_tmp):
    # merge avoiding edge artifacts by subtracting the offset difference
    # between sessions

    last_samp_raw = raw.get_data()[:, -1]
    first_samp_tmp = raw_tmp.get_data()[:, 0]
    diff = first_samp_tmp - last_samp_raw
    diff = diff.reshape(len(diff), 1)

    raw_tmp._data = raw_tmp.get_data()-diff
    raw = mne.concatenate_raws([raw, raw_tmp])
    return raw


def make_bads_info(dir_epo, fname_save):
    import glob
    import os
    import mne
    import json
    import natsort

    files = os.listdir(dir_epo)
    files.sort()

    subjs = [f.split('_')[0] for f in files]
    subjs = set(subjs)
    dat = {s: {} for s in subjs}

    for s in subjs:
        subj_epos = glob.glob(os.path.join(dir_epo,  s + '*'))

        for f in subj_epos:
            ses = os.path.split(f)[-1].replace('-epo.fif', '')
            epo = mne.read_epochs(f, preload=False)
            dat[s][ses] = {}
            dat[s][ses]['bad_ch'] = natsort.natsorted(epo.info['bads'])
            dat[s][ses]['good_epo'] = epo.selection.tolist()

    dat = dict(sorted(dat.items()))
    with open(fname_save, 'w') as outfile:
        json.dump(dat, outfile, indent=4)
