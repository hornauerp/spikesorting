import os
import pickle

from glob import glob
from pathlib import Path

import numpy as np
import spikeinterface.full as si
import UnitMatchPy.default_params as default_params
import UnitMatchPy.utils as util
import UnitMatchPy.metric_functions as mf
import UnitMatchPy.overlord as ov
from kneed import KneeLocator


def load_objects(path_list):
    """
    Load recordings and sortings from the specified paths.

    Arguments:
        path_list - list of Path objects

    Returns:
        recordings - list of RecordingExtractor
        sortings - list of SortingExtractor
    """

    recordings = []
    sortings = []

    for sorting_path in path_list:
        json_path = os.path.join(
            Path(sorting_path).parent.absolute(), "spikeinterface_recording.json"
        )

        recordings.append(si.load_extractor(json_path, base_folder=True))
        sortings.append(si.KiloSortSortingExtractor(sorting_path))

    return recordings, sortings


def select_good_units(sorting):
    """
    Select only good units from a sorting extractor.

    Arguments:
        sorting - SortingExtractor

    Returns:
        SortingExtractor with only good units
    """
    unit_ids_tmp = sorting.get_property("original_cluster_id")
    # If the sorting has a "bc_unitType" property, we use it to select good units
    if "bc_unitType" in sorting.get_property_keys():
        is_good_tmp = np.bool_(
            sorting.get_property("bc_unitType") == "GOOD"
        ) & np.bool_(sorting.get_property("KSLabel") == "good")
        keep = unit_ids_tmp[is_good_tmp]
        # Otherwise, we use only the KSLabel property
    else:
        is_good_tmp = sorting.get_property("KSLabel")
        keep = unit_ids_tmp[is_good_tmp == "good"]

    return sorting.select_units(keep)


def perform_cv_split(recordings, sortings):
    """
    Split recordings and sortings into 2 halves.

    Arguments:
        recordings - list of RecordingExtractor
        sortings - list of SortingExtractor

    Returns:
        split_recordings - list of 2 RecordingExtractor
        split_sortings - list of 2 SortingExtractor
    """
    split_sortings = []
    split_recordings = []

    # Split each recording/sorting into 2 halves
    for i, sorting in enumerate(sortings):
        split_idx = recordings[i].get_num_samples() // 2

        split_sorting = []
        split_sorting.append(sorting.frame_slice(start_frame=0, end_frame=split_idx))
        split_sorting.append(
            sorting.frame_slice(
                start_frame=split_idx, end_frame=recordings[i].get_num_samples()
            )
        )

        split_sortings.append(split_sorting)

    for i, recording in enumerate(recordings):
        split_idx = recording.get_num_samples() // 2

        split_recording = []
        split_recording.append(
            recording.frame_slice(start_frame=0, end_frame=split_idx)
        )
        split_recording.append(
            recording.frame_slice(
                start_frame=split_idx, end_frame=recording.get_num_samples()
            )
        )

        split_recordings.append(split_recording)

    return split_recordings, split_sortings


def extract_waveforms(split_recordings, split_sortings):
    """
    Extract waveforms from split recordings and sortings.

    Arguments:
        split_recordings - list of 2 RecordingExtractor
        split_sortings - list of 2 SortingExtractor

    Returns:
        all_waveforms - list of waveforms (nUnits, Time Points, Channels, CV)
    """
    # create sorting analyzer for each pair
    analysers = []
    for i, _ in enumerate(split_recordings):
        split_analysers = []

        split_analysers.append(
            si.create_sorting_analyzer(
                split_sortings[i][0], split_recordings[i][0], sparse=False
            )
        )
        split_analysers.append(
            si.create_sorting_analyzer(
                split_sortings[i][1], split_recordings[i][1], sparse=False
            )
        )
        analysers.append(split_analysers)

    all_waveforms = []
    for i, _ in enumerate(analysers):
        for half in range(2):
            analysers[i][half].compute(
                [
                    "random_spikes",
                    "waveforms",
                    "templates",
                ],
                extension_params={
                    "random_spikes": {"max_spikes_per_unit": 500, "method": "uniform"}
                    # "waveforms": {"ms_before": cutout_ms[0], "ms_after": cutout_ms[1]},
                },
            )
        templates_first = analysers[i][0].get_extension("templates")
        templates_second = analysers[i][1].get_extension("templates")
        t1 = templates_first.get_data()
        t2 = templates_second.get_data()
        all_waveforms.append(np.stack((t1, t2), axis=-1))

    return all_waveforms


def generate_templates(path_list, extract_good_units_only=True, load_if_exists=True):
    """
    Generate templates from the specified sortings.

    Arguments:
        path_list - list of Path objects
        extract_good_units_only - bool, default True

    Returns:
        waveforms - list of waveforms (nUnits, Time Points, Channels, CV)
        channel_pos - list of channel positions (nUnits, nChannels, 3)
    """
    # Generate save path from the first sorting path
    parts = Path(path_list[0]).parts
    save_path = Path(os.path.join(*parts[:-3], "UM_data", parts[-2]))

    # Load waveforms if they exist
    if load_if_exists and save_path.joinpath("waveforms.npz").exists():
        waveforms, channel_pos = load_waveforms(save_path)
        print(f"Waveforms and channel positions loaded from {save_path}")
        return waveforms, channel_pos
    
    # Load recordings and sortings
    recordings, sortings = load_objects(path_list)

    # Select only good units
    if extract_good_units_only:
        for i, sorting in enumerate(sortings):
            sortings[i] = select_good_units(sorting)

    # Preprocess recordings
    for recording in recordings:
        recording = si.highpass_filter(recording)  # highpass

    # Perform CV split
    split_recordings, split_sortings = perform_cv_split(recordings, sortings)

    print("Extracting waveforms...")
    # Extract waveforms
    all_waveforms = extract_waveforms(split_recordings, split_sortings)

    # Zero center waveforms
    waveforms = [zero_center_waveform(wf) for wf in all_waveforms]

    # Extract channel positions
    channel_pos = [r.get_channel_locations() for r in recordings]
    channel_pos = [np.insert(cp, 0, np.ones(cp.shape[0]), axis=1) for cp in channel_pos]

    save_waveforms(waveforms, channel_pos, save_path)

    return waveforms, channel_pos


def zero_center_waveform(waveform):
    """
    Centers waveform about zero, by subtracting the mean of the first 15 time points.
    This function is useful for Spike Interface where the waveforms are not centered about 0.

    Arguments:
        waveform - ndarray (nUnits, Time Points, Channels, CV)

    Returns:
        Zero centered waveform
    """
    waveform = waveform - np.broadcast_to(
        waveform[:, :5, :, :].mean(axis=1)[:, np.newaxis, :, :], waveform.shape
    )
    return waveform


def get_sorting_path_list(path_parts):
    """
    Get a list of sorting paths matching the description, sorted by modification time.

    Arguments:
        path_parts - list of strings

    Returns:
        path_list - list of Path objects
    """

    path_pattern = os.path.join(*path_parts)
    path_list = glob(path_pattern)
    path_list.sort(key=os.path.getmtime)  # sort by modification time
    path_list = [
        Path(p).parent.absolute() for p in path_list
    ]  # get the parent directory (the actual sorting output directory)
    print(
        f"Found {len(path_list)} sorting paths matching the description:\n{path_pattern}\n"
    )

    return path_list


def save_waveforms(waveforms, channel_pos, um_save_path):
    """
    Save waveforms and channel positions to a .npz file.

    Arguments:
        waveforms - list of waveforms (nUnits, Time Points, Channels, CV)
        channel_pos - list of channel positions (nUnits, nChannels, 3)
        um_save_path - Path object
    """
    output_path = os.path.join(um_save_path, "waveforms.npz")
    if not os.path.exists(um_save_path):
        os.makedirs(um_save_path)
    np.savez(
        output_path, waveforms=np.array(waveforms, dtype=object), channel_pos=channel_pos, allow_pickle=True
    )
    print(f"Waveforms and channel positions saved to {output_path}")


def load_waveforms(um_save_path):
    """
    Load waveforms and channel positions from a .npz file.

    Arguments:
        um_save_path - Path object

    Returns:
        waveforms - list of waveforms (nUnits, Time Points, Channels, CV)
        channel_pos - list of channel positions (nUnits, nChannels, 3)
    """
    output_path = os.path.join(um_save_path, "waveforms.npz")
    data = np.load(output_path, allow_pickle=True)
    waveforms = data["waveforms"]
    channel_pos = data["channel_pos"]
    return waveforms, channel_pos


def generate_waveform_array(waveforms):
    """
    Generate a single array of waveforms from a list of waveforms.

    Arguments:
        waveforms - list of waveforms (nUnits, Time Points, Channels, CV)

    Returns:
        waveform - ndarray (nUnits*nCVs, Time Points, Channels)
    """

    for i, wf in enumerate(waveforms):
        if i == 0:
            waveform = wf
        else:
            waveform = np.concatenate((waveform, wf), axis=0)

    return waveform


def generate_um_params(param, waveforms):
    """
    Generate the parameters for the UnitMatch algorithm.

    Arguments:
        param - dictionary (specified by the user)
        waveforms - list of waveforms (nUnits, Time Points, Channels, CV)

    Returns:
        param - dictionary
        clus_info - dictionary
    """
    param = default_params.get_default_param(param)
    n_units_per_session = np.array([wf.shape[0] for wf in waveforms])
    param["n_units"], session_id, session_switch, param["n_sessions"] = (
        util.get_session_data(n_units_per_session)
    )
    param["within_session"] = util.get_within_session(session_id, param)
    param["n_channels"] = waveforms[0].shape[2]
    param["n_units_per_session"] = n_units_per_session

    good_units = [list(range(u)) for u in n_units_per_session]

    clus_info = {
        "good_units": good_units,
        "session_switch": session_switch,
        "session_id": session_id,
        "original_ids": np.concatenate(good_units),
    }

    return param, clus_info


def find_best_matches_across_sessions(score_matrix, session_switch, th=0.1, sel_units = None):
    """
    Find the best matching units across sessions.

    Arguments:
        score_matrix - ndarray (nUnits, nUnits)
        session_switch - ndarray or list (session switch indices in # of units)
        th - float, default 0.1 (minimum threshold for matching units)

    Returns:
        unit_probs - ndarray (nUnits, nSessions)
        unit_paths - ndarray (nUnits, nSessions)
    """
    
    sel_ci = session_switch
    
    max_prob = np.max(np.stack((score_matrix, score_matrix.T), axis=2), axis=2)

    sel_prob = max_prob[sel_ci[0]:sel_ci[-1], sel_ci[0]:sel_ci[-1]].copy()
    unit_probs = np.zeros((sel_ci[1], len(sel_ci) - 2)) - 1
    unit_paths = np.zeros((sel_ci[1], len(sel_ci) - 1),dtype=int) - 1
    for u in range(sel_ci[1]):
        session_id = 1 # session ID to be matched (session_id - 1, session_id)
        session_prob = sel_prob[(sel_ci[session_id - 1]): sel_ci[session_id], sel_ci[session_id] : sel_ci[session_id + 1]]
        # We essentially sort the units by the maximum probability of matching
        matched_ids = np.unravel_index(np.argmax(session_prob), session_prob.shape) # (unit_id before, unit_id after)
        unit_id = int(matched_ids[0] + sel_ci[session_id - 1])
        if sel_units is not None and unit_id not in sel_units:
            sel_prob[unit_id, :] = 0
            continue
        unit_paths[u, session_id - 1] = unit_id
        while session_id < (len(sel_ci) - 1):
            unit_prob = sel_prob[unit_id, sel_ci[session_id] : sel_ci[session_id + 1]].copy()
            # We prevent the same unit from being matched twice
            sel_prob[unit_id, :] = 0
            
            if unit_prob.max() < th:
                break
            unit_id = int(np.argmax(unit_prob) + sel_ci[session_id])
            unit_probs[u, session_id - 1] = unit_prob.max()
            unit_paths[u, session_id] = unit_id
            session_id += 1

    return unit_probs, unit_paths


def infer_match_threshold(unit_probs, n_misses = 1):
    """
    Infer the threshold for matching units.

    Arguments:
        unit_probs - ndarray (nUnits, nSessions)

    Returns:
        threshold - float
    """
    th_array = np.arange(0, 1, 0.01)
    n_units_kept = []
    for th in th_array:
        # keep only units that are matched in all but n_misses sessions
        n_units_kept.append(
            (np.sum(unit_probs > th, axis=1) >= (unit_probs.shape[1] - n_misses)).sum()
        )
    kn = KneeLocator(th_array, n_units_kept, curve="concave", direction="decreasing")
    return kn.knee


def threshold_matches(unit_probs, unit_paths, threshold, n_misses = 1):
    """
    Keep matches based on the specified threshold.

    Arguments:
        unit_probs - ndarray (nUnits, nSessions)
        threshold - float

    Returns:
        unit_probs - ndarray (nUnits, nSessions)
        unit_paths - ndarray (nUnits, nSessions)
    """
    # keep only units that are matched in all but one sessions
    keep_idx = np.logical_and(np.sum(unit_probs > threshold, axis=1) >= (unit_probs.shape[1] - n_misses), 
                              np.all(unit_probs > -1, axis=1))
    
    unit_probs = unit_probs[keep_idx, :]
    unit_paths = unit_paths[keep_idx, :]
    return unit_probs, unit_paths


def match_across_session(score_matrix, clus_info, min_th=0.5, n_misses = 1, sel_units = None):
    """
    Match units across sessions.

    Arguments:
        score_matrix - ndarray (nUnits, nUnits)
        clus_info - dictionary

    Returns:
        unit_probs - ndarray (nUnits, nSessions)
        unit_paths - ndarray (nUnits, nSessions)
    """
    unit_probs, unit_paths = find_best_matches_across_sessions(score_matrix, clus_info, min_th, sel_units)
    threshold = infer_match_threshold(unit_probs, n_misses)
    unit_probs, unit_paths = threshold_matches(unit_probs, unit_paths, threshold, n_misses)

    print(
        f"Threshold for matching units: {threshold}, Number of matched units: {unit_probs.shape[0]}"
    )

    return unit_probs, unit_paths
   

def convert_unit_ids(unit_ids, session_switch, new_session_switch):
    return

def identify_session(unit_id, session_switch):
    """
    Identify the session of a unit.
    
    Arguments:
        unit_id - int
        session_switch - ndarray or list
        
    Returns:
        session_id - int
    """
    
    return np.argwhere(unit_id < session_switch)[0][0] - 1


def select_sessions(score_matrix, session_switch, sel_idx):
    """
    Select sessions from the score matrix.
    
    Arguments:
        score_matrix - ndarray (nUnits, nUnits)
        session_switch - ndarray or list
        sel_idx - list of session indices
        
    Returns:
        sel_matrix - ndarray (nUnits, nUnits)
        sel_switch - ndarray or list
    """
    ranges = [range(session_switch[i],session_switch[i+1]) for i in sel_idx]
    idx = np.r_[*ranges]
    sel_matrix = score_matrix[idx,:][:,idx]
    sel_switch = np.cumsum([len(range) for range in ranges])
    sel_switch = np.insert(sel_switch,0,0)
    return sel_matrix, sel_switch

def get_score_matrix(waveforms, channel_pos, param, load_if_exists=True):
    """
    Get the score matrix for the specified waveforms.
    
    Arguments:
        waveforms - list of waveforms (nUnits, Time Points, Channels, CV)
        channel_pos - list of channel positions (nUnits, nChannels, 3)
        param - dictionary
        load_if_exists - bool, default True
        
    Returns:
        score_matrix - ndarray (nUnits, nUnits)
        clus_info - dictionary
        param - dictionary
    """
    if load_if_exists and param["save_path"].joinpath("score_matrix.pkl").exists():
        with open(param["save_path"].joinpath("score_matrix.pkl"), "rb") as f:
            score_matrix, clus_info, param = pickle.load(f)
        print(f"Score matrix loaded from {param['save_path'].joinpath('score_matrix.pkl')}")
        return score_matrix, clus_info, param
    
    param, clus_info = generate_um_params(param, waveforms)
    waveform = generate_waveform_array(waveforms)
    extracted_wave_properties = ov.extract_parameters(waveform, channel_pos, clus_info, param)
    avg_waveform_per_tp_flip = mf.flip_dim(extracted_wave_properties['avg_waveform_per_tp'], param)
    euclid_dist = mf.get_Euclidean_dist(avg_waveform_per_tp_flip, param)
    centroid_dist, centroid_var = mf.centroid_metrics(euclid_dist, param)
    score_matrix = np.max(np.stack((centroid_dist, centroid_dist.T), axis=2), axis=2)
    
    save_score_matrix(score_matrix, clus_info, param)
    
    return score_matrix, clus_info, param


def save_score_matrix(score_matrix, clus_info, param):
    """
    Save the score matrix to a .pkl file.
    
    Arguments:
        score_matrix - ndarray (nUnits, nUnits)
        clus_info - dictionary
        param - dictionary
    """
    save_file = os.path.join(param["save_path"], "score_matrix.pkl")
    with open(save_file, "wb") as f:
        pickle.dump((score_matrix, clus_info, param), f)
    print(f"Score matrix saved to {save_file}")