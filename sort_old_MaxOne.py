import os, time, shutil, pickle
from pathlib import Path
from glob import glob
import numpy as np
import spikeinterface.full as si

sorter = 'kilosort2_5'
si.Kilosort2_5Sorter.set_kilosort2_5_path('/home/phornauer/Git/Kilosort_2020b')
sorter_params = si.get_default_sorter_params(si.Kilosort2_5Sorter)
sorter_params['n_jobs'] = -1
sorter_params['detect_threshold'] = 5.5
sorter_params['minFR'] = 0.01
sorter_params['minfr_goodchannels'] = 0.01
sorter_params['keep_good_only'] = False
sorter_params['do_correction'] = False

with open("file_path_list.pkl", "rb") as output:
    path_list = pickle.load(output)
save_root = '/net/bs-filesvr02/export/group/hierlemann/intermediate_data/Mea1k/phornauer/DeePhysSortings/'
for p_idx, p in enumerate(path_list):
    path_parts = p.split('/')
    recording_path = os.path.join("/",save_root, *path_parts[7:9],path_parts[-1])
    output_folder = Path(os.path.join("/",*recording_path.split('/')[:-1],"well000","sorted"))
    if not os.path.exists(os.path.join(output_folder,'amplitudes.npy')):
        #print(f"\n\n---------Sorting recording {path_parts[7:9]}---------\n\n")
        output_folder.mkdir(parents=True, exist_ok=True)
        raw_file = os.path.join(output_folder,'recording.dat')
        wh_file = os.path.join(output_folder, 'temp_wh.dat')
        try:
            rec = si.MaxwellRecordingExtractor(recording_path)
            print(f"DURATION: {rec.get_num_frames() / rec.get_sampling_frequency()} s -- "
              f"NUM. CHANNELS: {rec.get_num_channels()}")
        except Exception:
            continue

        try:
            t_start_sort = time.time()
            sorting = si.run_sorter(sorter, rec, output_folder=output_folder, verbose=False,
                                    **sorter_params)
            print(f"\n\nSpike sorting elapsed time {time.time() - t_start_sort} s")
            if os.path.exists(wh_file):
                os.remove(wh_file)
            if os.path.exists(raw_file):
                os.remove(raw_file)
        except Exception:
            if os.path.exists(wh_file):
                os.remove(wh_file)
            if os.path.exists(raw_file):
                os.remove(raw_file)
            continue
        if (p_idx + 1)%10 == 0:
            print(f'Finished {p_idx+1}/{len(path_list)} recordings')
