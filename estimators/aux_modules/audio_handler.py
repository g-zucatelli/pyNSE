import os
import numpy as np
import soundfile as sf
from tqdm.contrib.concurrent import process_map
import resampy

class AudioHandler(object):
    
    @staticmethod
    def read_wav(filepath, sample_rate):
        data, SR = sf.read(filepath)

        if len(data.shape)>1:
            data = data[:,1]

        data_rs = resampy.resample(data, SR, sample_rate)
        return data_rs, sample_rate
    
    @staticmethod
    def soxi_get_length(filepath):
        cmd = f"soxi -D {filepath}"
        length = float(os.popen(cmd).read().replace("\n",""))
        return (filepath, length)

    @staticmethod
    def get_file_lengths(list_wav_files):
        print(f"Lenght-Based Preprocessing:")
        list_filepath_lengths = process_map(AudioHandler.soxi_get_length,
                                   list_wav_files)
        
        list_lengths = [args[1] for args in list_filepath_lengths]
        list_lengths = list(set(list_lengths))
        return list_filepath_lengths, list_lengths

    @staticmethod
    def slice_signal(signal, window_size, hop_size):
        start_window = np.arange(window_size)
        num_windows = int( (len(signal)-window_size) // hop_size + 1)
        window_indeces = np.array([start_window + idx*hop_size for idx in range(num_windows)])
        return signal[window_indeces]