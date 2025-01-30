import csv
from estimators.aux_modules.audio_handler import AudioHandler

AUDIO_SAMPLE_RATES = [8000, 16000, 32000, 441000, 48000, 96000]

class BaseEstimator(object):
    def __init__(self, sample_rate=16000, scales=[0.2]):
        self.config = {}

        self.set_sample_rate(sample_rate)
        self.set_scales(scales)
        self.audio_handler = AudioHandler()

    def set_sample_rate(self, sample_rate):
        assert sample_rate in AUDIO_SAMPLE_RATES, f"Check audio sample rates in {AUDIO_SAMPLE_RATES}."
        self.config['sample_rate'] = sample_rate
    
    def set_scales(self, scales):
        assert all(scale > 0 and scale < 1 for scale in scales), "Observable scales should be a list of positive values in (0,1]."
        self.config['scales'] = scales

    def get_info(self):
        for key, value in self.config.items():
            print(f"{key}: {value}")
        return self.config
    
    def write_ns_eval(self, csv_path, ns_eval):
        with open(csv_path, "w", newline="") as csv_file:
            w = csv.writer(csv_file)
            w.writerow(ns_eval.keys())
            for idx in range(len(self.config['scales'])):
                local_eval = [ns_eval[key][idx] for key in ns_eval.keys()]
                w.writerow(local_eval)

    def call(self):
        raise NotImplementedError
