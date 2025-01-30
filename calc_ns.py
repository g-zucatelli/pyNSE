import os
import sys
import argparse
import numpy as np
from tqdm.contrib.concurrent import process_map

from estimators.ins import INS

ESTIMATORS = {'INS': INS}
AUDIO_SAMPLE_RATES = [8000, 16000, 32000, 441000, 48000, 96000]

def get_args():
    parser = argparse.ArgumentParser(description="Arguments to run pyNSE (Python Non-Stationary Estimator Toolbox).")

    parser.add_argument('-p', '--path', type=str, help="Path to .WAV signal or .TXT list of WAV files.")
    parser.add_argument('-e', '--estimator', type=str, default='INS', help=f'Estimator name. Available options are {ESTIMATORS.keys()}')
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000, help=f'Sample rate. Available options are {ESTIMATORS.keys()}')
    parser.add_argument('-obs', '--observed_scales', nargs='+', default=[0.2], help='List of observed scales relative to total length, values in (0,1]')
    parser.add_argument('-ns', '--num_surrogates', type=int, default=50, help="Number of surrogates for INS estimator.")

    return vars(parser.parse_args(args=sys.argv[1:]))


if __name__ == '__main__':
    args = get_args()

    input_path = args['path']
    estimator_name = args['estimator']
    sample_rate = args['sample_rate']
    scales = np.array([float(obs) for obs in args['observed_scales']])

    if estimator_name == 'INS':
        num_surrogates = args['num_surrogates']
        estimator = ESTIMATORS[estimator_name](sample_rate, scales, num_surrogates)
    else:
        pass

    _, file_extension = os.path.splitext(input_path)

    if file_extension == '.wav':
        estimator(input_path)
    else:
        with open(input_path) as file:
            files = [line.rstrip() for line in file]
        process_map(estimator, files)



    