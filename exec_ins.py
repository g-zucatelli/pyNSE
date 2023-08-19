import sys
import argparse
import numpy as np

from pyINS_class import INS

def get_args():
    parser = argparse.ArgumentParser(description="Arguments to run pyINS (python Index of Nons-Stationary).")
    parser.add_argument('-p', '--path', type=str, help="Path to signal.")
    parser.add_argument('-l', '--pathlist', type=str, help="Path to list of signals.")
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000, help="Sample rate for audio files.")
    parser.add_argument('-w', '--window_length', nargs='+', help='List of INS analysis windows relative to total length, values in [0,1].', default=[0.2])
    parser.add_argument('-n', '--num_surrogates', type=int, default=50, help="Number of Surrogates.")
    parser.add_argument('-f', '--feature', type=bool, default=False, help="Extract INS as feature.")
    parser.add_argument('-fw', '--feature_window', type=str, default='0.4', help="Feature window duration in seconds (s).")
    parser.add_argument('-fh', '--feature_hop', type=str, default='0.2', help="Feature hop duration in seconds (s).")
    parser.add_argument('-s', '--save_spec', type=bool, default=False, help="Save Multitaper Spectrogram.")

    return vars(parser.parse_args(args=sys.argv[1:]))

def check_input_args(args):
    assert bool(args['path']) != bool(args['pathlist']), "Assert '-p' or '-l' is part of input flags."

if __name__ == '__main__':
    args = get_args()
    check_input_args(args)

    if args['path']:
        PATH = [args['path']]
    else:
        PATH = [line.strip() for line in open(args['pathlist'])]
    
    SAMPLE_RATE = args['sample_rate']
    WIN_LEN = np.array([float(w) for w in args['window_length']])
    NUM_SUR = args['num_surrogates']
    FEAT = args['feature']
    FEAT_WIN = float(args['feature_window'])
    FEAT_HOP = float(args['feature_hop'])
    SAVE_SPEC = args['save_spec']


    ins = INS(list_wav_files=PATH,
              sample_rate=SAMPLE_RATE,
              analysis_windows=WIN_LEN,
              number_surrogates=NUM_SUR,
              extract_feature=FEAT,
              feature_window=FEAT_WIN,
              feature_hop=FEAT_HOP,
              save_specs=False) # TO DO
    ins.exec_ins_calc()

    