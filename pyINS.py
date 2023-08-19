import argparse
import sys
import soundfile as sf
import numpy as np
import scipy
import pickle
import time

from tqdm import tqdm

from data_and_stats import tf_representation as tf_rep
from data_and_stats import stationary_test as stat_test

def get_args():
    parser = argparse.ArgumentParser(description="Arguments to run pyINS (python Index of Nons-Stationary).")
    parser.add_argument('-p', '--path', type=str, help="Path to signal")
    parser.add_argument('-w', '--window_length', nargs='+', help='List of windows relative to total length, values in [0,1]', default=[0.2])
    parser.add_argument('-n', '--num_surrogates', type=int, default=50, help="Number of Surrogates.")
    parser.add_argument('-s', '--save_spec', type=bool, default=False, help="Save Multitaper Spectrogram.")

    return vars(parser.parse_args(args=sys.argv[1:]))


if __name__ == '__main__':
    args = get_args()

    PATH = args['path']
    WIN_LEN = np.array([float(w) for w in args['window_length']])
    NUM_SUR = args['num_surrogates']
    SAVE_SPEC = args['save_spec']

    # Read Audio
    [x, sr] = sf.read(PATH)

    if len(x.shape)>1:
        x = x[:,1]

    x = x/np.std(x)
    Nx = len(x)

    # TRF Options
    choixtfr = 'mtfr' # MultiTaper Choices
    Mh = 5
    tm = 5

    # Test Options
    opt_dist = 8
    JJ = NUM_SUR
    doBS = 0
    fa_rate = 0.05
    Nhist = 20

    # Results in a List
    ins_list = ["Nh0,INS_thresh,INS\n"]

    for Nh0 in tqdm(WIN_LEN):
        Nh = np.round(Nx*Nh0/2)*2-1
        Nfft = int(2 ** np.ceil(np.log2(Nh)))
        dt = (Nh+1) // 8
        sides = (Nh + 1) / 2
        tt = np.arange(sides, Nx-sides+1, dt)
        ttred = tt

        # TF Representation
        x_hilbert = scipy.signal.hilbert(x)
        MSp = tf_rep.tfrsp_hm(x_hilbert, tt, Nfft, Nh, Mh, tm)
        tfrx = np.mean(MSp, axis=2)
        tfr = tfrx[:Nfft // 2, :]

        if (Nh0 == WIN_LEN[0]) and (SAVE_SPEC):
            path_out = PATH.replace('.wav', '.pkl')

            with open(path_out, 'wb') as handle:
                pickle.dump(tfr, handle)

        # Test Statistics
        #tfr = tfrx[:Nfft//2,:]
        [theta1, Cn_dist, Cn_mean] = stat_test.statio_test_theta(tfr, ttred)

        # Surrogate Reference Measures
        theta0 = np.zeros(JJ)

        for jj in np.arange(JJ):
            z = tf_rep.phasemodul(x, Nx)

            z_hilbert = scipy.signal.hilbert(z)
            MSp = tf_rep.tfrsp_hm(z_hilbert, tt, Nfft, Nh, Mh, tm)
            tfrz = np.mean(MSp, axis=2)

            tfr = tfrz[:Nfft // 2, :]
            [theta_o, Cn_dist, Cn_mean] = stat_test.statio_test_theta(tfr, ttred)
            theta0[jj] = theta_o


        # Gamma Modelling
        gamma_hat = scipy.stats.rv_continuous.fit(scipy.stats.gamma, theta0, floc=0)
        (gamma_a, gamma_loc, gamma_scale) = gamma_hat

        gamma_thresh = scipy.stats.gamma.ppf(1-fa_rate, gamma_a, gamma_loc, gamma_scale)

        # Non-Stationary Evaluation
        test_result = (theta1 > gamma_thresh)

        INS = np.sqrt(theta1/np.mean(theta0))
        INS_thresh = np.sqrt(gamma_thresh / (gamma_a * gamma_scale))

        # Saving Results
        str_result = str(Nh0)+","+str(INS_thresh)+","+str(INS)+"\n"
        ins_list.append(str_result)


    # Saving Results in CSV file
    path_out = PATH.replace('.wav','.csv')

    with open(path_out, 'w') as f:
        f.writelines(ins_list)