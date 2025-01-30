import os 
import scipy
import numpy as np
from tqdm import tqdm

from estimators.base_estimator import BaseEstimator 

from estimators.aux_modules import tf_representation as tf_rep
from estimators.aux_modules import stationary_test as stat_test


class INS(BaseEstimator):
    def __init__(self, sample_rate=16000, scales=..., num_surrogates=50):
        super().__init__(sample_rate, scales)
        
        self.set_num_surrogates(num_surrogates)
        self.set_ins_test()

    def set_num_surrogates(self, num_surrogates):
        assert isinstance(num_surrogates, int) and num_surrogates > 0, "Number surrogates must be positive integer."
        self.config['num_surrogates'] = num_surrogates

    def set_ins_test(self):
        # TRF Options
        self.config['choixtfr'] = 'mtfr' # MultiTaper Choices
        self.config['Mh'] = 5
        self.config['tm'] = 5

        # Test Options
        self.config['opt_dist'] = 8
        self.config['JJ'] = self.config['num_surrogates']
        self.config['doBS'] = 0
        self.config['fa_rate'] = 0.05
        self.config['Nhist'] = 20


    def __call__(self, audio_path):
        assert os.path.exists(audio_path), FileNotFoundError
        x, _ = self.audio_handler.read_wav(audio_path, self.config['sample_rate'])
        x = x/np.std(x)
        Nx = len(x)

        # Results in a List
        ns_eval = {'Nh0':[], 'INS_thresh':[], 'INS':[], 'Eval':[]}

        for Nh0 in tqdm(self.config['scales']):
            Nh = np.round(Nx*Nh0/2)*2-1
            Nfft = int(2 ** np.ceil(np.log2(Nh)))
            dt = (Nh+1) // 8
            sides = (Nh + 1) / 2
            tt = np.arange(sides, Nx-sides+1, dt)
            ttred = tt

            # TF Representation
            x_hilbert = scipy.signal.hilbert(x)
            MSp = tf_rep.tfrsp_hm(x_hilbert, tt, Nfft, Nh, self.config['Mh'], self.config['tm'])
            tfrx = np.mean(MSp, axis=2)
            tfr = tfrx[:Nfft // 2, :]

            # Test Statistics
            #tfr = tfrx[:Nfft//2,:]
            [theta1, Cn_dist, Cn_mean] = stat_test.statio_test_theta(tfr, ttred)

            # Surrogate Reference Measures
            theta0 = np.zeros(self.config['num_surrogates'])

            for jj in np.arange(self.config['num_surrogates']):
                z = tf_rep.phasemodul(x, Nx)

                z_hilbert = scipy.signal.hilbert(z)
                MSp = tf_rep.tfrsp_hm(z_hilbert, tt, Nfft, Nh, self.config['Mh'], self.config['tm'])
                tfrz = np.mean(MSp, axis=2)

                tfr = tfrz[:Nfft // 2, :]
                [theta_o, Cn_dist, Cn_mean] = stat_test.statio_test_theta(tfr, ttred)
                theta0[jj] = theta_o


            # Gamma Modelling
            gamma_hat = scipy.stats.rv_continuous.fit(scipy.stats.gamma, theta0, floc=0)
            (gamma_a, gamma_loc, gamma_scale) = gamma_hat

            gamma_thresh = scipy.stats.gamma.ppf(1-self.config['fa_rate'], gamma_a, gamma_loc, gamma_scale)

            # Local Non-Stationary Evaluation
            test_result = (theta1 > gamma_thresh)

            INS = np.sqrt(theta1/np.mean(theta0))
            INS_thresh = np.sqrt(gamma_thresh / (gamma_a * gamma_scale))

            # Saving Results
            ns_eval['Nh0'].append(Nh0)
            ns_eval['INS_thresh'].append(INS_thresh)
            ns_eval['INS'].append(INS)
            ns_eval['Eval'].append(test_result)


        # Saving Results in CSV file
        path_out = audio_path.replace('.wav','.csv')
        self.write_ns_eval(path_out, ns_eval)
