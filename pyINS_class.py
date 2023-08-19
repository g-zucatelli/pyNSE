import scipy
import pickle
import numpy as np
from tqdm.contrib.concurrent import process_map

from data_and_stats.audio_handler import AudioHandler 
from data_and_stats.time_freq_rep import TimeFreqRep 
from data_and_stats import stationary_test as stat_test

class INS:
    def __init__(self, list_wav_files,
                 sample_rate,
                 analysis_windows,
                 number_surrogates,
                 extract_feature,
                 feature_window,
                 feature_hop,
                 save_specs=False):
        
        # Data and Statistic Classes
        self.audio_handler = AudioHandler()
        self.tf_rep = TimeFreqRep()

        # Current Fixed Parameters:
        self.choixtfr = 'mtfr' # MultiTaper Choices
        self.Mh = 5
        self.tm = 5

        self.opt_dist = 8
        self.doBS = 0
        self.fa_rate = 0.05
        self.Nhist = 20


        # Execution Parameters:
        self.list_wav_files = list_wav_files
        self.sample_rate = sample_rate
        self.analysis_windows = analysis_windows
        self.JJ = number_surrogates
        self.extract_feature = extract_feature
        self.feature_window = feature_window
        self.feature_hop = feature_hop
        self.save_spec = save_specs 

        self.filepath_lengths, self.key_lengths = self.audio_handler.get_file_lengths(self.list_wav_files)
        self.dict_params = self.__prep_dict_params()
        
    def __prep_dict_params(self):
        params = {}
        length_keys = self.key_lengths
        
        if self.extract_feature:
            length_keys = [self.feature_window]

        for length in length_keys:
            params[length] = self.tf_rep.get_length_based_params(length, 
                                                          self.sample_rate,
                                                          self.analysis_windows, 
                                                          self.Mh,
                                                          self.tm) 
        
        return params
    
    def tfrsp_hm(self, x_hilbert, length, Nh0):
        [h, Dh, _] = self.dict_params[length][Nh0]["hermf"]
        S = self.dict_params[length][Nh0]["tfrsp_hm_S"]

        for k in np.arange(1,self.Mh+1):
            spt = self.tfrsp_h(x_hilbert, self.dict_params[length][Nh0]["tt"],
                          self.dict_params[length][Nh0]["Nfft"],
                          h[k-1],
                          Dh[k],
                          length,
                          Nh0)
            S[:,:,k-1] = spt

        return S

    def tfrsp_h(self, x_hilbert, tt, Nfft, h_k, Dh_k, length, Nh0):
        xrow = self.dict_params[length][Nh0]["tfrsp_h_xrow"] 
        tcol = self.dict_params[length][Nh0]["tfrsp_h_tcol"] 
        Lh = self.dict_params[length][Nh0]["tfrsp_h_Lh"] 

        if (tcol==1):
            Dt = 1
        else:
            Deltat = tt[1:tcol]-tt[0:tcol-1]

            Mini = np.min(Deltat)
            Maxi = np.max(Deltat)
            if Mini != Maxi:
                raise Exception('The time instants must be regularly sampled.')
            else:
                Dt = Mini

        S = self.dict_params[length][Nh0]["tfrsp_h_S_tf2_tf3"]
        tf2 = self.dict_params[length][Nh0]["tfrsp_h_S_tf2_tf3"]
        tf3 = self.dict_params[length][Nh0]["tfrsp_h_S_tf2_tf3"]

        Th = h_k * np.arange(-Lh, Lh+1)

        for icol in np.arange(tcol):
            ti = int(tt[icol])
            range_o = np.min([np.round(Nfft/2)-1, Lh, ti-1])
            range_f = np.min([np.round(Nfft/2)-1, Lh, xrow-ti])
            tau = np.arange(-int(range_o), int(range_f+1))
            indices = (Nfft + tau) % Nfft

            norm_h = np.linalg.norm(h_k[Lh + tau])

            S[indices, icol] = x_hilbert[ti + tau - 1] * np.conjugate(h_k[Lh + tau]) / norm_h
            tf2[indices, icol] = x_hilbert[ti + tau - 1] * np.conjugate(Th[Lh + tau]) / norm_h
            tf3[indices, icol] = x_hilbert[ti + tau - 1] * np.conjugate(Dh_k[Lh + tau]) / norm_h

        S = scipy.fft.fft(S.T).T
        tf2 = scipy.fft.fft(tf2.T).T
        tf3 = scipy.fft.fft(tf3.T).T

        avoid_warn = np.where(S != 0)

        tf2[avoid_warn] = np.round(np.real(tf2[avoid_warn]/S[avoid_warn]/Dt))
        tf3[avoid_warn] = np.round(np.imag(Nfft*tf3[avoid_warn] / S[avoid_warn] / (2*np.pi)))
        S = np.abs(S) ** 2

        return S

    def calc_individual_ins(self, args):
        filepath, length = args
        dict_ins = {"Nh0": self.analysis_windows, "INS_thresh":[], "INS":[]}
        
        x = self.audio_handler.read_wav(filepath, self.sample_rate)
        
        if self.extract_feature:
            length = self.feature_window
            feature_window = int(self.feature_window * self.sample_rate)
            feature_hop = int(self.feature_hop * self.sample_rate)
            
            x_sliced = self.audio_handler.slice_signal(signal=x,
                                                       window_size=feature_window,
                                                       hop_size=feature_hop)
        else:
            x_sliced = [x]
        
        for x in x_sliced:
            x_hilbert = scipy.signal.hilbert(x)
            _ins_vec = []
            _thr_vec = []

            for idx, Nh0 in enumerate(self.analysis_windows):
                MSp = self.tfrsp_hm(x_hilbert, length, Nh0)
                tfrx = np.mean(MSp, axis=2)
                tfr = tfrx[:self.dict_params[length][Nh0]["Nfft"] // 2, :]

                if idx==0 and self.save_spec and not self.extract_feature:
                    path_out = filepath.replace('.wav', '_spec.pkl')
                    with open(path_out, 'wb') as handle:
                        pickle.dump(tfr, handle)

                # Test Statistics
                [theta1, Cn_dist, Cn_mean] = stat_test.statio_test_theta(tfr,
                                                                        self.dict_params[length][Nh0]["ttred"])

                # Surrogate Reference Measures
                theta0 = np.zeros(self.JJ)

                for jj in np.arange(self.JJ):
                    z = self.tf_rep.phasemodul(x)
                    z_hilbert = scipy.signal.hilbert(z)
                    
                    MSp = self.tfrsp_hm(z_hilbert, length, Nh0)
                    tfrz = np.mean(MSp, axis=2)
                    tfr = tfrz[:self.dict_params[length][Nh0]["Nfft"] // 2, :]

                    [theta_o, Cn_dist, Cn_mean] = stat_test.statio_test_theta(tfr,
                                                                            self.dict_params[length][Nh0]["ttred"])
                    theta0[jj] = theta_o


                # Gamma Modelling
                gamma_hat = scipy.stats.rv_continuous.fit(scipy.stats.gamma, theta0, floc=0)
                (gamma_a, gamma_loc, gamma_scale) = gamma_hat

                gamma_thresh = scipy.stats.gamma.ppf(1-self.fa_rate, gamma_a, gamma_loc, gamma_scale)

                # Non-Stationary Evaluation
                # test_result = (theta1 > gamma_thresh)

                _INS_thresh = np.sqrt(gamma_thresh / (gamma_a * gamma_scale))
                _INS = np.sqrt(theta1/np.mean(theta0))
                
                _thr_vec.append(_INS_thresh)
                _ins_vec.append(_INS)
            
            dict_ins["INS_thresh"].append(_thr_vec)
            dict_ins["INS"].append(_ins_vec)

        # Saving Results
        path_out = filepath.replace('.wav','_ins.pkl')
        with open(path_out, 'wb') as handle:
            pickle.dump(dict_ins, handle)

    def exec_ins_calc(self):
        print("Executing INS calculation:")
        process_map(self.calc_individual_ins, self.filepath_lengths)



