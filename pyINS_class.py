import os
import scipy
import math
import tqdm
import pickle
import numpy as np
import soundfile as sf
from tqdm.contrib.concurrent import process_map
import resampy

from auxiliary_functions import tf_representation as tf_rep
from auxiliary_functions import stationary_test as stat_test

class INS:
    def __init__(self, list_wav_files,
                 sample_rate,
                 analysis_windows,
                 number_surrogates,
                 save_specs=False):
        
        # Current Fixed Parameters:
        self.choixtfr = 'mtfr' # MultiTaper Choices
        self.Mh = 5
        self.tm = 5

        self.opt_dist = 8
        self.doBS = 0
        self.fa_rate = 0.05
        self.Nhist = 20


        # Execution Parameters:
        # self.list_wav_files = list_wav_files
        self.list_wav_files = [line.strip() for line in open(list_wav_files)]
        self.sample_rate = sample_rate
        self.analysis_windows = analysis_windows
        self.JJ = number_surrogates
        self.save_spec = save_specs

        self.filepath_lengths, self.key_lengths = self.get_file_lengths()
        self.dict_params = self.__prep_dict_params()

    def soxi_get_length(self, filepath):
        cmd = f"soxi -D {filepath}"
        length = float(os.popen(cmd).read().replace("\n",""))
        return (filepath, length)

    def get_file_lengths(self):
        print(f"Lenght-Based Preprocessing:")
        list_filepath_lengths = process_map(self.soxi_get_length,
                                   self.list_wav_files)
        
        list_lengths = [args[1] for args in list_filepath_lengths]
        list_lengths = list(set(list_lengths))
        return list_filepath_lengths, list_lengths

    def get_length_based_params(self, length):
        length_params = {}
        Nx = int(length * self.sample_rate)

        for Nh0 in self.analysis_windows:
            length_params[Nh0] = {}
            length_params[Nh0]["Nh"] = np.round(Nx*Nh0/2)*2-1
            length_params[Nh0]["Nfft"] = int(2 ** np.ceil(np.log2(length_params[Nh0]["Nh"])))
            length_params[Nh0]["dt"] = (length_params[Nh0]["Nh"]+1) // 8
            length_params[Nh0]["sides"] = (length_params[Nh0]["Nh"] + 1) / 2
            length_params[Nh0]["tt"] = np.arange(length_params[Nh0]["sides"], Nx-length_params[Nh0]["sides"]+1, length_params[Nh0]["dt"])
            length_params[Nh0]["ttred"] = length_params[Nh0]["tt"]
            
            length_params[Nh0]["hermf"] = tf_rep.hermf(length_params[Nh0]["Nh"], self.Mh, self.tm)
            length_params[Nh0]["tfrsp_hm_S"] = np.zeros([length_params[Nh0]["Nfft"], len(length_params[Nh0]["tt"]), self.Mh])

            length_params[Nh0]["tfrsp_h_xrow"] = Nx
            length_params[Nh0]["tfrsp_h_tcol"] = np.shape(length_params[Nh0]["tt"])[0]
            length_params[Nh0]["tfrsp_h_hrow"] = np.shape(length_params[Nh0]["hermf"][0][0])[0]
            length_params[Nh0]["tfrsp_h_Lh"] = int((length_params[Nh0]["tfrsp_h_hrow"]-1)/2)
            hlength = np.floor(length_params[Nh0]["Nfft"]/4)
            length_params[Nh0]["tfrsp_h_hlength"] = hlength + 1 - (hlength % 2)
            
            Szero = np.zeros([length_params[Nh0]["Nfft"], length_params[Nh0]["tfrsp_h_tcol"]], dtype=complex)
            length_params[Nh0]["tfrsp_h_S_tf2_tf3"] = Szero
        
        return length_params
        
    def __prep_dict_params(self):
        params = {}
        for length in self.key_lengths:
            params[length] = self.get_length_based_params(length) 
        
        return params

    def read_wav(self, filepath):
        data, SR = sf.read(filepath)

        if len(data.shape)>1:
            data = data[:,1]

        data_rs = resampy.resample(data, SR, self.sample_rate)
        return data_rs

    @staticmethod
    def phasemodul(x):
        Nfft = len(x)
        A = np.zeros(Nfft)
        idx = np.arange(int(np.ceil(Nfft / 2)))

        y = np.fft.fft(x, Nfft)
        A[idx] = np.abs(y[idx])

        phase0 = (2*np.pi)*np.random.rand(Nfft)
        z = np.real(np.fft.ifft(2 * A * np.exp(1j * phase0)))
        return z
    
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
        xrow = self.dict_params[length][Nh0]["tfrsp_h_xrow"] #np.shape(x_hilbert)[0]
        tcol = self.dict_params[length][Nh0]["tfrsp_h_tcol"] #np.shape(tt)[0]
        #hrow = self.dict_params[length][Nh0]["tt"] #np.shape(h_k)[0]
        Lh = self.dict_params[length][Nh0]["tfrsp_h_Lh"] #int((hrow-1)/2)
        #hlength = self.dict_params[length][Nh0]["tfrsp_h_hlength"]

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

        #dict_ins = {"Nh0": self.analysis_windows, "INS_thresh":[], "INS":[]}
        ins_list = ["Nh0,INS_thresh,INS\n"]

        x = self.read_wav(filepath)
        x_hilbert = scipy.signal.hilbert(x)

        for idx, Nh0 in enumerate(self.analysis_windows):
            MSp = self.tfrsp_hm(x_hilbert, length, Nh0)
            tfrx = np.mean(MSp, axis=2)
            tfr = tfrx[:self.dict_params[length][Nh0]["Nfft"] // 2, :]

            if idx==0 and self.save_spec:
                path_out = filepath.replace('.wav', '.pkl')

                with open(path_out, 'wb') as handle:
                    pickle.dump(tfr, handle)

            # Test Statistics
            [theta1, Cn_dist, Cn_mean] = stat_test.statio_test_theta(tfr,
                                                                     self.dict_params[length][Nh0]["ttred"])

            # Surrogate Reference Measures
            theta0 = np.zeros(self.JJ)

            for jj in np.arange(self.JJ):
                z = self.phasemodul(x)
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
            
            #dict_ins["INS_thresh"].append(_INS_thresh)
            #dict_ins["INS"].append(_INS)
            
            # Saving Results
            str_result = str(Nh0)+","+str(_INS_thresh)+","+str(_INS)+"\n"
            ins_list.append(str_result)

        # Saving Results in CSV file
        path_out = filepath.replace('.wav','_class.csv')

        with open(path_out, 'w') as f:
            f.writelines(ins_list)

    def exec_ins_calc(self):
        print("Executing INS calculation:")
        process_map(self.calc_individual_ins, self.filepath_lengths)


