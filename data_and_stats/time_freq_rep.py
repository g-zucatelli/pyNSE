import math
import numpy as np

class TimeFreqRep(object):
    
    @staticmethod
    def hermf(Nh, Mh, tm):
        """
        Calculate a set of Orthonormal Hermitian Functions
        :param Nh: Number of Points
        :param Mh: Maximum order
        :param tm: half time support

        :return h: Hermite Functions (MxN)
        :return Dh: H' (MxN)
        :return tt: time vector (1xN)
        """
        h_dt = 2 * tm / (Nh-1)
        h_tt = np.linspace(-tm, tm, int(Nh))

        g = np.exp(-(h_tt ** 2)/2)

        P = np.array([np.zeros(int(Nh)), np.ones(int(Nh)), 2*h_tt])
        for k in np.arange(3, Mh+2):
            temp = [2 * h_tt * P[k-1] - 2 * (k-2) * P[k-2]]
            P = np.concatenate([P, temp])

        Htemp = [np.zeros(int(Nh))]
        for k in np.arange(1,Mh+2):
            temp = [P[k] * g / np.sqrt(np.sqrt(np.pi) * (2 ** (k-1)) * math.gamma(k)) * np.sqrt(h_dt)]
            Htemp = np.concatenate([Htemp, temp])

        h = Htemp[1:Mh+1]

        Dh = [np.zeros(int(Nh))]
        for k in np.arange(1, (Mh+1)):
            temp = [(h_tt * Htemp[k] - np.sqrt(2*k) * Htemp[k + 1])*h_dt]
            Dh = np.concatenate([Dh, temp])

        return h, Dh, h_tt
    
    @staticmethod
    def get_length_based_params(length, sample_rate, analysis_windows, Mh, tm):
        length_params = {}
        Nx = int(length * sample_rate)

        for Nh0 in analysis_windows:
            length_params[Nh0] = {}
            length_params[Nh0]["Nh"] = np.round(Nx*Nh0/2)*2-1
            length_params[Nh0]["Nfft"] = int(2 ** np.ceil(np.log2(length_params[Nh0]["Nh"])))
            length_params[Nh0]["dt"] = (length_params[Nh0]["Nh"]+1) // 8
            length_params[Nh0]["sides"] = (length_params[Nh0]["Nh"] + 1) / 2
            length_params[Nh0]["tt"] = np.arange(length_params[Nh0]["sides"], Nx-length_params[Nh0]["sides"]+1, length_params[Nh0]["dt"])
            length_params[Nh0]["ttred"] = length_params[Nh0]["tt"]
            
            length_params[Nh0]["hermf"] = TimeFreqRep.hermf(length_params[Nh0]["Nh"], Mh, tm)
            length_params[Nh0]["tfrsp_hm_S"] = np.zeros([length_params[Nh0]["Nfft"], len(length_params[Nh0]["tt"]), Mh])

            length_params[Nh0]["tfrsp_h_xrow"] = Nx
            length_params[Nh0]["tfrsp_h_tcol"] = np.shape(length_params[Nh0]["tt"])[0]
            length_params[Nh0]["tfrsp_h_hrow"] = np.shape(length_params[Nh0]["hermf"][0][0])[0]
            length_params[Nh0]["tfrsp_h_Lh"] = int((length_params[Nh0]["tfrsp_h_hrow"]-1)/2)
            hlength = np.floor(length_params[Nh0]["Nfft"]/4)
            length_params[Nh0]["tfrsp_h_hlength"] = hlength + 1 - (hlength % 2)
            
            Szero = np.zeros([length_params[Nh0]["Nfft"], length_params[Nh0]["tfrsp_h_tcol"]], dtype=complex)
            length_params[Nh0]["tfrsp_h_S_tf2_tf3"] = Szero
        
        return length_params
    
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