import numpy as np
import scipy
import math


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


def phasemodul(x, Nx):
    Nfft = Nx
    A = np.zeros(Nfft)
    idx = np.arange(int(np.ceil(Nfft / 2)))

    y = np.fft.fft(x, Nfft)
    A[idx] = np.abs(y[idx])

    phase0 = (2*np.pi)*np.random.rand(Nfft)

    z = np.real(np.fft.ifft(2 * A * np.exp(1j * phase0)))

    return z

def tfrsp_hm(x_hilbert, tt, Nfft, Nh, Mh, tm):
    [h, Dh, d_tt] = hermf(Nh, Mh, tm)

    S = np.zeros([Nfft, len(tt), Mh])

    for k in np.arange(1,Mh+1):
        spt = tfrsp_h(x_hilbert, tt, Nfft, h[k-1], Dh[k])
        S[:,:,k-1] = spt

    return S

def tfrsp_h(x_hilbert, tt, Nfft, h_k, Dh_k):
    xrow = np.shape(x_hilbert)[0]
    tcol = np.shape(tt)[0]
    hrow = np.shape(h_k)[0]
    Lh = int((hrow-1)/2)

    hlength = np.floor(Nfft/4)
    hlength = hlength + 1 - (hlength % 2)

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

    S = np.zeros([Nfft, tcol], dtype=complex)
    tf2 = np.zeros([Nfft, tcol], dtype=complex)
    tf3 = np.zeros([Nfft, tcol], dtype=complex)

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