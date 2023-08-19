import numpy as np

def statio_test_theta(tfr, ttred, opt_dist=8, a=0, b=0.5):
    [l, c] = np.shape(tfr)
    Cn_dist = dist_locvsglob(tfr, ttred, opt_dist, a, b)
    Cn_mean = np.mean(Cn_dist)
    theta = np.sum((Cn_dist - Cn_mean) ** 2) / len(ttred)

    return theta, Cn_dist, Cn_mean


def dist_locvsglob(S, t, opt_dist=8, a=0, b=0.5, lamb=1):
    [Nfftover2, Nx] = np.shape(S)
    dS = np.zeros(len(t), dtype='float64')
    B = np.arange(int(np.max([np.round(a * Nfftover2 * 2), 1])-1), int(np.round(b * Nfftover2 * 2)))

    ymargS = np.mean(S, axis=1)

    if opt_dist == 8:
        ymargSN = np.abs(ymargS)/np.sum(np.abs(ymargS))

        for n in np.arange(len(t)):
            ytapS = S[:, n]
            ytapSN = np.abs(ytapS) / np.sum(np.abs(ytapS))

            D_kl  = np.sum((ytapSN[B]-ymargSN[B]) * np.log(ytapSN[B]/ymargSN[B]))
            D_lsd = np.sum(np.abs(np.log(ytapS[B]/ymargS[B])))
            dS[n] = D_kl*(1+lamb*D_lsd)

    else:
        # TO DO - Normalization + Switch Case for OPT_DIST distances
        raise Exception("Other Distances Not Available")

    return dS

