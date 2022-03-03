#%% import
import numpy as np

from cachetools import cached

#%% Haar Transform

SQRTTWO = np.sqrt(2)

class WaveletTransform:
    wc = None ## Wavelet coefficients
    ws = None ## Approximation space sizes
    renorm = None ## renormalization scale
    J = None ## Maximal scale
    gamma = 1 ## renormalization factor


    def copy(self):
        H = WaveletTransform()
        H.wc = self.wc.copy()
        H.ws = self.ws.copy()
        H.renorm = self.renorm.copy()
        H.J = self.J
        return(H)

    def _haar_forward_step_1d(self, wc_cur, dim: int, j):
        wc_swap = wc_cur.swapaxes(dim, 0)
        wc_lowcur = wc_swap
        
        n = wc_lowcur.shape[0]
        n_ori = self.wc.shape[dim]
        n_low = np.ceil(n/2).astype('int')

        if (n == 1):
            wc_comb = wc_lowcur
        else:
            if (n % 2 == 0):
                wc_low = (wc_lowcur[0::2] + wc_lowcur[1::2]) / 2
                if (n_ori != n * 2 ** j):
                    delta = n_ori/(2**j) - (n - 1)
                    wc_low[-1] = (wc_lowcur[-2] + delta * wc_lowcur[-1]) / ( 1+ delta)
                wc_high = wc_lowcur[0::2] - wc_low
            else:
                wc_low = (wc_lowcur[0:-1:2] + wc_lowcur[1::2]) / 2
                wc_high = wc_lowcur[0:-1:2] - wc_low
                wc_low = np.concatenate([wc_low , wc_lowcur[[-1]]])

            wc_comb = np.concatenate([wc_low, wc_high])
        
        wc_swap[:] = wc_comb
        self.ws[dim].insert(0, n_low)

    def _haar_forward_step(self, j):
        wc_lowcur = self.wc[tuple([slice(None, s[0]) for s in self.ws])]
        for i,dd in enumerate(self.wc.shape):
            self._haar_forward_step_1d(wc_lowcur, i, j)


    def _haar_backward_step_1d(self, wc_cur, dim: int, j):

        wc_swap = wc_cur.swapaxes(dim, 0)
        n = self.ws[dim][1]
        n_ori = self.wc.shape[dim]
        n_low = self.ws[dim][0]
        wc_low = wc_swap[:n_low]
        wc_high = wc_swap[n_low:]

        wc_cur = np.zeros(shape=wc_swap.shape)

        if (n == 1):
            wc_cur = wc_low
        elif (n % 2 == 0):
            wc_cur[0::2] = wc_low + wc_high
            wc_cur[1::2] = 2 * wc_low - wc_cur[0::2]
            if (n_ori != n * 2 ** j):
                    delta = n_ori/(2**j) - (n - 1)
                    wc_cur[-1] = ((1 + delta) * wc_low[-1] - wc_cur[-2])/ delta
        else:
            wc_cur[0:-1:2] = wc_low[:-1] + wc_high
            wc_cur[-1] = wc_low[-1]
            wc_cur[1::2] = 2 * wc_low[:-1] - wc_cur[0:-1:2]

        wc_swap[:] = wc_cur
        self.ws[dim] = self.ws[dim][1:]

    def _haar_backward_step(self, j):
        wc_cur = self.wc[tuple([slice(None, s[1]) for s in self.ws])]
        for i,dd in enumerate(self.wc.shape):
            self._haar_backward_step_1d(wc_cur, i, j)

    def haar_forward(self, X, J = None, gamma = 1):
        self.wc = X.copy()
        self.gamma = gamma
        self.ws = [ [d] for d in self.wc.shape]
        if J is None:
            J = np.ceil(np.log2(np.max(self.wc.shape))).astype('int')
        self.J = J
        
        for j in range(J):
            self._haar_forward_step(j)

        if np.abs(self.gamma)>1e-8 :
            self.renorm = haar_FWT_renorm_shape(X.shape, J=J)
        else:
            self.renorm = np.ones_like(self.wc)

        self.wc = self.wc * np.power(self.renorm, self.gamma)

        return(self)


    def haar_backward_inplace(self):
        self.wc = self.wc / np.power(self.renorm, self.gamma)
        for j in range(self.J):
            self._haar_backward_step(self.J-j-1)
        self.J = 0
        return(self)

    def haar_backward(self):
        wc = self.wc.copy()
        ws = self.ws.copy()
        J = self.J

        self.haar_backward_inplace()

        X = self.wc.copy()
        self.wc = wc
        self.ws = ws
        self.J = J
        return(X)

    def _haar_coeff_scale_1D(self, i, s):
        delta = [s - i for s in s]
        delta_test = [delta > 0  for delta in delta]
        if all(delta_test):
            delta_J = 0
        else:
            delta_pos = delta_test.index(True)
            delta_J = delta_pos - 1
        
        return(delta_J)

    def _haar_coeff_pos_1D(self, i, s, delta_J):
        
        if (i < s[delta_J]):
            pos = i * (2 ** (self.J - delta_J))
        else:
            pos = (i - s[delta_J]) * (2 ** (self.J - delta_J))
        return pos

    def  _haar_coeff_type_1D(self, i, s, delta_J):
        
        if (i < s[delta_J]):
            type = "L" 
        else:
            type = "H"
        return type

    def haar_coeff_pos_type(self, ind):

        delta_J = [self._haar_coeff_scale_1D(i, s) for i,s in zip(ind, self.ws)]
        delta_J = max(delta_J)

        pos = [self._haar_coeff_pos_1D(i, s, delta_J) for i,s in zip(ind, self.ws)]
        type = [self._haar_coeff_type_1D(i, s, delta_J) for i,s in zip(ind, self.ws)]
        return (pos, type, self.J-delta_J)

    def __repr__(self):
        return "Haar Transform\nCoeffs: " + str(self.wc) +\
             "\n" + "Sizes (scale=" + str(self.J) + "): " + str(self.ws)
 


def haar_forward(X, J = None, gamma = 1):
    haar = WaveletTransform()
    haar.haar_forward(X, J, gamma)
    return(haar)

def haar_backward_transpose(X, J = None, gamma = 1):
    haar = WaveletTransform()
    haar.haar_forward(X, J, gamma)
    IWT_transpose = haar_IWT_mat_shape(X.shape, J, gamma).T
    haar.wc.flat = IWT_transpose @ X.flat
    return(haar) 


@cached(cache={})
def haar_IWT_mat_shape(shape, J = None, gamma = 1):
    X = np.zeros(shape=shape)
    return(haar_IWT_mat(X, J, gamma))

def haar_IWT_mat(X, J = None, gamma = 1):
    haar = haar_forward(X, J, gamma)
    nc = np.prod(haar.wc.shape)
    M = np.zeros(shape = (nc, nc))
    ws_ori = haar.ws.copy()
    for i in range(nc):
        haar.wc = np.zeros(shape=haar.wc.shape)
        haar_flat = haar.wc.flat
        haar_flat[i] = 1.
        haar.ws = ws_ori.copy()
        M[:,i] = haar.haar_backward().flatten()
    return(M)

@cached(cache={})
def haar_FWT_renorm_shape(shape, J = None):
    renorm = np.zeros(shape=shape)
    FWT = haar_FWT_mat_shape(shape, J=J, gamma=0)
    renorm.flat = 1 / np.sqrt(np.sum(FWT * FWT, axis=1))
    return(renorm)

@cached(cache={})
def haar_FWT_mat_shape(shape, J = None, gamma = 1):
    X = np.zeros(shape=shape)
    return(haar_FWT_mat(X, J, gamma))

def haar_FWT_mat(X, J = None, gamma = 1):
    nc = np.prod(X.shape)
    M = np.zeros(shape = (nc, nc))
    for i in range(nc):
        XX = np.zeros(X.shape)
        XX.flat[i] = 1.
        haar = haar_forward(XX, J, gamma)
        M[:,i] = haar.wc.flatten()
    return(M)






# %%
S = np.reshape(np.linspace(0,27,40), newshape=(4,10))
haar = haar_forward(S, J=2, gamma=1)
haar.haar_backward() - S

# %%
S = np.reshape(np.linspace(0,27,16*16), newshape=(16,16))
haar = haar_forward(S, gamma=1)
haar.haar_backward() - S

# %%
haar_backward_transpose(S, J=2, gamma=2)
# %%
haar._haar_coeff_scale_1D(3, haar.ws[0])
# %%

