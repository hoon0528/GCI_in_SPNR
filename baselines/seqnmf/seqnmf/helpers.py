import numpy as np
import warnings

def get_shapes(W, H, force_full=False):
    N = W.shape[0]
    T = H.shape[1]
    K = W.shape[1]
    L = W.shape[2]

    #trim zero padding along the L and K dimensions
    if not force_full:
        W_sum = W.sum(axis=0).sum(axis=1)
        H_sum = H.sum(axis=1)
        K = 1
        for k in np.arange(W.shape[1]-1, 0, -1):
            if (W_sum[k] > 0) or (H_sum[k] > 0):
                K = k+1
                break

        L = 2
        for l in np.arange(W.shape[2]-1, 2, -1):
            W_sum = W.sum(axis=1).sum(axis=0)
            if W_sum[l] > 0:
                L = l+1
                break

    return N, K, L, T

def trim_shapes(W, H, N, K, L, T):
    return W[:N, :K, :L], H[:K, :T]

def reconstruct(W, H):
    N, K, L, T = get_shapes(W, H, force_full=True)
    W, H = trim_shapes(W, H, N, K, L, T)

    H = np.hstack((np.zeros([K, L]), H, np.zeros([K, L])))
    T += 2 * L
    X_hat = np.zeros([N, T])

    for t in np.arange(L):
        X_hat += np.dot(W[:, :, t], np.roll(H, t - 1, axis=1))

    return X_hat[:, L:-L]


def shift_factors(W, H):
    warnings.simplefilter('ignore') #ignore warnings for nan-related errors

    N, K, L, T = get_shapes(W, H, force_full=True)
    W, H = trim_shapes(W, H, N, K, L, T)

    if L > 1:
        center = int(np.max([np.floor(L / 2), 1]))
        Wpad = np.concatenate((np.zeros([N, K, L]), W, np.zeros([N, K, L])), axis=2)

        for i in np.arange(K):
            temp = np.sum(np.squeeze(W[:, i, :]), axis=0)
            # return temp, temp
            try:
                cmass = int(np.max(np.floor(np.sum(temp * np.arange(1, L + 1)) / np.sum(temp)), axis=0))
            except ValueError:
                cmass = center
            Wpad[:, i, :] = np.roll(np.squeeze(Wpad[:, i, :]), center - cmass, axis=1)
            H[i, :] = np.roll(H[i, :], cmass - center, axis=0)

    return Wpad[:, :, L:-L], H


def compute_loadings_percent_power(V, W, H):
    N, K, L, T = get_shapes(W, H)
    W, H = trim_shapes(W, H, N, K, L, T)

    loadings = np.zeros(K)
    var_v = np.sum(np.power(V, 2))

    for i in np.arange(K):
        WH = reconstruct(np.reshape(W[:, i, :], [W.shape[0], 1, W.shape[2]]),\
                         np.reshape(H[i, :], [1, H.shape[1]]))
        loadings[i] = np.divide(np.sum(np.multiply(2 * V.flatten(), WH.flatten()) - np.power(WH.flatten(), 2)), var_v)

    loadings[loadings < 0] = 0
    return loadings

