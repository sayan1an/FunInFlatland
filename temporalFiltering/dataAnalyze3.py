import numpy as np
import matplotlib.pyplot as plt

data = np.load('OccluderMotion.npz')

windowSize = data['ws']
sampleCount = data['sc']

n0 = data['n0']
mc = data['mc']
ref = data['r']

print(sampleCount)

def mvAvgFilter(windowSz : int, data):
    window = np.zeros(shape=windowSz)
    filtered = np.zeros(shape=data.shape)

    for i in range(data.shape[0]):
        window[i % windowSz] = data[i]
        filtered[i] = np.sum(window) / windowSz
    
    return filtered

mvAvg = mvAvgFilter(windowSize, n0)
mvAvgMc = mvAvgFilter(windowSize, mc)

delay = - windowSize // 2
plt.plot(mvAvg[windowSize:550], label="Coherent samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvg[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1]), np.mean((mvAvg[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1])**2)**0.5) )
plt.plot(mvAvgMc[windowSize:550], label="MC samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvgMc[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1]), np.mean((mvAvgMc[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1])**2)**0.5))
plt.plot(ref[windowSize - windowSize // 2: 550 - windowSize // 2 + 1], label="Reference")
plt.legend()
plt.show()
