import numpy as np
import matplotlib.pyplot as plt

data = np.load('OccluderMotion.npz')

windowSize = data['ws']
sampleCount = data['sc']
n2 = data['n2']
n1 = data['n1']
n0 = data['n0']
p1 = data['p1']
p2 = data['p2']
mc = data['mc']
ref = data['r']

print(sampleCount)

for t in range(2, 500):
    ## https://en.wikipedia.org/wiki/Five-point_stencil
    ord1 = (-n2[t + 2] + 8 * n1[t + 1] - 8 * p1[t - 1] + p2[t - 2]) / 12.0
    ord2 = (-n2[t + 2] + 16 * n1[t + 1] -30 * n0[t] + 16 * p1[t - 1] - p2[t - 2]) / 12.0
    #print(str(n2[t + 2]) + " " + str(n1[t + 1]) + " " + str(n0[t]) + " " + str(p1[t - 1]) + " " + str(p2[t - 2]) )
    n2p = n0[t] + ord1 * 2
    print(str(n2[t + 2]) + " " + str(n2p) + " " + str(n2p + ord2 * 2))

def mvAvgFilter(windowSz : int, data):
    window = np.zeros(shape=windowSz)
    filtered = np.zeros(shape=data.shape)

    for i in range(data.shape[0]):
        window[i % windowSz] = data[i]
        filtered[i] = np.sum(window) / windowSz
    
    return filtered

def mvAvgFilterAdv(windowSz : int, N2, N1, N0, P1, P2):
    effWinSize = windowSz + 4
    wn2 = np.zeros(shape=effWinSize)
    wn1 = np.zeros(shape=effWinSize)
    wn0 = np.zeros(shape=effWinSize)
    wp1 = np.zeros(shape=effWinSize)
    wp2 = np.zeros(shape=effWinSize)
       
    filtered = np.zeros(shape=N0.shape)
    
    for i in range(N0.shape[0]):
        wn2[i % effWinSize] = N2[i]
        wn1[i % effWinSize] = N1[i]
        wn0[i % effWinSize] = N0[i]
        wp1[i % effWinSize] = P1[i]
        wp2[i % effWinSize] = P2[i]

        # most stale element
        start = (i + 3) % effWinSize
        accum = 0.0
        for j in range(windowSz):
            en2 = wn2[(start + j + 2) % effWinSize]
            en1 = wn1[(start + j + 1) % effWinSize]
            en0 = wn0[(start + j) % effWinSize]
            ep1 = wp1[(start + effWinSize + j - 1) % effWinSize]
            ep2 = wp2[(start + effWinSize + j - 2) % effWinSize]

            ord1 = (-en2 + 8 * en1 - 8 * ep1 + ep2) / 12.0
            ord2 = (-en2 + 16 * en1 -30 * en0 + 16 * ep1 - ep2) / 12.0
            h = windowSz // 2 - j
            accum = accum  + en0 + h * ord1 + ord2 * h * h / 2.0

        filtered[i] = accum / windowSz 
    
    return filtered

#plt.plot(mvAvgFilter(windowSize,n2)[windowSize:550])
#plt.plot(mvAvgFilter(windowSize,n1)[windowSize:550])
#plt.plot(mvAvgFilter(windowSize,n0)[windowSize:550])
#plt.plot(mvAvgFilter(windowSize,p1)[windowSize:550])
#plt.plot(mvAvgFilter(windowSize,p2)[windowSize:550])

mvAvg = mvAvgFilter(windowSize, n0)
delayedMvAvg = mvAvgFilterAdv(windowSize, n2, n1, n0, p1, p2)

plt.plot(mvAvg[13:550])
plt.plot(delayedMvAvg[13:550])
plt.plot(ref[13:550])
#plt.plot(ref[windowSize - windowSize // 2:550 -  windowSize // 2])
plt.show()

print(str(np.std(mvAvgFilter(windowSize,n0)[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1])))
print(str(np.std(mvAvgFilter(windowSize,mc)[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1])))