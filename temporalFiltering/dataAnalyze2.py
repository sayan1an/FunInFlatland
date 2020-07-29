import numpy as np
import matplotlib.pyplot as plt

data = np.load('OccluderMotion_baseline_t2.npz')

windowSize = data['ws']
sampleCount = data['sc']
n2 = data['n2']
n1 = data['n1']
n0 = data['n0']
p1 = data['p1']
p2 = data['p2']
mc = data['mc']
ref = data['r']
delay_w2 = windowSize // 2
print(sampleCount)

#for t in range(2, 500):
#    print(str(n2[t + 2]) + " " + str(n1[t + 1]) + " " + str(n0[t]) + " " + str(p1[t - 1]) + " " + str(p2[t - 2]))

# for t in range(2, 500):
#     ## https://en.wikipedia.org/wiki/Five-point_stencil
#     ord1 = (-n2[t + 2] + 8 * n1[t + 1] - 8 * p1[t - 1] + p2[t - 2]) / 12.0
#     ord2 = (-n2[t + 2] + 16 * n1[t + 1] -30 * n0[t] + 16 * p1[t - 1] - p2[t - 2]) / 12.0
#     #print(str(n2[t + 2]) + " " + str(n1[t + 1]) + " " + str(n0[t]) + " " + str(p1[t - 1]) + " " + str(p2[t - 2]) )
#     n2p = n0[t] + ord1 * 2
#     print(str(n2[t + 2]) + " " + str(n2p) + " " + str(n2p + ord2 * 2))

def mvAvgFilter(windowSz : int, data):
    window = np.zeros(shape=windowSz)
    filtered = np.zeros(shape=data.shape)

    for i in range(data.shape[0]):
        window[i % windowSz] = data[i]
        filtered[i] = np.sum(window) / windowSz
    
    return filtered

mvAvg = mvAvgFilter(windowSize, (n2 + n1 + n0 + p1 + p2) / 5.0)
mvAvgMc = mvAvgFilter(windowSize, mc)

plt.plot(mvAvg[windowSize:550], label="Coherent samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvg[windowSize:] - ref[windowSize - delay_w2: - delay_w2]), np.mean((mvAvg[windowSize:] - ref[windowSize - delay_w2: - delay_w2])**2)**0.5) )
plt.plot(mvAvgMc[windowSize:550], label="MC samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvgMc[windowSize:] - ref[windowSize - delay_w2: - delay_w2]), np.mean((mvAvgMc[windowSize:] - ref[windowSize - delay_w2: - delay_w2])**2)**0.5))
plt.plot(ref[windowSize - delay_w2: 550 - delay_w2], label="Reference")
plt.xlabel('time')
plt.ylabel('visibility')
plt.legend()
plt.show()

def mvAvgFilterAdv(windowSz : int, N2, N1, N0, P1, P2):
    effWinSize = windowSz + 4
    wn2 = np.zeros(shape=effWinSize)
    wn1 = np.zeros(shape=effWinSize)
    wn0 = np.zeros(shape=effWinSize)
    wp1 = np.zeros(shape=effWinSize)
    wp2 = np.zeros(shape=effWinSize)
       
    filtered = np.zeros(shape=N0.shape)
    predictedError = np.zeros(shape=N0.shape)
    predictedErrorOrd2 = np.zeros(shape=N0.shape)
    
    for i in range(N0.shape[0]):
        wn2[i % effWinSize] = N2[i]
        wn1[i % effWinSize] = N1[i]
        wn0[i % effWinSize] = N0[i]
        wp1[i % effWinSize] = P1[i]
        wp2[i % effWinSize] = P2[i]

        # most stale element
        start = (i + 3) % effWinSize
        accum = wn2[(start + windowSz // 2) % effWinSize] + wn1[(start + windowSz // 2) % effWinSize] + wp1[(start + windowSz // 2) % effWinSize] + wp2[(start + windowSz // 2) % effWinSize]
        correction = 0.0
        correction2 = 0.0
        for j in range(windowSz):
            en2 = wn2[(start + j + 2) % effWinSize]
            en1 = wn1[(start + j + 1) % effWinSize]
            en0 = wn0[(start + j) % effWinSize]
            ep1 = wp1[(start + effWinSize + j - 1) % effWinSize]
            ep2 = wp2[(start + effWinSize + j - 2) % effWinSize]

            weight = 0.5
            #ord1 = -(-en2 + 8 * en1 - 8 * ep1 + ep2) / 12.0
            ord1 = weight * (ep1 - en1) / 2 + (1.0 - weight) * (ep2 - en2) / 4
            #ord2 = (-en2 + 16 * en1 -30 * en0 + 16 * ep1 - ep2) / 12.0
            ord2 = 0 * (en1 + ep1 - 2 * en0) + 1*(en2 + ep2 - en1 - ep1) / 4
            h = np.clip(windowSz // 2 - j, -2.5, 2.5)
            accum += en0 
            correction -= h * ord1 
            correction2 -= ord2 * h * h / 2.0

        filtered[i] = accum / (windowSz + 4)
        predictedError[i] = correction / windowSz
        predictedErrorOrd2[i] = correction2 / windowSz

    return filtered, predictedError, predictedErrorOrd2

mvAvg = mvAvgFilter(windowSize, (n2 + n1 + n0 + p1 + p2) / 5.0 )
delayedMvAvg, prdErr, prdErr2 = mvAvgFilterAdv(windowSize, n2, n1, n0, p1, p2)

#plt.plot(delayedMvAvg[2*windowSize: 500], label="Coherent")
#plt.plot(ref[2*windowSize - delay_w2 - 3: 500- delay_w2 - 3], label='Reference')
#plt.legend()
#plt.show()

actualErrorPlt = ref[2*windowSize -delay_w2 - 3: -delay_w2 - 3] - delayedMvAvg[2*windowSize:]
predictedErrorPlt = prdErr[2*windowSize - 3: -3]

print(np.mean((actualErrorPlt - predictedErrorPlt)**2)**0.5)

plt.plot(predictedErrorPlt[:500], label="Predicted error")
plt.plot(actualErrorPlt[:500], label="Actual error")
plt.xlabel('time')
plt.ylabel('Error')
plt.legend()
plt.title('Order 1 error')
plt.show()

plt.plot(delayedMvAvg[2*windowSize:550] + prdErr[2*windowSize - 3: 550 - 3], label="Coherent samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(delayedMvAvg[2*windowSize:] + prdErr[2*windowSize - 3:-3]- ref[2*windowSize -delay_w2 -3: -delay_w2 - 3]), np.mean((delayedMvAvg[2*windowSize:] + prdErr[2*windowSize - 3:-3]- ref[2*windowSize -delay_w2 -3: -delay_w2 - 3])**2)**0.5) )
plt.plot(mvAvgMc[2*windowSize - 3:550 - 3], label="MC samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvgMc[windowSize:] - ref[windowSize - delay_w2: -delay_w2]), np.mean((mvAvgMc[windowSize:] - ref[windowSize - delay_w2: -delay_w2])**2)**0.5))
plt.plot(ref[2*windowSize - delay_w2 - 3: 550 - delay_w2 - 3], label="Reference")
plt.xlabel('time')
plt.ylabel('Visibility')
plt.legend()
plt.title('Order 1 correction')
plt.show()

# Order 2 error
actualError2 = ref[2*windowSize -delay_w2 - 3: -delay_w2 - 3] - delayedMvAvg[2*windowSize:] - prdErr[2*windowSize - 3: -3]
predictedError2Plt = prdErr[2*windowSize - 5: -5]
# Find best offset
#for i in range(1, windowSize):
#    print(i)
#    print(np.mean((actualError2 - prdErr2[2*windowSize -i: -i])**2)**0.5)
#    print(np.mean((actualError2[:-i] - prdErr2[2*windowSize +i:])**2)**0.5)


plt.plot(predictedError2Plt[2*windowSize-5:500-5], label="Predicted error")
plt.plot(actualError2[:500], label="Actual error")
plt.xlabel('time')
plt.ylabel('Error')
plt.legend()
plt.title('Order 2 error')
plt.show()

plt.plot(delayedMvAvg[2*windowSize:550] + prdErr[2*windowSize - 3: 550 - 3] + prdErr2[2*windowSize - 5: 550 - 5], label="Coherent samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(delayedMvAvg[2*windowSize:] + prdErr[2*windowSize - 3:-3] + prdErr2[2*windowSize - 5:-5] - ref[2*windowSize -delay_w2 -3: -delay_w2 - 3]), np.mean((delayedMvAvg[2*windowSize:] + prdErr[2*windowSize - 3:-3] + prdErr2[2*windowSize - 5:-5] - ref[2*windowSize -delay_w2 -3: -delay_w2 - 3])**2)**0.5) )
plt.plot(mvAvgMc[2*windowSize - 3:550 - 3], label="MC samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvgMc[windowSize:] - ref[windowSize - delay_w2: -delay_w2]), np.mean((mvAvgMc[windowSize:] - ref[windowSize - delay_w2: -delay_w2])**2)**0.5))
plt.plot(ref[2*windowSize - delay_w2 - 3: 550 - delay_w2 - 3], label="Reference")
plt.xlabel('time')
plt.ylabel('Visibility')
plt.legend()
plt.title('Order 2 correction')
plt.show()

# With first order gradient compensation
def mvAvgFilterAdv2(windowSz : int, N2, N1, N0, P1, P2, order):
    effWinSize = windowSz + 4
    wn2 = np.zeros(shape=effWinSize)
    wn1 = np.zeros(shape=effWinSize)
    wn0 = np.zeros(shape=effWinSize)
    wp1 = np.zeros(shape=effWinSize)
    wp2 = np.zeros(shape=effWinSize)
       
    filtered = np.zeros(shape=N0.shape)
    predictedError = np.zeros(shape=N0.shape)
    predictedErrorOrd2 = np.zeros(shape=N0.shape)
    
    for i in range(N0.shape[0]):
        wn2[i % effWinSize] = N2[i]
        wn1[i % effWinSize] = N1[i]
        wn0[i % effWinSize] = N0[i]
        wp1[i % effWinSize] = P1[i]
        wp2[i % effWinSize] = P2[i]

        # most stale element
        start = (i + 3) % effWinSize
        accum = wn2[(start + windowSz // 2) % effWinSize] + wn1[(start + windowSz // 2) % effWinSize] + wp1[(start + windowSz // 2) % effWinSize] + wp2[(start + windowSz // 2) % effWinSize]
        correction = 0.0
        correction2 = 0.0
        for j in range(windowSz):
            en2 = wn2[(start + j + 2) % effWinSize]
            en1 = wn1[(start + j + 1) % effWinSize]
            en0 = wn0[(start + j) % effWinSize]
            ep1 = wp1[(start + effWinSize + j - 1) % effWinSize]
            ep2 = wp2[(start + effWinSize + j - 2) % effWinSize]

            weight = 0.5
            #ord1 = -(-en2 + 8 * en1 - 8 * ep1 + ep2) / 12.0
            ord1 = weight * (ep1 - en1) / 2 + (1.0 - weight) * (ep2 - en2) / 4
            #ord2 = -(-en2 + 16 * en1 -30 * en0 + 16 * ep1 - ep2) / 12.0
            ord2 = 0 * (en1 + ep1 - 2 * en0) + 1*(en2 + ep2 - en1 - ep1) / 4
            h = np.clip(windowSz // 2 - j, -2.5, 2.5)
            accum += en0 
            correction += h * ord1 
            correction2 -= ord2 * h * h / 2.0
        
        predictedError[i] = -correction / windowSz
        predictedErrorOrd2[i] = correction2 / windowSz
        filtered[i] = accum / (windowSz + 4)

        if i > 3 and order > 0:
            filtered[i] += predictedError[i - 3]
        if i > 5 and order > 1:
            filtered[i] += predictedErrorOrd2[i - 5]

    return filtered, predictedError, predictedErrorOrd2

mvAvg,_,_ =  mvAvgFilterAdv2(windowSize, n2, n1, n0, p1, p2, 1)
plt.plot(mvAvg[2*windowSize:550], label="Coherent samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvg[2*windowSize:] - ref[2*windowSize-delay_w2 -3: -delay_w2 - 3]), np.mean((mvAvg[2*windowSize:] - ref[2*windowSize-delay_w2 -3: -delay_w2 - 3])**2)**0.5) )
plt.plot(mvAvgMc[2*windowSize - 3:550 - 3], label="MC samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvgMc[2*windowSize:] - ref[2*windowSize - delay_w2: -delay_w2]), np.mean((mvAvgMc[2*windowSize:] - ref[2*windowSize - delay_w2: -delay_w2])**2)**0.5))
plt.plot(ref[2*windowSize - delay_w2 - 3: 550 - delay_w2 - 3], label="Reference")
plt.xlabel('time')
plt.ylabel('Visibility')
plt.title('Order 1 correction')
plt.legend()
plt.show()


mvAvg,_,_ =  mvAvgFilterAdv2(windowSize, n2, n1, n0, p1, p2, 2)
plt.plot(mvAvg[2*windowSize:550], label="Coherent samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvg[2*windowSize:] - ref[2*windowSize-delay_w2 -3: -delay_w2 - 3]), np.mean((mvAvg[2*windowSize:] - ref[2*windowSize-delay_w2 -3: -delay_w2 - 3])**2)**0.5) )
plt.plot(mvAvgMc[2*windowSize - 3:550 - 3], label="MC samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvgMc[2*windowSize:] - ref[2*windowSize - delay_w2: -delay_w2]), np.mean((mvAvgMc[2*windowSize:] - ref[2*windowSize - delay_w2: -delay_w2])**2)**0.5))
plt.plot(ref[2*windowSize - delay_w2 - 3: 550 - delay_w2 - 3], label="Reference")
plt.xlabel('time')
plt.ylabel('Visibility')
plt.title('Order 2 correction')
plt.legend()
plt.show()
