import numpy as np
import matplotlib.pyplot as plt

data = np.load('OccluderMotion_baseline_t3.npz')

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

plt.plot(mvAvg[windowSize:550], label="Coherent samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvg[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1]), np.mean((mvAvg[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1])**2)**0.5) )
plt.plot(mvAvgMc[windowSize:550], label="MC samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvgMc[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1]), np.mean((mvAvgMc[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1])**2)**0.5))
plt.plot(ref[windowSize - windowSize // 2: 550 - windowSize // 2 + 1], label="Reference")
plt.xlabel('time')
plt.ylabel('visibility')
plt.legend()
plt.show()

def mvAvgFilterAdv(windowSz : int, data):
    window = np.zeros(shape=windowSz)
    windowOld = np.zeros(shape=windowSz)
    filtered = np.zeros(shape=data.shape)
    predictErr = np.zeros(shape=data.shape)
    for i in range(data.shape[0]):
        windowOld[i % windowSz] = window[i % windowSz]
        window[i % windowSz] = data[i]
        
        correction = 0
        for j in range(windowSz):
            grad = (window[ (i + j + 1) % windowSz] - windowOld[ (i + j + 1) % windowSz]) / windowSz
            correction += grad * np.clip(windowSz // 2 - j, -100, 100)
        
        predictErr[i] = correction / windowSz
        filtered[i] = (np.sum(window)) / windowSz

    return filtered, predictErr


mvAvg, predictedError = mvAvgFilterAdv(windowSize, n0)
mvAvgMc = mvAvgFilter(windowSize, mc)
predictedErrorPlt = predictedError[2*windowSize + windowSize // 2:]
actualErrorPlt = ref[2*windowSize - windowSize // 2: - windowSize // 2 + 1]- mvAvg[2*windowSize:]#ref[2*windowSize:]

trim = 2*windowSize
print(np.mean(((actualErrorPlt[:- windowSize // 2 + 1] - predictedErrorPlt))[:-trim]**2)**0.5)

plt.plot(predictedErrorPlt[:500], label="Predicted error")
plt.plot(actualErrorPlt[:500], label="Actual error")
plt.xlabel('time')
plt.ylabel('Error')
plt.legend()
plt.show()

plt.plot(mvAvg[2*windowSize:550] + predictedError[2*windowSize + windowSize // 2: 550 + windowSize // 2], label="Coherent samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvg[2*windowSize:-windowSize // 2 + 1] + predictedError[2*windowSize + windowSize // 2:]- ref[2*windowSize - windowSize // 2: - windowSize // 2 + 1][:- windowSize // 2 + 1]), np.mean((mvAvg[2*windowSize:-windowSize // 2 + 1] + predictedError[2*windowSize + windowSize // 2:]- ref[2*windowSize - windowSize // 2: - windowSize // 2 + 1][:- windowSize // 2 + 1])**2)**0.5) )
plt.plot(mvAvgMc[2*windowSize:550], label="MC samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvgMc[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1]), np.mean((mvAvgMc[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1])**2)**0.5))
plt.plot(ref[2*windowSize - windowSize // 2: 550 - windowSize // 2 + 1], label="Reference")
plt.xlabel('time')
plt.ylabel('Visibility')
plt.legend()
plt.show()

def mvAvgFilterAdv2(windowSz : int, data):
    window = np.zeros(shape=windowSz)
    windowOld = np.zeros(shape=windowSz)
    filtered = np.zeros(shape=data.shape)
  
    for i in range(data.shape[0]):
        windowOld[i % windowSz] = window[i % windowSz]
        window[i % windowSz] = data[i]
        
        correction = 0
        for j in range(windowSz):
            grad = (window[ (i + j + 1) % windowSz] - windowOld[ (i + j + 1) % windowSz]) / windowSz
            correction += grad * np.clip(windowSz // 2 - j, -100, 100)
        
        filtered[i] = np.sum(window) / windowSz

        if i >= windowSz // 2:
            filtered[i - (windowSz // 2)] += correction / windowSz

    return filtered

#A more efficeint version of mvAvgFilterAdv2
#Does not contain the inner for loop over gradients
def mvAvgFilterAdv2Eff(windowSz : int, data):
    window = np.zeros(shape=windowSz)
    windowGrad = np.zeros(shape=windowSz)
    windowFilteredNoGrad = np.zeros(shape=windowSz//2)
    filtered = np.zeros(shape=data.shape)
        
    gradSum = 0
    correctionLast = 0
    for i in range(data.shape[0]):
        inNew = data[i]
        gradOld = windowGrad[i % windowSz]
        oldFiltered = windowFilteredNoGrad[i%(windowSz//2)]
        gradNew = (inNew - window[i % windowSz]) / windowSz
        windowGrad[i % windowSz] = gradNew
        window[i % windowSz] = inNew
        
        correction = correctionLast - ((windowSz // 2) + 1) * gradOld - (windowSz // 2) * gradNew + gradSum
        windowFilteredNoGrad[i%(windowSz//2)] = windowFilteredNoGrad[(i + (windowSz//2) - 1) % (windowSz//2)] + gradNew
        correctionLast = correction
            
        gradSum += gradNew - gradOld
        if i >= windowSz // 2:
            filtered[i - windowSz // 2] = oldFiltered + correction / windowSz
           
    return filtered

mvAvg = mvAvgFilterAdv2(windowSize, n0)
mvAvg2 = mvAvgFilterAdv2Eff(windowSize, n0)

print("Efficient gradient error:")
print(np.sqrt(np.mean((mvAvg - mvAvg2)[2*windowSize:-trim]**2)))

plt.plot(mvAvg2[windowSize:550], label="Coherent samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std((mvAvg2[windowSize:] - ref[windowSize - windowSize // 2:- windowSize // 2 + 1])[:-trim]), np.mean((mvAvg2[windowSize:] - ref[windowSize - windowSize // 2:- windowSize // 2 + 1])[:-trim]**2)**0.5) )
plt.plot(mvAvgMc[windowSize:550], label="MC samples, variance:{0:0.6f}, MSE:{1:1.6f}".format(np.std(mvAvgMc[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1]), np.mean((mvAvgMc[windowSize:] - ref[windowSize - windowSize // 2: - windowSize // 2 + 1])**2)**0.5))
plt.plot(ref[windowSize - windowSize // 2: 550 - windowSize // 2 + 1], label="Reference")
plt.xlabel('time')
plt.ylabel('Visibility')
plt.legend()
plt.show()