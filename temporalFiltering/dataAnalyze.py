import numpy as np
import matplotlib.pyplot as plt

data = np.load('OccluderMotion.npz')

windowSize = data['ws']
movingData = data['c']
sampleCountCurrentFrame = data['sc']
staticData = data['g']
sampleCountGradient = data['sg']
mcData = data['mc']
referenceData = data['r']

print(windowSize)

plt.plot(movingData[0:550], label="Moving samples, error:" + str(np.std(movingData - referenceData)))
plt.plot(staticData[0:550], label='Static samples, error:' + str(np.std(staticData - referenceData))) 
plt.plot(mcData[0:550], label='MonteCarlo, error:' + str(np.std(mcData - referenceData)))
plt.plot(referenceData[0:550], label='Reference')
plt.legend()
plt.show()

def mvAvgFilter(windowSz : int, data):
    window = np.zeros(shape=windowSz)
    filtered = np.zeros(shape=data.shape)

    for i in range(data.shape[0]):
        window[i % windowSz] = data[i]
        filtered[i] = np.sum(window) / windowSz
    
    return filtered

filteredMovingData = mvAvgFilter(windowSize, movingData)
filteredStaticData = mvAvgFilter(windowSize, staticData)
filteredMcData = mvAvgFilter(windowSize, mcData)

plt.plot(filteredMovingData[windowSize:550], label='filtered moving samples, error:' + str(np.std(filteredMovingData[windowSize:] - referenceData[windowSize:])))
plt.plot(filteredStaticData[windowSize:550], label='static samples, error:' + str(np.std(filteredStaticData[windowSize:] - referenceData[windowSize:])))
plt.plot(filteredMcData[windowSize:550], label='filtered MC samples, error:' + str(np.std(filteredMcData[windowSize:] - referenceData[windowSize:])))
plt.plot(referenceData[windowSize:550], label='Reference')
plt.legend()
plt.show()

filteredCurrentAndStaticData = (filteredMovingData + staticData) / 2.0

plt.plot(filteredCurrentAndStaticData[windowSize:550], label='filtered moving samples + static samples, error:' + str(np.std(filteredCurrentAndStaticData[windowSize:] - referenceData[windowSize:])))
plt.plot(filteredMcData[windowSize:550], label='filtered MC samples, error:' + str(np.std(filteredMcData[windowSize:] - referenceData[windowSize:])))
plt.plot(referenceData[windowSize:550], label='Reference')
plt.legend()
plt.show()

def mvAvgFilterAdv(windowSz : int, data, gradData):
    windowData = np.zeros(shape=windowSz)
    windowGrad = np.zeros(shape=windowSz)
    filtered = np.zeros(shape=data.shape)
    mid = windowSz // 2

    for i in range(data.shape[0]):
        windowData[i % windowSz] = data[i]
        windowGrad[i % windowSz] = gradData[i]
        midElement = windowGrad[(i + mid + 1) % windowSz]
        correction = (2 * mid + 1) * midElement - np.sum(windowGrad)
        filtered[i] = (np.sum(windowData) + correction) / windowSz
    
    return filtered

filteredMovingDataWithGrad = mvAvgFilterAdv(windowSize, movingData, referenceData)
plt.plot(filteredMovingDataWithGrad[windowSize:550], label="filtered with gradient, error:" + str(np.std(filteredMovingDataWithGrad[windowSize:] - referenceData[windowSize:])))
plt.plot(filteredMovingData[windowSize:550], label='filtered moving samples, error:' + str(np.std(filteredMovingData[windowSize:] - referenceData[windowSize:])))
plt.plot(referenceData[windowSize:550], label='Reference')
plt.legend()
plt.show()
