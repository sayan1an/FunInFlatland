import matplotlib.pyplot as plt
import numpy as np

sampleCount = np.array([10, 20, 30, 40, 80, 160])
regularFixedWindow = np.array([(0.0126 + 0.0121 + 0.0117)/3.0, (0.00731 + 0.00844) / 2.0, (0.00646 + 0.00564) / 2.0, (0.00708 + 0.00719 + 0.00744) / 3.0, (0.00404 + 0.00547) / 2.0, (0.00226 + 0.00315) / 2.0])
mcFixedWindow = np.array([(0.0257 + 0.0352 + 0.0414) / 3.0, (0.0223 + 0.0260 + 0.0248) / 3.0, (0.0215 + 0.0252 + 0.0210) / 3.0, (0.0167 + 0.0151)/2.0, (0.0122 + 0.0134) / 2.0, (0.0122 + 0.00698 + 0.0107) / 3.0])
plt.plot(sampleCount, regularFixedWindow / 0.181818 * 100, label="Regular")
plt.plot(sampleCount, mcFixedWindow / 0.181818 * 100, label="MC")
plt.xlabel("Sample count")
plt.ylabel("% error")
plt.legend()
plt.show()

windowSize = np.array([10, 20, 40])
regularFixedSample = np.array([(0.00652 + 0.00926) / 2, (0.00364 + 0.00485) / 2.0, (0.00294 + 0.00308) / 2.0])
mcFixedSample = np.array([(0.0223 + 0.0260 + 0.0248) / 3.0, (0.0216 + 0.0212) / 2.0, (0.0131 + 0.0151) / 2.0])
plt.plot(windowSize, regularFixedSample / 0.181818 * 100, label="Regular")
plt.plot(windowSize, mcFixedSample / 0.181818 * 100, label="MC")
plt.xlabel("Window size")
plt.ylabel("% error")
plt.legend()
plt.show()
