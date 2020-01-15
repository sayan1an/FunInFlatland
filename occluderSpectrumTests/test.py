import sys
sys.path.append('../')

from fifl import *
import matplotlib.pyplot as plt
from PIL import Image

testname = "OccluderSpectrumTest"

#screenWidth, screenHeight, center horizontalShift, center vertical shift
screenSetup(800, 800, 0, 0, testname)

def saveImage(name, primalImage):
    formatted = (primalImage * 255 / np.max(primalImage)).astype('uint8')
    img = Image.fromarray(np.flip(formatted, 0))
    img.show()
    img.save(name)

def getSpectrum(primalImage):
    ftimage = np.fft.fft2(primalImage)
    ftimage = np.fft.fftshift(ftimage)
    
    return np.abs(ftimage)
    
def getCovarianceMat(imageMat):
    ySize = imageMat.shape[0]
    xSize = imageMat.shape[1]
    
    # construct A, I
    A = np.zeros((xSize * ySize, 3))
    I = np.zeros((xSize * ySize, 1))
    mu = np.array([xSize / 2.0, ySize / 2.0]).reshape((2,1))

    # Initialize
    for x in range(xSize):
        for y in range(ySize):
            if imageMat[y,x] > 1e-6:
                I[y * xSize + x] = np.log((0.0 + imageMat[y,x]) * 2 * np.pi)
                A[y * xSize + x, 0] = 0.5* (x - mu[0])**2
                A[y * xSize + x, 1] = (x - mu[0]) * (y - mu[1])
                A[y * xSize + x, 2] = 0.5 * (y - mu[1])**2

    A_T = np.transpose(A)
    A_T_A_inv = np.linalg.inv(np.dot(A_T, A)) + np.eye(3) * 0.0001
    det_sigma_inv = 1

    # Iterative solve
    for i in range(2):
        # Add the correction factor
        I_iter = I - 0.5 * np.log(det_sigma_inv)
        A_T_I = np.dot(A_T, I_iter)
        s = -np.dot(A_T_A_inv, A_T_I)
        # Update determinant
        det_sigma_inv = s[0] * s[2] - s[1] * s[1]
        print(det_sigma_inv)
  
    cov_inv = np.zeros((2,2))
    cov_inv[0, 0] = s[0]
    cov_inv[1, 1] = s[2]
    cov_inv[0, 1] = cov_inv[1, 0] = s[1]

    return np.linalg.inv(cov_inv)

def getCovarianceMatEmperical(imageMat):
    ySize = imageMat.shape[0]
    xSize = imageMat.shape[1]
        
    mu = np.array([xSize / 2.0, ySize / 2.0]).reshape((2,1))
    powerAvailable = np.sum(imageMat**2)

    sigmaSq_x = 0
    sigmaSq_y = 0
    sigmaSq_xy = 0
    total = 0
    powerUsed = 0
    for x in range(xSize):
        for y in range(ySize):
            if imageMat[y, x] > 0.025:
                total = total +  imageMat[y, x]
                powerUsed = powerUsed + imageMat[y, x]**2
                sigmaSq_x = sigmaSq_x + imageMat[y, x] * (x - mu[0])**2
                sigmaSq_y = sigmaSq_y + imageMat[y, x] * (y - mu[1])**2
                sigmaSq_xy = sigmaSq_xy + imageMat[y, x] * (x - mu[0]) * (y - mu[1])

    sigmaSq_x = sigmaSq_x / total
    sigmaSq_y = sigmaSq_y / total
    sigmaSq_xy = sigmaSq_xy / total

    print(powerUsed/powerAvailable)
    covMat = np.array([sigmaSq_x, sigmaSq_xy, sigmaSq_xy, sigmaSq_y]).reshape((2,2))

    print(np.linalg.eig(covMat))
    return np.array([sigmaSq_x, sigmaSq_xy, sigmaSq_xy, sigmaSq_y]).reshape((2,2))
    

def generateRays(light, receiver):
    rayEndPoints = light.sample(np.arange(0.0001, 1, 1.0 / float(emitterLength * emitterDensity)))
    rayOrigins = receiver.sample(np.arange(0.0001, 1.0, 1.0 / float(receiverLength * receiverDensity)))
   
    rays = []
    i = 0
    for origin in rayOrigins:
        j = 0
        for endPoint in rayEndPoints:
            direction = Vector(endPoint.pos[0] - origin.pos[0], endPoint.pos[1] - origin.pos[1])
            direction.normalize()
            rays.append([i, j, Ray(origin, direction)])
            j = j + 1
        i = i + 1
    return rays, len(rayOrigins)

def primaryToSecondaryVar(emitterPosition, emitterOrientation, emitterLength, emitterDensity, reciverPosition, receiverLength, receiverDensity, occluderPosition, occluderHScale, occluderVScale):
    eh = np.abs(emiiterPosition.pos[0] - receiverPosition.pos[0])
    ev = np.abs(emiiterPosition.pos[1] - receiverPosition.pos[1])
    eTheta = emitterOrientation * np.pi / 180.0
    oh = np.abs(occluderPosition.pos[0] - receiverPosition.pos[0])
    ov = np.abs(occluderPosition.pos[1] - receiverPosition.pos[1])
    oh_min = oh - occluderHScale * 50.0 / 2.0
    oh_max = oh + occluderHScale * 50.0 / 2.0
    ov_min = ov - occluderVScale * 50.0 / 2.0
    ov_max = ov + occluderVScale * 50.0 / 2.0
    return (emitterLength, eh, ev, eTheta, oh_min, oh_max, ov_min, ov_max)


def primalVisibility(emitterPosition, emitterOrientation, emitterLength, emitterDensity, reciverPosition, receiverLength, receiverDensity, occluderPosition, occluderHScale, occluderVScale):
    sceneName = testname
    scene = Scene(sceneName)

    # Light source
    light = Light("line", emitterPosition, emitterLength, emitterOrientation)
    scene.append(light)
    #drawText("Emitter", -390, 0, "Black", 15)

    # Receiver
    receiver = Line(Point(receiverPosition.pos[0] - receiverLength / 2.0, receiverPosition.pos[1]), Point(receiverPosition.pos[0] + receiverLength / 2.0, receiverPosition.pos[1]), material=Material(0.0))
    scene.append(receiver)
    #drawText("Receiver plane", 0, -380, "Black", 15)
    
    # Occluder
    scene.append(Box(position=occluderPosition, hScale=occluderHScale, vScale=occluderVScale, orientation=0))
    #drawText("Occluder", -100, 10, "Black", 15)

    (eL, eh, ev, eTheta, oh_min, oh_max, ov_min, ov_max) = primaryToSecondaryVar(emitterPosition, emitterOrientation, emitterLength, emitterDensity, reciverPosition, receiverLength, receiverDensity, occluderPosition, occluderHScale, occluderVScale)
    drawText("eL:{:0.1f}".format(eL), 310, 350, "Black", 15)
    drawText("eh:{:0.1f}".format(eh), 310, 330, "Black", 15)
    drawText("ev:{:0.1f}".format(ev), 310, 310, "Black", 15)
    drawText("eTheta:{:0.2f}".format(eTheta), 310, 290, "Black", 15)
    drawText("oh_min:{:0.1f}".format(oh_min), 310, 270, "Black", 15)
    drawText("oh_max:{:0.1f}".format(oh_max), 310, 250, "Black", 15)
    drawText("ov_min:{:0.1f}".format(ov_min), 310, 230, "Black", 15)
    drawText("ov_max:{:0.1f}".format(ov_max), 310, 210, "Black", 15)

    scene.draw()

    rays, reciverSamples = generateRays(light, receiver)
  
    primalVisibilityData = np.zeros((len(rays), 3), dtype=int)
    ctr = 0
    for ray in rays:
        (i,o) = scene.intersect(ray[2])
        primalVisibilityData[ctr, 0] = ray[0]
        primalVisibilityData[ctr, 1] = ray[1]
        if i.hit and isinstance(o, Light):
            primalVisibilityData[ctr, 2] = 1
        elif i.hit:
            primalVisibilityData[ctr, 2] = 0
        ctr = ctr + 1
        #ray[2].draw()
    
    return primalVisibilityData, reciverSamples

folder = "results/"
emitterLength = 1000
receiverLength = 1000
emitterDensity = 0.05
receiverDensity = 0.05
emitterOrientation = 90.0
emiiterPosition = Point(-350, -350 + emitterLength / 2.0)
receiverPosition = Point(-350 + receiverLength / 2.0, -350)
occluderPosition = Point(-100, -100)
occluderHScale = 4.0
occluderVScale = 4.0

for i in range(0,1):
    emiiterPosition = Point(emiiterPosition.pos[0] + 100, emiiterPosition.pos[1])
    tl.clearscreen()
    primalVisibilityData, nReceiverSamples = primalVisibility(emiiterPosition, emitterOrientation, emitterLength, 
        emitterDensity, receiverPosition, receiverLength, receiverDensity, occluderPosition, occluderHScale, occluderVScale)
    
    screenshot(folder + "scene_" + str(i))
    
    nEmitterSamples = int(primalVisibilityData.shape[0] / nReceiverSamples)
    primalImage = np.zeros((nEmitterSamples, nReceiverSamples)) # rows == y-axis, cols = = x-axis

    for index in range(primalVisibilityData.shape[0]):
        x = primalVisibilityData[index, 0]
        y = primalVisibilityData[index, 1]
        if (primalVisibilityData[index, 2] > 0):
            plt.scatter(x, y, color='red')
            primalImage[y, x] = 1.0
        else:
            plt.scatter(x, y, color='blue')
            primalImage[y, x] = 0.0
    
    plt.savefig(folder  + "primal_" + str(i) + ".png")
    saveImage(folder + "primal_pil_" + str(i) + ".bmp", primalImage)
    spectrum = getSpectrum(primalImage)
    spectrum_modified = spectrum / np.max(spectrum)
    saveImage(folder + "spectral_pil_" + str(i) + ".bmp", spectrum_modified)
    saveImage(folder + "spectral_pil_log_" + str(i) + ".bmp", np.log(1 + spectrum_modified))
    print(getCovarianceMatEmperical(spectrum_modified))


print("Finished")
#tl.done()