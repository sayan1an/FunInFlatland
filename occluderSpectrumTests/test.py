import sys
sys.path.append('../')

from fifl import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2
import glob
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

def getCovarianceMatEmperical(in_imageMat):
    ySize = in_imageMat.shape[0]
    xSize = in_imageMat.shape[1]

    imageMat = in_imageMat / np.max(in_imageMat)

    mu = np.array([xSize / 2.0, ySize / 2.0]).reshape((2,1))
    powerAvailable = np.sum(imageMat**2)

    sigmaSq_x = 0
    sigmaSq_y = 0
    sigmaSq_xy = 0
    searchStep =  0.005

    for level in np.arange(0.005, 0.1, searchStep):
        clippedImageMat = (imageMat > level).astype(int) * imageMat
        powerUsed = np.sum(clippedImageMat**2)
      
        if powerUsed / powerAvailable < 0.95:
            newLevel = level - searchStep / 2.0
            clippedImageMat = (imageMat > newLevel).astype(int) * imageMat
            print ("Retaining {:0.3f} % of power".format(100*np.sum(clippedImageMat**2)/powerAvailable))
            total = np.sum(clippedImageMat)
            for x in range(xSize):
                for y in range(ySize):
                    sigmaSq_x = sigmaSq_x + clippedImageMat[y, x] * (x - mu[0])**2
                    sigmaSq_y = sigmaSq_y + clippedImageMat[y, x] * (y - mu[1])**2
                    sigmaSq_xy = sigmaSq_xy + clippedImageMat[y, x] * (x - mu[0]) * (y - mu[1])
            
            sigmaSq_x = sigmaSq_x / total
            sigmaSq_y = sigmaSq_y / total
            sigmaSq_xy = sigmaSq_xy / total

            break

       
    covMat = np.array([sigmaSq_x, sigmaSq_xy, sigmaSq_xy, sigmaSq_y]).reshape((2,2))
    eigVal, eigVec = np.linalg.eig(covMat)

    firstPrincipalAxis = 0
    secondPrincipalAxis = 1
    if eigVal[1] > eigVal[0]:
        firstPrincipalAxis = 1
        secondPrincipalAxis = 0

    stdMajor = np.sqrt(eigVal[firstPrincipalAxis])
    stdMinor = np.sqrt(eigVal[secondPrincipalAxis])

    eigVecMajor = eigVec[:, firstPrincipalAxis]
    

    return stdMajor, stdMinor, np.arccos(eigVecMajor[0]) * 180 / np.pi
    

def generateRays(light, receiver, emitterLength, emitterDensity, receiverLength, receiverDensity):
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

def primaryToSecondaryVar(emitterPosition, emitterOrientation, emitterLength, emitterDensity, receiverPosition, receiverLength, receiverDensity, occluderPosition, occluderHScale, occluderVScale):
    eh = np.abs(emitterPosition.pos[0] - receiverPosition.pos[0])
    ev = np.abs(emitterPosition.pos[1] - receiverPosition.pos[1])
    eTheta = emitterOrientation * np.pi / 180.0
    oh = np.abs(occluderPosition.pos[0] - receiverPosition.pos[0])
    ov = np.abs(occluderPosition.pos[1] - receiverPosition.pos[1])
    oh_min = oh - occluderHScale * 50.0 / 2.0
    oh_max = oh + occluderHScale * 50.0 / 2.0
    ov_min = ov - occluderVScale * 50.0 / 2.0
    ov_max = ov + occluderVScale * 50.0 / 2.0
    return (emitterLength, eh, ev, eTheta, oh_min, oh_max, ov_min, ov_max)

def primalVisibility(emitterPosition, emitterOrientation, emitterLength, emitterDensity, receiverPosition, receiverLength, receiverDensity, occluderPosition, occluderHScale, occluderVScale):
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

    (eL, eh, ev, eTheta, oh_min, oh_max, ov_min, ov_max) = primaryToSecondaryVar(emitterPosition, emitterOrientation, emitterLength, emitterDensity, receiverPosition, receiverLength, receiverDensity, occluderPosition, occluderHScale, occluderVScale)
    drawText("eL:{:0.1f}".format(eL), 310, 350, "Black", 15)
    drawText("eh:{:0.1f}".format(eh), 310, 330, "Black", 15)
    drawText("ev:{:0.1f}".format(ev), 310, 310, "Black", 15)
    drawText("eTheta:{:0.2f}".format(eTheta), 310, 290, "Black", 15)
    drawText("oh_min:{:0.1f}".format(oh_min), 310, 270, "Black", 15)
    drawText("oh_max:{:0.1f}".format(oh_max), 310, 250, "Black", 15)
    drawText("ov_min:{:0.1f}".format(ov_min), 310, 230, "Black", 15)
    drawText("ov_max:{:0.1f}".format(ov_max), 310, 210, "Black", 15)

    scene.draw()

    rays, reciverSamples = generateRays(light, receiver, emitterLength, emitterDensity, receiverLength, receiverDensity)
  
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

def runExperiment(experimentSeriesName, experimentIndex, emiiterPosition, emitterOrientation, emitterLength, emitterDensity, receiverPosition, receiverLength, receiverDensity, occluderPosition, occluderHScale, occluderVScale):
    tl.clearscreen()
    primalVisibilityData, nReceiverSamples = primalVisibility(emiiterPosition, emitterOrientation, emitterLength, 
        emitterDensity, receiverPosition, receiverLength, receiverDensity, occluderPosition, occluderHScale, occluderVScale)
    
    # Save the screen setup
    screenshot(experimentSeriesName + "_scene_" + str(experimentIndex))
    
    nEmitterSamples = int(primalVisibilityData.shape[0] / nReceiverSamples)
    primalImage = np.zeros((nEmitterSamples, nReceiverSamples)) # rows == y-axis, cols = = x-axis

    for index in range(primalVisibilityData.shape[0]):
        x = primalVisibilityData[index, 0]
        y = primalVisibilityData[index, 1]
        if (primalVisibilityData[index, 2] > 0):
            #plt.scatter(x, y, color='red')
            primalImage[y, x] = 1.0
        else:
            #plt.scatter(x, y, color='blue')
            primalImage[y, x] = 0.0
    
    # Save primal image of occluder
    plt.xlabel("Receiver(x)")
    plt.ylabel("Emitter(y)")
    plt.imshow(primalImage, cmap='hot', interpolation='nearest', origin='lower')
    plt.savefig(experimentSeriesName + "_primal_" + str(experimentIndex) + ".png")
    plt.close()

    # Save spectral image of occluder
    spectrum = getSpectrum(primalImage)
    plt.xlabel("Receiver($\Omega_x$)")
    plt.ylabel("Emitter($\Omega_y$)")
    plt.imshow(spectrum, cmap='hot', interpolation='nearest', origin='lower')
    stdMajor, stdMinor, angle = getCovarianceMatEmperical(spectrum)
    ax = plt.gca()
    e1 = Ellipse((spectrum.shape[0]/2.0, spectrum.shape[1]/2.0), width=2*stdMajor, height=2*stdMinor, angle=angle, edgecolor='green', facecolor='green', linewidth=1)
    e1.set_alpha(0.4)
    ax.add_patch(e1)
    e2 = Ellipse((spectrum.shape[0]/2.0, spectrum.shape[1]/2.0), width=6*stdMajor, height=6*stdMinor, angle=angle, edgecolor='yellow', facecolor='yellow', linewidth=1)
    e2.set_alpha(0.15)
    ax.add_patch(e2)
    plt.text(spectrum.shape[0] - 20, spectrum.shape[1] - 3, "Std Major : {:0.3f}".format(stdMajor), color='white')
    plt.text(spectrum.shape[0] - 20, spectrum.shape[1] - 6, "Std Minor : {:0.3f}".format(stdMinor), color='white')
    plt.text(spectrum.shape[0] - 20, spectrum.shape[1] - 9, "Angle (deg) : {:0.1f}".format(angle), color='white')
    plt.savefig(experimentSeriesName + "_spectral_" + str(experimentIndex) + ".png")
    plt.close()

def readFrames(preName, numFiles):
    img_array = []
    for i in range(numFiles):
        filename = preName + str(i) + ".png"
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    return img_array, size
    
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def generateVideos(expSeriesName):
    checkNumFileScene = 0
    checkNumFilePrimal = 0
    checkNumFileSpectral = 0

    for filename in glob.glob(expSeriesName + "_scene_*.png"):
        checkNumFileScene = checkNumFileScene + 1
    for filename in glob.glob(expSeriesName + "_primal_*.png"):
        checkNumFilePrimal = checkNumFilePrimal + 1
    for filename in glob.glob(expSeriesName + "_spectral_*.png"):
        checkNumFileSpectral = checkNumFileSpectral + 1
    
    if checkNumFileScene != checkNumFilePrimal or checkNumFileScene != checkNumFileSpectral:
        print("Image files are not discovered correctly!")
        return

    sceneFrames, sceneSize = readFrames(expSeriesName + "_scene_", checkNumFileScene)
    primalFrames, primalSize = readFrames(expSeriesName + "_primal_", checkNumFileScene)
    spectralFrames, spectralSize = readFrames(expSeriesName + "_spectral_", checkNumFileScene)
    
    im_h_resize = hconcat_resize_min([sceneFrames[0], primalFrames[0], spectralFrames[0]])
    out = cv2.VideoWriter(expSeriesName + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 4, (im_h_resize.shape[1], im_h_resize.shape[0]))
 
    for i in range(checkNumFileScene):
        im_h_resize = hconcat_resize_min([sceneFrames[i], primalFrames[i], spectralFrames[i]])
        out.write(im_h_resize)

    out.release()

experimentSeriesName = "results/emitterTrans"
_emitterLength = 1000
_receiverLength = 1000
_emitterDensity = 0.05
_receiverDensity = 0.05
_emitterOrientation = 90.0
_emiiterPosition = Point(-350, -350 + _emitterLength / 2.0)
_receiverPosition = Point(-350 + _receiverLength / 2.0, -350)
_occluderPosition = Point(-100, -100)
_occluderHScale = 4.0
_occluderVScale = 4.0

for i in range(0,40):
    new_emiiterPosition = Point(_emiiterPosition.pos[0] + i * 5, _emiiterPosition.pos[1])
    runExperiment(experimentSeriesName, i, new_emiiterPosition, _emitterOrientation, _emitterLength, _emitterDensity, _receiverPosition, _receiverLength, _receiverDensity, _occluderPosition, _occluderHScale, _occluderVScale)

generateVideos(experimentSeriesName)

print("Finished")
#tl.done()