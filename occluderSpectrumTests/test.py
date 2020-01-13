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

for i in range(0,4):
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
    saveImage(folder + "spectral_pil_" + str(i) + ".bmp", getSpectrum(primalImage))



print("Finished")
#tl.done()