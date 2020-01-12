import sys
sys.path.append('../')

from fifl import *
import matplotlib.pyplot as plt

testname = "OccluderSpectrumTest"

#screenWidth, screenHeight, center horizontalShift, center vertical shift
screenSetup(800, 800, 0, 0, testname)

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
    return rays

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


def spectrumTester(emitterPosition, emitterOrientation, emitterLength, emitterDensity, reciverPosition, receiverLength, receiverDensity, occluderPosition, occluderHScale, occluderVScale):
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
    scene.draw()

    rays = generateRays(light, receiver)
  
    scatter = []
    for ray in rays:
        (i,o) = scene.intersect(ray[2])
        if i.hit and isinstance(o, Light):
            plt.scatter(ray[0], ray[1], color="red")
        elif i.hit:
            plt.scatter(ray[0], ray[1], color="blue")
        ray[2].draw()
 
    #plt.show()
    #tl.done()

emitterLength = 1000
receiverLength = 1000
emitterDensity = 0.01
receiverDensity = 0.01
emitterOrientation = 90.0
emiiterPosition = Point(-350, -350 + emitterLength / 2.0)
receiverPosition = Point(-350 + receiverLength / 2.0, -350)
occluderPosition = Point(-100, -100)
occluderHScale = 4.0
occluderVScale = 4.0

for i in range(0,1):
    emiiterPosition = Point(emiiterPosition.pos[0] + i * 100, emiiterPosition.pos[1])
    tl.clearscreen()
    spectrumTester(emiiterPosition, emitterOrientation, emitterLength, 
        emitterDensity, receiverPosition, receiverLength, receiverDensity, occluderPosition, occluderHScale, occluderVScale)

tl.done()