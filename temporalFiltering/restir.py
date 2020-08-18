import sys
sys.path.append('../')
from fifl import *
import numpy as np
import matplotlib.pyplot as plt

testname = "OccluderMotionRestir"

class Reservoir:
    def __init__(self):
        self.M = 0 # Counts the number of samples seen
        self.w = 0.0 # Cumulative weights of samples seen
        self.y =  -1.0 # reservoir sample
        self.W = -1.0 # corresponding sample weights

    def update(self, xi, wi):
        self.w += wi
        self.M += 1

        if wi == 0.0:
            return

        if np.random.uniform() < wi / self.w:
            self.y = xi

def RIS(M : int):
    r = Reservoir()

    for i in range(M):
        xi = np.random.uniform()
        r.update(xi, 1.0)
    
    r.W = r.w / r.M

    return r

def combine(rList):
    s = Reservoir()

    for r in rList:
        s.update(r.y, r.W * r.M)

    s.M = 0
    for r in rList:
        s.M += r.M

    s.W = s.w / s.M

    #if s.M > 200:
    #   s.M = 0.1
    
    return s

def restirRun(M :int, rPrev : Reservoir, scene : Scene, lightSrc : Light, receiverPos : Point):
    rsvr = RIS(M)

    if rayTrace(scene, lightSrc, receiverPos, np.array([rsvr.y])) < 0.5:
        rsvr.W = 0
    
    rsvr = combine([rsvr, rPrev])

    #print(rsvr.y)
    return rsvr, rayTrace(scene, lightSrc, receiverPos, np.array([rsvr.y]))


def sceneSetup(testName, time):
    sceneName = testName
    scene = Scene(sceneName)
    
    emitterPosition = Point(0,500)
    emitterOrientation = 0
    emitterLength = 1000
    receiverPosition = Point(0,-500)
    receiverLength = 1000

    occluderMotion = 12
    xPos = np.sin(occluderMotion * time * 2 * np.pi / 1000) * np.cos(occluderMotion * time * 2 * np.pi / 1000) * 500
    yPos =  np.sin(occluderMotion * time * 2 * np.pi / 500) * 250
    rot = ((occluderMotion * time) % 360)
    occluderPosition = Point(xPos, yPos)
    occluderHScale = 6
    occluderVScale = 6

    # Light source
    light = Light("line", emitterPosition, emitterLength, emitterOrientation)
    scene.append(light)
    #drawText("Emitter", -390, 0, "Black", 15)

    # Receiver
    receiver = Line(Point(receiverPosition.pos[0] - receiverLength / 2.0, receiverPosition.pos[1]), Point(receiverPosition.pos[0] + receiverLength / 2.0, receiverPosition.pos[1]), material=Material(0.0))
    scene.append(receiver)
    #drawText("Receiver plane", 0, -380, "Black", 15)
    
    # Occluder
    scene.append(Box(position=occluderPosition, hScale=occluderHScale, vScale=occluderVScale, orientation=rot))
    #drawText("Occluder", -100, 10, "Black", 15)

    #scene.draw()
   
    return scene, light, receiverPosition

def rayTrace(scene, lightSource, receiverPosition, pattern, draw = False, color = 'blue'):
    rayEndPoints = lightSource.sample(pattern)

    rays = []
    i = 0
    for endPoint in rayEndPoints:
        direction = Vector(endPoint.pos[0] - receiverPosition.pos[0], endPoint.pos[1] - receiverPosition.pos[1])
        direction.normalize()
        rays.append(Ray(receiverPosition, direction))

    occlusion = 0
    for ray in rays:
        i, o = scene.intersect(ray)
        if i.hit and isinstance(o, Light):
            occlusion = occlusion + 1

        if draw:
            ray.color = color
            ray.draw()
           
    return float(occlusion) / len(rays)

def generateFrames(start:int, end:int):
    pResv = Reservoir()

    valList = []

    for t in range(start, end):
        s, l, r = sceneSetup(testname, t)
        pResv, val = restirRun(10, pResv, s, l, r)

        valList.append(val)
    
    return np.array(valList)

def bruteForce(start : int, end : int, valArr):
    referencePattern = np.arange(0.0001, 1, 1.0 / 3000.0)

    for t in range(start, end):
        s, l, r = sceneSetup(testname, t)
        valArr[t] = rayTrace(s, l, r, referencePattern)

#np.random.seed(51)



if __name__ == '__main__':
    processes = []
    nTimeSteps = 600
    chucnkSize = 150
   
    reference = multiprocessing.Array("f", nTimeSteps, lock=False)
    for idx in range(0, nTimeSteps, chucnkSize):
        p = multiprocessing.Process(target=bruteForce, args=(idx, idx + chucnkSize, reference))
        processes.append(p)
        p.start()
   
    for p in processes:
        p.join()
    
    print("Reference done")
    accum = np.zeros(nTimeSteps)
    nInstances = 5 # 2 RT per instance

    for i in range(nInstances):
        accum += generateFrames(0, nTimeSteps)

    accum /= nInstances

    plt.plot(accum[50:], label="RIS+WRS")
    plt.plot(reference[50:], label="Reference")
    plt.legend()
    plt.show()

    #np.savez_compressed(testname, n0=n0, mc=monteCarlo, r=reference, sc = 10, ws = 11)

    #tl.done()