import sys
sys.path.append('../')
from fifl import *

testname = "OccluderMotion"

#screenWidth, screenHeight, center horizontalShift, center vertical shift, drawZoom
screenSetup(800, 800, 0, 0, testname, 0.5)

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


def sceneSetup(testName, time, seed, sampleCount, windowSz):
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
   
    referencePattern = np.arange(0.0001, 1, 1.0 / 3000.0)
    np.random.seed(time % windowSz)
    patternStatic = np.arange(0.5 / sampleCount, 1, 1.0 / sampleCount)
    jitter = (np.random.uniform(size=sampleCount) - 1) * 0.8
    pattern_0, _ = np.modf(patternStatic + jitter / sampleCount + time / float(windowSz))
      
    np.random.seed(time)
    randomPattern = np.random.uniform(size=sampleCount)
    
    return rayTrace(scene, light, receiverPosition, pattern_0), rayTrace(scene, light, receiverPosition, randomPattern), rayTrace(scene, light, receiverPosition, referencePattern)

def multiProcFn(start:int, end:int, stn0, stmc, stRef):
    windowSize = 11
    sampleCnt = 10
    seed = 51
    for t in range(start, end):
        n0, mc, ref = sceneSetup(testname, t, seed, sampleCnt, windowSize)
        stn0[t] = n0
        stmc[t] = mc
        stRef[t] = ref

        #screenshot(testname + "_scene_" + str(t))

        #lock.acquire()
        #tl.clearscreen()
        #lock.release()
        
        if start == 0:
            print(t)

if __name__ == '__main__':
    processes = []
    nTimeSteps = 5000
    chucnkSize = 1250
    n0 = multiprocessing.Array("f", nTimeSteps, lock=False)
    monteCarlo = multiprocessing.Array("f", nTimeSteps, lock=False)
    reference = multiprocessing.Array("f", nTimeSteps, lock=False)
    for idx in range(0, nTimeSteps, chucnkSize):
        p = multiprocessing.Process(target=multiProcFn, args=(idx, idx + chucnkSize, n0, monteCarlo, reference))
        processes.append(p)
        p.start()
   
    for p in processes:
        p.join()
    
    #sc - sampleCount for current frame
    #sg - sampleCount for gradient frame
    #ws - window size
    np.savez_compressed(testname, n0=n0, mc=monteCarlo, r=reference, sc = 10, ws = 11)

    #tl.done()