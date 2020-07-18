import sys
sys.path.append('../')
from fifl import *

testname = "OccluderMotion"

#screenWidth, screenHeight, center horizontalShift, center vertical shift, drawZoom
screenSetup(800, 800, 0, 0, testname, 0.5)

def rayTrace(scene, lightSource, receiverPosition, pattern, draw = False):
    rayEndPoints = lightSource.sample(pattern)

    rays = []
    i = 0
    for endPoint in rayEndPoints:
        direction = Vector(endPoint.pos[0] - receiverPosition.pos[0], endPoint.pos[1] - receiverPosition.pos[1])
        direction.normalize()
        rays.append(Ray(receiverPosition, direction))

    occlusion = 0
    k = 0
    for ray in rays:
        i, o = scene.intersect(ray)
        if i.hit and isinstance(o, Light):
            occlusion = occlusion + 1

        if draw:
            if k == 5:
                ray.color = 'red'
            ray.draw()
        k = k + 1   
    return float(occlusion) / len(rays) 


def sceneSetup(testName, time, seed, sampleCount, velocity):
    sceneName = testName
    scene = Scene(sceneName)
    
    emitterPosition = Point(0,500)
    emitterOrientation = 0
    emitterLength = 1000
    receiverPosition = Point(0,-500)
    receiverLength = 1000

    occluderMotion = 1
    xPos = np.sin(occluderMotion * time * 2 * np.pi / 1000) * 500
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
    np.random.seed(seed)
    jitter = (np.random.uniform(size=sampleCount) - 1) * 0.8
    patternStatic = np.arange(0.5 / sampleCount, 1, 1.0 / sampleCount)
    pattern = patternStatic + jitter / sampleCount + velocity * time
    pattern,_ = np.modf(pattern)
    
    np.random.seed(time)
    randomPattern = np.random.uniform(size=2*sampleCount)
    
    return rayTrace(scene, light, receiverPosition, pattern), rayTrace(scene, light, receiverPosition, patternStatic), rayTrace(scene, light, receiverPosition, randomPattern), rayTrace(scene, light, receiverPosition, referencePattern)


def multiProcFn(start:int, end:int, storeCurrent, storeGradient, storeMC, storeReference):
    windowSize = 11
    sampleCnt = 10
    seed = 51
    for t in range(start, end):
        currentFrameEstimate, gradientEstimate, mcEstimate, referenceEstimate = sceneSetup(testname, t, seed, sampleCnt, 1.0 / windowSize)
        #currentFrameEstimate, gradientEstimate, mcEstimate = sceneSetup(testname, t, seed, sampleCnt, 1.0 / windowSize)
        storeCurrent[t] = currentFrameEstimate
        storeGradient[t] = gradientEstimate
        storeMC[t] = mcEstimate
        storeReference[t] = referenceEstimate

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
    current = multiprocessing.Array("f", nTimeSteps, lock=False)
    gradient = multiprocessing.Array("f", nTimeSteps, lock=False)
    monteCarlo = multiprocessing.Array("f", nTimeSteps, lock=False)
    reference = multiprocessing.Array("f", nTimeSteps, lock=False)
    for idx in range(0, nTimeSteps, chucnkSize):
        p = multiprocessing.Process(target=multiProcFn, args=(idx, idx + chucnkSize, current, gradient,monteCarlo, reference))
        processes.append(p)
        p.start()
   
    for p in processes:
        p.join()
    
    #sc - sampleCount for current frame
    #sg - sampleCount for gradient frame
    #ws - window size
    np.savez_compressed(testname, c=current, sc = 10,  g=gradient, sg = 10, ws = 11, mc=monteCarlo, r=reference)

    #tl.done()