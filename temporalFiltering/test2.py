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
    samplePerTrajectory = sampleCount // 5 # Assuming 5 trajectories
    patternStatic = np.arange(0.5 / samplePerTrajectory, 1, 1.0 / samplePerTrajectory)
    jitter = (np.random.uniform(size=samplePerTrajectory) - 1) * 0.8
    pattern_n2, _ = np.modf(patternStatic + jitter / sampleCount + velocity * (time - 2))
    pattern_n1, _ = np.modf(patternStatic + jitter / sampleCount + velocity * (time - 1))
    pattern_0, _ = np.modf(patternStatic + jitter / sampleCount + velocity * time)
    pattern_p1, _ = np.modf(patternStatic + jitter / sampleCount + velocity * (time + 1))
    pattern_p2, _ = np.modf(patternStatic + jitter / sampleCount + velocity * (time + 2))
   
    np.random.seed(time)
    randomPattern = np.random.uniform(size=sampleCount)
    
    #return rayTrace(scene, light, receiverPosition, pattern_n2, True, 'blue'), rayTrace(scene, light, receiverPosition, pattern_n1, True, 'green'), rayTrace(scene, light, receiverPosition, pattern_0, True, 'yellow'), rayTrace(scene, light, receiverPosition, pattern_p1, True, 'magenta'), rayTrace(scene, light, receiverPosition, pattern_p2, True, 'red'), rayTrace(scene, light, receiverPosition, randomPattern), rayTrace(scene, light, receiverPosition, referencePattern)
    return rayTrace(scene, light, receiverPosition, pattern_n2), rayTrace(scene, light, receiverPosition, pattern_n1), rayTrace(scene, light, receiverPosition, pattern_0), rayTrace(scene, light, receiverPosition, pattern_p1), rayTrace(scene, light, receiverPosition, pattern_p2), rayTrace(scene, light, receiverPosition, randomPattern), rayTrace(scene, light, receiverPosition, referencePattern)



def multiProcFn(start:int, end:int, stn2, stn1, stn0, stp1, stp2, stmc, stRef):
    windowSize = 11
    sampleCnt = 40
    seed = 51
    for t in range(start, end):
        n2, n1, n0, p1, p2, mc, ref = sceneSetup(testname, t, seed, sampleCnt, 1.0 / windowSize)
        stn2[t] = n2
        stn1[t] = n1
        stn0[t] = n0
        stp1[t] = p1
        stp2[t] = p2
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
    n2 = multiprocessing.Array("f", nTimeSteps, lock=False)
    n1 = multiprocessing.Array("f", nTimeSteps, lock=False)
    n0 = multiprocessing.Array("f", nTimeSteps, lock=False)
    p1 = multiprocessing.Array("f", nTimeSteps, lock=False)
    p2 = multiprocessing.Array("f", nTimeSteps, lock=False)
    monteCarlo = multiprocessing.Array("f", nTimeSteps, lock=False)
    reference = multiprocessing.Array("f", nTimeSteps, lock=False)
    for idx in range(0, nTimeSteps, chucnkSize):
        p = multiprocessing.Process(target=multiProcFn, args=(idx, idx + chucnkSize, n2, n1, n0, p1, p2, monteCarlo, reference))
        processes.append(p)
        p.start()
   
    for p in processes:
        p.join()
    
    #sc - sampleCount for current frame
    #sg - sampleCount for gradient frame
    #ws - window size
    np.savez_compressed(testname, n2=n2, n1=n1, n0=n0, p1=p1, p2=p2, mc=monteCarlo, r=reference, sc = 40, ws = 11)

    #tl.done()