import sys
sys.path.append('../')
from fifl import *

testname = "OccluderSpectrumTest"

#screenWidth, screenHeight, center horizontalShift, center vertical shift, drawZoom
screenSetup(800, 800, 0, 0, testname, 0.5)

def rayTrace(scene, lightSource, receiverPosition, pattern):
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

        if pattern.shape[0] < 100:
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

    scene.draw()
   
    referencePattern = np.arange(0.0001, 1, 1.0 / 3000.0)
    np.random.seed(seed)
    pattern = np.random.uniform(size=sampleCount)
    pattern = pattern + velocity * time
    pattern,_ = np.modf(pattern)
    
    return rayTrace(scene, light, receiverPosition, pattern)#, rayTrace(scene, light, receiverPosition, referencePattern)


def multiProcFn(start:int, end:int):
    windowSize = 40
    sampleCnt = 20
    seed = 5
    for t in range(start, end):
        currentFrameEstimate = sceneSetup(testname, t, seed, sampleCnt, 1.0 / windowSize)
        screenshot(testname + "_scene_" + str(t))
        lock.acquire()
        tl.clearscreen()
        lock.release()

if __name__ == '__main__':
    processes = []
    for idx in range(0, 5000, 125):
        p = multiprocessing.Process(target=multiProcFn, args=(idx, idx + 125,))
        processes.append(p)
        p.start()
   
    for p in processes:
        p.join()
 

    tl.done()