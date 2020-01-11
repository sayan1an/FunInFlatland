import sys
sys.path.append('../')

from fifl import *
import matplotlib.pyplot as plt

testname = "OccluderSpectrumTest"

#screenWidth, screenHeight, center horizontalShift, center vertical shift
screenSetup(800, 800, 0, 0, testname)

sceneName = testname
scene = Scene(sceneName)

# scene params
emitterLength = 1000
receiverLength = 1000
emitterDensity = 50.0
receiverDensity = 50.0
emiiterPointOfOrigin = Point(-350, -350)
receiverPointOfOrigin = Point(-350, -350)
occluderPosition = Point(-100, -100)

# Light source
scene.append(Light(Line(emiiterPointOfOrigin, Point(emiiterPointOfOrigin.pos[0], emiiterPointOfOrigin.pos[1] + emitterLength), flipNormal=True)))
# Reciver
scene.append(Line(receiverPointOfOrigin, Point(receiverPointOfOrigin.pos[0] + receiverLength, receiverPointOfOrigin.pos[1]), material=Material(0.0)))

# Occluder
scene.append(Box(position=occluderPosition, hScale=4, vScale=4, orientation=0))

scene.draw()

def generateRays():
    rayEndPoints = np.arange(5, emitterLength, float(emitterLength) / emitterDensity)
    rayOrigins = np.arange(5, receiverLength, float(receiverLength) / receiverDensity)

    rays = []
    for origin in rayOrigins:
        for endPoint in rayEndPoints:
            o = Point(receiverPointOfOrigin.pos[0] + origin, receiverPointOfOrigin.pos[1])
            e = Point(emiiterPointOfOrigin.pos[0], emiiterPointOfOrigin.pos[1] + endPoint)

            direction = Vector(e.pos[0] - o.pos[0], e.pos[1] - o.pos[1])
            direction.normalize()

            rays.append([origin, endPoint, Ray(o, direction)])
    
    return rays

rays = generateRays()

scatter = []
for ray in rays:
    (i,o) = scene.intersect(ray[2])
    if i.hit and isinstance(o, Light):
        plt.scatter(ray[0], ray[1], color="red")
    elif i.hit:
        plt.scatter(ray[0], ray[1], color="blue")
    #ray[2].draw()
 
plt.show()
tl.done()