from fifl import *

scene = Scene()
#reciver plane
scene.append(Line(Point(-100,-75), Point(100, -75)))
scene.draw()

def generatePrimaryRays(nRays):
    cameraFoucus_x = 0
    cameraFoucus_y = -75

    angles = np.arange(1.0/(nRays + 1), 1.0 - 0.00001, 1.0/(nRays + 1)) * np.pi * 0.5

    cameraDistance = 500
    cameraRayDirection_x = np.sin(angles)
    cameraRayDirection_y = -np.cos(angles)

    cameraPosition_x = -cameraRayDirection_x * cameraDistance + cameraFoucus_x
    cameraPosition_y = -cameraRayDirection_y * cameraDistance + cameraFoucus_y
    
    primaryRays = []
    for i in range(angles.shape[0]):
        primaryRays.append(Ray(Point(cameraPosition_x[i], cameraPosition_y[i]), Vector(cameraRayDirection_x[i], cameraRayDirection_y[i])))
    
    return primaryRays

for ray in generatePrimaryRays(5):
    ray.t = 600
    ray.draw()

tl.done()