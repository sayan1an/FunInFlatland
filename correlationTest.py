from fifl import *

######################################### Define Scenes #######################

scenes = []
alphas = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1]

# Scene 1 - Environment Light
for alpha in alphas:
    scene = Scene(str(alpha))
    #reciever plane
    scene.append(Line(Point(-100,-75), Point(100, -75), material=Material(1.0, specularAlpha=alpha)))
    # Occluder
    #scene.append(Line(Point(-75,125), Point(25, 125), material=Material(1.0, specularAlpha=0.5), mask=0xf0))
    #environment light
    scene.append(Light(radiance=0.4, mask=0xff))
    scenes.append(scene)
    
def generatePrimaryRays(nRays):
    cameraFoucus_x = 0
    cameraFoucus_y = -75

    angles = np.arange(1.0/(nRays + 1), 1.0 - 0.00001, 1.0 / (nRays + 1)) * np.pi * 0.5

    cameraDistance = 100
    cameraRayDirection_x = np.sin(angles)
    cameraRayDirection_y = -np.cos(angles)

    cameraPosition_x = -cameraRayDirection_x * cameraDistance + cameraFoucus_x
    cameraPosition_y = -cameraRayDirection_y * cameraDistance + cameraFoucus_y
    
    primaryRays = []
    for i in range(angles.shape[0]):
        primaryRays.append(Ray(Point(cameraPosition_x[i], cameraPosition_y[i]), Vector(cameraRayDirection_x[i], cameraRayDirection_y[i])))
    
    return (primaryRays, angles)

def shade(scene, primaryRay, nSecondaryRays):
  (intersection, obj) = scene.intersect(primaryRay)
  #primaryRay.draw()

  if not intersection.hit:
    return 0
  
  if isinstance(obj, Light):
    return obj.radiance

  (angles, weights) = sample("uniform", nSecondaryRays)
  shadowRays = angleToRays(angles, intersection)
  lightRays = angleToRays(angles, intersection, 0x0f)
 
  shadowSamples = []
  formFactorSamples = []
  lightSamples = []
  actualSamples = []

  for i in range(len(angles)):
    formFactorSamples.append(obj.material.eval(intersection.normal, Vector(-primaryRay.pos[0], -primaryRay.pos[1]),  Vector(lightRays[i].pos[0], lightRays[i].pos[1])))
    
    (iSec, oSec) = scene.intersect(lightRays[i])
    if isinstance(oSec, Light):
        lightSamples.append(oSec.radiance * weights[i]) 
    else:
        lightSamples.append(0)

    (iSec, oSec) = scene.intersect(shadowRays[i])
    #shadowRays[i].color = "red"
    #shadowRays[i].draw()
    if isinstance(oSec, Light):
        shadowSamples.append(1)
        actualSamples.append(formFactorSamples[i] * oSec.radiance * weights[i])
    else:
        shadowSamples.append(0)
        actualSamples.append(0)

  return angles, shadowSamples, formFactorSamples, lightSamples, actualSamples

def computeCorrelations(shadowSamples, formFactorSamples, lightSamples, actualSamples):
    avgShadow = 0
    avgFormFactor = 0

    lightNorm = 0
    for i in range(len(shadowSamples)):
        avgShadow = avgShadow + shadowSamples[i] * lightSamples[i]
        avgFormFactor = avgFormFactor + formFactorSamples[i] * lightSamples[i]
        lightNorm = lightNorm + lightSamples[i]
    
    avgShadow = avgShadow / lightNorm
    avgFormFactor = avgFormFactor / lightNorm

    correlationTerm = 0
    actualTerm = 0
    for i in range(len(shadowSamples)):
        actualTerm = actualTerm + actualSamples[i]
        correlationTerm = correlationTerm + (formFactorSamples[i] - avgFormFactor) * (shadowSamples[i] - avgShadow) * lightSamples[i]

    approxTerm = avgShadow * avgFormFactor * lightNorm
    approxTermIncorrect = avgShadow * avgFormFactor * (lightNorm ** 2)
    #print(correlationTerm)
    #print(approxTerm)
    #print(actualTerm)

    if not (np.abs(correlationTerm + approxTerm - actualTerm) < 1e-10):
        print("Incorrect caluculations")

    return np.abs(correlationTerm / actualTerm) * 100

#computeCorrelations(shadowSamples, formFactorSamples, lightSamples, actualSamples)

# (primaryRays, viewingAngles) = generatePrimaryRays(500)
# (incidentAngles, shadowSamples, formFactorSamples, lightSamples, actualSamples)  = shade(primaryRays[0], 1000)
# plt.plot(incidentAngles, shadowSamples)
# plt.plot(incidentAngles, formFactorSamples)
# plt.plot(incidentAngles, lightSamples)
# plt.plot(incidentAngles, actualSamples)
# plt.show()

errorVaryAlpha = []
nPrimaryRays = 20
nSecondaryRays = 500
for scene in scenes:
    errorVaryView = []
    index = 0
    (primaryRays, viewingAngles) = generatePrimaryRays(nPrimaryRays)
    for primaryRay in primaryRays:
        (incidentAngles, shadowSamples, formFactorSamples, lightSamples, actualSamples)  = shade(scene, primaryRay, nSecondaryRays)
        errorVaryView.append(computeCorrelations(shadowSamples, formFactorSamples, lightSamples, actualSamples))
        #print(index)
        index = index + 1
    errorVaryAlpha.append(errorVaryView)

(primaryRays, viewingAngles) = generatePrimaryRays(nPrimaryRays)
for trajectory in errorVaryAlpha:
    plt.plot(viewingAngles, trajectory)

plt.ylim(0, 50)
plt.show()
tl.done()