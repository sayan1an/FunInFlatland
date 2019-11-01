import sys
sys.path.append('../')

from fifl import *

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

    plt.plot(angles, np.array(lightSamples) * 10000, label="lightsamples")
    plt.plot(angles, shadowSamples, label="shadowsamples")
    plt.plot(angles, formFactorSamples, label="FormFactorSamples")
    plt.legend()
    plt.show()

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
    correlationError = []

    for i in range(len(shadowSamples)):
        actualTerm = actualTerm + actualSamples[i]
        correlationError.append((formFactorSamples[i] - avgFormFactor) * (shadowSamples[i] - avgShadow) * lightSamples[i])
        correlationTerm = correlationTerm + (formFactorSamples[i] - avgFormFactor) * (shadowSamples[i] - avgShadow) * lightSamples[i]

    approxTerm = avgShadow * avgFormFactor * lightNorm
    approxTermIncorrect = avgShadow * avgFormFactor * (lightNorm ** 2)
    #print(correlationTerm)
    #print(approxTerm)
    #print(actualTerm)
    if not (np.abs(correlationTerm + approxTerm - actualTerm) < 1e-10):
        print("Incorrect caluculations")

    return np.abs(correlationTerm / actualTerm) * 100, (correlationError / actualTerm) * 100
