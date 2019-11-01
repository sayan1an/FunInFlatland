from base import *

######################################### Define Scenes #######################
alpha = 0.1
shift = -200
shiftLight = -150

# Scene 1 - Environment Light
scene = Scene(str(alpha))
#reciever plane
scene.append(Line(Point(-100,-75), Point(100, -75), material=Material(1.0, specularAlpha=alpha)))
# Occluder
scene.append(Line(Point(-50+shift,125), Point(50+shift, 125), material=Material(1.0, specularAlpha=0.5), mask=0xf0))
#environment light
#scene.append(Light(radiance=0.4, mask=0xff))
scene.append(Light("line", Point(shiftLight, 200), 150, radiance=0.4))
scene.draw()

nPrimaryRays = 5
nSecondaryRays = 5000
index = 0
errorVaryView = []
(primaryRays, viewingAngles) = generatePrimaryRays(nPrimaryRays)
for primaryRay in primaryRays:
    (incidentAngles, shadowSamples, formFactorSamples, lightSamples, actualSamples)  = shade(scene, primaryRay, nSecondaryRays)
    (netError, correlationError) = computeCorrelations(shadowSamples, formFactorSamples, lightSamples, actualSamples)
    errorVaryView.append(netError)
    print(netError, viewingAngles[index])
    plt.plot(incidentAngles, correlationError, label=str(viewingAngles[index]))
    plt.show()
    index = index + 1
    
plt.legend()
plt.show()

plt.plot(viewingAngles, errorVaryView)
plt.ylim(0, 50)
plt.show()
tl.done()