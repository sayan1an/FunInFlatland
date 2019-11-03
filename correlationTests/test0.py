from base import *

testname = "Test10"

#screenWidth, screenHeight, center horizontalShift, center vertical shift
screenSetup(800, 800, -75, 0, testname)

######################################### Define Scenes #######################
alpha = 0.1
shift = 400
shiftLight = 400

# Scene 1 - Environment Light
scene = Scene(str(alpha))
#reciever plane
scene.append(Line(Point(-100,-75), Point(100, -75), material=Material(1.0, specularAlpha=alpha)))
drawText("Receiver", 0, -100, "black", 16)
# Occluder
scene.append(Line(Point(-50+shift,165), Point(50+shift, 165), material=Material(1.0, specularAlpha=0.5), mask=0xf0))
drawText("Occluder", shift, 140, "black", 16)
#environment light
#scene.append(Light(radiance=0.4, mask=0xff))
scene.append(Light("line", Point(shiftLight, 200), 150, radiance=0.4))
drawText("Area Light", shiftLight, 175, "black", 16)
scene.draw()

drawText("View Directions", -150, 0, "black", 16)

nPrimaryRays = 5
nSecondaryRays = 5000
index = 0
errorVaryView = []
(primaryRays, viewingAngles) = generatePrimaryRays(nPrimaryRays)

plt.figure(figsize=(7.2, 7)) # 7*dpi, 7*dpi, default dpi = 80    

for primaryRay in primaryRays:
    (incidentAngles, shadowSamples, formFactorSamples, lightSamples, actualSamples)  = shade(scene, primaryRay, nSecondaryRays)
    (netError, correlationError) = computeCorrelations(shadowSamples, formFactorSamples, lightSamples, actualSamples)
    errorVaryView.append(netError)
    plt.plot(incidentAngles, correlationError, label="ViewDir: {view:.3f}, I_error(%): {error:.2f}".format(view=viewingAngles[index], error=netError))
    index = index + 1


plt.legend()
plt.title("Correlation error for: Alpha:{alpha:0.2f} and Area Light".format(alpha=alpha))
plt.ylabel("i_error or error contribution (%)")
plt.xlabel("Incident direction theta (w.r.t. surface normal)")
plt.savefig(testname + ".png")
plt.show()

plt.plot(viewingAngles, errorVaryView)
plt.ylim(0, 50)
plt.show()

screenshot(testname + "setup")

tl.done()