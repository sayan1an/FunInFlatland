from base import *

testname = "Test16"

#screenWidth, screenHeight, center horizontalShift, center vertical shift
screenSetup(800, 800, 150, 0, testname)

######################################### Define Scenes #######################
alpha = 0.1
shift = 500
shiftLight = 500

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

drawText("View Direction", -150, 0, "black", 16)

nPrimaryRays = 5
nSecondaryRays = 10000

(primaryRays, viewingAngles) = generatePrimaryRays(nPrimaryRays)
rayIdx = 3
(incidentAngles, shadowSamples, formFactorSamples, lightSamples, actualSamples)  = shade(scene, primaryRays[rayIdx], nSecondaryRays)
(netError, correlationError) = computeCorrelations(shadowSamples, formFactorSamples, lightSamples, actualSamples)

fig, axs = plt.subplots(4)
axs[0].plot(incidentAngles, correlationError, label="Combined error i.e. i_error, I_error(%):{error:0.2f}".format(error=netError), color="black")
axs[0].set_xticklabels([])
axs[0].legend()
axs[1].plot(incidentAngles, np.array(lightSamples), label="L", color="red")
axs[1].set_xticklabels([])
axs[1].legend()
axs[2].plot(incidentAngles, np.array(shadowSamples) - np.average(shadowSamples), label="V - E[V]", color="green")
axs[2].set_xticklabels([])
axs[2].legend()
axs[3].plot(incidentAngles, np.array(formFactorSamples) - np.average(formFactorSamples), label="F - E[F]", color="blue")
axs[3].legend()
axs[3].set_xlabel("Incident direction theta (w.r.t. surface normal)")
fig.suptitle("Correlation error for: Alpha:{alpha:0.2f} and View Dir:{view:0.2f}".format(alpha=alpha, view=viewingAngles[rayIdx]))
fig.figsize=(7.2, 7)
fig.savefig(testname + ".png")
fig.show()

screenshot(testname + "setup")

tl.done()