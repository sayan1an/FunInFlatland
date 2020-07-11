# World coordinates - Positive Y axis towards screen top. Positive X axis towards screen right.
# Env Light - Env light is defined in the upper semi-circle with center at world (0,0). 

from abc import ABC, abstractmethod
import turtle as tl
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image #conda install pillow, manuall install ghostscript, point Path to gs/bin/ 
import multiprocessing

SCENE_BOUND = 10000.0
lock = multiprocessing.RLock()

def screenSetup(screenWidth, screenHeight, hShift, vShift, title, drawZoom=1.0):
    lock.acquire() 
    screen = tl.Screen()
    screen.setup(screenWidth, screenHeight)
    screen.reset()
    screen.setworldcoordinates(-screenWidth / 2 + hShift, -screenHeight / 2 + vShift, screenWidth / 2 + hShift, screenHeight / 2 + vShift)
    screen.title(title)
    Drawable.drawZoom = drawZoom
    lock.release()

def screenshot(filename):
    lock.acquire()
    tl.hideturtle()
    cnv = tl.getscreen().getcanvas() 
    ps = cnv.postscript(colormode = 'color')
    im = Image.open(io.BytesIO(ps.encode('utf-8')))
    im.save(filename + '.png')
    lock.release()

def drawText(text, xPos, yPos, color, fontSize):
    lock.acquire()
    style = ('Arial', fontSize, 'normal')
    pen = tl.Turtle()
    pen.speed(0)
    pen.hideturtle()
    pen.color(color)
    pen.up()
    pen.goto(xPos, yPos)
    pen.down()
    pen.write(text, font=style, align="center")
    lock.release()

def rotMat(ang):
  ang = ang * np.pi / 180.0
  return np.array([[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
def scaleMat(hScale, vScale):
  return np.array([[hScale, 0, 0], [0, vScale, 0], [0, 0, 1]])
def translationMat(hTrans, vTrans):
  return np.array([[1, 0, hTrans], [0, 1, vTrans], [0, 0, 1]])

# Abstract class Drawable
# Base class for all drawable quantities
class Drawable(ABC):
  drawZoom = 1.0 # class or static variable
  @abstractmethod
  def draw(self):
        pass

# Abstract class Intersectable
# Base class for any shape or object that implements ray-shape intersection
class Intersectable(ABC):
  mask = None
  @abstractmethod
  def intersect(self, ray): # Method returns an Intersection object, the normals and tangents must be unit length
      pass

class Sampleable(ABC):
  @abstractmethod
  def sample(self, sampleArray): # Method takes as input an array of numbers between [0, 1] and returns a set of points
      pass

class CameraRayPayload:
  camRay = None
  value = None # Final radiance at screen pixel
  jitter = None # Jiterr from pixel start boundary
  primaryHitPoint = None
  secondayDirection = None
  secondaryHitPoint = None

class Point(Drawable):
  size = None
  color = None
  pos = None

  def __init__(self, x, y, z = 1):
    self.size = 3
    self.color = "black"
    self.pos = np.array([x, y, z], dtype=float)
  
  def sub(self, point):
    return Point(self.pos[0] - point.pos[0], self.pos[1] - point.pos[1])
  
  def add(self, point):
    return Point(self.pos[0] + point.pos[0], self.pos[1] + point.pos[1])

  def dot(self, point):
    return self.pos[0] * point.pos[0] + self.pos[1] * point.pos[1]
 
  def scale(self, scalar):
    return Point(self.pos[0] * scalar, self.pos[1] * scalar)
  
  def getTangent(self):
    return Point(-self.pos[1], self.pos[0])

  def length(self):
      return np.sqrt(self.pos[0]**2 + self.pos[1]**2)

  def draw(self):
    lock.acquire()
    pen = tl.Turtle()
    pen.speed(0)
    pen.hideturtle()
    pen.width(self.size)
    pen.color(self.color)
    pen.up()
    pen.goto(self.pos[0], self.pos[1])
    pen.down()
    pen.dot()
    lock.release()

class Vector(Point):
  t = None
  origin = None
    
  def __init__(self, x, y, z=0):
    self.color = "green"
    self.pos = np.array([x, y, z], dtype=float)
    self.origin = Point(0, 0)
    self.t = 50 * self.length()
  
  def normalize(self):
      l = self.length()
      self.pos[0] = self.pos[0] / l
      self.pos[1] = self.pos[1] / l
      self.t = 50

  def draw(self):
    lock.acquire()
    pen = tl.Turtle()
    pen.speed(0)
    pen.shape("classic")
    pen.width(self.size)
    pen.color(self.color)
    pen.up()
    pen.goto(self.origin.pos[0] * Drawable.drawZoom, self.origin.pos[1] * Drawable.drawZoom)
    pen.down()
    pen.setheading(np.angle([self.pos[0] + self.pos[1] * 1.0j], deg=True))
    pen.forward(self.t * Drawable.drawZoom)
    lock.release()
 
  def scale(self, vec):
    v = super().scale(vec)
    return Vector(v.pos[0], v.pos[1])

  def add(self, vec):
    v = super().add(vec)
    return Vector(v.pos[0], v.pos[1])

  def sub(self, vec):
    v = super().sub(vec)
    return Vector(v.pos[0], v.pos[1])

  def getTangent(self):
    v = super().getTangent()
    return Vector(v.pos[0], v.pos[1])

class Ray(Vector):
  mask = None
  def __init__(self, origin, direction, mask=0xff):
    self.color = "blue"
    self.origin = Point(origin.pos[0], origin.pos[1])
    self.pos = np.array([direction.pos[0], direction.pos[1], 0], dtype=float)
    self.normalize()
    self.t = 0
    self.mask = mask
  
  def draw(self):
    if self.t > 0:
        self.origin.color = self.color
        self.origin.size = self.size
        self.origin.draw()
        super().draw()

class Intersection(Drawable):
  color = None
  size = None
  hit = None
  
  intersection = None # Point of intersection
  normal = None # Normal and tangent at point of intersection
  tangent = None

  def __init__(self, hit=False , intersection=Point(0,0), normal=Vector(0,0), tangent=Vector(0,0)):
    self.color = "red"
    self.size = 2
    self.hit = hit
    self.intersection = Point(intersection.pos[0], intersection.pos[1])
    self.normal = Vector(normal.pos[0], normal.pos[1])
    self.tangent = Vector(tangent.pos[0], tangent.pos[1])

  def draw(self):
    if self.hit:
      self.intersection.color = self.color
      self.intersection.size = self.size
      self.intersection.draw()
      
      self.normal.origin = self.intersection
      self.normal.color = self.color
      self.normal.size = self.size
      self.normal.draw()

      self.tangent.origin = self.intersection
      self.tangent.color = self.color
      self.tangent.size = self.size
      self.tangent.draw()

class Camera(Drawable, Intersectable):
  worldToCamera = None # View transform
  cameraToScreen = None # Projective Transform
  screenToRaster = None # Clip, NDC, raster - Usually implemented in graphics hardware

  cameraToWorld = None
  screenToCamera = None
  rasterToScreen = None

  screenRes = None

  @abstractmethod
  def generateRays(self):
    pass

  def intersect(self):
    return Intersection()

class PerspectiveCamera(Camera):
  near = None
  def __init__(self, cameraPos, cameraFront, fov, near, far, resolution):
    cameraFront = Vector(cameraFront.pos[0], cameraFront.pos[1])
    cameraFront.normalize()
    self.setWordlToCamera(cameraPos, cameraFront)
    self.setCameraToScreen(fov, near, far)
    self.setScreenToRaster(resolution)
    self.near = near
    self.screenRes = resolution

  def setWordlToCamera(self, cameraPos, cameraFront):
    # Simalar to Vulkan, camera pointing towards -z
    cameraRight = cameraFront.getTangent()
    self.worldToCamera = np.array([[cameraRight.pos[0], cameraRight.pos[1], -cameraRight.dot(cameraPos)],
        [-cameraFront.pos[0], -cameraFront.pos[1], cameraFront.dot(cameraPos)],
        [0, 0, 1]], dtype=float)
    self.cameraToWorld = np.linalg.inv(self.worldToCamera)

  def setCameraToScreen(self, fov, near, far):
    # Similar to Vulkan
    # Bring x in viewing frustum of camera space between -1 to 1
    # Bring y or depth values between 0 to 1, i.e. after perspective transform followed by perspective division 
    self.cameraToScreen = np.array([[1.0/np.tan(fov * np.pi/360.0), 0, 0],
        [0, -far / (far - near), - far * near / (far - near)],
        [0, -1, 0]])
    self.screenToCamera = np.linalg.inv(self.cameraToScreen)

  def setScreenToRaster(self, screenRes):
    self.rasterToScreen = np.array([[2.0 / screenRes , 0, -1], [0, 1, 0], [0, 0, 1]])
    self.screenToRaster = np.linalg.inv(self.rasterToScreen)

  def screenToRay(self, screenCoord):
    # Transform point (0,0,1) in camera space to world space
    rayOrigin = np.dot(self.cameraToWorld, np.array([0, 0, 1]))

    temp = np.dot(self.screenToCamera, np.dot(self.rasterToScreen, np.array([screenCoord, -self.near * 5, 1])))
    temp[0] = temp[0] / temp[2]
    temp[1] = temp[1] / temp[2]
    temp[2] = 0

    rayDir = np.dot(self.cameraToWorld, temp)
    return Ray(Point(rayOrigin[0], rayOrigin[1]), Vector(rayDir[0], rayDir[1]))

  def generateRays(self, spp):
    rays = [[] for i in range(self.screenRes)]
    
    if spp % 2 == 0: # If even make it odd
      spp = spp + 1
    
    for s in range(-int(spp/2), int(spp/2) + 1):
      jitter = 0.5 + (s / float(spp))
      for i in range(self.screenRes):
        c = CameraRayPayload()
        c.camRay = self.screenToRay(i + jitter)
        c.jitter = jitter
        rays[i].append(c)

    return rays

  def draw(self):
    cameraPos = np.dot(self.cameraToWorld, np.array([0, 0, 1]))
    cameraRight = np.dot(self.cameraToWorld, np.array([1, 0, 0]))
    cameraFront = np.dot(self.cameraToWorld, np.array([0, -1, 0]))

    i = Intersection(True, Point(cameraPos[0], cameraPos[1]), Vector(cameraFront[0], cameraFront[1]), Vector(cameraRight[0], cameraRight[1]))
    i.draw()

    # In camera coordinates
    screenDistance = -self.near * 2.5
    temp = np.dot(self.cameraToScreen, np.array([0, screenDistance, 1]))
    # Depth of screen after perspective division
    normalDepth = temp[1] / temp[2]
    
    temp = np.dot(self.screenToCamera, np.dot(self.rasterToScreen, np.array([self.screenRes - 1, normalDepth, 1])))
    temp[0] = temp[0] / temp[2]
    temp[1] = temp[1] / temp[2]
    temp[2] = 1

    screenEnd = np.dot(self.cameraToWorld, temp)

    temp = np.dot(self.screenToCamera, np.dot(self.rasterToScreen, np.array([0, normalDepth, 1])))
    temp[0] = temp[0] / temp[2]
    temp[1] = temp[1] / temp[2]
    temp[2] = 1

    screenStart = np.dot(self.cameraToWorld, temp)

    screen = Line(Point(screenStart[0],  screenStart[1]), Point(screenEnd[0],  screenEnd[1]))
    screen.color = "purple"
    screen.draw()

class Material:
  diffuseColor = None # Usually a texture
  specularColor = None # Can also be a texture but usaully 1 - avg(diffuseColor) is used.
  specularAlpha = None # Use beckmann brdf

  def __init__(self, specularColor, specularAlpha = None):
    self.diffuseColor = 1 - specularColor
    self.specularAlpha = specularAlpha
    self.specularColor = specularColor

  def D(self, cosThetaHalf):
    cosThetaHalf_2 = cosThetaHalf * cosThetaHalf
    tanThetaHalf_2 = (1 - cosThetaHalf_2) / cosThetaHalf_2
    alpha_2 = self.specularAlpha * self.specularAlpha

    return np.exp(-tanThetaHalf_2/alpha_2) / (np.pi * alpha_2 * cosThetaHalf_2 * cosThetaHalf_2)
  
  def evalDiffuse(self, surfNorm, outDir):
    return self.diffuseColor * outDir.dot(surfNorm) / np.pi
  
  def evalSpecular(self, surfNorm, viewDir, outDir):
    wHalf = viewDir.add(outDir)
    wHalf.normalize()
    cosThetaHalf = wHalf.dot(surfNorm)
    if (cosThetaHalf <= 1e-15):
        return 0
    
    return self.specularColor * self.D(cosThetaHalf) / (4 * viewDir.dot(surfNorm))

  def eval(self, surfNorm, viewDir, outDir):
    if self.specularAlpha == None:
      return self.evalDiffuse(surfNorm, outDir)
    else:
      return self.evalSpecular(surfNorm, viewDir, outDir) + self.evalDiffuse(surfNorm, outDir)

class Line(Drawable, Intersectable, Sampleable):
  flipNormal = None
  size = None
  color = None
  start = None
  end = None
  material = None
    
  def __init__(self, start, end, material=Material(0.0), flipNormal=False, mask=0xff, drawColor="magenta"):
    self.flipNormal = flipNormal
    self.start = Point(start.pos[0], start.pos[1])
    self.end = Point(end.pos[0], end.pos[1])
    self.material = Material(material.specularColor, material.specularAlpha)
    self.size = 3
    self.color = drawColor
    self.mask = mask
      
  def drawNormal(self):
    v = self.end.sub(self.start)
    normal = Vector(-v.pos[1], v.pos[0])
    normal.normalize()
    
    if self.flipNormal:
      normal = normal.scale(-1)
    normal.origin = self.end.add(self.start).scale(0.5)
    normal.color = self.color
    normal.size = self.size

    normal.draw()

  def draw(self):
    lock.acquire()
    pen = tl.Turtle()
    pen.speed(0)
    pen.hideturtle()
    pen.width(self.size)
    pen.color(self.color)
    pen.up()
    pen.goto(self.start.pos[0] * Drawable.drawZoom, self.start.pos[1] * Drawable.drawZoom)
    pen.down()
    pen.goto(self.end.pos[0] * Drawable.drawZoom, self.end.pos[1] * Drawable.drawZoom)
    #self.drawNormal()
    lock.release()
    
  def intersect(self, ray):
    if not (self.mask & ray.mask):
        return Intersection()
    v1 = ray.origin.sub(self.start)
    v2 = self.end.sub(self.start)
    v3 = Point(-ray.pos[1], ray.pos[0])
    
    t2 = v1.dot(v3) / v2.dot(v3)
    
    if t2 < 0 or t2 > 1:
      return Intersection()
    
    intersect = self.start.add(v2.scale(t2))
    ray.t = intersect.sub(ray.origin).dot(ray)

    if ray.t < 0:
      return Intersection()
    
    tangent = Vector(v2.pos[0], v2.pos[1])
    tangent.normalize()

    normal = tangent.getTangent()
    normal.normalize()
  
    if self.flipNormal:
        normal = normal.scale(-1)
    return Intersection(True, intersect, normal, tangent)

  def sample(self, sampleArray):
    nSamples = sampleArray.shape[0]

    if (sampleArray >= 0).sum() != nSamples and (sampleArray <= 1).sum() != nSamples:
      raise Exception("All elements of the input array should be between 0 and 1")

    pointList = []
    direction = self.end.sub(self.start)
    
    for i in range(0, nSamples):
      num = sampleArray[i]
      pointList.append(self.start.add(direction.scale(num)))

    return pointList

class Box(Drawable, Intersectable):
  left = None
  right = None
  top = None
  bottom = None
  mask = None

  def __init__(self,  position=Point(0,0), size=50, hScale=1.0, vScale=1.0, orientation=0, mask=0xff, drawColor="magenta"):
    transMat = np.dot(rotMat(orientation), scaleMat(hScale, vScale))
    transMat = np.dot(translationMat(position.pos[0], position.pos[1]), transMat)
    
    start = np.dot(transMat, np.array([-size/2, -size/2, 1]))
    end = np.dot(transMat, np.array([-size/2, size/2, 1]))
    self.left = Line(Point(start[0], start[1]), Point(end[0], end[1]), drawColor=drawColor)

    start = np.dot(transMat, np.array([size/2, -size/2, 1]))
    end = np.dot(transMat, np.array([size/2, size/2, 1]))
    self.right = Line(Point(start[0], start[1]), Point(end[0], end[1]), flipNormal=True, drawColor=drawColor)

    start = np.dot(transMat, np.array([-size/2, size/2, 1]))
    end = np.dot(transMat, np.array([size/2, size/2, 1]))
    self.top = Line(Point(start[0], start[1]), Point(end[0], end[1]), drawColor=drawColor)

    start = np.dot(transMat, np.array([-size/2, -size/2, 1]))
    end = np.dot(transMat, np.array([size/2, -size/2, 1]))
    self.bottom = Line(Point(start[0], start[1]), Point(end[0], end[1]), flipNormal=True, drawColor=drawColor)

    self.mask = mask

  def draw(self):
    self.left.draw()
    self.right.draw()
    self.top.draw()
    self.bottom.draw()

  def intersect(self, ray):
    if not (self.mask & ray.mask):
      return Intersection()

    intersectObjList = [self.left, self.right, self.top, self.bottom]

    min_t = SCENE_BOUND
    intersection = Intersection()
    for o in intersectObjList:
      i = o.intersect(ray)
      if i.hit and ray.t < min_t and ray.t > 1e-10:
        min_t = ray.t
        intersection = i
    
    ray.t = min_t

    return intersection

class Light(Drawable, Intersectable, Sampleable):
  geometry = None
  radiance = None
  # Generic constructor
  def __init__(self,  geometryString="env", position=Point(0,0), length=50, orientation=0, radiance=1.0, mask=0xff, drawColor="orange"):
    if geometryString == "line":
      start = np.dot(rotMat(orientation), np.array([length/2, 0, 1]))
      end = np.dot(rotMat(orientation), np.array([-length/2, 0, 1]))
      start = position.add(Point(start[0], start[1]))
      end = position.add(Point(end[0], end[1]))
      self.geometry = Line(start, end, flipNormal=True, drawColor=drawColor)
    elif geometryString == "env":
      self.geometry = "env"
    else:
      self.geometry = "NotImplemented"

    self.mask = mask  
    self.radiance = radiance
  
  def draw(self):
    if self.geometry == "env" or self.geometry == "NotImplemented":
      return
   
    self.geometry.draw()

  def intersect(self, ray):
    if not (self.mask & ray.mask):
      return Intersection()

    if self.geometry == "env":
      if ray.pos[1] < 0:
        return Intersection()
      
      normal = Vector(-ray.pos[0], -ray.pos[1])
      tangent = normal.getTangent()
      ray.t = SCENE_BOUND - 1
      return Intersection(True, normal, tangent)

    return self.geometry.intersect(ray)
  
  def sample(self, sampleArray):
    if self.geometry == "env":
      raise Exception("Sampling env-light is not implemented")

    return self.geometry.sample(sampleArray)
  
class Scene(Drawable, Intersectable):
  objects = None
  name = None
  
  def __init__(self, name="A scene"):
    self.objects = [] 
    self.name = name

  def append(self, o):
    self.objects.append(o)
    
  def draw(self):
    for o in self.objects:
      o.draw()
    
  def intersect(self, ray):
    min_t = SCENE_BOUND
    intersection = Intersection()
    retObj = None
    for o in self.objects:
      i = o.intersect(ray)
      if i.hit and ray.t < min_t and ray.t > 1e-10:
        min_t = ray.t
        intersection = i
        retObj = o

    ray.t = min_t
    return (intersection, retObj)    
  
def angleToRays(angles, intersection, mask=0xff):
  cosTheta = np.cos(angles)
  sinTheta = np.sin(angles)

  xWorld = intersection.tangent.pos[0] * sinTheta + intersection.normal.pos[0] * cosTheta
  yWorld = intersection.tangent.pos[1] * sinTheta + intersection.normal.pos[1] * cosTheta
  
  secondaryRays = []
  
  for i in range(angles.shape[0]):
    secondaryRays.append(Ray(intersection.intersection, Vector(xWorld[i], yWorld[i]), mask=mask))

  return secondaryRays

# returns an array of angle(s) and weights
# Note that angles are between -pi/2 to pi/2
def sample(type, nSamples):
  if type == "uniform":
    angles = np.arange(0.5 * np.pi/nSamples, np.pi - 0.0001, np.pi/nSamples) - np.pi/2.0
    return (angles, np.ones(angles.shape) * np.pi/nSamples)
  if type == "uniformRandom":
    angles = np.random.uniform(0, 1, nSamples) * np.pi  - np.pi / 2.0
    return (angles, np.ones(angles.shape) * np.pi/ nSamples)
  if type == "cosine":
    randList = np.random.uniform(0, 1, nSamples)
    angles = np.arcsin(2*randList-1.0)
    return (angles, np.ones(angles.shape) * 2.0 / (np.cos(angles) * nSamples))
  else:
    print("Sampler type not found")

# def shade(cameraRayPayload):
#   cameraRayPayload.camRay.draw()
#   (intersection, obj) = scene.intersect(cameraRayPayload.camRay)
  
#   if not intersection.hit:
#     cameraRayPayload.value = 0.0
#     return
  
#   cameraRayPayload.primaryHitPoint = intersection.intersection

#   if isinstance(obj, Light):
#     cameraRayPayload.value = o.radiance
#     return

#   (angles, weights) = sample("uniform", 20)
#   secondaryRays = angleToRays(angles, intersection)
 
#   value = 0
#   for i in range(len(secondaryRays)):
#     (iSec, oSec) = scene.intersect(secondaryRays[i])
#     #secondaryRays[i].draw()
#     if isinstance(oSec, Light):
#        brdfEval = obj.material.eval(intersection.normal, Vector(-cameraRayPayload.camRay.pos[0], -cameraRayPayload.camRay.pos[1]),  Vector(secondaryRays[i].pos[0], secondaryRays[i].pos[1]))
#        value = value + oSec.radiance * brdfEval * weights[i] 
  
#   cameraRayPayload.value = value

# ## Setup scene
# camera = PerspectiveCamera(Vector(-300, 70), Vector(2, -2), 60, 10, 1000, 20)
# camera.draw()
# cameraRays = camera.generateRays(100)

# scene = Scene()
# scene.append(Line(Point(-100,-75), Point(100, -50)))
# scene.append(Light(radiance=0.4)) #Environment light
# #scene.append(Light("line", Point(0, 75), 100, 0))
# scene.append(Line(Point(-100,-100), Point(100, -100)))
# scene.append(Line(Point(-100,-75), Point(100, -100)))
# scene.append(Line(Point(-100, 100), Point(100,  100), flipNormal=True))
# scene.draw()

# ## Raytrace
# for iPixel in cameraRays:
#   for c in iPixel:
#     shade(c)

# ## Collect radiance and convert into image
# image = np.zeros([len(cameraRays)])

# for iPixel in range(len(cameraRays)):
#   pixelSamples = cameraRays[iPixel]
#   spp = len(pixelSamples)

#   for c in pixelSamples:
#     image[iPixel] = image[iPixel] + c.value / spp

# plt.plot(image)
# plt.show()

# tl.done()