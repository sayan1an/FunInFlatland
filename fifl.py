from abc import ABC, abstractmethod
import turtle as tl
import numpy as np

SCENE_BOUND = 10000

# Abstract class Drawable
# Base class for all drawable quantities
class Drawable(ABC):
  @abstractmethod
  def draw(self):
        pass

# Abstract class Intersectable
# Base class for any shape or object that implements ray-shape intersection 
class Intersectable(ABC):
  @abstractmethod
  def intersect(self, ray): # Method returns an Intersection object, the normals and tangents must be unit length
      pass
  
class Point(Drawable):
  size = 1.0
  color = "black"
  pos = None

  def __init__(self, x, y, z = 1):
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
    pen = tl.Turtle()
    pen.hideturtle()
    pen.width(self.size)
    pen.color(self.color)
    pen.up()
    pen.goto(self.pos[0], self.pos[1])
    pen.speed(0)
    pen.down()
    pen.dot()

class Vector(Point):
  t = 0.0
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
    pen = tl.Turtle()
    pen.shape("classic")
    pen.width(self.size)
    pen.color(self.color)
    pen.up()
    pen.goto(self.origin.pos[0], self.origin.pos[1])
    pen.speed(0)
    pen.down()
    pen.setheading(np.angle([self.pos[0] + self.pos[1] * 1.0j], deg=True))
    pen.forward(self.t)
 
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
  def __init__(self, origin, direction):
    self.color = "blue"
    self.origin = Point(origin.pos[0], origin.pos[1])
    self.pos = np.array([direction.pos[0], direction.pos[1], 1], dtype=float)
    self.normalize()
    self.t = 0
  
  def draw(self):
    if self.t > 0:
        self.origin.color = self.color
        self.origin.size = self.size
        self.origin.draw()
        super().draw()

class Intersection(Drawable):
  color = "red"
  size = 2
  hit = False
  
  intersection = None
  normal = None
  tangent = None

  def __init__(self, hit=False , intersection=Point(0,0), normal=Vector(0,0), tangent=Vector(0,0)):
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

  def generateRays(self):
    rays = []
    
    # Transform point (0,0,1) in camera space to world space
    rayOrigin = np.dot(self.cameraToWorld, np.array([0, 0, 1]))
    for i in range(self.screenRes):
      temp = np.dot(self.screenToCamera, np.dot(self.rasterToScreen, np.array([i, -self.near * 5, 1])))
      temp[0] = temp[0] / temp[2]
      temp[1] = temp[1] / temp[2]
      temp[2] = 0

      rayDir = np.dot(self.cameraToWorld, temp)
      rays.append(Ray(Point(rayOrigin[0], rayOrigin[1]), Vector(rayDir[0], rayDir[1])))
    
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

class Line(Drawable, Intersectable):
  flipNormal = False
  size = 3.0
  color = "magenta"
  start = Point(0.0, 0.0)
  end = Point(0.0, 0.0)
  
  def __init__(self, start, end, flipNormal = False):
    self.flipNormal = flipNormal
    self.start = start
    self.end = end
      
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
    pen = tl.Turtle()
    pen.hideturtle()
    pen.width(self.size)
    pen.color(self.color)
    pen.up()
    pen.goto(self.start.pos[0], self.start.pos[1])
    pen.speed(0)
    pen.down()
    pen.goto(self.end.pos[0], self.end.pos[1])
    
  def intersect(self, ray):
    v1 = ray.origin.sub(self.start)
    v2 = self.end.sub(self.start)
    v3 = Point(-ray.pos[1], ray.pos[0])
    
    t2 = v1.dot(v3) / v2.dot(v3)
    
    if t2 < 0 or t2 > 1:
        return Intersection()

    intersect = self.start.add(v2.scale(t2))
    ray.t = intersect.sub(ray.origin).length()
    
    tangent = Vector(v2.pos[0], v2.pos[1])
    tangent.normalize()

    normal = tangent.getTangent()
    normal.normalize()
  
    if self.flipNormal:
        normal = normal.scale(-1)
    return Intersection(True, intersect, normal, tangent)

class Scene(Drawable, Intersectable):
    objects = []

    def append(self, o):
        self.objects.append(o)
    
    def draw(self):
        for o in self.objects:
            o.draw()
    
    def intersect(self, ray):
      min_t = SCENE_BOUND
      intersection = Intersection()
      for o in self.objects:
        i = o.intersect(ray)
        if i.hit and ray.t < min_t:
            min_t = ray.t
            intersection = i

      ray.t = min_t
      return intersection    
  
tl.Screen().title("2D Renderer")

c = PerspectiveCamera(Vector(-300, 70), Vector(2, -2), 60, 10, 1000, 20)
c.draw()
rays = c.generateRays()

scene = Scene()
#scene.append(Line(Point(-100,-75), Point(100, -50)))
scene.append(Line(Point(-100,-100), Point(100, -100)))
scene.append(Line(Point(-100,-75), Point(100, -100)))
scene.append(Line(Point(-100, 100), Point(100,  100), True))
scene.draw()

for r in rays:
  scene.intersect(r)
  r.draw()

tl.done()
  