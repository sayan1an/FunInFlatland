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
  x = 0.0
  y = 0.0
  
  def __init__(self, x, y):
    self.x = x
    self.y = y
  
  def sub(self, point):
    return Point(self.x - point.x, self.y - point.y)
  
  def add(self, point):
    return Point(self.x + point.x, self.y + point.y)

  def dot(self, point):
    return self.x * point.x + self.y * point.y
 
  def scale(self, scalar):
      return Point(self.x * scalar, self.y * scalar)

  def length(self):
      return np.sqrt(self.x**2 + self.y**2)

  def draw(self):
    pen = tl.Turtle()
    pen.hideturtle()
    pen.width(self.size)
    pen.color(self.color)
    pen.up()
    pen.goto(self.x, self.y)
    pen.speed(0)
    pen.down()
    pen.dot()

class Vector(Point):
  t = 0.0
  origin = Point(0.0, 0.0)
    
  def __init__(self, x, y):
    self.color = "green"
    self.x = x
    self.y = y
    self.t = 50 * self.length()
  
  def normalize(self):
      l = self.length()
      self.x = self.x / l
      self.y = self.y / l
      self.t = 50

  def draw(self):
    pen = tl.Turtle()
    pen.shape("classic")
    pen.width(self.size)
    pen.color(self.color)
    pen.up()
    pen.goto(self.origin.x, self.origin.y)
    pen.speed(0)
    pen.down()
    pen.setheading(np.angle([self.x + self.y * 1.0j], deg=True))
    pen.forward(self.t)
 
  def scale(self, vec):
    v = super().scale(vec)
    return Vector(v.x, v.y)

  def add(self, vec):
    v = super().add(vec)
    return Vector(v.x, v.y)

  def sub(self, vec):
    v = super().sub(vec)
    return Vector(v.x, v.y)

class Ray(Vector):
  def __init__(self, origin, direction):
    self.color = "blue"
    self.origin.x = origin.x
    self.origin.y = origin.y
    self.x = direction.x
    self.y = direction.y
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
  intersection = Point(0, 0)
  normal = Vector(0, 0)
  tangent = Vector(0, 0)
    
  def __init__(self, hit=False , intersection=Point(0,0), normal=Vector(0,0), tangent=Vector(0,0)):
    self.hit = hit
    self.intersection = intersection
    self.normal = normal
    self.tangent = tangent

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
    normal = Vector(-v.y, v.x)
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
    pen.goto(self.start.x, self.start.y)
    pen.speed(0)
    pen.down()
    pen.goto(self.end.x, self.end.y)
    self.drawNormal()
    
  def intersect(self, ray):
    v1 = ray.origin.sub(self.start)
    v2 = self.end.sub(self.start)
    v3 = Point(-ray.y, ray.x)

    t2 = v1.dot(v3) / v2.dot(v3)
    if t2 < 0 or t2 > 1:
        return Intersection()

    intersect = self.start.add(v2.scale(t2))
    ray.t = intersect.sub(ray.origin).length()
    
    tangent = Vector(v2.x, v2.y)
    tangent.normalize()

    normal = Vector(-v2.y, v2.x)
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

r = Ray(Point(0,0), Point(-7,-5))

scene = Scene()

scene.append(Line(Point(-100,-75), Point(100, -50)))
scene.append(Line(Point(-100,-100), Point(100, -100)))
scene.append(Line(Point(-100,-75), Point(100, -100)))
scene.append(Line(Point(-100, 100), Point(100,  100), True))
scene.draw()
i = scene.intersect(r)
i.draw()
r.draw()

tl.done()
  