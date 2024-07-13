import torch
import itertools

# convex hull of d unit vectors
def solver_Simplex(objective: torch.FloatTensor):
  n = len(list(objective))
  max = float('-inf')

  for i in range(n):
    vertex = torch.zeros(n)
    vertex[i] = 1
    if torch.dot(objective, vertex) > max:
      max = torch.dot(objective, vertex)
      best_point = vertex

  print("Max is ", float(max), ", attained by ", best_point.tolist())

# convex set is d-dim unit square
def solver_Square(objective: torch.FloatTensor):
  n = len(list(objective))
  max = float('-inf')
  vertices = list(itertools.product([0, 1], repeat=n))

  for vertex in vertices:
    vertex = torch.tensor(vertex, dtype=torch.float32)
    if torch.dot(objective, vertex) > max:
      max = torch.dot(objective, vertex)
      best_point = vertex

  print("Max is ", float(max), ", attained by ", best_point.tolist())

# convex hull of a given set of d-dim vectors, points is a list containing vectors (lists)
def solver_ConvHull(objective: torch.FloatTensor, points: list):
  max = float('-inf')
  
  for i in range(len(points)):
    vertex = torch.FloatTensor(points[i])
    if torch.dot(objective, vertex) > max:
      max = torch.dot(objective, vertex)
      best_point = torch.FloatTensor(points[i])

  print("Max is ", float(max), ", attained by ", best_point.tolist())
