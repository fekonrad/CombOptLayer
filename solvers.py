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

#---------------------------------------------------------------------

import numpy as np

def valid_node(node, size_of_grid):
    """Checks if node is within the grid boundaries."""
    if node[0] < 0 or node[0] >= size_of_grid:
        return False
    if node[1] < 0 or node[1] >= size_of_grid:
        return False
    return True

def up(node):
    return (node[0]-1,node[1])

def down(node):
    return (node[0]+1,node[1])

def left(node):
    return (node[0],node[1]-1)

def right(node):
    return (node[0],node[1]+1)

def backtrack(distances):
    initial_node = [0,0]
    desired_node = [weights.shape[0],weights.shape[0]]
    # idea start at the last node then choose the least number of steps to go back
    # last node
    path = [desired_node]

    size_of_grid = distances.shape[0]

    while True:
        # check up down left right - choose the direction that has the least distance
        potential_distances = []
        potential_nodes = []

        directions = [up,down,left,right]

        for direction in directions:
            node = direction(path[-1])
            if valid_node(node, size_of_grid):
                potential_nodes.append(node)
                potential_distances.append(distances[node[0],node[1]])

        least_distance_index = np.argmin(potential_distances)
        path.append(potential_nodes[least_distance_index])

        if path[-1][0] == initial_node[0] and path[-1][1] == initial_node[1]:
            break

    return list(reversed(path))

def dijkstra(weights):
    initial_node = [0,0]
    desired_node = [weights.shape[0],weights.shape[0]]
    """Dijkstras algorithm for finding the shortest path between two nodes in a graph.

    Args:
        initial_node (list): [row,col] coordinates of the initial node
        desired_node (list): [row,col] coordinates of the desired node
        weights (array 2d): 2d numpy array of grid vertex weights

    Returns:
        list[list]: list of list of nodes that form the shortest path
    """
    # initialize vertex weights
    weights = weights.copy()
    # source and destination are free
    weights[initial_node[0],initial_node[1]]
    weights[desired_node[0],desired_node[1]]


    # initialize maps for distances and visited nodes
    size_of_floor = weights.shape[0]

    # we only want to visit nodes once
    visited = np.zeros([size_of_floor,size_of_floor],bool)

    # initiate matrix to keep track of distance to source node
    # initial distance to nodes is infinity so we always get a lower actual distance
    distances = np.ones([size_of_floor,size_of_floor]) * np.inf
    # initial node has a distance of 0 to itself
    distances[initial_node[0],initial_node[1]] = 0

    # start algorithm
    current_node = [initial_node[0], initial_node[1]]
    while True:

        directions = [up, down, left, right]
        for direction in directions:
            potential_node = direction(current_node)
            if valid_node(potential_node, size_of_floor): # boundary checking
                if not visited[potential_node[0],potential_node[1]]: # check if we have visited this node before
                    # update distance to node
                    distance = distances[current_node[0], current_node[1]] + weights[potential_node[0],potential_node[1]]

                    # update distance if it is the shortest discovered
                    if distance < distances[potential_node[0],potential_node[1]]:
                        distances[potential_node[0],potential_node[1]] = distance


        # mark current node as visited
        visited[current_node[0]  ,current_node[1]] = True

        # select next node
        # by choosing the the shortest path so far
        t=distances.copy()
        # we don't want to visit nodes that have already been visited
        t[np.where(visited)]=np.inf
        # choose the shortest path
        node_index = np.argmin(t)

        # convert index to row,col.
        node_row = node_index//size_of_floor
        node_col = node_index%size_of_floor
        # update current node.
        current_node = (node_row, node_col)

        # stop if we have reached the desired node
        if current_node[0] == desired_node[0] and current_node[1]==desired_node[1]:
            break

    # backtrack to construct path
    return backtrack(distances)



### Usage:
'''
import matplotlib.pyplot as plt
weights = np.array([[2,0,1,4,0,1,3,2,0,0],
                      [0,0,0,0,0,2,0,3,0,0],
                      [0,0,0,0,1,0,0,0,0,0],
                      [0,0,2,1,0,1,0,0,0,0],
                      [1,1,1,1,0,1,1,0,4,0],
                      [0,0,0,1,0,1,0,0,0,0],
                      [0,0,0,0,0,1,1,0,0,0],
                      [0,0,2,0,0,1,0,0,3,0],
                      [0,0,0,0,1,0,0,0,0,0],
                      [0,0,2,0,2,0,0,4,0,0]], dtype=float)
path = dijkstra(obstacles)
print(path)

    
p = np.zeros(shape=obstacles.shape)
for i in range(len(path)):
    p[path[i][0],path[i][1]] = np.nan

plt.imshow(p+obstacles, cmap='jet')
plt.show()
'''
