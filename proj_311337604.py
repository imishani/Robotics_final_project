"""
requierments:
packages:
    networkx (using pip:  'pip install networkx')
    seaborn
    math
"""

from matplotlib import pyplot as plt
import networkx as nx
import seaborn as sns
import math
import random

x_s = [1.01, 1.2] #1.01, 1.2
x_g = [7.9, 8.3]#[7.9, 8.3]
'''
press the start button.. it will take a few seconds 
                '''


class RRT_star:

    def __init__(self, start, goal, obstacles, rand_area, robot_radius=0.35,
                 expand_dis=2., path_resolution=0.5, goal_sample_rate=15.0, max_iter=7000):
        self.G = nx.DiGraph()
        self.attr = {}  # parent, x path, y path
        self.robot = robot_radius
        self.start = start
        self.goal = goal
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.circle_obstacle = obstacles['circles']
        self.rectangle_obstacle = obstacles['rectangle']
        self.node_list = []
        if not self.collision_checker(self.goal, [self.goal[0]], [self.goal[1]]):
            raise ValueError('Input Error! goal point is not allowed!')
        if not self.collision_checker(self.start, [self.start[0]], [self.start[1]]):
            raise ValueError('Input Error! start point is not allowed!')

    def collision_checker(self, node, path_x, path_y):
        '''circle checker'''
        for (ox, oy, size) in self.circle_obstacle:
            dx_list = [ox - x for x in path_x]
            dy_list = [oy - y for y in path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
            if min(d_list) <= (size + self.robot) ** 2:
                return False

        '''rectangle checker'''
        for rec in self.rectangle_obstacle:
            for (xp, yp) in zip(path_x, path_y):
                if (rec[0] - self.robot) <= xp and (rec[2] + self.robot) >= xp and (rec[1] - self.robot) <= yp and (
                        rec[3] + self.robot) >= yp:
                    return False

        return True

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node[0] - from_node[0]
        dy = to_node[1] - from_node[1]
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def gen_new_node(self, from_node, to_node, extend_length=float('inf')):

        d, theta = self.calc_distance_and_angle(from_node, to_node)
        new_node_x, new_node_y = from_node[0], from_node[1]
        path_x = [new_node_x]
        path_y = [new_node_y]

        if extend_length > d:
            extend_length = d

        for _ in range(math.floor(extend_length / self.path_resolution)):
            new_node_x += self.path_resolution * math.cos(theta)
            new_node_y += self.path_resolution * math.sin(theta)
            path_x.append(new_node_x)
            path_y.append(new_node_y)

        d, _ = self.calc_distance_and_angle([new_node_x, new_node_y], to_node)
        if d <= self.path_resolution:
            path_x.append(to_node[0])
            path_y.append(to_node[1])
            new_node_x = to_node[0]
            new_node_y = to_node[1]

        return (new_node_x, new_node_y), from_node, path_x, path_y

    def generate_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rand = (random.uniform(self.min_rand, self.max_rand),
                    random.uniform(self.min_rand, self.max_rand))
        else:
            rand = self.goal
        return rand

    def get_nearest_node(self, rand_node):
        nodes_lst = list(self.G.nodes)
        distance = [(node[0] - rand_node[0]) ** 2 + (node[1] - rand_node[1]) ** 2 for node in nodes_lst]
        min_ind = distance.index(min(distance))
        return nodes_lst[min_ind]

    def find_neighbors_fixed_radius(self, new_node, r=3.0):
        lst = [node for node in list(self.G.nodes) if math.hypot(node[0] - new_node[0], node[1] - new_node[1]) <= r]
        if len(lst) > 0:
            return lst
        return None

    def choose_parent(self, new_node, neighbors):

        costs = []
        for n in neighbors:
            node, parent, path_x, path_y = self.gen_new_node(n, new_node)
            if self.collision_checker(node, path_x, path_y) and len(path_x) > 1:
                d, _ = self.calc_distance_and_angle(n, node)
                costs.append(d + self.G.node[n]['cost'])
            else:
                costs.append(float('inf'))

        minimum_cost = min(costs)
        if minimum_cost == float('inf'):
            print('There is no good path')
            return None, None
        parent = neighbors[costs.index(minimum_cost)]
        new_node, parent, path_x, path_y = self.gen_new_node(parent, new_node)  # , self.expand_dis
        if parent != self.goal:
            self.G.add_node(new_node, parent=parent, path=(path_x, path_y), cost=minimum_cost)
        return new_node, parent

    def rewire(self, new_node, neighbors):
        for near in neighbors:
            re_node, parent, path_x, path_y = self.gen_new_node(new_node, near)
            d, _ = self.calc_distance_and_angle(new_node, near)
            cost = d + self.G.node[near]['cost']
            if self.collision_checker(re_node, path_x, path_y) and self.G.node[near]['cost'] > cost:
                self.G = nx.relabel_nodes(self.G, {near: re_node})
                self.G.node[re_node]['cost'] = cost
                self.G.node[re_node]['parent'] = parent
                self.G.node[re_node]['path'] = (path_x, path_y)
                self.update_costs(new_node)

    def update_costs(self, parent):
        for node in list(self.G.nodes):
            if self.G.node[node]['parent'] == parent:
                d, _ = self.calc_distance_and_angle(parent, node)
                self.G.node[node]['cost'] = d + self.G.node[parent]['cost']
                self.update_costs(node)


    def path_planning(self):
        self.G.add_node(self.start, parent=None, cost=0)
        for i in range(self.max_iter):
            rand_node = self.generate_random_node()
            nearest = self.get_nearest_node(rand_node)
            new_node, parent, path_x, path_y = self.gen_new_node(nearest, rand_node, self.expand_dis)

            # create new node with attributes:
            if self.collision_checker(new_node, path_x, path_y):
                neighbors = self.find_neighbors_fixed_radius(new_node)
                if neighbors is None:
                    neighbors = [parent]
                new_node, parent = self.choose_parent(new_node, neighbors)
                if new_node is not None:
                    self.rewire(new_node, neighbors)


        print("last iteration")
        to_goal = [n for n in list(self.G.nodes) if
                   math.hypot(n[0] - self.goal[0], n[1] - self.goal[1]) <= self.expand_dis]
        free = []
        costs = []
        parents = []
        path = []
        for x in to_goal:
            new_node, parent, path_x, path_y = self.gen_new_node(x, self.goal)
            d, _ = self.calc_distance_and_angle(new_node, x)
            cost = d + self.G.node[x]['cost']
            if self.collision_checker(new_node, path_x, path_y) and x != self.goal:
                free.append(x)
                costs.append(cost)
                parents.append(parent)
                path.append((path_x, path_y))
        minimum = min(costs)
        min_ind = costs.index(minimum)
        self.G.node[self.goal]['parent'] = free[min_ind]
        self.G.node[self.goal]['path'] = path[min_ind]
        return self.G

print('Calculating path... It might take a while =)')
circles = [
    (2.4, 20, 1.2),
    (6.4, 20, 1.2),
    (10.4, 20, 1.2),
    (14.4, 20, 1.2),
    (18.4, 20, 1.2),
    (22.4, 20, 1.2),
    (12, 7, 1.5)
]

rectangles = [
    [2, 2.5, 3, 9],
    [3, 2.5,  13.5, 3.5],
    [15.5, 2.5, 21, 3.5],
    [21, 2.5, 22, 9],
    [8 ,9.25, 8.5, 14.75],
    [15.5 , 9.25, 16, 14.75],
    [8, 14.75, 16, 15.25]
]
rrt = RRT_star(start=tuple(x_s),
            goal=tuple(x_g),
            rand_area=[0.35, 23.65],
            obstacles={'circles': circles, 'rectangle': rectangles})
path = rrt.path_planning()


x, y =[x_g[0]], [x_g[1]]
parents = path.node[tuple(x_g)]['parent']

while parents != None:
    x.append(parents[0])
    y.append(parents[1])
    parents = path.node[parents]['parent']


sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(12, 12))
for node in list(path.nodes):
    if path.nodes[node]['parent'] != None:
        plt.plot(path.nodes[node]['path'][0], path.nodes[node]['path'][1], "-g", alpha=0.3)
circle1 = plt.Circle((2.4, 20), 1.2)
circle2 = plt.Circle((6.4, 20), 1.2)
circle3 = plt.Circle((10.4, 20), 1.2)
circle4 = plt.Circle((14.4, 20), 1.2)
circle5 = plt.Circle((18.4, 20), 1.2)
circle6 = plt.Circle((22.4, 20), 1.2)
circle7 = plt.Circle((12., 7), 1.5)
rectangle1 = plt.Rectangle((2, 2.5), 1, 6.5, edgecolor='#1f77b4')
rectangle2 = plt.Rectangle((3, 2.5), 10.5, 1, edgecolor='#1f77b4')
rectangle3 = plt.Rectangle((15.5, 2.5), 5.5, 1, edgecolor='#1f77b4')
rectangle4 = plt.Rectangle((21, 2.5), 1, 6.5, edgecolor='#1f77b4')
rectangle5 = plt.Rectangle((8, 9.25), 0.5, 5.5, edgecolor='#1f77b4')
rectangle6 = plt.Rectangle((15.5, 9.25), 0.5, 5.5, edgecolor='#1f77b4')
rectangle7 = plt.Rectangle((8, 14.75), 8, 0.5, edgecolor='#1f77b4')
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)
ax.add_artist(circle5)
ax.add_artist(circle6)
ax.add_artist(circle7)
ax.add_artist(rectangle1)
ax.add_artist(rectangle2)
ax.add_artist(rectangle3)
ax.add_artist(rectangle4)
ax.add_artist(rectangle5)
ax.add_artist(rectangle6)
ax.add_artist(rectangle7)
start = plt.Circle(tuple(x_s), 0.35, color='r')
ax.add_artist(start)
goal = plt.Circle(tuple(x_g), 0.35, color='r')
ax.add_artist(goal)
plt.text(x_s[0], x_s[1] - 1, 'Start', fontsize=13, fontweight='bold', bbox={'facecolor': '#1f77b4', 'alpha': 0.5, 'pad': 3})
plt.text(x_g[0], x_g[1] + 1, 'Goal', fontsize=13, fontweight='bold', bbox={'facecolor': '#1f77b4', 'alpha': 0.5, 'pad': 3})
ax.plot(x, y, 'm', alpha=0.3, lw=15)
plt.plot(x, y, 'k', linestyle='dashed')
plt.axvline(x=0, lw=10)
plt.axvline(x=24, lw=10)
plt.axhline(y=0, lw=10)
plt.axhline(y=24, lw=10)
plt.xlim(0, 24)
plt.ylim(0, 24)
plt.grid(True)
plt.show()
