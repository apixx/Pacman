import bisect
import sys


class Problem:
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        raise NotImplementedError

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def value(self):
        raise NotImplementedError

    def h(self, node):
        raise NotImplementedError


class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):

        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0  # search depth
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):

        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):

        next_state = problem.result(self.state, action)
        return Node(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next_state))

    def solution(self):

        return [node.action for node in self.path()[1:]]

    def solve(self):

        return [node.state for node in self.path()[0:]]

    def path(self):

        x, result = self, []
        while x:
            result.append(x)
            x = x.parent
        result.reverse()
        return result

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


class Queue:

    def __init__(self):
        raise NotImplementedError

    def append(self, item):
        raise NotImplementedError

    def extend(self, items):
        raise NotImplementedError

    def pop(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __contains__(self, item):
        raise NotImplementedError


class Stack(Queue):

    def __init__(self):
        self.data = []

    def append(self, item):
        self.data.append(item)

    def extend(self, items):
        self.data.extend(items)

    def pop(self):
        return self.data.pop()

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return item in self.data


class FIFOQueue(Queue):

    def __init__(self):
        self.data = []

    def append(self, item):
        self.data.append(item)

    def extend(self, items):
        self.data.extend(items)

    def pop(self):
        return self.data.pop(0)

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return item in self.data


class PriorityQueue(Queue):

    def __init__(self, order=min, f=lambda x: x):

        assert order in [min, max]
        self.data = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort_right(self.data, (self.f(item), item))

    def extend(self, items):
        for item in items:
            bisect.insort_right(self.data, (self.f(item), item))

    def pop(self):
        if self.order == min:
            return self.data.pop(0)[1]
        return self.data.pop()[1]

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.data)

    def __getitem__(self, key):
        for _, item in self.data:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.data):
            if item == key:
                self.data.pop(i)


def tree_search(problem, fringe):
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        print(node.state)
        if problem.goal_test(node.state):
            return node
        fringe.extend(node.expand(problem))
    return None


def memoize(fn, slot=None):
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if args not in memoized_fn.cache:
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}
    return memoized_fn


def best_first_graph_search(problem, f):
    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None


def astar_search(problem, h=None):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


def breadth_first_tree_search(problem):
    return tree_search(problem, FIFOQueue())


def depth_first_tree_search(problem):
    return tree_search(problem, Stack())


def graph_search(problem, fringe):
    closed = set()
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        if problem.goal_test(node.state):
            return node
        if node.state not in closed:
            closed.add(node.state)
            fringe.extend(node.expand(problem))
    return None


def breadth_first_graph_search(problem):
    return graph_search(problem, FIFOQueue())


def depth_first_graph_search(problem):
    return graph_search(problem, Stack())


def depth_limited_search(problem, limit=50):
    def recursive_dls(node, problem, limit):

        cutoff_occurred = False
        if problem.goal_test(node.state):
            return node
        elif node.depth == limit:
            return 'cutoff'
        else:
            for successor in node.expand(problem):
                result = recursive_dls(successor, problem, limit)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
        if cutoff_occurred:
            return 'cutoff'
        return None

    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result


def uniform_cost_search(problem):
    return graph_search(problem, PriorityQueue(min, lambda a: a.path_cost))


class Pacman(Problem):
    def __init__(self, obstacles, initial, goal=None):
        super().__init__(initial, goal)
        self.obstacles = obstacles

    def successor(self, state):

        successors = dict()
        pacman_x = state[0]
        pacman_y = state[1]
        direction = state[2]
        goals = state[3]

        if direction == "west":
            if pacman_x - 1 >= 0 and (pacman_x - 1, pacman_y) not in self.obstacles:
                food = list(goals)
                if (pacman_x - 1, pacman_y) in food:
                    food.remove((pacman_x - 1, pacman_y))
                newSide = 'west'
                successors['ContinueForward'] = (pacman_x - 1, pacman_y, newSide, tuple(food))
            if pacman_x + 1 < 10 and (pacman_x + 1, pacman_y) not in self.obstacles:
                food = list(goals)
                if (pacman_x + 1, pacman_y) in food:
                    food.remove((pacman_x + 1, pacman_y))
                newSide = 'east'
                successors['ContinueBackward'] = (pacman_x + 1, pacman_y, newSide, tuple(food))

            if 0 <= pacman_y - 1 and (pacman_x, pacman_y - 1) not in self.obstacles:
                food = list(goals)
                if (pacman_x, pacman_y - 1) in food:
                    food.remove((pacman_x, pacman_y - 1))
                newSide = 'south'
                successors['TurnLeft'] = (pacman_x, pacman_y - 1, newSide, tuple(food))

            if pacman_y + 1 < 10 and (pacman_x, pacman_y + 1) not in self.obstacles:
                food = list(goals)
                if (pacman_x, pacman_y + 1) in food:
                    food.remove((pacman_x, pacman_y + 1))
                newSide = 'north'
                successors['TurnRight'] = (pacman_x, pacman_y + 1, newSide, tuple(food))

        if direction == "east":
            if pacman_x + 1 < 10 and (pacman_x + 1, pacman_y) not in self.obstacles:
                food = list(goals)
                if (pacman_x + 1, pacman_y) in food:
                    food.remove((pacman_x + 1, pacman_y))
                newSide = 'east'
                successors['ContinueForward'] = (pacman_x + 1, pacman_y, newSide, tuple(food))
            if pacman_x - 1 >= 0 and (pacman_x - 1, pacman_y) not in self.obstacles:
                food = list(goals)
                if (pacman_x - 1, pacman_y) in food:
                    food.remove((pacman_x - 1, pacman_y))
                newSide = 'west'
                successors['ContinueBackward'] = (pacman_x - 1, pacman_y, newSide, tuple(food))
            if pacman_y + 1 < 10 and (pacman_x, pacman_y + 1) not in self.obstacles:
                food = list(goals)
                if (pacman_x, pacman_y + 1) in food:
                    food.remove((pacman_x, pacman_y + 1))
                newSide = 'north'
                successors['TurnLeft'] = (pacman_x, pacman_y + 1, newSide, tuple(food))
            if pacman_y - 1 >= 0 and (pacman_x, pacman_y - 1) not in self.obstacles:
                food = list(goals)
                if (pacman_x, pacman_y - 1) in food:
                    food.remove((pacman_x, pacman_y - 1))
                newSide = 'south'
                successors['TurnRight'] = (pacman_x, pacman_y - 1, newSide, tuple(food))

        if direction == "south":
            if pacman_y - 1 >= 0 and (pacman_x, pacman_y - 1) not in self.obstacles:
                food = list(goals)
                if (pacman_x, pacman_y - 1) in food:
                    food.remove((pacman_x, pacman_y - 1))
                newSide = 'south'
                successors['ContinueForward'] = (pacman_x, pacman_y - 1, newSide, tuple(food))
            if pacman_y + 1 < 10 and (pacman_x, pacman_y + 1) not in self.obstacles:
                food = list(goals)
                if (pacman_x, pacman_y + 1) in food:
                    food.remove((pacman_x, pacman_y + 1))
                newSide = 'north'
                successors['ContinueBackward'] = (pacman_x, pacman_y + 1, newSide, tuple(food))
            if pacman_x + 1 < 10 and (pacman_x + 1, pacman_y) not in self.obstacles:
                food = list(goals)
                if (pacman_x + 1, pacman_y) in food:
                    food.remove((pacman_x + 1, pacman_y))
                newSide = 'east'
                successors['TurnLeft'] = (pacman_x + 1, pacman_y, newSide, tuple(food))

            if pacman_x - 1 >= 0 and (pacman_x - 1, pacman_y) not in self.obstacles:
                food = list(goals)
                if (pacman_x - 1, pacman_y) in food:
                    food.remove((pacman_x - 1, pacman_y))
                newSide = 'west'
                successors['TurnRight'] = (pacman_x - 1, pacman_y, newSide, tuple(food))

        if direction == "north":
            if pacman_y + 1 < 10 and (pacman_x, pacman_y + 1) not in self.obstacles:
                food = list(goals)
                if (pacman_x, pacman_y + 1) in food:
                    food.remove((pacman_x, pacman_y + 1))
                newSide = 'north'
                successors['ContinueForward'] = (pacman_x, pacman_y + 1, newSide, tuple(food))
            if pacman_y - 1 >= 0 and (pacman_x, pacman_y - 1) not in self.obstacles:
                food = list(goals)
                if (pacman_x, pacman_y - 1) in food:
                    food.remove((pacman_x, pacman_y - 1))
                newSide = 'south'
                successors['ContinueBackward'] = (pacman_x, pacman_y - 1, newSide, tuple(food))
            if pacman_x - 1 >= 0 and (pacman_x - 1, pacman_y) not in self.obstacles:
                food = list(goals)
                if (pacman_x - 1, pacman_y) in food:
                    food.remove((pacman_x - 1, pacman_y))
                newSide = 'west'
                successors['TurnLeft'] = (pacman_x - 1, pacman_y, newSide, tuple(food))

            if pacman_x + 1 < 10 and (pacman_x + 1, pacman_y) not in self.obstacles:
                food = list(goals)
                if (pacman_x + 1, pacman_y) in food:
                    food.remove((pacman_x + 1, pacman_y))
                newSide = 'east'
                successors['TurnRight'] = (pacman_x + 1, pacman_y, newSide, tuple(food))

        return successors

    def actions(self, state):

        return self.successor(state).keys()

    def result(self, state, action):

        return self.successor(state)[action]

    def goal_test(self, state):

        listDots = state[3]
        return len(listDots) == 0

    def path_cost(self, c, state1, action, state2):

        return c + 1

    def value(self):

        raise NotImplementedError

    def h(self, node):
        goal = node.state[3]
        pac_x = node.state[0]
        pac_y = node.state[1]
        ind = 0
        res = 0
        p = {}
        for i in range(len(goal)):
            p[ind] = goal[i]
        for v in p.values():
            x = v[0]
            y = v[1]
            menheten = abs(pac_x - x) + abs(pac_y - y)
            res += menheten

        return res


if __name__ == '__main__':

    x = int(input())
    y = int(input())
    direction = input()
    n = int(input())
    array = []
    for i in range(n):
        food = [int(element) for element in input().split(",")]
        array.append(food)

    for i in range(n):
        array[i] = tuple(array[i])

    array = tuple(array)

    obstacles = [(0, 7), (1, 7), (2, 5), (3, 0), (3, 1), (3, 3), (3, 4), (3, 5), (4, 3), (5, 3), (6, 3), (6, 4), (6, 5),
                 (6, 6), (7, 4), (8, 4)]

    pacman = Pacman(obstacles, (x, y, direction, array))
    result = astar_search(pacman)
    print(result.solution())

"""
{Input Example}
0
0
east
6
0,6
0,8
2,4
4,0
5,4
7,3"""
