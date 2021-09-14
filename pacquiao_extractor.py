from misio.pacman.featureExtractors import FeatureExtractor
from misio.pacman.game import Actions
from misio.pacman.util import CustomCounter
import numpy as np

class PacQuiaoExtractor(FeatureExtractor):

    def __init__(self):
        self.upper_bound = 9
        self.max_func_value = 1
        self.will_eat_reward = 1
        self.half_solid_area = 3

    def getFeatures(self, state, action):
        pass

    def getFeaturesAndFeatureState(self, state, action, actionToClosestFood):
        features = CustomCounter()
        num_ghosts_in_future_position, will_eat, scared_ghosts = self.calcFeatures(state, action)
        #features["closest-ghost"] = 1.0 / closest_ghost if closest_ghost != 0 else self.max_func_value
        features["num-ghost-in-next-pos"] = num_ghosts_in_future_position * (1 - scared_ghosts)#if scared_ghosts == 0.0 else 0
        features["eats-food"] = will_eat
        features["scared-ghosts"] = scared_ghosts
        features["closest-food-direction"] = int(actionToClosestFood == action)
        #features["closest-food"] = 1.0 / closest_food if closest_food != 0 else self.max_func_value
        #features["freedom-degree"] = freedom_degree
        #features["closest-capsule"] = 1.0 / closest_capsule if closest_capsule != 0 else self.max_func_value
        #return (closest_food, num_ghosts_in_future_position, will_eat, scared_ghosts , action), features
        #print(closest_food_direction,action,features["closest-food-direction"])

        return ( num_ghosts_in_future_position, will_eat, scared_ghosts, actionToClosestFood, action), features

    def getFeatureState(self, state, action, actionToClosestFood):
        closest_ghost, num_ghosts_in_future_position, will_eat, scared_ghosts = self.calcFeatures(state, action)
        return (closest_ghost, num_ghosts_in_future_position, will_eat, scared_ghosts, actionToClosestFood, action)

    def calcFeatures(self, state, action):
        xp, yp = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        pac_x, pac_y = int(xp + dx), int(yp + dy)

        ghost_positions = state.getGhostPositions()
        ghost_states = state.getGhostStates()
        current_food = state.getFood()
        walls = state.getWalls()
        capsule_positions = state.getCapsules()

        #closest_ghost, closest_ghost_state = self.calcTraitsOfClosestGhost(pac_x, pac_y, ghost_states)
        #scared_ghosts = self.willGhostsBeScared(pac_x,pac_y,capsule_positions, ghost_states)
        #closest_ghost = min(self.upper_bound, closest_ghost)
        num_ghosts_in_future_position = sum((pac_x, pac_y) in Actions.getLegalNeighbors(g, walls) for g in ghost_positions)
        will_eat = self.will_eat_reward if current_food[pac_x][pac_y] == True else 0.0
        ratio_scared_ghosts = self.calcRatioScaredGhosts(ghost_states)

        # closest_food, closest_food_position = self.calcShortestDistanceFromFood(pac_x, pac_y, current_food)
        # closest_capsule = self.calcShortestDistanceFromObject(pac_x, pac_y, capsule_positions)
        # closest_food = min(self.upper_bound, closest_food)
        #closest_capsule = min(self.upper_bound, closest_capsule)
        # num_food = self.calcNumFoodInCloseArea(pac_x, pac_y, current_food)
        #freedom_degree = len(Actions.getLegalNeighbors((pac_x,pac_y), walls))

        return num_ghosts_in_future_position, will_eat, ratio_scared_ghosts

    def willGhostsBeScared(self,pac_x,pac_y,capsule_positions, ghost_states):
        if (len(capsule_positions)>0):
            for pos in capsule_positions:
                if (pac_x == pos[0] and pac_y == pos[1]):
                    return 1.0
        for gh in ghost_states:#any ghost
            if (gh.scaredTimer > 1):
                return 1.0
        return 0.0

    def calcRatioScaredGhosts(self, ghost_states):
        res = 0
        for gh in ghost_states:
            if gh.scaredTimer > 0:
                res += 1.0
        return res / len(ghost_states)

    def calcTraitsOfClosestGhost(self, pac_x, pac_y, ghost_states):
        min_dist = np.inf
        state = 1  # 1 - normal , 0 - scared
        for gh in ghost_states:
            pos_x, pos_y = gh.configuration.getPosition()
            distance = abs(pac_x - pos_x) + abs(pac_y - pos_y)
            if distance < min_dist:
                min_dist = distance
                if (gh.scaredTimer > 0):
                    state = 0
                else:
                    state = 1

        return min_dist, state

    def calcShortestDistanceFromObject(self, pac_x, pac_y, positions):
        min_dist = np.inf
        for pos in positions:
            distance = abs(pac_x - pos[0]) + abs(pac_y - pos[1])
            if distance < min_dist:
                min_dist = distance
        return min_dist

    def calcShortestDistanceFromFood(self, pac_x, pac_y, grid_food):
        min_dist = grid_food.width * grid_food.height
        closest_food_position = (None,None)
        for x in range(grid_food.width):
            for y in range(grid_food.height):
                if grid_food[x][y] == True:
                    distance = abs(pac_x - x) + abs(pac_y - y)
                    if distance < min_dist:
                        min_dist = distance
                        closest_food_position = (x,y)
        return min_dist, closest_food_position

    def calcNumFoodInCloseArea(self, pac_x, pac_y, grid_food):
        count = 0
        half = self.half_solid_area
        for x in range(max(pac_x - half, 0), min(pac_x + half + 1, grid_food.width)):
            for y in range(max(pac_y - half, 0), min(pac_y + half + 1, grid_food.height)):
                if grid_food[x][y] == True:
                    count += 1
        return count
