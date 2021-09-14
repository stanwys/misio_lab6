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
        features["num-ghost-in-next-pos"] = num_ghosts_in_future_position * (1 - scared_ghosts)#if scared_ghosts == 0.0 else 0
        features["eats-food"] = will_eat
        features["scared-ghosts"] = scared_ghosts
        features["closest-food-direction"] = int(actionToClosestFood == action)
        
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
        
        num_ghosts_in_future_position = sum((pac_x, pac_y) in Actions.getLegalNeighbors(g, walls) for g in ghost_positions)
        will_eat = self.will_eat_reward if current_food[pac_x][pac_y] == True else 0.0
        ratio_scared_ghosts = self.calcRatioScaredGhosts(ghost_states)

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
