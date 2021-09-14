from misio.pacman.learningAgents import ReinforcementAgent
from misio.pacman.util import CustomCounter
from pacquiao_extractor import PacQuiaoExtractor
from misio.pacman.game import Actions
import random
import numpy as np

class PacQuiaoAgent(ReinforcementAgent):

    def __init__(self,
                 epsilon=0.25,
                 gamma=0.8,
                 alpha=0.001,
                 numTraining=200,
                 extractor=PacQuiaoExtractor(),
                 num_ghost_w=-10.0,
                 eats_food_w=15.0,
                 scared_ghosts_w = 5.0,
                 closest_direction_w = 20.0,
                 **args):

        self.featExtractor = extractor
        self.index = 0  # This is always Pacman
        self.weights = self.initializeWeights(num_ghost_w, eats_food_w,
                                              scared_ghosts_w, closest_direction_w)
        self.actionToClosestFood = None
        self.Q = {}
        ReinforcementAgent.__init__(self, epsilon=epsilon,
                                    gamma=gamma,
                                    alpha=alpha,
                                    numTraining=numTraining)

    def calculateDirectionOfGoingToClosestFood(self,state):

        def getVertexWithMinDist(vertices):
            min_dist = np.inf
            vertex = None
            for vert in vertices:
                if distance[vert] < min_dist:
                    min_dist = distance[vert]
                    vertex = vert
            return vertex

        pac_x, pac_y = state.getPacmanPosition()
        current_food = state.getFood()
        walls = state.getWalls()
        closest_food, closest_food_position = self.featExtractor.calcShortestDistanceFromFood(pac_x, pac_y, current_food)
        distance = {}
        previous = {}
        vertices = []
        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y]:
                    distance[(x,y)] = np.inf
                    previous[(x,y)] = None
                    vertices.append((x,y))
        distance[(pac_x,pac_y)] = 0
        end = False

        while len(vertices) > 0 and (end==False):
            u = getVertexWithMinDist(vertices)
            vertices.remove(u)
            if u == closest_food_position:
                end = True
                break

            neighbours = Actions.getLegalNeighbors(u, walls)
            new_dist = distance[u] + 1
            for nei in neighbours:
                if nei in vertices and new_dist < distance[nei]:
                    distance[nei]=new_dist
                    previous[nei]=u

        while previous[u] != (pac_x,pac_y):
            u = previous[u]
        vector = (u[0] - pac_x,u[1] - pac_y)

        return Actions.vectorToDirection(vector)

    def initializeWeights(self, num_ghost_w, eats_food_w, scared_ghosts_w, closest_direction_w):
        weights = CustomCounter()
        weights["num-ghost-in-next-pos"] = num_ghost_w
        weights["eats-food"] = eats_food_w
        weights["scared-ghosts"] = scared_ghosts_w
        weights["closest-food-direction"] = closest_direction_w

        return weights

    def getQValue(self, state, action):
        feature_state, feature_values = self.featExtractor.getFeaturesAndFeatureState(
            state, action, self.actionToClosestFood)
        if feature_state in self.Q:
            return self.Q[feature_state]
        else:
            new_value = self.computeQValue(feature_values)
            self.Q[feature_state] = new_value
            return new_value

    def computeValueFromQValues(self, state):

        legal_actions = self.getLegalActions(state)
        value = -np.inf
        if len(legal_actions) > 0:
            for action in legal_actions:
                new_val = self.getQValue(state, action)
                if new_val > value:
                    value = new_val
        else:
            return 0.0

        return value

    def computeActionFromQValues(self, state):

        action = None
        legal_actions = self.getLegalActions(state)
        if (len(legal_actions) > 0):
            value = -np.inf
            for act in legal_actions:
                new_val = self.getQValue(state, act)
                if new_val > value:
                    value = new_val
                    action = act
        return action

    def getAction(self, state):

        action = None
        legalActions = self.getLegalActions(state)

        if (len(legalActions) > 0):
            if (random.random() < self.epsilon):
                action = random.choice(legalActions)
            else:
                self.actionToClosestFood = self.calculateDirectionOfGoingToClosestFood(state)
                action = self.computeActionFromQValues(state)

        self.doAction(state, action)
        return action

    def update(self, state, action, nextState, reward):

        feature_state, feature_values = self.featExtractor.getFeaturesAndFeatureState(
            state, action, self.actionToClosestFood)
        q_value = self.computeQValue(feature_values)

        self.actionToClosestFood = self.calculateDirectionOfGoingToClosestFood(nextState)

        q_target = reward + self.discount * self.computeValueFromQValues(nextState)
        difference = q_target - q_value

        self.Q[feature_state] = q_value + self.alpha * difference
        self.updateWeights(feature_values, difference)

    def updateWeights(self, feature_values, difference):
        for key in feature_values:
            self.weights[key] += self.alpha * difference * feature_values[key]

    def computeQValue(self, feature_values):
        res = 0
        for key in self.weights:
            res += feature_values[key] * self.weights[key]
        return res

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getWeights(self):
        return self.weights

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        ReinforcementAgent.final(self, state)

        if self.episodesSoFar % 50 == 0:
            print(self.weights, len(self.Q))
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
        #     you might want to print your weights here for debugging
            print("WEIGHTS")
            print(self.weights, len(self.Q))
