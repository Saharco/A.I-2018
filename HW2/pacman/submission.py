import random, util
from game import Agent
from pacman import GameState
import numpy as np
import ghostAgents
from time import time
#     ********* Reflex agent- sections a and b *********

class OriginalReflexAgent(Agent):

    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current GameState (pacman.py) and the proposed action
        and returns a number, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return scoreEvaluationFunction(successorGameState)


class ReflexAgent(Agent):

    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current GameState (pacman.py) and the proposed action
        and returns a number, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
    """
    return gameState.getScore()


######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
    """

    The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

    A GameState specifies the full game state, including the food, capsules, agent configurations and more.
    Following are a few of the helper methods that you can use to query a GameState object to gather information about
    the present state of Pac-Man, the ghosts and the maze:

    gameState.getLegalActions():
    gameState.getPacmanState():
    gameState.getGhostStates():
    gameState.getNumAgents():
    gameState.getScore():
    The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
    """

    foodGrid = gameState.getFood()
    pacmanPosition = gameState.getPacmanPosition()

    # Minimize the returned value if pacman loses in this state
    for ghostPosition in gameState.getGhostPositions():
        if ghostPosition == gameState.getPacmanPosition():
            return 0


    ghostDists = sum(
        [util.manhattanDistance(ghost.getPosition(), pacmanPosition) for ghost in gameState.getGhostStates() if
         ghost.scaredTimer <= 0]) / gameState.getNumAgents()

    foodDistanceScore = 1.5 * max([foodGrid.height + foodGrid.width - util.manhattanDistance(xy, pacmanPosition) for xy in
                     foodGrid.asList()] + [0])

    gridSize = foodGrid.height * foodGrid.width

    foodScore = (foodGrid.height + foodGrid.width) * (gridSize - gameState.getNumFood())

    scaredAvg = 0
    if len(gameState.getGhostStates()) > 0:
        scaredAvg = sum([5 for ghost in gameState.getGhostStates() if ghost.scaredTimer > 0])

    result = gameState.getScore()\
             + foodDistanceScore \
             + len(gameState.getLegalActions())\
             + ghostDists\
             + foodScore\
             + scaredAvg\
             + random.normalvariate(0, 2)

    return result


#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def isTerminal(self, gameState, currentDepth):
        if gameState.isWin() or gameState.isLose() or len(
                gameState.getLegalActions()) == 0 or self.depth < currentDepth:
            return True


######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent
    """

    def getActionAux(self, gameState, agentIndex: int, currentDepth, action):
        if self.isTerminal(gameState, currentDepth):
            return self.evaluationFunction(gameState), action
        if agentIndex == 0:
            if currentDepth == 0:
                return max([self.getActionAux(gameState.generateSuccessor(agentIndex, possibleAction), agentIndex + 1,
                                          currentDepth, possibleAction)
                        for possibleAction in gameState.getLegalActions(0)], key=lambda p: (lambda x, y: int(x))(*p))
            else:
                return max([self.getActionAux(gameState.generateSuccessor(agentIndex, possibleAction), agentIndex + 1,
                                              currentDepth, action)
                            for possibleAction in gameState.getLegalActions(0)],
                           key=lambda p: (lambda x, y: int(x))(*p))
        if agentIndex >= gameState.getNumAgents():
            return self.getActionAux(gameState, 0, currentDepth + 1, action)
        else:
            return min([self.getActionAux(gameState.generateSuccessor(agentIndex, possibleAction), agentIndex + 1,
                                          currentDepth, action)
                        for possibleAction in gameState.getLegalActions(agentIndex)],
                       key=lambda p: (lambda x, y: int(x))(*p))

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.getScore():
            Returns the score corresponding to the current state of the game

          gameState.isWin():
            Returns True if it's a winning state

          gameState.isLose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue

        """

        # BEGIN_YOUR_CODE
        return self.getActionAux(gameState, 0, 0, [])[1]

        # END_YOUR_CODE


######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning
    """

    def getActionAux(self, gameState, agentIndex: int, currentDepth, action, alpha, beta):
        if self.isTerminal(gameState, currentDepth):
            return self.evaluationFunction(gameState), action
        if agentIndex == 0:
            currMax = np.NINF, action
            if currentDepth == 0:
                for childAction in gameState.getLegalPacmanActions():
                    value = self.getActionAux(gameState.generatePacmanSuccessor(childAction), 1, currentDepth,
                                              childAction, alpha, beta)
                    currMax = max(value, currMax, key=lambda p: (lambda x, y: float(x))(*p))
                    alpha = max(currMax[0], alpha)
                    if currMax[0] >= beta:
                        return np.inf, childAction
                return currMax

            else:
                for childAction in gameState.getLegalPacmanActions():
                    value = self.getActionAux(gameState.generatePacmanSuccessor(childAction), 1, currentDepth,
                                              action, alpha, beta)
                    currMax = max(value, currMax, key=lambda p: (lambda x, y: float(x))(*p))
                    alpha = max(currMax[0], alpha)
                    if currMax[0] >= beta:
                        return np.inf, action
                return currMax
        if agentIndex >= gameState.getNumAgents():
            return self.getActionAux(gameState, 0, currentDepth + 1, action, alpha, beta)
        else:
            currMin = np.inf, action
            for childAction in gameState.getLegalActions(agentIndex):
                value = self.getActionAux(gameState.generateSuccessor(agentIndex, childAction), agentIndex + 1,
                                          currentDepth,
                                          action, alpha, beta)
                currMin = min(value, currMin, key=lambda p: (lambda x, y: float(x))(*p))
                beta = min(currMin[0], beta)
                if currMin[0] <= alpha:
                    return np.NINF, action
            return currMin

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        # BEGIN_YOUR_CODE
        return self.getActionAux(gameState, 0, 0, [], np.NINF, np.inf)[1]
        # END_YOUR_CODE


######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getActionAux(self, gameState, agentIndex: int, currentDepth, firstAction):
        if self.isTerminal(gameState, currentDepth):
            return self.evaluationFunction(gameState), firstAction
        if agentIndex == 0:
            if currentDepth == 0:
                return max([self.getActionAux(gameState.generateSuccessor(agentIndex, possibleAction), agentIndex + 1,
                                              currentDepth, possibleAction)
                            for possibleAction in gameState.getLegalActions(0)],
                           key=lambda p: (lambda x, y: int(x))(*p))
            else:
                return max([self.getActionAux(gameState.generateSuccessor(agentIndex, possibleAction), agentIndex + 1,
                                              currentDepth, firstAction)
                            for possibleAction in gameState.getLegalActions(0)], key=lambda p: (lambda x, y: int(x))(*p))
        if agentIndex >= gameState.getNumAgents():
            return self.getActionAux(gameState, 0, currentDepth + 1, firstAction)
        else:
            # probabilistic behavior
            currentGhost = ghostAgents.RandomGhost(agentIndex)
            ghostDist = currentGhost.getDistribution(gameState)
            return sum([prob * self.getActionAux(gameState.generateSuccessor(agentIndex, action), agentIndex + 1,
                                                 currentDepth, firstAction)[0]
                        for action, prob in ghostDist.items()]), firstAction

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their legal moves.
        """

        # BEGIN_YOUR_CODE
        return self.getActionAux(gameState, 0, 0, [])[1]
        # END_YOUR_CODE


######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """
    def getActionAux(self, gameState, agentIndex: int, currentDepth, firstAction):
        if self.isTerminal(gameState, currentDepth):
            return self.evaluationFunction(gameState), firstAction
        if agentIndex == 0:
            if currentDepth == 0:
                return max([self.getActionAux(gameState.generateSuccessor(agentIndex, possibleAction), agentIndex + 1,
                                              currentDepth, possibleAction)
                            for possibleAction in gameState.getLegalActions(0)], key=lambda p: (lambda x, y: int(x))(*p))
            else:
                return max([self.getActionAux(gameState.generateSuccessor(agentIndex, possibleAction), agentIndex + 1,
                                              currentDepth, firstAction)
                            for possibleAction in gameState.getLegalActions(0)],
                           key=lambda p: (lambda x, y: int(x))(*p))
        if agentIndex >= gameState.getNumAgents():
            return self.getActionAux(gameState, 0, currentDepth + 1, firstAction)
        else:
            # probabilistic behavior
            currentGhost = ghostAgents.DirectionalGhost(agentIndex)
            ghostDist = currentGhost.getDistribution(gameState)
            return sum([prob * self.getActionAux(gameState.generateSuccessor(agentIndex, action), agentIndex + 1,
                                                 currentDepth, firstAction)[0]
                        for action, prob in ghostDist.items()]), firstAction
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
        """

        # BEGIN_YOUR_CODE
        return self.getActionAux(gameState, 0, 0, [])[1]
        # END_YOUR_CODE

######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
    """
      Your competition agent
    """

    def getAction(self, gameState):
        """
          Returns the action using self.depth and self.evaluationFunction

        """

        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE
