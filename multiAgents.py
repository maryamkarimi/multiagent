# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        score = successorGameState.getScore()

        foodSum = 0
        # Calculate the sum of distances from current position to every food position
        for food in newFood.asList():
          foodDistance = util.manhattanDistance(newPos, food)
          foodSum += foodDistance

        ghostSum = 0
        # Calculate the sum of distances from current position to every ghost position
        for ghost in newGhostStates:
          ghostDistance = util.manhattanDistance(newPos, ghost.getPosition())
          ghostSum += ghostDistance
          # if a ghost is too close
          if (ghostDistance < 2):
            return -float('inf')
  
        # if all the food is already eaten
        if foodSum == 0:
          return float('inf')

        # Dividing ghost distances sum by food distances sum
        # this is because the bigger ghost sum, the farther ghosts are and the better the score should be
        # and the smaller the food sum is, the closer we are to food positions
        return successorGameState.getScore() + ghostSum / foodSum


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        score, action = self.value(gameState, 0, 0)
        
        # return the action
        return action

    def value(self, gameState, depth, agentIndex):
      
      legalActions = gameState.getLegalActions(agentIndex)
      
      # if the state is a terminal state
      if depth == self.depth or len(legalActions) == 0:
        return gameState.getScore(), ""

      # if the agent is MAX (agentIndex = 0 means Pacman)
      if agentIndex == 0:
        return self.maxValue(gameState, depth, agentIndex)

      # if the agent is MIN (ghosts are >= 1)
      else:
        return self.minValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth, agentIndex):
      # set max to negative infinity at first
      maxValue = -float('inf')

      # for every legal action of agent
      for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
        # successorIndex will be agentIndex + 1 as it will be next player's turn now
        successorIndex = agentIndex + 1
        successorDepth = depth

        # If all of the agents have finished playing their turn, increase the depth by one and change successorIndex to zero (Pacman)
        if successorIndex == gameState.getNumAgents():
          successorIndex = 0
          successorDepth = successorDepth + 1

        # Get the minimax score of successor
        currValue = self.value(successor, successorDepth, successorIndex)[0]
        
        # If current value is greater than maxValue, update maxValue and maxAction
        if currValue > maxValue:
          maxValue = currValue
          maxAction = action

      return maxValue, maxAction
      
    def minValue(self, gameState, depth, agentIndex):
      # set min to infinity at first
      minValue = float('inf')

      # for every legal action of agent
      for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
        # successorIndex will be agentIndex + 1 as it will be next player's turn now
        successorIndex = agentIndex + 1
        successorDepth = depth
        
        # If all of the agents have finished playing their turn, increase the depth by one and change successorIndex to zero (Pacman)
        if successorIndex == gameState.getNumAgents():
          successorIndex = 0
          successorDepth = successorDepth + 1

        # Get the minimax score of successor 
        currValue = self.value(successor, successorDepth, successorIndex)[0]
        
        # If current value is smaller than minValue, update minValue and minAction
        if currValue < minValue:
          minValue = currValue
          minAction = action

      return minValue, minAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
