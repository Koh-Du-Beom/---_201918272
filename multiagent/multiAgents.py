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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """ 
    def max_val(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = (None, -float("inf"))
        actions = gameState.getLegalActions(agentIndex)
        #agentIndex에 따라 어떤 방향으로 움직일지가 나온다. 이떄 반복문을 돌려서 actions를 검사
        for action in actions :
            Actions = (action, self.checkAction(gameState.generateSuccessor(agentIndex, action), (depth+1)%gameState.getNumAgents(), depth+1, alpha, beta))
            #getLegalActions로 가져온 successor의 정보를 기반으로 bestAction이 될 후보들을 만든다.
            bestAction = max(bestAction, Actions, key = lambda x:x[1])
            #bestAction은 가장 좋은 actions, bestAction중에서 가장 좋은(높은 값을 찾으려는)action이어야 한다.
            #이때 key = lambda x:x[1]은 x[1]인 value를 기준으로 가장 좋은 action을 찾기위해 설정
            if bestAction[1] > beta :
                return bestAction 
                #bestAction의 value값이 beta보다 크면 bestAction이 맞으므로 그대로 리턴
            else:
                alpha = max(alpha, bestAction[1])
                #그렇지 않다면 최소가치인 alpha를 업데이트
        return bestAction    
        
    def min_val(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = (None, float("inf"))
        actions = gameState.getLegalActions(agentIndex)
        #max_val알고리즘과 유사한 동작이다.
        for action in actions :
            Actions = (action, self.checkAction(gameState.generateSuccessor(agentIndex, action), (depth+1)%gameState.getNumAgents(),depth+1 ,alpha, beta))   
            #getLegalActions로 가져온 successor의 정보를 기반으로 bestAction이 될 후보들을 만든다.
            bestAction = min(bestAction, Actions, key = lambda x:x[1])
            #max_val알고리즘과 비슷하지만 이땐 가장 좋은(낮은 값을 찾으려는)action이 다르므로 min함수 이용
            
            if bestAction[1] < alpha:
                return bestAction
                #bestAction의 value값이 alpha보다 작으면 bestAction이 맞으므로 그대로 리턴
            else:
                beta = min(beta, bestAction[1])
                #이것도 역시 max_val알고리즘과 유사하지만 이땐 beta를 업데이트해준다.
        return bestAction
    
    def checkAction(self, gameState, agentIndex ,depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
            #최대로 탐색할 수 있는 depth까지 갔거나, 이겼거나, 졌을 경우엔 탐색을 종료.
            #이때, evaluationFunction을 사용해 게임 종료 시 최종점수를 환산하기 위함
        
        if agentIndex == 0 :
            return self.max_val(gameState, agentIndex, depth, alpha, beta)[1]
            #agentIndex가 0인 경우 Pacman(플레이어인 우리)의 차례이므로, 점수를 최대화하려는 움직임을 취함.
            #이때 max_val함수는 action과 value의 두 값을 리턴하는데, value를 리턴하기 위해 [1]을 사용.
        else:
            return self.min_val(gameState, agentIndex, depth, alpha, beta)[1]
            #agentIndex가 0이 아닌경우 Ghost(우리를 적대하는 AI)의 차례이므로, 점수를 최소화하려는 움직임을 취함.
            #이때 min_val함수도 마찬가지로 value를 리턴함.
    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max_val(gameState, 0, 0, -float("inf"), float("inf"))[0]
        #AgentIndex를 0으로 시작해서 maxval함수가 시작되게 하고, max_val함수가 돌려주는 bestAction에서 action(행동)을 가져오기
        

    
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
