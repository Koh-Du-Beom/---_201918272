# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random
import util
import math
import copy


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_values = {}
        #Q-value를 dictionary형태로서 초기화한다.
    
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if(state, action) in self.q_values:
          return self.q_values[(state, action)]
          #(state, action)에 해당하는 값이 q_values의 딕셔너리 안에 있다면 그에 대한 q_value를 리턴
        else:
          return 0.0
          #Agent가 이전에 보지 못한 특정 상태와 action에 대한 Q-value가 0이어야한다 
        
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        tmpQvalues = {} #getQValue함수로 가져온 데이터를 담음
        actions = self.getLegalActions(state) #가능한 action을 담은 변수
        if len(actions) == 0:
          return 0
        #legal action이 없으면 0을 리턴해야함.
        for action in actions:
          tmpQvalues[action] = self.getQValue(state, action)
          #action에 대한 Q값을 tmpQvalues변수에 저장해둔다.
        return max(tmpQvalues.values())
        #저장된 값 중에서 value값이 가장 높은 것을 리턴
        
    
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state) #가능한 action을 담은 변수
        if len(actions) == 0:
          return None
        #leagal action이 없다면 None을 리턴
        bestValue = self.getValue(state)
        #최적의 value를 가져온다.
        bestActions = []
        for action in actions:
          if self.getQValue(state, action) == bestValue :
            bestActions.append(action)
        #bestActions에 bestValue와 getQValue의 리턴값이 같은 데이터들을 추가한다.
        return random.choice(bestActions)
        #더 좋은 동작을 위해 랜덤한 액션을 선택하는 것
    
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        
        "*** YOUR CODE HERE ***"
        epsilon = self.epsilon
        if util.flipCoin(epsilon):
          #위에서 probability = epsilon으로 지정을 해주었고 filpcoin함수 사용해서 성공확률이 epsilon인 이진변수를 시뮬레이션
          action = random.choice(legalActions)
        else:
          action = self.getPolicy(state)
          #1-epsilon의 확률로 현재상태에서 정책을 얻어와 최적의 행동을 하게됨
        
        return action
    
    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        discount = self.discount
        alpha = self.alpha
        qvalue = self.getQValue(state, action)
        nextValue = self.getValue(nextState) 
        #nextState에 대한 value
        
        if nextState:
          newValue = (1-alpha) * qvalue + alpha *(reward + discount * nextValue)
          self.setValue(state, action, newValue)
          #nextState가 True인 경우엔 nextValue가 유효하므로 위 식을 통해 value를 업데이트
        else:
          newValue = (1-alpha) * qvalue + alpha * reward
          self.setValue(state, action, newValue)
          #nextState가 False인 경우엔 nextValue가 유효하지 않으므로 위 식을 통해 value를 업데이트
        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
    def setValue(self, state, action, value):
      self.q_values[(state, action)] = value


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        
        features = self.featExtractor.getFeatures(state, action)
        #feature들을 featExtractor의 getFeatures()함수를 통해 가져온다.
        q_value = 0
        for feature in features:
           q_value += features[feature] * self.weights[feature]
          #Q(s,a) = f(s,a)w의 합으로 구성되며, 이를 표현한 식. 
        return q_value
        #계산된 q_value를 리턴한다.

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        alpha = self.alpha
        discount = self.discount
        difference = (reward + discount * self.getValue(nextState)) - (self.getQValue(state, action))
        #difference를 구현
        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
          self.weights[feature] += alpha * difference * features[feature]
        #가중치 벡터를 업데이트해준다.
    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print("weight = {}".format(self.weights))
            #weight를 출력해보기
              
