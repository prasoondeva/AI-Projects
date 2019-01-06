# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util, pdb

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        values_k= util.Counter()
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(0,iterations):
            for state in self.mdp.getStates():
                possibleActions = self.mdp.getPossibleActions(state)
                Q = []
                Values_k_next = None
                for action in possibleActions:
                    possibleTransitions = self.mdp.getTransitionStatesAndProbs(state,action)
                    sum = 0.0
                    for transition in possibleTransitions:
                        S_prime = transition[0]
                        transitionProbability = transition[1]
                        sum = sum + transitionProbability * (self.mdp.getReward(state,action,S_prime) + discount * values_k[S_prime])
                    Q.append(sum)
                if(self.mdp.isTerminal(state)):
                    Q.append(0)
                self.values[state] = max(Q)
            values_k=util.Counter()
            for key in self.values:
                values_k[key]=self.values[key]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        transitions=self.mdp.getTransitionStatesAndProbs(state,action)
        sum=0.0
        for transition in transitions:
            nextState=transition[0]
            value_k=self.values[nextState]
            sum=sum+transition[1]*(self.mdp.getReward(state,action,nextState)+self.discount*value_k)

        return sum;

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.mdp.getPossibleActions(state)
        if(len(possibleActions) == 0):
            return None
        argMax = None
        finalAction = None

        for action in possibleActions:
            transitions = self.mdp.getTransitionStatesAndProbs(state,action)
            sum = 0.0
            for transition in transitions:
                nextState = transition[0]
                value_k = self.values[nextState]
                sum = sum + transition[1]*(self.mdp.getReward(state,action,nextState) + self.discount*value_k)
            if (argMax is None):
                argMax = sum
                finalAction = action
            elif sum > argMax:
                argMax = sum
                finalAction = action

        return finalAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)