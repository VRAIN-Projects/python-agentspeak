from __future__ import print_function
from typing import Union, Tuple, Iterator
from enum import Enum

import sys
import collections
import copy
import functools
import os.path
import time
import threading
import asyncio
import concurrent.futures
import random
import numpy as np
import math


import agentspeak
import agentspeak.runtime
import agentspeak.stdlib
import agentspeak.parser
import agentspeak.lexer
import agentspeak.util

from agentspeak import UnaryOp, BinaryOp, AslError, asl_str


LOGGER = agentspeak.get_logger(__name__)
C = {}

class PADExpression():
    """
    This class is used to represent the PAD expressions.
    
    PAD = {Pleasure, Arousal, Dominance}
    """
         
    def __init__(self, pThres, op1, aThres, op2, dThres):
        """
        Constructor of the PADExpression class.
        
        Args:
            pThres (float): Pleasure threshold.
            op1 (str): Operator 1.
            aThres (float): Arousal threshold.
            op2 (str): Operator 2.
            dThres (float): Dominance threshold.
        """
        self.PThreshold = pThres
        self.operator1 = op1
        self.AThreshold = aThres
        self.operator2 = op2
        self.DThreshold = dThres
        
    def evaluate(self, p, a, d) -> bool: 
        """
        This method is used to evaluate the PAD expression.

        Args:
            p (float): Pleasure value.
            a (float): Arousal value.
            d (float): Dominance value.
        Returns:
            bool: True if the PAD expression is evaluated, False otherwise.
        """
        
        result = True
        if self.operator1 == "and":
            result = ((p <= self.PThreshold) and (p <= self.AThreshold))
        else:
            result = ((p <= self.PThreshold) or (p <= self.AThreshold))
        if self.operator2 == "and":
            result = (result and (p <= self.AThreshold))
        else:
            result = (result or (p <= self.AThreshold))
        return result

class Personality():
    """ 
    This class is used to represent the personality of the agent.
    
    """
        
    def __init__(self):
        """
        Constructor of the Personality class.
        """
        self.traitsLabels = []
        self.traits = []
        self.rationalityLevel = 0.0
        self.copingStrategies = []
        self.init()
        
    def init(self):
        """ 
        This method is used to initialize the personality of the agent.
        """
        self.traitsLabels = []
        self.traits = []
        self.setTraitsLabels([])
        if self.traitsLabels != None:
            self.traits = [0.0] * len(self.traitsLabels)
     
    def clone(self):
        """
        This method is used to clone the personality of the agent.
        """
        return self
     
    def getRationalityLevel(self):
        """ 
        This method is used to get the rationality level of the agent.

        Returns:
            float: Rationality level of the agent.
        """
        return self.rationalityLevel
     
    def setRationalityLevel(self, rationalityLevel) -> None:
        """
        This method is used to set the rationality level of the agent.
        
        Args:
            rationalityLevel (float): Rationality level of the agent.
        """
        self.rationalityLevel = rationalityLevel
    
    def getCopingStrategies(self):
        """
        This method is used to get the coping strategies of the agent.
        
        Returns:
            list: Coping strategies of the agent.
        """
        return self.copingStrategies
    
    def setCopingStrategies(self, copingStrategies) -> None:
        """
        This method is used to set the coping strategies of the agent.
        
        Args:
            copingStrategies (list): Coping strategies of the agent.
        """
        self.copingStrategies = copingStrategies
         
    def setTraitsLabels(self, traitsLab):
        """
        This method is used to set the traits labels of the agent.
         
        Args:
            traitsLab (list): Traits labels of the agent.
        """
        self.traitsLabels = traitsLab   
         
    def getTraitsLabels(self):
        """ 
        This method is used to get the traits labels of the agent.
        
        Returns:
            list: Traits labels of the agent.
        """
        return self.traitsLabels
    
    def getTraits(self):
        """
        This method is used to get the traits of the agent.
        
        Returns:
            list: Traits of the agent.
        """
        return self.traits
    
    def set_traits(self, traits):
        """
        This method is used to set the traits of the agent.
        
        Args:
            traits (list): Traits of the agent.
        """
        self.traits = traits
        
        
class TermQuery(agentspeak.runtime.TermQuery):
    
    def execute_concern(self, agent, intention, concern):
        # Boolean constants.
        term = agentspeak.evaluate(self.term, intention.scope)
        if term is True:
            yield
            return
        elif term is False:
            return

        choicepoint = object()

        concern = copy.deepcopy(concern)
        intention.stack.append(choicepoint)
        

        if agentspeak.unify(term, concern.head, intention.scope, intention.stack):
            for _ in concern.query.execute(agent, intention):
                yield

        agentspeak.reroll(intention.scope, intention.stack, choicepoint)
        
class AffectiveAgent(agentspeak.runtime.Agent):
    """
    This class is a subclass of the Agent class. 
    It is used to add the affective layer to the agent.
    """
    def __init__(self, env: agentspeak.runtime.Environment, name: str, beliefs = None, rules = None, plans = None, concerns = None):
        """
        Constructor of the AffectiveAgent class.
        
        Args:
            env (agentspeak.runtime.Environment): Environment of the agent.
            name (str): Name of the agent.
            beliefs (dict): Beliefs of the agent.
            rules (dict): Rules of the agent.
            plans (dict): Plans of the agent.
        
        Attributes:
            env (agentspeak.runtime.Environment): Environment of the agent.
            name (str): Name of the agent.
            beliefs (dict): Beliefs of the agent.
            rules (dict): Rules of the agent.
            plans (dict): Plans of the agent.
            current_step (str): Current step of the agent.
            T (dict): is the temporary information of the current 
            rational cycle consisting of a dictionary containing:
                - "p": Applicable plan.
                - "Ap": Applicable plans.
                - "i": Intention.
                - "R": Relevant plans.
                - "e": Event.
             
        """
        super(AffectiveAgent, self).__init__(env, name, beliefs, rules, plans)
        
        self.current_step = ""
        self.T = {}
        
        # Circunstance initialization
        self.C = {}
        self.C["I"] = collections.deque()
        
        self.Ag = {"P": Personality(), "cc": []} # Personality and concerns definition
        
        self.Ta = {"mood": {}, "emotion":{}} # Temporal affective state definition
        
        self.Mem = {} # Affective memory definition (⟨event 𝜀, affective value av⟩)
        
        self.concerns = collections.defaultdict(lambda: []) if concerns is None else concerns
        
        self.event_queue = []
        self.AV = {"desirability": None} 
         
        self.initAffectiveThreshold()
         
        self.fulfilledExpectations = []
        self.notFulfilledExpectations = []
        
    def initAffectiveThreshold(self):
        """
        This method is used to initialize the affective thresholds of the agent.
        """
        
        self.DISPLACEMENT = 0.5
        self.affRevEventThreshold = []
        self.affRevEventThreshold.append(PADExpression(0.8, "or", 0.8, "and", 0.0))
        
    def add_concern(self, concern):
        """ 
        This method is used to add a concern to the agent.
        
        Args:
            concern (Concern): Concern to be added.
        """
        self.concerns[(concern.head.functor, len(concern.head.args))].append(concern)
        
    def test_concern(self, term, intention, concern):
        """This function is used to know the value of a concern

        Args:
            term (Literal): Term of the concern
            intention (Intention): Intention of the agent
            concern (Concern): Concern of the agent

        Raises:
            AslError:  If the term is not a Literal

        Returns:
            OR[bool, str]: If the concern is not found, return False. If the concern is found, return the value of the concern
        """
        term = agentspeak.evaluate(term, intention.scope)

        if not isinstance(term, agentspeak.Literal):
            raise AslError("expected concern literal, got: '%s'" % term)

        query = TermQuery(term)

        try:
            next(query.execute_concern(self, intention, concern))
            concern_value = " ".join(asl_str(agentspeak.freeze(t, intention.scope, {})) for t in term.args)
            return concern_value
        except StopIteration:
            return False
        
    def call(self, trigger: agentspeak.Trigger, goal_type:agentspeak.GoalType, term: agentspeak.Literal, calling_intention: agentspeak.runtime.Intention, delayed: bool = False):
        """ This method is used to call an event.

        Args:
            trigger (agentspeak.Trigger): Trigger of the event.
            goal_type (agentspeak.GoalType): Type of the event.
            term (agentspeak.Literal): Term of the event.
            calling_intention (agentspeak.runtime.Intention): Intention of the agent.
            delayed (bool, optional): Delayed event. Defaults to False.

        Raises:
            AslError: "expected literal" if the term is not a literal.
            AslError: "expected literal term" if the term is not a literal term.
            AslError: "no applicable plan for" + str(term)" if there is no applicable plan for the term.
            log.error: "expected end of plan" if the plan is not finished. The plan finish with a ".".

        Returns:
            bool: True if the event is called.
        
        If the event is a belief, we add or remove it.
        
        If the event is a goal addition, we start the reasoning cycle.
        
        If the event is a goal deletion, we remove it from the intentions queue.
        
        If the event is a tellHow addition, we tell the agent how to do it.
        
        If the event is a tellHow deletion, we remove the tellHow from the agent.
        
        If the event is a askHow addition, we ask the agent how to do it.
        """
        # Modify beliefs.       
        if goal_type == agentspeak.GoalType.belief: 
            # We recieve a belief and the affective cycle is activated.
            # We need to provide to the sunction the term and the Trigger type.
            self.event_queue.append((term, trigger))
            self.appraisal((term, trigger),0)
            if trigger == agentspeak.Trigger.addition:
                self.add_belief(term, calling_intention.scope)
            else: 
                found = self.remove_belief(term, calling_intention) 
                if not found: 
                    return True 

        # Freeze with caller scope.
        frozen = agentspeak.freeze(term, calling_intention.scope, {}) 

        if not isinstance(frozen, agentspeak.Literal): 
            raise AslError("expected literal") 

        # Wake up waiting intentions.
        for intention_stack in self.C["I"]: 
            if not intention_stack: 
                continue 
            intention = intention_stack[-1] 

            if not intention.waiter or not intention.waiter.event: 
                continue
            event = intention.waiter.event

            if event.trigger != trigger or event.goal_type != goal_type: 
                continue 

            if agentspeak.unifies_annotated(event.head, frozen): 
                intention.waiter = None 

        if goal_type == agentspeak.GoalType.achievement and trigger == agentspeak.Trigger.addition:
            
            self.C["E"] = [term] if "E" not in self.C else self.C["E"] + [term]
            self.current_step = "SelEv"
            self.applySemanticRuleDeliberate()
            return True
            
        
        if goal_type == agentspeak.GoalType.achievement and trigger == agentspeak.Trigger.addition: 
            raise AslError("no applicable plan for %s%s%s/%d" % (
                trigger.value, goal_type.value, frozen.functor, len(frozen.args))) 
        elif goal_type == agentspeak.GoalType.test:
            return self.test_belief(term, calling_intention) 

        # If the goal is an achievement and the trigger is an removal, then the agent will delete the goal from his list of intentions
        if goal_type == agentspeak.GoalType.achievement and trigger == agentspeak.Trigger.removal: 
            if not agentspeak.is_literal(term):
                raise AslError("expected literal term") 

            # Remove a intention passed by the parameters.
            for intention_stack in self.C["I"]: 
                if not intention_stack: 
                    continue 

                intention = intention_stack[-1] 

                if intention.head_term.functor == term.functor: 
                    if agentspeak.unifies(term.args, intention.head_term.args):
                        intention_stack.remove(intention)   

        # If the goal is an tellHow and the trigger is an addition, then the agent will add the goal received as string to his list of plans
        if goal_type == agentspeak.GoalType.tellHow and trigger == agentspeak.Trigger.addition:
            
            str_plan = term.args[2] 

            tokens = [] 
            tokens.extend(agentspeak.lexer.tokenize(agentspeak.StringSource("<stdin>", str_plan), agentspeak.Log(LOGGER), 1)) # extend the tokens with the tokens of the string plan
            
            # Prepare the conversion from tokens to AstPlan
            first_token = tokens[0] 
            log = agentspeak.Log(LOGGER) 
            tokens.pop(0) 
            tokens = iter(tokens) 

            # Converts the list of tokens to a Astplan
            if first_token.lexeme in ["@", "+", "-"]: 
                tok, ast_plan = agentspeak.parser.parse_plan(first_token, tokens, log) 
                if tok.lexeme != ".": 
                    raise log.error("", tok, "expected end of plan")
            
            # Prepare the conversión of Astplan to Plan
            variables = {} 
            actions = agentspeak.stdlib.actions
            
            head = ast_plan.event.head.accept(agentspeak.runtime.BuildTermVisitor(variables)) 

            if ast_plan.context: 
                context = ast_plan.context.accept(BuildQueryVisitor(variables, actions, log)) 
            else: 
                context = TrueQuery() 

            body = agentspeak.runtime.Instruction(agentspeak.runtime.noop) 
            body.f = agentspeak.runtime.noop 
            if ast_plan.body: 
                ast_plan.body.accept(BuildInstructionsVisitor(variables, actions, body, log)) 
                 
            #Converts the Astplan to Plan
            plan = agentspeak.runtime.Plan(ast_plan.event.trigger, ast_plan.event.goal_type, head, context, body,ast_plan.body,ast_plan.annotations) 
            
            if ast_plan.args[0] is not None:
                plan.args[0] = ast_plan.args[0]

            if ast_plan.args[1] is not None:
                plan.args[1] = ast_plan.args[1]
            
          
            # Add the plan to the agent
            self.add_plan(plan) 

        # If the goal is an askHow and the trigger is an addition, then the agent will find the plan in his list of plans and send it to the agent that asked
        if goal_type == agentspeak.GoalType.askHow and trigger == agentspeak.Trigger.addition: 
           self.T["e"] =  term.args[2]
           return self._ask_how(term)

        # If the goal is an unTellHow and the trigger is a removal, then the agent will delete the goal from his list of plans   
        if goal_type == agentspeak.GoalType.tellHow and trigger == agentspeak.Trigger.removal:

            label = term.args[2]

            delete_plan = []
            plans = self.plans.values()
            for plan in plans:
                for differents in plan:                    
                    if ("@" + str(differents.annotation[0].functor)).startswith(label):
                        delete_plan.append(differents)
            for differents in delete_plan:
                plan.remove(differents)

        return True 
    
    def applySelEv(self) -> bool:
        """
        This method is used to select the event that will be executed in the next step

        Returns:
            bool: True if the event was selected
        """
        
        #self.term = self.ast_goal.atom.accept(agentspeak.runtime.BuildTermVisitor({}))
        if "E" in self.C and len(self.C["E"]) > 0:
            # Select one event from the list of events and remove it from the list without using pop
            self.T["e"] = self.C["E"][0]
            self.C["E"] = self.C["E"][1:]
            self.frozen = agentspeak.freeze(self.T["e"], agentspeak.runtime.Intention().scope, {}) 
            self.T["i"] = agentspeak.runtime.Intention()
            self.current_step = "RelPl"
            self.delayed = True
        else:
            self.current_step = "SelEv"
            self.delayed = False
            return False
            
        return True
    
    def applyRelPl(self) -> bool:
        """
        This method is used to find the plans that are related to the current goal.
        We say that a plan is related to a goal if both have the same functor

        Returns:
            bool: True if the plans were found, False otherwise
            
        - If the plans were found, the dictionary T["R"] will be filled with the plans found and the current step will be changed to "AppPl"
        - If not plans were found, the current step will be changed to "SelEv" to select a new event
        """
        RelPlan = collections.defaultdict(lambda: [])
        plans = self.plans.values()
        for plan in plans:
            for differents in plan:
                if self.T["e"].functor in differents.head.functor:
                    RelPlan[(differents.trigger, differents.goal_type, differents.head.functor, len(differents.head.args))].append(differents)
         
        if not RelPlan:
            self.current_step = "SelEv"
            return False
        self.T["R"] = RelPlan
        self.current_step = "AppPl"
        return True
    
    def applyAppPl(self) -> bool:
        """
        This method is used to find the plans that are applicable to the current goal.
        We say that a plan is applicable to a goal if both have the same functor, 
        the same number of arguments and the context are satisfied

        Returns:
            bool: True if the plans were found, False otherwise
        
        - If the plans were found, the dictionary T["Ap"] will be filled with the plans found and the current step will be changed to "SelAppl"
        - If not plans were found, return False
        """
        self.T["Ap"] = self.T["R"][(agentspeak.Trigger.addition, agentspeak.GoalType.achievement, self.frozen.functor, len(self.frozen.args))] 
        self.current_step = "SelAppl"
        return self.T["Ap"] != []
    
    def applySelAppl(self) -> bool:
        """ 
        This method is used to select the plan that is applicable to the current goal.
        We say that a plan is applicable to a goal if both have the same functor, 
        the same number of arguments and the context are satisfied 
        
        We select the first plan that is applicable to the goal in the dict of 
        applicable plans

        Returns:
            bool: True if the plan was found, False otherwise
            
        - If the plan was found, the dictionary T["p"] will be filled with the plan found and the current step will be changed to "AddIM"
        - If not plan was found, return False
        """
        for plan in self.T["Ap"]: 
                for _ in agentspeak.unify_annotated(plan.head, self.frozen, self.T["i"].scope, self.T["i"].stack): 
                    for _ in plan.context.execute(self, self.T["i"]):   
                        self.T["p"] = plan
                        self.current_step = "AddIM"
                        return True
        return False
    
    def applyAddIM(self) -> bool:
        """
        This method is used to add the intention to the intention stack of the agent

        Returns:
            bool: True if the intention is added to the intention stack
        
        - When  the intention is added to the intention stack, the current step will be changed to "SelEv"
        """
        self.T["i"].head_term = self.frozen 
        self.T["i"].instr = self.T["p"].body 
        self.T["i"].calling_term = self.T["e"] 

        if not self.delayed and self.C["I"]: 
            for intention_stack in self.C["I"]: 
                if intention_stack[-1] == self.delayed: 
                    intention_stack.append(self.T["i"]) 
                    return True
        new_intention_stack = collections.deque() 
        new_intention_stack.append(self.T["i"]) 
        
        # Add the event and the intention to the Circumstance
        self.C["I"].append(new_intention_stack) 
        
        self.current_step = "SelInt"
        return True      
    
    def applySemanticRuleDeliberate(self):
        """
        This method is used to apply the first part of the reasoning cycle.
        This part consists of the following steps:
        - Select an event
        - Find the plans that are related to the event
        - Find the plans that are applicable to the event
        - Select the plan that is applicable to the event
        - Add the intention to the intention stack of the agent
        """
        options = {
            "SelEv": self.applySelEv,
            "RelPl": self.applyRelPl,
            "AppPl": self.applyAppPl,
            "SelAppl": self.applySelAppl,
            "AddIM": self.applyAddIM
        }
        if self.current_step in options:
            flag = options[self.current_step]()
            if flag:
                self.applySemanticRuleDeliberate()
            else:
                return True
        return True
    
    def affectiveTransitionSystem(self):
        """
        This method is used to apply the second part of the reasoning cycle.
        This part consists of the following steps:
        - Appraisal
        - Update Aff State
        - Select Coping Strategy
        - Coping
        """
        
        options = {
            "Appr" : self.applyAppraisal,
            "UpAs" : self.applyUpdateAffState,
            "SelCs" : self.applySelectCopingStrategy,
            "Cope" : self.applyCope
        }
        
        runingAffectiveCycle = True
        
        if self.current_step in options:
            flag = options[self.current_step]()
            if flag:
                self.affectiveTransitionSystem()
            else:
                return True
        return True
    
    def appraisal(self, event, concern_value):
        """
        This method is used to apply the appraisal process.
        
        Args:
            event (tuple): Event to be appraised.
            concern_value (float): Concern value of the event.
        
        Returns:
            bool: True if the event was appraised, False otherwise.
        """
        selectingCs = True
        result = False
        if event != None:
                # Calculating desirability
                if len(self.concerns):
                    desirability =  self.desirability(event)
                    self.AV["desirability"] = desirability

                # Calculating likelihood. 
                likelihood = self.likelihood(event)
                self.AV["likelihood"] = likelihood

                # Calculating causal attribution
                causal_attribution = self.causalAttribution(event)
                self.AV["causal_attribution"] = causal_attribution

                # Calculating controllability: 
                if len(self.concerns):
                    controllability = self.controllability(event,concern_value,desirability)
                    self.AV["controllability"] = controllability
                    pass
                result = True
        else:
            self.AV["desirability"] = None
            self.AV["expectedness"] = None
            self.AV["likelihood"] = None
            self.AV["causal_attribution"] = None
            self.AV["controllability"] = None
        return result
    
    def controllability(self, event, concernsValue, desirability):
        """
        This method is used to calculate the controllability of the event.
        
        Args:
            event (tuple): Event to be appraised.
            concernsValue (float): Concern value of the event.
            desirability (float): Desirability of the event.
             
        Returns:
            float: Controllability of the event.
        """
        result = None
        if desirability != None and concernsValue != None:
            result = desirability - concernsValue
            result = ((result + 1)/2)
        return result
    
        
        
        
    def causalAttribution(self, event):
        """
        This method is used to calculate the causal attribution of the event.
        
        Args:
            event (tuple): Event to be appraised.
            
        Returns:
            str: Causal attribution of the event.
        """
        ca = None
        if any([annotation.functor == "source" for annotation in event[0].annots]):
            ca = "other"
        else:
            ca = "self"
        return ca
    
    def likelihood(self, event):
        """
        This method is used to calculate the likelihood of the event.
        
        Args:
            event (tuple): Event to be appraised.
            
        Returns:
            float: Likelihood of the event.
        """        
        result = None
        if event != None and event[1].name == "addition":
            result = 1.0
        return result
     
        
    def expectedness(self, event, remove):
        """
        This method is used to calculate the expectedness of the event.
         
        Args:
            event (tuple): Event to be appraised.
            remove (bool): True if the event is removed, False otherwise.
             
        Returns:
            float: Expectedness of the event.
        """
       
        result1 = None
        result2 = None
        result = None
        
        if event != None:
            index = self.fulfilledExpectations.index(event[0])
            if index != -1:
                result1 = self.fulfilledExpectations[index][1]
                if remove:
                    self.fulfilledExpectations.pop(index)
            else:
                index = self.notFulfilledExpectations.index(event[0])
                if index != -1:
                    result1 = -1 * self.notFulfilledExpectations[index][1]
                    if remove:
                        self.notFulfilledExpectations.pop(index)
         
        #processing events that "didn't happen" and were expected in this affective cycle
        #Averaging negative expectedness and removing this value from the previous result
        av = 0
        count = 0
        for i in range(len(self.notFulfilledExpectations)):
            av = av + self.notFulfilledExpectations[i][1]
            count = count + 1
        if remove:
            self.notFulfilledExpectations = []
        if count > 0:
            result2 = (av/count) 
         
        if result1 != None and result2 != None:
            result = max(-1,result1 - result2)
         
        return result # range [-1,1]
    
    def desirability(self, event):
        """
        This method is used to calculate the desirability of the event.
         
        Args:
            event (tuple): Event to be appraised.
            
        Returns:
            float: Desirability of the event.
        """
        concernVal = None
        concern = self.concerns[("concern__",1)][0] # This function return the first concern of the agent
        
         
        if concern != None:
            if event[1].name == "addition":
                # adding the new literal if the event is an addition of a belief
                concernVal = self.applyConcernForAddition(event,concern) 
            else:
                concernVal = self.applyConcernForDeletion(event,concern) 
                
            if concernVal != None:
                if float(concernVal) < 0 or float(concernVal) > 1:
                    concernVal = 0
                    
        return float(concernVal)
    
    def applyConcernForAddition(self, event, concern):
        """ 
        This method is used to apply the concern for the addition of a belief.
        
        Args:
            event (tuple): Event to be appraised.
            concern (Concern): Concern to be applied.
            
        Returns:
            float: Concern value of the event.
        """
        
        # We add the new belief to the agent's belief base, so we can calculate the concern value
        self.add_belief(event[0], agentspeak.runtime.Intention().scope)
        # We calculate the concern value
        concern_value = self.test_concern(concern.head, agentspeak.runtime.Intention(), concern)
        # We remove the belief from the agent's belief base again
        self.remove_belief(event[0], agentspeak.runtime.Intention())

        return concern_value 
        
        
        
        
    def applyConcernForDeletion(self, event, concern):
        """ 
        This method is used to apply the concern for the deletion of a belief.
        
        Args:
            event (tuple): Event to be appraised.
            concern (Concern): Concern to be applied.
            
        Returns:
            float: Concern value of the event.
        """

        # We remove the belief from the agent's belief base, so we can calculate the concern value
        self.remove_belief(event[0], agentspeak.runtime.Intention())
        # We calculate the concern value
        concern_value = self.test_concern(concern.head, agentspeak.runtime.Intention(), concern)
        # We add the belief to the agent's belief base again
        self.add_belief(event[0], agentspeak.runtime.Intention())
        
        return concern_value 
        
         
            
    
    def applyAppraisal(self) -> bool:
        """
        This method is used to apply the appraisal process.
        """
        
        ped = PairEventDesirability(None)
        if True: # while self.lock instead of True for the real implementation
            if self.event_queue:
                ped.event = self.event_queue.pop()
            
        if ped.event == None:
            self.appraisal(None, 0.0) 
            self.currentEvent = None
            self.eventProcessedInCycle = False
        else:
            self.appraisal(ped.event, random.random()) 
            self.currentEvent = ped.event
            self.eventProcessedInCycle = True
            
        if self.cleanAffectivelyRelevantEvents(): 
            self.Mem = {}  
        
        # The next step is Update Aff State
        self.current_step = "UpAs"
        return True
    
    def applyCope(self):
        """
        This method is used to apply the coping process.
        """
        
        SelectingCs = True
        while SelectingCs() and self.C["CS"]:
            SelectingCs = self.cope()
        self.current_step = "Appr"
        return True
    
    def cope(self):
        """
        This method is used to apply the coping process.
         
        Returns:
            bool: True if the coping strategy was applied, False otherwise.
        """
        
        if self.C["CS"]:
            cs = self.C["CS"].pop(0)
            self["CS"].append(cs)
            return True
        else:
            return False
        
    def applySelectCopingStrategy(self):
        """
        This method is used to select the coping strategy.
        Personality parser is not implemented yet.
         
        Returns:
            bool: True if the coping strategy was selected, False otherwise.
        """
       
        self.C["CS"] = []
        self.selectCs() 
        self.current_step = "Cope"
        return True
    
    def selectCs(self):
        """
        This method is used to select the coping strategy.
        """
        
        AClabel = self.C["AfE"]
        logCons = False
        asContainsCs = False
        if len(AClabel) != 0 and self.Ag["P"].getCopingStrategies() != None:
            for cs in self.Ag["P"].getCopingStrategies():
                if cs.getAffectCategory().name in AClabel:
                    logCons = cs.getContext().logicalConsequence(self.ag).hasNext()
                    if logCons:
                        self.C["CS"].append(cs)
                    asContainsCs = True
            if asContainsCs:
                self.C["AfE"] = []
                
    
    
    def applyUpdateAffState(self):
        """
        This method is used to update the affective state.
        """
        
        if self.eventProcessedInCycle:
            self.UpdateAS() 
        if self.isAffectRelevantEvent(self.currentEvent): 
            self.Mem.append(self.currentEvent)
        self.current_step = "SelCs"
        return True
    
    def isAffectRelevantEvent(self, currentEvent):
        """
        This method is used to check if the event is affect relevant.
         
        Args:
            currentEvent (tuple): Event to be checked.
            
        Returns:
            bool: True if the event is affect relevant, False otherwise.
        """
        result = currentEvent != None
        for ex in self.affRevEventThreshold: 
            result = result and ex.evaluate(self.C["AS"]["P"], self.C["AS"]["A"], self.C["AS"]["D"])
        return result
    
    def UpdateAS(self):
        """
        This method is used to update the affective state.
        """
        self.DISPLACEMENT = 0.5
        
        if isinstance(self.AS, PAD): 
            pad = PAD()
            calculated_as = self.deriveASFromAppraisalVariables() 
            if calculated_as != None:
                 # PAD current_as = (PAD) getAS(); 
                current_as = self.AS
                
                tmpVal = None
                vDiff_P = None
                vDiff_A = None
                vDiff_D = None
                vectorToAdd_P = None
                vectorToAdd_A = None
                vectorToAdd_D = None
                lengthToAdd  = None
                VEC = calculated_as

                # Calculating the module of VEC
                VECmodule = math.sqrt( math.pow(VEC.getP(),2) + math.pow(VEC.getA(),2) + math.pow(VEC.getD(),2) )
                 
                # 1 Applying the pull and push of ALMA 
                if PAD.betweenVECandCenter( current_as, VEC) or not PAD.sameOctant(current_as, VEC):
                    vDiff_P = VEC.getP() - current_as.getP()
                    vDiff_A = VEC.getA() - current_as.getA()
                    vDiff_D = VEC.getD() - current_as.getD()
                else:
                    vDiff_P = current_as.getP() - VEC.getP()
                    vDiff_A = current_as.getA() - VEC.getA()
                    vDiff_D = current_as.getD() - VEC.getD()
                    
                # 2 The module of the vector VEC () is multiplied by the DISPLACEMENT and this
                # is the length that will have the vector to be added to 'as'
                lengthToAdd = VECmodule * self.DISPLACEMENT
                
                # 3 Determining the vector to add
                vectorToAdd_P = vDiff_P * self.DISPLACEMENT
                vectorToAdd_A = vDiff_A * self.DISPLACEMENT
                vectorToAdd_D = vDiff_D * self.DISPLACEMENT
                 
                # 4 The vector vectorToAdd is added to 'as' and this is the new value of
                # the current affective state
                tmpVal = current_as.getP() + vectorToAdd_P
                if tmpVal > 1:
                    tmpVal = 1.0
                else:
                    if tmpVal < -1:
                        tmpVal = -1.0
                pad.setP( round(tmpVal * 10.0) / 10.0 )
                tmpVal = current_as.getA() + vectorToAdd_A
                if tmpVal > 1:
                    tmpVal = 1.0
                else:
                    if tmpVal < -1:
                        tmpVal = -1.0
                pad.setA(  round(tmpVal * 10.0) / 10.0 )
                tmpVal = current_as.getD() + vectorToAdd_D
                if tmpVal > 1:
                    tmpVal = 1.0
                else:
                    if tmpVal < -1:
                        tmpVal = -1.0
                pad.setD(  round(tmpVal * 10.0) / 10.0 )
                 
                self.AS = pad
                AClabel = self.getACLabel(self.AS) 
                self.AfE = AClabel
        pass
                 
    def getACLabel(self):
        """
        This method is used to get the affective category label.
        
        Returns:
            list: Affective category label.
        """
        
        result = []
        matches = True
        r = None
         
        for acl in self.affectiveCategories.keys():
            matches = True
            if self.affectiveCategories[acl] != None:
                if len(self.affectiveCategories[acl]) == len(self.AS):
                    for i in range(len(self.AS)):
                        r = self.affectiveCategories[acl][i]
                        matches = matches and self.AS[i] >= r.getMin() and self.AS[i] <= r.getMax()
                else:
                    try:
                        raise Exception("The number of components for the affective category " + acl + " must be the same as the number of the components for the affective state")
                    except Exception as e:
                        print(e)
            if matches:
                result.append(acl)
        return result
                        
                
    def deriveASFromAppraisalVariables(self):
        """
        This method is used to derive the affective state from the appraisal variables.
        
        Returns:
            PAD: Affective state.
        """
        em = []
        if self.AV["expectedness"] != None and self.AV["expectedness"] < 0:
            em.append("surprise")
        if self.AV["desirability"] != None and self.AV["likelihood"] != None:
            if self.AV["desirability"] > 0.5:
                if self.AV["likelihood"] < 1:
                    em.append("hope")
                elif self.AV["likelihood"] == 1:
                    em.append("joy")
            else:
                if self.AV["likelihood"] < 1:
                    em.append("fear")
                elif self.AV["likelihood"] == 1:
                    em.append("sadness")
                if self.AV["causal_attribution"] != None and self.AV["controllability"] != None and self.AV["causal_attribution"] == "other" and self.AV["controllability"] > 0.7:
                    em.append("anger")
        result = PAD()
        result.setP(0.0)
        result.setA(0.0)
        result.setD(0.0)
        for e in em:
            if e == "anger":
                result.setP(result.getP()-0.51)
                result.setA(result.getA()+0.59)
                result.setD(result.getD()+0.25)
            elif e == "fear":
                result.setP(result.getP()-0.64)
                result.setA(result.getA()+0.60)
                result.setD(result.getD()-0.43)
            elif e == "hope":
                result.setP(result.getP()+0.2)
                result.setA(result.getA()+0.2)
                result.setD(result.getD()-0.1)
            elif e == "joy":
                result.setP(result.getP()+0.76)
                result.setA(result.getA()+0.48)
                result.setD(result.getD()+0.35)
            elif e == "sadness":
                result.setP(result.getP()-0.63)
                result.setA(result.getA()-0.27)
                result.setD(result.getD()-0.33)
            elif e == "surprise":
                result.setP(result.getP()+0.4)
                result.setA(result.getA()+0.67)
                result.setD(result.getD()-0.13)
         
        # Averaging
        if len(em) > 0:
            result.setP(result.getP()/len(em))
            result.setA(result.getA()/len(em))
            result.setD(result.getD()/len(em))
        else:
            result = None
        return result
    
    
    def cleanAffectivelyRelevantEvents(self) -> bool:
        return True    
            
    def step(self) -> bool:
        """
        This method is used to apply the second part of the reasoning cycle.
        This method consist in selecting the intention to execute and executing it.
        This part consists of the following steps:
        - Select the intention to execute
        - Apply the clear intention
        - Apply the execution intention

        Raises:
            log.error: If the agent has no intentions
            log.exception: If the agent raises a python exception

        Returns:
            bool: True if the agent executed or cleaned an intention, False otherwise
        """
        options = {
            "SelInt": self.applySelInt,
            "CtlInt": self.applyCtlInt,
            "ExecInt": self.applyExecInt
        }
        if self.current_step in options:
            flag = options[self.current_step]()
            if not flag:
                return False
            else:
                return True
        else:
            return True

    def applySelInt(self) -> bool:
        """
        This method is used to select the intention to execute

        Raises:
            RuntimeError:  If the agent has no intentions

        Returns:
            bool: True if the agent has intentions, False otherwise
            
        - If the intention not have instructions, the current step will be changed to "CtlInt" to clear the intention
        - If the intention have instructions, the current step will be changed to "ExecInt" to execute the intention
         
        """
        while self.C["I"] and not self.C["I"][0]: 
            self.C["I"].popleft() 

        for intention_stack in self.C["I"]: 
            if not intention_stack:
                continue
            intention = intention_stack[-1]
            if intention.waiter is not None:
                if intention.waiter.poll(self.env):
                    intention.waiter = None
                else:
                    continue
            break
        else:
            return False
        
        if not intention_stack:
            return False
        
        instr = intention.instr
        self.intention_stack = intention_stack
        self.intention_selected = intention
        
        if not instr: 
            self.current_step = "CtlInt"
        else:
            self.current_step = "ExecInt"
        self.step()
        return True
    
    def applyExecInt(self) -> bool:
        """
        This method is used to execute the instruction

        Raises:
            AslError: If the plan fails
            
        Returns:
            bool: True if the instruction was executed
        """
        try:
            if self.intention_selected.instr.f(self, self.intention_selected):
                self.intention_selected.instr = self.intention_selected.instr.success # We set the intention.instr to the instr.success
            else:
                self.intention_selected.instr = self.intention_selected.instr.failure # We set the intention.instr to the instr.failure
                if not self.T["i"].instr: 
                    raise AslError("plan failure") 
                
        except AslError as err:
            log = agentspeak.Log(LOGGER)
            raise log.error("%s", err, loc=self.T["i"].instr.loc, extra_locs=self.T["i"].instr.extra_locs)
        except Exception as err:
            log = agentspeak.Log(LOGGER)
            raise log.exception("agent %r raised python exception: %r", self.name, err,
                                loc=self.T["i"].instr.loc, extra_locs=self.T["i"].instr.extra_locs)
        return True
    
    def applyCtlInt(self) -> True:
        """
        This method is used to control the intention
        
        Returns:
            bool: True if the intention was cleared
        """
        self.intention_stack.pop() 
        if not self.intention_stack:
            self.C["I"].remove(self.intention_stack) 
        elif self.intention_selected.calling_term:
            frozen = self.intention_selected.head_term.freeze(self.intention_selected.scope, {})
            
            calling_intention = self.intention_stack[-1]
            if not agentspeak.unify(self.intention_selected.calling_term, frozen, calling_intention.scope, calling_intention.stack):
                raise RuntimeError("back unification failed")
        return True
    
    def run(self) -> None:
        """
        This method is used to run the step cycle of the agent
        We run the second part of the reasoning cycle until the agent has no intentions
        """
        self.current_step = "SelInt"
        while self.step():
            pass

    def waiters(self) -> Iterator[agentspeak.runtime.Waiter]    :
        """
        This method is used to get the waiters of the intentions

        Returns:
            Iterator[agentspeak.runtime.Waiter]: The waiters of the intentions
        """
        return (intention[-1].waiter for intention in self.C["I"]
                if intention and intention[-1].waiter)


class Environment(agentspeak.runtime.Environment):
    """
    This class is used to represent the environment of the agent

    Args:
        agentspeak.runtime.Environment: The environment of the agent defined in the agentspeak library
    """
    def build_agent_from_ast(self, source, ast_agent, actions, agent_cls=agentspeak.runtime.Agent, name=None):
        """
        This method is used to build the agent from the ast

        Returns:
            Tuple[ast_agent, Agent]: The ast of the agent and the agent
            
        """
        
        agent_cls = AffectiveAgent
        
        log = agentspeak.Log(LOGGER, 3)
        agent = agent_cls(self, self._make_name(name or source.name))

        # Add rules to agent prototype.
        for ast_rule in ast_agent.rules:
            variables = {}
            head = ast_rule.head.accept(agentspeak.runtime.BuildTermVisitor(variables))
            consequence = ast_rule.consequence.accept(BuildQueryVisitor(variables, actions, log))
            rule = agentspeak.runtime.Rule(head, consequence)
            agent.add_rule(rule)
        

        # Add plans to agent prototype.
        for ast_plan in ast_agent.plans:
            variables = {}

            head = ast_plan.event.head.accept(agentspeak.runtime.BuildTermVisitor(variables))

            if ast_plan.context:
                context = ast_plan.context.accept(BuildQueryVisitor(variables, actions, log))
            else:
                context = TrueQuery()

            body = agentspeak.runtime.Instruction(agentspeak.runtime.noop)
            body.f = agentspeak.runtime.noop
            if ast_plan.body:
                ast_plan.body.accept(BuildInstructionsVisitor(variables, actions, body, log))

            str_body = str(ast_plan.body)

            plan = agentspeak.runtime.Plan(ast_plan.event.trigger, ast_plan.event.goal_type, head, context, body, ast_plan.body, ast_plan.annotations)
            if ast_plan.args[0] is not None:
                plan.args[0] = ast_plan.args[0]

            if ast_plan.args[1] is not None:
                plan.args[1] = ast_plan.args[1]
            agent.add_plan(plan)
            
        # Add beliefs to agent prototype.
        for ast_belief in ast_agent.beliefs:
            belief = ast_belief.accept(agentspeak.runtime.BuildTermVisitor({}))
            agent.call(agentspeak.Trigger.addition, agentspeak.GoalType.belief,
                       belief, agentspeak.runtime.Intention(), delayed=True)

        # Call initial goals on agent prototype. This is init of the reasoning cycle.
        # ProcMsg
        self.ast_agent = ast_agent
        
        for ast_goal in ast_agent.goals:
            # Start the first part of the reasoning cycle.
            agent.current_step = "SelEv"
            term = ast_goal.atom.accept(agentspeak.runtime.BuildTermVisitor({}))
            agent.C["E"] = [term] if "E" not in agent.C else agent.C["E"] + [term]
                   
         # Add rules to agent prototype.
        for concern in ast_agent.concerns:
            variables = {}
            head = concern.head.accept(agentspeak.runtime.BuildTermVisitor(variables))
            consequence = concern.consequence.accept(BuildQueryVisitor(variables, actions, log))
            concern = Concern(head, consequence)
            agent.add_concern(concern)
            concern_value = agent.test_concern(head, agentspeak.runtime.Intention(), concern)
            
            

        # Trying different ways to multiprocess the cycles of the agents
        multiprocesing = "asyncio2" # threading, asyncio, concurrent.futures, NO
        rc = 1 # number of cycles
        
        if multiprocesing == "asyncio":
            async def hola_thread():
                tiempo_inicial = time.time()
                
                await self.agent_funcs_done
                t = time.time() - tiempo_inicial

            async def agent_func():
                # Ejecutar la regla semántica
                if "E" in agent.C:
                    for i in range(len(agent.C["E"])):
                        agent.current_step = "Appr"
                        agent.affectiveTransitionSystem() # 
                # Sleep 5 seconds
                await asyncio.sleep(0.001)

            async def main():
                self.agent_funcs_done = asyncio.gather(*[agent_func() for i in range(rc)])
                await asyncio.gather(hola_thread(), self.agent_funcs_done)

            asyncio.run(main())
            
        elif multiprocesing == "asyncio2":
            import asyncio

            async def main():
                async def affective():
                    # This function will just sleep for 3 seconds and then set an event
                    #await asyncio.sleep(3)
                    await asyncio.sleep(3)
                    agent.current_step = "SelEv"
                    agent.applySemanticRuleDeliberate()
                    await asyncio.sleep(5)
                    event.set()

                async def rational():
                    # This function will wait for the event to be set before continuing its execution
                    if "E" in agent.C:
                        for i in range(len(agent.C["E"])):
                            agent.current_step = "SelEv"
                            agent.applySemanticRuleDeliberate()
                    await event.wait()

                # Create the event that will be used to synchronize the two functions
                event = asyncio.Event()

                # Create the two tasks that will run the functions
                task1 = asyncio.create_task(affective())
                task2 = asyncio.create_task(rational())

                # Wait for both tasks to complete
                await asyncio.gather(task1, task2)

            # Call the main() function using asyncio.run()
            asyncio.run(main())
            
        else: 
            if "E" in agent.C:
                for i in range(len(agent.C["E"])):   
                    agent.applySemanticRuleDeliberate()

        # Report errors.
        log.throw()

        self.agents[agent.name] = agent
        return ast_agent, agent
    
    def run_agent(self, agent: AffectiveAgent):
        """
        This method is used to run the agent
         
        Args:
            agent (AffectiveAgent): The agent to run
        """
        more_work = True
        while more_work:
            # Start the second part of the reasoning cycle.
            agent.current_step = "SelInt"
            more_work = agent.step()
            if not more_work:
                # Sleep until the next deadline.
                wait_until = agent.shortest_deadline()
                if wait_until:
                    time.sleep(wait_until - self.time())
                    more_work = True
    def run(self):
        """ 
        This method is used to run the environment
         
        """
        maybe_more_work = True
        while maybe_more_work:
            maybe_more_work = False
            for agent in self.agents.values():
                # Start the second part of the reasoning cycle.
                agent.current_step = "SelInt"
                if agent.step():
                    maybe_more_work = True
            if not maybe_more_work:
                deadlines = (agent.shortest_deadline() for agent in self.agents.values())
                deadlines = [deadline for deadline in deadlines if deadline is not None]
                if deadlines:
                    time.sleep(min(deadlines) - self.time())
                    maybe_more_work = True
                    
def call(trigger: agentspeak.Trigger, goal_type: agentspeak.GoalType, term: agentspeak.Literal, agent: AffectiveAgent, intention: agentspeak.runtime.Intention):
    """
    This method is used to call the agent

    Args:
        trigger (agentspeak.Trigger): The trigger of the agent
        goal_type (agentspeak.GoalType): The goal type of the agent
        term  (agentspeak.Literal): The term of the agent
        agent  (AffectiveAgent): The agent to call
        intention (agentspeak.runtime.Intention): The intention of the agent

    """
    return agent.call(trigger, goal_type, term, intention, delayed=False)

class BuildQueryVisitor(agentspeak.runtime.BuildQueryVisitor):
    
    def visit_literal(self, ast_literal):
        term = ast_literal.accept(agentspeak.runtime.BuildTermVisitor(self.variables))
        try:
            arity = len(ast_literal.terms)
            action_impl = self.actions.lookup(ast_literal.functor, arity)
            return ActionQuery(term, action_impl)
        except KeyError:
            if "." in ast_literal.functor:
                self.log.warning("no such action '%s/%d'", ast_literal.functor, arity,
                                 loc=ast_literal.loc,
                                 extra_locs=[t.loc for t in ast_literal.terms])
            return agentspeak.runtime.TermQuery(term)

class TrueQuery(agentspeak.runtime.TrueQuery):
    def __str__(self):
        return "true"

class ActionQuery(agentspeak.runtime.ActionQuery):
    
    def execute(self, agent, intention):
        agent.C["A"] = [(self.term, self.impl)] if "A" not in agent.C else agent.C["A"] + [(self.term, self.impl)]
        for _ in self.impl(agent, self.term, intention):
            yield

class BuildInstructionsVisitor(agentspeak.runtime.BuildInstructionsVisitor):
    def visit_formula(self, ast_formula):
        if ast_formula.formula_type == agentspeak.FormulaType.add:
            term = ast_formula.term.accept(agentspeak.runtime.BuildTermVisitor(self.variables))
            self.add_instr(functools.partial(agentspeak.runtime.add_belief, term),
                           loc=ast_formula.loc, extra_locs=[ast_formula.term.loc])
        elif ast_formula.formula_type == agentspeak.FormulaType.remove:
            term = ast_formula.term.accept(agentspeak.runtime.BuildTermVisitor(self.variables))
            self.add_instr(functools.partial(agentspeak.runtime.remove_belief, term))
        elif ast_formula.formula_type == agentspeak.FormulaType.test:
            term = ast_formula.term.accept(agentspeak.runtime.BuildTermVisitor(self.variables))
            self.add_instr(functools.partial(agentspeak.runtime.test_belief, term),
                           loc=ast_formula.loc, extra_locs=[ast_formula.term.loc])
        elif ast_formula.formula_type == agentspeak.FormulaType.replace:
            removal_term = ast_formula.term.accept(agentspeak.runtime.BuildReplacePatternVisitor())
            self.add_instr(functools.partial(agentspeak.runtime.remove_belief, removal_term))

            term = ast_formula.term.accept(agentspeak.runtime.BuildTermVisitor(self.variables))
            self.add_instr(functools.partial(agentspeak.runtime.add_belief, term),
                           loc=ast_formula.loc, extra_locs=[ast_formula.term.loc])
        elif ast_formula.formula_type == agentspeak.FormulaType.achieve:
            term = ast_formula.term.accept(agentspeak.runtime.BuildTermVisitor(self.variables))
            self.add_instr(functools.partial(call, agentspeak.Trigger.addition, agentspeak.GoalType.achievement, term),
                           loc=ast_formula.loc, extra_locs=[ast_formula.term.loc])
        elif ast_formula.formula_type == agentspeak.FormulaType.achieve_later:
            term = ast_formula.term.accept(agentspeak.runtime.BuildTermVisitor(self.variables))
            self.add_instr(functools.partial(agentspeak.runtime.call_delayed, agentspeak.Trigger.addition, agentspeak.GoalType.achievement, term),
                           loc=ast_formula.loc, extra_locs=[ast_formula.term.loc])
        elif ast_formula.formula_type == agentspeak.FormulaType.term:
            query = ast_formula.term.accept(BuildQueryVisitor(self.variables, self.actions, self.log))
            self.add_instr(functools.partial(agentspeak.runtime.push_query, query))
            self.add_instr(agentspeak.runtime.next_or_fail, loc=ast_formula.term.loc)
            self.add_instr(agentspeak.runtime.pop_query)

        return self.tail
    

class Concern:
    """
    This class is used to represent the concern of the agent
    """ 
    def __init__(self, head, query):         
        self.head = head
        self.query = query
        

    def __str__(self):
        return "%s :- %s" % (self.head, self.query)
    
                
class PairEventDesirability:
        def __init__(self, event):
            self.event = event
            self.AV = {"desirability": None} 

class AffectiveState:

    """
    This class is used to represent the affective state of the agent
    """                          
    
    components = None
    affectiveLabels = None
     
    def __init__(self):
        self.init()
        
    def init(self):
        self.affectiveLabels = []
        self.components = []
        self.setAffectiveLabels()
        if self.affectiveLabels:
            for i in range(len(self.affectiveLabels)):
                self.components.append(0.0)
        self.components = [0.0 for i in range(len(self.affectiveLabels))]
    
    def setAffectiveLabels(self):
        """
        This method is used to set the affective labels
        
        """
        pass
      
    
    def getAffectiveLabels(self):
        """
        This method is used to get the affective labels

        Returns:
            list: Affective labels
        """
        return self.affectiveLabels
    
    def getComponents(self):
        """
        This method is used to get the components of the affective state
        
        Returns:
            list: Components of the affective state
        """
        return self.components
    
    def setComponents(self, comp):
        """
        This method is used to set the components of the affective state
         
        Args:
            comp (list): Components of the affective state
        """
        if comp:
            if self.propperSize(len(comp)):
                self.components = comp
            else:
                raise Exception("Incorrect input data size")
            
    def getComponentsNumber(self):
        """
        This method is used to get the number of components of the affective state
        
        Returns:
            int: Number of components of the affective state
        """
        nr = 0
        if self.components:
            nr = len(self.components)
        return nr
    
    def propperSize(self, size):
        """
        This method is used to check if the size of the affective state is correct
        
        Args:
            size (int): Size of the affective state
             
        Returns: 
            bool: True if the size of the affective state is correct, False otherwise
        """
        result = False
        if self.components:
            result = (size == len(self.components))
        else:
            result = True
        return result
    
    def clone(self):
        pass
     
class PAD(AffectiveState):
    """
    This class is used to represent the PAD of the agent    
    """
     
     
    class PADlabels(Enum):
        pleassure = 0
        arousal = 1
        dominance = 2
        
    def __init__(self, P=None, A=None, D=None):
        super().__init__()
        if (P is not None) and (A is not None) and (D is not None):
            self.setP(P)
            self.setA(A)
            self.setD(D)
            
    def setAffectiveLabels(self):
        """
        This method is used to set the affective labels
        """
        self.affectiveLabels.append(self.PADlabels.pleassure.name)
        self.affectiveLabels.append(self.PADlabels.arousal.name)
        self.affectiveLabels.append(self.PADlabels.dominance.name)
        
    def clone(self):
        """
        This method is used to clone the PAD
         
        Returns:
            PAD: Cloned PAD
        """
        pad = PAD()
        pad.init()
        for i in range(self.getComponentsNumber()):
            pad.getComponents().append(self.getComponents()[i])
        return pad
     
    @staticmethod
    def sameOctant(as1, as2):
        """
        This method is used to check if two affective states are in the same octant
         
        Args:
            as1 (PAD): Affective state 1
            as2 (PAD): Affective state 2
            
        Returns:
            bool: True if the affective states are in the same octant, False otherwise
        """
        result = False
        if as1 is not None and as2 is not None:
            result = (  np.sign(as1.getP()) == np.sign(as2.getP()) and 
                        np.sign(as1.getA()) == np.sign(as2.getA()) and
                        np.sign(as1.getD()) == np.sign(as2.getD()))
        return result
    
    @staticmethod
    def betweenVECandCenter(as1, as2):
        """
        This method is used to check if two affective states are between the vector and the center
        
        Args:
            as1 (PAD): Affective state 1
            as2 (PAD): Affective state 2
            
        Returns:
            bool: True if the affective states are between the vector and the center, False otherwise
        """
        result = False
        if as1 is not None and as2 is not None:
            result = ((as1.getP() < 0 and as2.getP() > as1.getP()) or (as1.getP() > 0 and as2.getP() < as1.getP())) or \
                     ((as1.getA() < 0 and as2.getA() > as1.getA()) or (as1.getA() > 0 and as2.getA() < as1.getA())) or \
                     ((as1.getD() < 0 and as2.getD() > as1.getD()) or (as1.getD() > 0 and as2.getD() < as1.getD()))
        			
        return result
     
    def getP(self):
        return self.components[self.PADlabels.pleassure.value]
     
    def setP(self, P):
        self.components[self.PADlabels.pleassure.value] = P
         
    def getA(self):
        return self.components[self.PADlabels.arousal.value]
    
    def setA(self, A):
        self.components[self.PADlabels.arousal.value] = A
        
    def getD(self):
        return self.components[self.PADlabels.dominance.value]
     
    def setD(self, D):
        self.components[self.PADlabels.dominance.value] = D
        
        
