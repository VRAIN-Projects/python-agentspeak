from __future__ import print_function
from typing import Union, Tuple
import sys
import collections
import copy
import functools
import os.path
import time

import agentspeak
import agentspeak.runtime
import agentspeak.stdlib
import agentspeak.parser
import agentspeak.lexer
import agentspeak.util

from agentspeak import UnaryOp, BinaryOp, AslError, asl_str


LOGGER = agentspeak.get_logger(__name__)


class AffectiveAgent(agentspeak.runtime.Agent):
    """
    This class is a subclass of the Agent class. 
    It is used to add the affective layer to the agent.
    """
    def __init__(self, env: agentspeak.runtime.Environment, name: str, beliefs = None, rules = None, plans = None):
        """
        Constructor of the AffectiveAgent class.
        
        :param env: Environment of the agent.
        :param name: Name of the agent.
        :param beliefs: Beliefs of the agent.
        :param rules: Rules of the agent.
        :param plans: Plans of the agent.
        
        We initialize the intentions queue and the current step.
        """
        self.env = env
        self.name = name

        self.beliefs = collections.defaultdict(lambda: set()) if beliefs is None else beliefs
        self.rules = collections.defaultdict(lambda: []) if rules is None else rules
        self.plans = collections.defaultdict(lambda: []) if plans is None else plans

        self.intentions = collections.deque()
        
        self.current_step = ""
        self.T = {}
        
        
    def add_rule(self, rule: agentspeak.runtime.Rule):
        """
        This method is used to add a rule to the agent.

        Args:
            rule (agentspeak.runtime.Rule): Rule to add.
        """
        print("AÑADIENDO REGLA")
        print(rule)
        super(AffectiveAgent, self).add_rule(rule)
    
    def add_plan(self, plan: agentspeak.runtime.Plan):
        """
        This method is used to add a plan to the agent.
        
        Args:
            plan (agentspeak.runtime.Plan): Plan to add.
        """
        print(self.name,"AÑADIENDO PLAN: ",agentspeak.runtime.plan_to_str(plan))
        super(AffectiveAgent, self).add_plan(plan)
        
    def add_belief(self, term: agentspeak.Literal, scope: dict):
        """This method is used to add a belief to the agent.

        Args:
            term (agentspeak.Literal): Belief to add.
            scope (dict): Dict with the scopes of each term.
        """
        print(self.name,"AÑADIENDO BELIEF: ",term)
        super(AffectiveAgent, self).add_belief(term, scope)
        
    def test_belief(self, term: agentspeak.Literal, intention: agentspeak.runtime.Intention):
        """
        This method is used to test a belief of the agent.

        Args:
            term (agentspeak.Literal): Belief to test.
            intention (agentspeak.runtime.Intention): Intention of the agent.
        """
        print(self.name,"TESTEANDO CREENCIA: ",term)
        super(AffectiveAgent, self).test_belief(term, intention)
    
    def remove_belief(self, term: agentspeak.Literal, intention: agentspeak.runtime.Intention) -> None:
        """
        This method is used to remove a belief of the agent.

        Args:
            term (agentspeak.Literal): Belief to remove.
            intention (agentspeak.runtime.Intention): Intention of the agent.
        """
        print(self.name,"BORRANDO CREENCIA: ",term)
        super(AffectiveAgent, self).remove_belief(term, intention)
        
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
        
        If the event is a goal addition, we add it to the intentions queue.
        
        If the event is a goal deletion, we remove it from the intentions queue.
        
        If the event is a tellHow addition, we tell the agent how to do it.
        
        If the event is a tellHow deletion, we remove the tellHow from the agent.
        
        If the event is a askHow addition, we ask the agent how to do it.
        """
        print(self.name, "We are in the event: ", term.functor)
        # Modify beliefs.        
        if goal_type == agentspeak.GoalType.belief: 
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
        for intention_stack in self.intentions: 
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

        # If the goal is an achievement and the trigger is an addition, then the agent will add the goal to his list of intentions
        if goal_type == agentspeak.GoalType.achievement and trigger == agentspeak.Trigger.addition:
            self.T["e"] = term
            self.frozen = agentspeak.freeze(term, calling_intention.scope, {}) 
            # RelPlan (remove if want to use directly the applicable plans)
            RelPl = self.applyRelPl()

            print(self.name, "ApplPl", frozen.functor)
            self.current_step = "ApplPl"
            #applicable_plans = self.applyAppPl(trigger, goal_type, term, calling_intention, delayed, frozen)
            applicable_plans = self.applyAppPl()
            self.T["i"] = agentspeak.runtime.Intention()

            self.current_step = "SelAppPl"
            plan = self.applySelAppl()
            print(self.name, "SelAppPl", plan)
            if plan is not None:
                print(self.name, "AddIm", plan) 
                self.current_step = "AddIm"
                self.delayed = delayed
                self.calling_intention = calling_intention
                self.applyAddIM()
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
            for intention_stack in self.intentions: 
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
                context = ast_plan.context.accept(agentspeak.runtime.BuildQueryVisitor(variables, actions, log)) 
            else: 
                context = agentspeak.runtime.TrueQuery() 

            body = agentspeak.runtime.Instruction(agentspeak.runtime.noop) 
            body.f = agentspeak.runtime.noop 
            if ast_plan.body: 
                ast_plan.body.accept(agentspeak.runtime.BuildInstructionsVisitor(variables, actions, body, log)) 
                 
            #Converts the Astplan to Plan
            plan = agentspeak.runtime.Plan(ast_plan.event.trigger, ast_plan.event.goal_type, head, context, body,ast_plan.body,ast_plan.dicts_annotations) 
            
            if ast_plan.args[0] is not None:
                plan.args[0] = ast_plan.args[0]

            if ast_plan.args[1] is not None:
                plan.args[1] = ast_plan.args[1]
            
          
            # Add the plan to the agent
            self.add_plan(plan) 

        # If the goal is an askHow and the trigger is an addition, then the agent will find the plan in his list of plans and send it to the agent that asked
        if goal_type == agentspeak.GoalType.askHow and trigger == agentspeak.Trigger.addition: 

           return self._ask_how(term)

        # If the goal is an unTellHow and the trigger is a removal, then the agent will delete the goal from his list of plans   
        if goal_type == agentspeak.GoalType.tellHow and trigger == agentspeak.Trigger.removal:

            label = term.args[2]

            delete_plan = []
            plans = self.plans.values()
            for plan in plans:
                for differents in plan:
                    strplan = agentspeak.runtime.plan_to_str(differents)
                    if strplan.startswith(label):
                        delete_plan.append(differents)
            for differents in delete_plan:
                plan.remove(differents)

        return True 
    
    def applyRelPl(self) -> collections.defaultdict:
        """
        This method is used to find the plans that are related to the goal received as parameter.
        We say that a plan is related to a goal if both have the same functor

        Args:
            trigger (agentspeak.Trigger): Trigger of the goal
            goal_type (agentspeak.GoalType): Type of the goal
            term (agentspeak.Literal): Goal received as parameter
            calling_intention (agentspeak.runtime.Intention): Calling intention of the goal
            delayed (bool): True if the goal is delayed, False otherwise
            frozen (agentspeak.Literal): Frozen goal 

        Returns:
            collections.defaultdict: Dictionary with the plans related to the goal
        """
        RelPlan = collections.defaultdict(lambda: [])
        plans = self.plans.values()
        for plan in plans:
            for differents in plan:
                strplan = agentspeak.runtime.plan_to_str(differents)
                if self.T["e"].functor in strplan.split(":")[0]:
                    RelPlan[(differents.trigger, differents.goal_type, differents.head.functor, len(differents.head.args))].append(differents)
         
        if not RelPlan:
            return False
        self.T["R"] = RelPlan
        return RelPlan
    
    def applyAppPl(self) -> collections.defaultdict:
        """
        This method is used to find the plans that are applicable to the goal received as parameter.
        We say that a plan is applicable to a goal if both have the same functor, the same number of arguments and the context are satisfied

        Args:
            trigger (agentspeak.Trigger): Trigger of the goal
            goal_type (agentspeak.GoalType): Type of the goal
            term (agentspeak.Literal): Goal received as parameter
            calling_intention (agentspeak.runtime.Intention): Calling intention of the goal
            delayed (bool): True if the goal is delayed, False otherwise
            frozen (agentspeak.Literal): Frozen goal
            applicable_plans (collections.defaultdict): Dictionary with the plans related to the goal

        Returns:
            collections.defaultdict: Dictionary with the plans applicable to the goal 
        """
        self.T["Ap"] = self.T["R"][(agentspeak.Trigger.addition, agentspeak.GoalType.achievement, self.frozen.functor, len(self.frozen.args))] 
        return self.T["Ap"] 
    
    def applySelAppl(self) -> agentspeak.runtime.Plan:
        """ 
        This method is used to select the plan that is applicable to the goal received as parameter.
        We say that a plan is applicable to a goal if both have the same functor, the same number of arguments and the context are satisfied
        We select the first plan that is applicable to the goal in the dict of applicable plans

        Args:
            trigger (agentspeak.Trigger): Trigger of the goal
            goal_type (agentspeak.GoalType): Type of the goal
            term (agentspeak.Literal): Goal received as parameter
            calling_intention (agentspeak.runtime.Intention): Calling intention of the goal
            delayed (bool): True if the goal is delayed, False otherwise
            frozen (agentspeak.Literal): Frozen goal
            applicable_plans (collections.defaultdict): Dictionary with the plans related to the goal
            intention (agentspeak.runtime.Intention): Intention of the goal

        Returns:
            agentspeak.runtime.Plan: Plan selected for achieving the goal
        """
        for plan in self.T["Ap"]: 
                for _ in agentspeak.unify_annotated(plan.head, self.frozen, self.T["i"].scope, self.T["i"].stack): 
                    for _ in plan.context.execute(self, self.T["i"]):   
                        self.T["p"] = plan
                        return plan
    def applyAddIM(self) -> bool:
        """
        This method is used to add the intention to the intention stack of the agent

        Args:
            intention (agentspeak.runtime.Intention): Intention of the agent
            plan (agentspeak.runtime.Plan): Plan selected for achieving the goal
            calling_intention (agentspeak.runtime.Intention): Calling intention of the goal
            delayed (bool): True if the goal is delayed, False otherwise
            frozen (agentspeak.Literal): Frozen goal
            term (agentspeak.Literal): Goal received as parameter

        Returns:
            bool: True if the intention is added to the intention stack
        """
        self.T["i"].head_term = self.frozen 
        self.T["i"].instr = self.T["p"].body 
        self.T["i"].calling_term = self.T["e"] 

        if not self.delayed and self.intentions: 
            for intention_stack in self.intentions: 
                if intention_stack[-1] == self.delayed: 
                    intention_stack.append(self.T["i"]) 
                    return True
        new_intention_stack = collections.deque() 
        new_intention_stack.append(self.T["i"]) 
        self.intentions.append(new_intention_stack) 
        return True      
    
    def step(self) -> bool:
        """
        This method is used to execute the agent's intentions

        Raises:
            log.error: If the agent has no intentions
            log.exception: If the agent raises a python exception

        Returns:
            true: If the agent has no intentions
        """
        self.current_step = "SelInt"
        selected = self.applySelInt() #intention, instr
        if isinstance(selected, bool):
            return selected
        else:
            intention, instr = selected
            self.intention_selected = intention
        print(self.name, "SelInt", str(intention))
        try: 
            print(self.name, "ExecInt", instr)
            self.current_step = "ExecInt"
            self.applyExecInt()
        except AslError as err:
            log = agentspeak.Log(LOGGER)
            raise log.error("%s", err, loc=instr.loc, extra_locs=instr.extra_locs)
        except Exception as err:
            log = agentspeak.Log(LOGGER)
            raise log.exception("agent %r raised python exception: %r", self.name, err,
                                loc=instr.loc, extra_locs=instr.extra_locs)

        return True

    def applySelInt(self) -> Union[bool, Tuple[agentspeak.runtime.Intention, agentspeak.runtime.Instruction]]:
        """
        This method is used to select the intention to execute

        Raises:
            RuntimeError:  If the agent has no intentions

        Returns:
            Union[bool, Tuple[agentspeak.runtime.Intention, agentspeak.runtime.Instruction]]: If the agent has no intentions, return False. Otherwise, return the intention and the instruction to execute
        """
        while self.intentions and not self.intentions[0]: # while self.intentions is not empty and the first element of self.intentions is empty
            self.intentions.popleft() # remove the first element of self.intentions

        for intention_stack in self.intentions: 
            # Check if the intention has no length
            if not intention_stack:
                continue
            
            # We select the last intention of the intention_stack ¿?
            intention = intention_stack[-1]

            # Suspended / waiting.
            if intention.waiter is not None:
                if intention.waiter.poll(self.env):
                    intention.waiter = None
                else:
                    continue
            break
        else:
            return False
        
        # Ignore if the intentiosn stack is empty
        if not intention_stack:
            return False
        
        
        instr = intention.instr
        
        if not instr: # If there is no instruction
            intention_stack.pop() # Remove the last element of the intention_stack
            if not intention_stack:
                self.intentions.remove(intention_stack) # Remove the intention_stack from the self.intentions
            elif intention.calling_term:
                frozen = intention.head_term.freeze(intention.scope, {})
                
                calling_intention = intention_stack[-1]
                if not agentspeak.unify(intention.calling_term, frozen, calling_intention.scope, calling_intention.stack):
                    raise RuntimeError("back unification failed")
            return True
        
        return (intention, instr)
    
    def applyExecInt(self) -> None:
        """
        This method is used to execute the instruction

        Args:
            intention (agentspeak.runtime.Intention): Intention of the agent
            instr (agentspeak.runtime.Instruction): Instruction to execute

        Raises:
            AslError: If the plan fails
        """
        if self.intention_selected.instr.f(self, self.intention_selected): # If the instruction is true
            # self.current_step = "CtlInt" ¿?
            self.intention_selected.instr = self.intention_selected.instr.success # We set the intention.instr to the instr.success
        else:
            self.intention_selected.instr = self.intention_selected.instr.failure # We set the intention.instr to the instr.failure
            if not self.T["i"].instr: # If there is no instr.failure
                raise AslError("plan failure") # We raise an error
    
    def applySemanticRuleSense(self, ast_agent):
        self.current_step = "SelEv"
        for ast_goal in ast_agent.goals:
            self.term = ast_goal.atom.accept(agentspeak.runtime.BuildTermVisitor({}))
            self.T["calling_intention"] = agentspeak.runtime.Intention()
            self.T["delayed"] = True
            self.applySemanticRuleDeliberate()
            #self.call(agentspeak.Trigger.addition, agentspeak.GoalType.achievement, self.term, agentspeak.runtime.Intention(), delayed=True)
        pass
    
    def applyCtlInt(self) -> None:
        """
        This method is used to control the intention (Not implemented)
        """
        pass
    def run(self) -> None:
        """
        This method is used to run the step cycle of the agent
        """
        print("Running agent", self.name)
        while self.step():
            pass

