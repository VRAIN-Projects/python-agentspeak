from __future__ import print_function

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
    def __init__(self, *args, **kwargs):
        super(AffectiveAgent, self).__init__(*args, **kwargs)
        self.current_step = ""
        print("ESTO ES UN AGENTE AFECTIVO")
        
    def add_rule(self, rule):
        print("AÑADIENDO REGLA")
        print(rule)
        super(AffectiveAgent, self).add_rule(rule)
    
    def add_plan(self, plan):
        print(self.name,"AÑADIENDO PLAN: ",agentspeak.runtime.plan_to_str(plan))
        super(AffectiveAgent, self).add_plan(plan)
        
    def add_belief(self, term, scope):
        print(self.name,"AddIm: ",term)
        self.current_step = "AddIm"
        super(AffectiveAgent, self).add_belief(term, scope)
        
    def test_belief(self, term, intention):
        print(self.name,"TESTEANDO CREENCIA: ",term)
        super(AffectiveAgent, self).test_belief(term, intention)
    
    def remove_belief(self, term, intention):
        print(self.name,"BORRANDO CREENCIA: ",term)
        super(AffectiveAgent, self).remove_belief(term, intention)
        
    def call(self, trigger, goal_type, term, calling_intention, delayed=False):
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

            print(self.name, "RelPl", frozen.functor)
            self.current_step = "RelPl"
            applicable_plans = self.plans[(trigger, goal_type, frozen.functor, len(frozen.args))] 
            intention = agentspeak.runtime.Intention()

            # Find matching plan.
            for plan in applicable_plans: 
                for _ in agentspeak.unify_annotated(plan.head, frozen, intention.scope, intention.stack): 
                    for _ in plan.context.execute(self, intention): 
                        print(self.name, "ApplPl", frozen.functor)
                        self.current_step = "ApplPl"
                        # Here we can implement the SelAppPl algorithm to select the plan to be applied
                        # self.current_step = "SelAppPl"
                        intention.head_term = frozen 
                        intention.instr = plan.body 
                        intention.calling_term = term 

                        if not delayed and self.intentions: 
                            for intention_stack in self.intentions: 
                                if intention_stack[-1] == calling_intention: 
                                    intention_stack.append(intention) 
                                    return True

                        new_intention_stack = collections.deque() 
                        new_intention_stack.append(intention) 
                        self.intentions.append(new_intention_stack) 
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
    
    def step(self):
        
        print(self.name, "CtlInt", self.intentions)
        self.current_step = "CtlInt"
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

        try: 
            print(self.name, "SelInt", str(intention))
            self.current_step = "SelInt"
            if instr.f(self, intention): # If the instruction is true
                print(self.name, "ExecInt", instr)
                self.current_step = "ExecInt"
                intention.instr = instr.success # We set the intention.instr to the instr.success
            else:
                intention.instr = instr.failure # We set the intention.instr to the instr.failure
                if not intention.instr: # If there is no instr.failure
                    raise AslError("plan failure") # We raise an error
        except AslError as err:
            log = agentspeak.Log(LOGGER)
            raise log.error("%s", err, loc=instr.loc, extra_locs=instr.extra_locs)
        except Exception as err:
            log = agentspeak.Log(LOGGER)
            raise log.exception("agent %r raised python exception: %r", self.name, err,
                                loc=instr.loc, extra_locs=instr.extra_locs)

        return True

    def run(self):
        print("Running agent", self.name)
        while self.step():
            pass

