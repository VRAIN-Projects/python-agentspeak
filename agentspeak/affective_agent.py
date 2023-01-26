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

C = {}

class AffectiveAgent(agentspeak.runtime.Agent):
    """
    This class is a subclass of the Agent class. 
    It is used to add the affective layer to the agent.
    """
    def __init__(self, env: agentspeak.runtime.Environment, name: str, beliefs = None, rules = None, plans = None):
        """
        Constructor of the AffectiveAgent class.
        
        Args:
            env (agentspeak.runtime.Environment): Environment of the agent.
            name (str): Name of the agent.
            beliefs (dict): Beliefs of the agent.
            rules (dict): Rules of the agent.
            plans (dict): Plans of the agent.
        
        We initialize the intentions queue.
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
        super(AffectiveAgent, self).add_rule(rule)
    
    def add_plan(self, plan: agentspeak.runtime.Plan):
        """
        This method is used to add a plan to the agent.
        
        Args:
            plan (agentspeak.runtime.Plan): Plan to add.
        """
        super(AffectiveAgent, self).add_plan(plan)
        
    def add_belief(self, term: agentspeak.Literal, scope: dict):
        """This method is used to add a belief to the agent.

        Args:
            term (agentspeak.Literal): Belief to add.
            scope (dict): Dict with the scopes of each term.
        """
        super(AffectiveAgent, self).add_belief(term, scope)
        
    def test_belief(self, term: agentspeak.Literal, intention: agentspeak.runtime.Intention):
        """
        This method is used to test a belief of the agent.

        Args:
            term (agentspeak.Literal): Belief to test.
            intention (agentspeak.runtime.Intention): Intention of the agent.
        """
        super(AffectiveAgent, self).test_belief(term, intention)
    
    def remove_belief(self, term: agentspeak.Literal, intention: agentspeak.runtime.Intention) -> None:
        """
        This method is used to remove a belief of the agent.

        Args:
            term (agentspeak.Literal): Belief to remove.
            intention (agentspeak.runtime.Intention): Intention of the agent.
        """
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

        if goal_type == agentspeak.GoalType.achievement and trigger == agentspeak.Trigger.addition:
            self.T["e"] = term
            self.frozen = agentspeak.freeze(term, calling_intention.scope, {}) 
            self.T["i"] = agentspeak.runtime.Intention()
            self.current_step = "RelPl"
            self.delayed = delayed
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
            
            head = ast_plan.event.head.accept(BuildTermVisitor(variables)) 

            if ast_plan.context: 
                context = ast_plan.context.accept(BuildQueryVisitor(variables, actions, log)) 
            else: 
                context = TrueQuery() 

            body = agentspeak.runtime.Instruction(agentspeak.runtime.noop) 
            body.f = agentspeak.runtime.noop 
            if ast_plan.body: 
                ast_plan.body.accept(BuildInstructionsVisitor(variables, actions, body, log)) 
                 
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
    
    def applySelEv(self) -> bool:
        """
        This method is used to select the event that will be executed in the next step

        Returns:
            bool: True if the event was selected, False otherwise
        """
        self.term = self.ast_goal.atom.accept(BuildTermVisitor({}))
        self.T["e"] = self.term
        self.frozen = agentspeak.freeze(self.term, agentspeak.runtime.Intention().scope, {}) 
        self.T["i"] = agentspeak.runtime.Intention()
        self.current_step = "RelPl"
        self.delayed = True
        return True
    
    def applyRelPl(self) -> bool:
        """
        This method is used to find the plans that are related to the goal received as parameter.
        We say that a plan is related to a goal if both have the same functor

        Returns:
            bool: True if the plans were found, False otherwise
        """
        RelPlan = collections.defaultdict(lambda: [])
        plans = self.plans.values()
        for plan in plans:
            for differents in plan:
                strplan = agentspeak.runtime.plan_to_str(differents)
                if self.T["e"].functor in strplan.split(":")[0]:
                    RelPlan[(differents.trigger, differents.goal_type, differents.head.functor, len(differents.head.args))].append(differents)
         
        if not RelPlan:
            self.current_step = "SelEv"
            return False
        self.T["R"] = RelPlan
        self.current_step = "AppPl"
        return True
    
    def applyAppPl(self) -> bool:
        """
        This method is used to find the plans that are applicable to the goal received as parameter.
        We say that a plan is applicable to a goal if both have the same functor, the same number of arguments and the context are satisfied

        Returns:
            bool: True if the plans were found, False otherwise
        """
        self.T["Ap"] = self.T["R"][(agentspeak.Trigger.addition, agentspeak.GoalType.achievement, self.frozen.functor, len(self.frozen.args))] 
        self.current_step = "SelAppl"
        return self.T["Ap"] != []
    
    def applySelAppl(self) -> bool:
        """ 
        This method is used to select the plan that is applicable to the goal received as parameter.
        We say that a plan is applicable to a goal if both have the same functor, the same number of arguments and the context are satisfied
        We select the first plan that is applicable to the goal in the dict of applicable plans

        Returns:
            bool: True if the plan was found, False otherwise
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
        C["E"] = [self.T["e"]] if "E" not in C else C["E"] + [self.T["e"]]
        C["I"] = [(self.T["i"].head_term,self.T["i"])] if "I" not in C else C["I"] + [(self.T["i"].head_term,self.T["i"])]
        print(C)
        self.current_step = "SelInt"
        return True      
    
    def applySemanticRuleDeliberate(self):
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
            
    def step(self) -> bool:
        """
        This method is used to execute the agent's intentions

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
        """
        while self.intentions and not self.intentions[0]: 
            self.intentions.popleft() 

        for intention_stack in self.intentions: 
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
        This method is used to control the intention (Not implemented)
        
        Returns:
            bool: True
        """
        self.intention_stack.pop() 
        if not self.intention_stack:
            self.intentions.remove(self.intention_stack) 
        elif self.intention_selected.calling_term:
            frozen = self.intention_selected.head_term.freeze(self.intention_selected.scope, {})
            
            calling_intention = self.intention_stack[-1]
            if not agentspeak.unify(self.intention_selected.calling_term, frozen, calling_intention.scope, calling_intention.stack):
                raise RuntimeError("back unification failed")
        return True
    
    def run(self) -> None:
        """
        This method is used to run the step cycle of the agent
        """
        self.current_step = "SelInt"
        while self.step():
            pass


class Environment(agentspeak.runtime.Environment):
    def build_agent_from_ast(self, source, ast_agent, actions, agent_cls=agentspeak.runtime.Agent, name=None):
        # This function is also called by the optimizer.
        agent_cls = AffectiveAgent
        
        log = agentspeak.Log(LOGGER, 3)
        agent = agent_cls(self, self._make_name(name or source.name))

        # Add rules to agent prototype.
        for ast_rule in ast_agent.rules:
            variables = {}
            head = ast_rule.head.accept(BuildTermVisitor(variables))
            consequence = ast_rule.consequence.accept(BuildQueryVisitor(variables, actions, log))
            agent.add_rule(agentspeak.runtime.Rule(head, consequence))

        # Add plans to agent prototype.
        for ast_plan in ast_agent.plans:
            variables = {}

            head = ast_plan.event.head.accept(BuildTermVisitor(variables))

            if ast_plan.context:
                context = ast_plan.context.accept(BuildQueryVisitor(variables, actions, log))
            else:
                context = TrueQuery()

            body = agentspeak.runtime.Instruction(agentspeak.runtime.noop)
            body.f = agentspeak.runtime.noop
            if ast_plan.body:
                ast_plan.body.accept(BuildInstructionsVisitor(variables, actions, body, log))

            str_body = str(ast_plan.body)

            plan = agentspeak.runtime.Plan(ast_plan.event.trigger, ast_plan.event.goal_type, head, context, body, ast_plan.body, ast_plan.dicts_annotations)
            if ast_plan.args[0] is not None:
                plan.args[0] = ast_plan.args[0]

            if ast_plan.args[1] is not None:
                plan.args[1] = ast_plan.args[1]
            agent.add_plan(plan)
        
        # Add beliefs to agent prototype.
        for ast_belief in ast_agent.beliefs:
            belief = ast_belief.accept(BuildTermVisitor({}))
            agent.call(agentspeak.Trigger.addition, agentspeak.GoalType.belief,
                       belief, agentspeak.runtime.Intention(), delayed=True)

        
        # Call initial goals on agent prototype. This is init of the reasoning cycle.
        # ProcMsg
        self.ast_agent = ast_agent
        
        for ast_goal in ast_agent.goals:
            agent.current_step = "SelEv"
            agent.ast_goal = ast_goal
            agent.applySemanticRuleDeliberate()
            #agent.call(agentspeak.Trigger.addition, agentspeak.GoalType.achievement,
                      # agent.term, agentspeak.runtime.Intention(), delayed=True)

        # Report errors.
        log.throw()

        self.agents[agent.name] = agent
        return ast_agent, agent
    
    def run_agent(self, agent):
            more_work = True
            while more_work:
                agent.current_step = "SelInt"
                more_work = agent.step()
                if not more_work:
                    # Sleep until the next deadline.
                    wait_until = agent.shortest_deadline()
                    if wait_until:
                        time.sleep(wait_until - self.time())
                        more_work = True
    def run(self):
            maybe_more_work = True
            while maybe_more_work:
                maybe_more_work = False
                for agent in self.agents.values():
                    agent.current_step = "SelInt"
                    if agent.step():
                        maybe_more_work = True
                    
                if not maybe_more_work:
                    deadlines = (agent.shortest_deadline() for agent in self.agents.values())
                    deadlines = [deadline for deadline in deadlines if deadline is not None]
                    if deadlines:
                        time.sleep(min(deadlines) - self.time())
                        maybe_more_work = True


class BuildTermVisitor:
    def __init__(self, variables):
        self.variables = variables

    def visit_literal(self, ast_literal):
        return agentspeak.Literal(ast_literal.functor,
            (t.accept(self) for t in ast_literal.terms),
            (t.accept(self) for t in ast_literal.annotations))

    def visit_const(self, ast_const):
        return ast_const.value

    def visit_list(self, ast_list):
        return tuple(t.accept(self) for t in ast_list.terms)

    def visit_linked_list(self, ast_linked_list):
        return agentspeak.LinkedList(
            ast_linked_list.head.accept(self),
            ast_linked_list.tail.accept(self))

    def visit_unary_op(self, ast_unary_op):
        return agentspeak.UnaryExpr(
            ast_unary_op.operator.value,
            ast_unary_op.operand.accept(self))

    def visit_binary_op(self, ast_binary_op):
        return agentspeak.BinaryExpr(
            ast_binary_op.operator.value,
            ast_binary_op.left.accept(self),
            ast_binary_op.right.accept(self))

    def visit_variable(self, ast_variable):
        try:
            return self.variables[ast_variable.name]
        except KeyError:
            if ast_variable.name == "_":
                var = agentspeak.Wildcard()
            else:
                var = agentspeak.Var()

            self.variables[ast_variable.name] = var
            return var


class BuildReplacePatternVisitor(BuildTermVisitor):
    def __init__(self):
        BuildTermVisitor.__init__(self, {})

    def visit_unary_op(self, ast_unary_op):
        return agentspeak.Wildcard()

    def visit_binary_op(self, ast_binary_op):
        return agentspeak.Wildcard()


class BuildQueryVisitor:
    def __init__(self, variables, actions, log):
        self.variables = variables
        self.actions = actions
        self.log = log

    def visit_literal(self, ast_literal):
        term = ast_literal.accept(BuildTermVisitor(self.variables))
        try:
            arity = len(ast_literal.terms)
            action_impl = self.actions.lookup(ast_literal.functor, arity)
            global C
            
            C["A"] = [(term, action_impl)] if "A" not in C else C["A"] + [(term, action_impl)]
            print(C)
            return ActionQuery(term, action_impl)
        except KeyError:
            if "." in ast_literal.functor:
                self.log.warning("no such action '%s/%d'", ast_literal.functor, arity,
                                 loc=ast_literal.loc,
                                 extra_locs=[t.loc for t in ast_literal.terms])
            return TermQuery(term)

    def visit_const(self, ast_const):
        if ast_const.value is True:
            return TrueQuery()
        elif ast_const.value is False:
            return FalseQuery()
        else:
            raise self.log.error("non-boolean const in query context: '%s'",
                                 ast_const.value, loc=ast_const.loc)

    def visit_binary_op(self, ast_binary_op):
        if ast_binary_op.operator == BinaryOp.op_and:
            return AndQuery(ast_binary_op.left.accept(self),
                            ast_binary_op.right.accept(self))
        elif ast_binary_op.operator == BinaryOp.op_or:
            return OrQuery(ast_binary_op.left.accept(self),
                           ast_binary_op.right.accept(self))
        elif ast_binary_op.operator == BinaryOp.op_unify:
            return UnifyQuery(ast_binary_op.left.accept(BuildTermVisitor(self.variables)),
                              ast_binary_op.right.accept(BuildTermVisitor(self.variables)))
        elif not ast_binary_op.operator.value.comp_op:
            self.log.error("invalid operator in query context: '%s'",
                           ast_binary_op.operator.value.lexeme,
                           loc=ast_binary_op.loc,
                           extra_locs=[ast_binary_op.left.loc, ast_binary_op.right.loc])

        return TermQuery(ast_binary_op.accept(BuildTermVisitor(self.variables)))

    def visit_unary_op(self, ast_unary_op):
        if ast_unary_op.operator == UnaryOp.op_not:
            return NotQuery(ast_unary_op.operand.accept(self))
        else:
            raise self.log.error("non-boolean unary operator in query context: '%s'",
                                 ast_unary_op.operator.lexeme, ast_unary_op.loc)

    def visit_variable(self, ast_variable):
        return TermQuery(ast_variable.accept(BuildTermVisitor(self.variables)))


class BuildEventVisitor(BuildTermVisitor):
    def __init__(self, log):
        super(BuildEventVisitor, self).__init__({})
        self.log = log

    def visit_event(self, ast_event):
        ast_event = ast_event.accept(agentspeak.parser.ConstFoldVisitor(self.log))
        return Event(ast_event.trigger, ast_event.goal_type, ast_event.head.accept(self))

    def visit_unary_op(self, op):
        raise self.log.error("event is supposed to be unifiable, but contains non-const expression", loc=op.loc)

    def visit_binary_op(self, op):
        raise self.log.error("event is supposed to be unifiable, but contains non-const expression", loc=op.loc)


class TrueQuery:
    def execute(self, agent, intention):
        yield
    
    def __str__(self):
        return "true"


class FalseQuery:
    def execute(self, agent, intention):
        return
        yield


class ActionQuery:
    def __init__(self, term, impl):
        self.term = term
        self.impl = impl

    def execute(self, agent, intention):
        for _ in self.impl(agent, self.term, intention):
            yield


class TermQuery:
    def __init__(self, term):
        self.term = term

    def execute(self, agent, intention):
        # Boolean constants.
        term = agentspeak.evaluate(self.term, intention.scope)
        if term is True:
            yield
            return
        elif term is False:
            return

        try:
            group = term.literal_group()
        except AttributeError:
            raise AslError("expected boolean or literal in query context, got: '%s'" % term)

        # Query on the belief base.
        for belief in agent.beliefs[group]:
            for _ in agentspeak.unify_annotated(term, belief, intention.scope, intention.stack):
                yield

        choicepoint = object()

        # Follow rules.
        for rule in agent.rules[group]:
            rule = copy.deepcopy(rule)

            intention.stack.append(choicepoint)

            if agentspeak.unify(term, rule.head, intention.scope, intention.stack):
                for _ in rule.query.execute(agent, intention):
                    yield
            # Check reroll
            agentspeak.reroll(intention.scope, intention.stack, choicepoint)

    def __str__(self):
        return str(self.term)


class AndQuery:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def execute(self, agent, intention):
        for _ in self.left.execute(agent, intention):
            for _ in self.right.execute(agent, intention):
                yield

    def __str__(self):
        return "(%s & %s)" % (self.left, self.right)


class OrQuery:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def execute(self, agent, intention):
        for _ in self.left.execute(agent, intention):
            yield

        for _ in self.right.execute(agent, intention):
            yield

    def __str__(self):
        return "(%s | %s)" % (self.left, self.right)


class NotQuery:
    def __init__(self, query):
        self.query = query

    def execute(self, agent, intention):
        choicepoint = object()
        intention.stack.append(choicepoint)

        success = any(True for _ in self.query.execute(agent, intention))

        agentspeak.reroll(intention.scope, intention.stack, choicepoint)

        if not success:
            yield


class UnifyQuery:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def execute(self, agent, intention):
        return agentspeak.unify_annotated(self.left, self.right, intention.scope, intention.stack)

    def __str__(self):
        return "(%s = %s)" % (self.left, self.right)
class BuildInstructionsVisitor:
    def __init__(self, variables, actions, tail, log):
        self.variables = variables
        self.actions = actions
        self.tail = tail
        self.log = log

    def add_instr(self, f, loc=None, extra_locs=()):
        self.tail.success = agentspeak.runtime.Instruction(f, loc, extra_locs)
        self.tail = self.tail.success
        return self.tail

    def visit_formula(self, ast_formula):
        if ast_formula.formula_type == agentspeak.FormulaType.add:
            term = ast_formula.term.accept(BuildTermVisitor(self.variables))
            self.add_instr(functools.partial(agentspeak.runtime.add_belief, term),
                           loc=ast_formula.loc, extra_locs=[ast_formula.term.loc])
        elif ast_formula.formula_type == agentspeak.FormulaType.remove:
            term = ast_formula.term.accept(BuildTermVisitor(self.variables))
            self.add_instr(functools.partial(agentspeak.runtime.remove_belief, term))
        elif ast_formula.formula_type == agentspeak.FormulaType.test:
            term = ast_formula.term.accept(BuildTermVisitor(self.variables))
            self.add_instr(functools.partial(agentspeak.runtime.test_belief, term),
                           loc=ast_formula.loc, extra_locs=[ast_formula.term.loc])
        elif ast_formula.formula_type == agentspeak.FormulaType.replace:
            removal_term = ast_formula.term.accept(BuildReplacePatternVisitor())
            self.add_instr(functools.partial(agentspeak.runtime.remove_belief, removal_term))

            term = ast_formula.term.accept(BuildTermVisitor(self.variables))
            self.add_instr(functools.partial(agentspeak.runtime.add_belief, term),
                           loc=ast_formula.loc, extra_locs=[ast_formula.term.loc])
        elif ast_formula.formula_type == agentspeak.FormulaType.achieve:
            term = ast_formula.term.accept(BuildTermVisitor(self.variables))
            self.add_instr(functools.partial(call, agentspeak.Trigger.addition, agentspeak.GoalType.achievement, term),
                           loc=ast_formula.loc, extra_locs=[ast_formula.term.loc])
        elif ast_formula.formula_type == agentspeak.FormulaType.achieve_later:
            term = ast_formula.term.accept(BuildTermVisitor(self.variables))
            self.add_instr(functools.partial(agentspeak.runtime.call_delayed, agentspeak.Trigger.addition, agentspeak.GoalType.achievement, term),
                           loc=ast_formula.loc, extra_locs=[ast_formula.term.loc])
        elif ast_formula.formula_type == agentspeak.FormulaType.term:
            query = ast_formula.term.accept(BuildQueryVisitor(self.variables, self.actions, self.log))
            self.add_instr(functools.partial(agentspeak.runtime.push_query, query))
            self.add_instr(agentspeak.runtime.next_or_fail, loc=ast_formula.term.loc)
            self.add_instr(agentspeak.runtime.pop_query)

        return self.tail

    def visit_for(self, ast_for):
        query = ast_for.generator.accept(BuildQueryVisitor(self.variables, self.actions, self.log))
        self.add_instr(functools.partial(agentspeak.runtime.push_query, query))

        for_head = self.add_instr(agentspeak.runtime.next_or_fail)

        last_in_loop = ast_for.body.accept(self)
        last_in_loop.success = for_head

        self.tail = agentspeak.runtime.Instruction(agentspeak.runtime.pop_query)
        for_head.failure = self.tail
        return self.tail

    def visit_if_then_else(self, ast_if_then_else):
        query = ast_if_then_else.condition.accept(BuildQueryVisitor(self.variables, self.actions, self.log))
        self.add_instr(functools.partial(agentspeak.runtime.push_query, query))
        test_instr = self.add_instr(agentspeak.runtime.next_or_fail)

        tail = agentspeak.runtime.Instruction(agentspeak.runtime.pop_query)

        if ast_if_then_else.if_body:
            if_tail = ast_if_then_else.if_body.accept(self)
            if_tail.success = tail
        else:
            test_instr.success = tail

        if ast_if_then_else.else_body:
            else_head = agentspeak.runtime.Instruction(agentspeak.runtime.noop)
            test_instr.failure = else_head
            self.tail = else_head
            ast_if_then_else.else_body.accept(self)
            self.tail.success = tail
        else:
            test_instr.failure = tail

        self.tail = tail
        return self.tail

    def visit_while(self, ast_while):
        tail = agentspeak.runtime.Instruction(agentspeak.runtime.pop_choicepoint)

        query = ast_while.condition.accept(BuildQueryVisitor(self.variables, self.actions, self.log))
        while_head = self.add_instr(functools.partial(agentspeak.runtime.push_query, query))
        self.add_instr(agentspeak.runtime.push_choicepoint)

        test_instr = self.add_instr(agentspeak.runtime.next_or_fail)
        test_instr.failure = tail

        self.add_instr(agentspeak.runtime.pop_query)

        ast_while.body.accept(self)
        while_tail = self.add_instr(agentspeak.runtime.pop_choicepoint)
        while_tail.success = while_head

        self.tail = tail
        return self.add_instr(agentspeak.runtime.pop_query)

    def visit_body(self, ast_body):
        for formula in ast_body.formulas:
            formula.accept(self)

        return self.tail
    
def call(trigger, goal_type, term, agent, intention):
    return agent.call(trigger, goal_type, term, intention, delayed=False)
