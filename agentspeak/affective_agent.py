from __future__ import print_function
from typing import Union, Tuple, Iterator

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
        self.env = env
        self.name = name

        self.beliefs = collections.defaultdict(lambda: set()) if beliefs is None else beliefs
        self.rules = collections.defaultdict(lambda: []) if rules is None else rules
        self.plans = collections.defaultdict(lambda: []) if plans is None else plans

        
        self.current_step = ""
        self.T = {}
        
        # Circunstance initialization
        self.C = {}
        self.C["I"] = collections.deque()
        
        
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
        
        If the event is a goal addition, we start the reasoning cycle.
        
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
        
        #self.term = self.ast_goal.atom.accept(BuildTermVisitor({}))
        if len(self.C["E"]) > 0:
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
                print(differents.head.functor,self.T["e"].functor )
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

            plan = agentspeak.runtime.Plan(ast_plan.event.trigger, ast_plan.event.goal_type, head, context, body, ast_plan.body, ast_plan.annotations)
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
            # Start the first part of the reasoning cycle.
            agent.current_step = "SelEv"
            term = ast_goal.atom.accept(BuildTermVisitor({}))
            agent.C["E"] = [term] if "E" not in agent.C else agent.C["E"] + [term]

        # Trying different ways to multiprocess the cycles of the agents
        multiprocesing = "NO" # threading, asyncio, concurrent.futures, NO
        rc = 500 # number of cycles
        import time 
        
        if multiprocesing == "threading":
        
            import threading
            import time

            condition = threading.Condition()
            agent.counter = 0

            def hola_thread():
                with condition:
                    print("Start of the thread 1")
                    tiempo_inicial = time.time()
                    while agent.counter < rc:
                        condition.wait()
                    t = time.time() - tiempo_inicial
                    print("End of the thread 1", t)
                    # Open a txt file to save the results
                    with open("results.txt", "a") as f:
                        f.write(f"{multiprocesing};{t};{rc} \n")

            def agent_func():
                with condition:
                    # Ejecutar la regla semántica (commented out since `agent` and `agent.C` are undefined in this code)
                    if "E" in agent.C:
                        for i in range(len(agent.C["E"])):
                            agent.applySemanticRuleDeliberate()
                    # Sleep 5 seconds
                    time.sleep(0.001)
                    print("End of the one thread like thread 2")
                    agent.counter += 1
                    if agent.counter == rc:
                        condition.notify()

            t1 = threading.Thread(target=hola_thread)
            t1.start()

            threads = []
            for i in range(rc):
                t2 = threading.Thread(target=agent_func)
                threads.append(t2)
                t2.start()

            for t in threads:
                t.join()
            t1.join()

        
        elif multiprocesing == "asyncio":
            import asyncio

            async def hola_thread():
                print("Start of the thread 1")
                tiempo_inicial = time.time()
                
                await self.agent_funcs_done
                t = time.time() - tiempo_inicial
                print("End of the thread 1", t)
                with open("results.txt", "a") as f:
                    f.write(f"{multiprocesing};{t};{rc} \n")

            async def agent_func():
                # Ejecutar la regla semántica
                if "E" in agent.C:
                    for i in range(len(agent.C["E"])):
                        agent.applySemanticRuleDeliberate()
                # Sleep 5 seconds
                await asyncio.sleep(0.001)
                print("End of the one thread like thread 2")

            async def main():
                self.agent_funcs_done = asyncio.gather(*[agent_func() for i in range(rc)])
                await asyncio.gather(hola_thread(), self.agent_funcs_done)

            asyncio.run(main())
            
        elif multiprocesing == "concurrent.futures":
            import concurrent.futures
            import time

            def hola_thread():
                print("Start of the thread 1")
                tiempo_inicial = time.time()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    agent_funcs = [executor.submit(agent_func) for i in range(rc)]
                    concurrent.futures.wait(agent_funcs)
                t = time.time() - tiempo_inicial
                print("End of the thread 1", t)
                with open("results.txt", "a") as f:
                    f.write(f"{multiprocesing};{t};{rc} \n")
            def agent_func():
                # Ejecutar la regla semántica
                if "E" in agent.C:
                    for i in range(len(agent.C["E"])):
                        agent.applySemanticRuleDeliberate()
                # Sleep 5 seconds
                time.sleep(0.001)
                print("End of the one thread like thread 2")

            hola_thread()
    
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

############################################################################################################
#################### Classes from the agentspeak library ###################################################
############################################################################################################

class BuildTermVisitor(agentspeak.runtime.BuildTermVisitor):
    pass
    
class BuildReplacePatternVisitor(agentspeak.runtime.BuildReplacePatternVisitor):
    pass

class BuildQueryVisitor(agentspeak.runtime.BuildQueryVisitor):
    
    def visit_literal(self, ast_literal):
        term = ast_literal.accept(BuildTermVisitor(self.variables))
        try:
            arity = len(ast_literal.terms)
            action_impl = self.actions.lookup(ast_literal.functor, arity)
            return ActionQuery(term, action_impl)
        except KeyError:
            if "." in ast_literal.functor:
                self.log.warning("no such action '%s/%d'", ast_literal.functor, arity,
                                 loc=ast_literal.loc,
                                 extra_locs=[t.loc for t in ast_literal.terms])
            return TermQuery(term)

class BuildEventVisitor(agentspeak.runtime.BuildEventVisitor):
    pass

class TrueQuery(agentspeak.runtime.TrueQuery):
    def __str__(self):
        return "true"
    
class FalseQuery(agentspeak.runtime.FalseQuery):
    pass

class ActionQuery(agentspeak.runtime.ActionQuery):
    
    def execute(self, agent, intention):
        agent.C["A"] = [(self.term, self.impl)] if "A" not in agent.C else agent.C["A"] + [(self.term, self.impl)]
        for _ in self.impl(agent, self.term, intention):
            yield

class TermQuery(agentspeak.runtime.TermQuery):
    pass

class AndQuery(agentspeak.runtime.AndQuery):
    pass

class OrQuery(agentspeak.runtime.OrQuery):
    pass

class NotQuery(agentspeak.runtime.NotQuery):
    pass

class UnifyQuery(agentspeak.runtime.UnifyQuery):
    pass

class BuildInstructionsVisitor(agentspeak.runtime.BuildInstructionsVisitor):
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
    

