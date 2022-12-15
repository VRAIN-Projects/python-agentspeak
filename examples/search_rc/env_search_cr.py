#!/usr/bin/env python

import agentspeak
import agentspeak.runtime
import agentspeak.stdlib

import os


env = agentspeak.runtime.Environment()

with open(os.path.join(os.path.dirname(__file__), "agente1.asl")) as source:
    agents = env.build_agents(source, 1, agentspeak.stdlib.actions)

if __name__ == "__main__":
    env.run()
