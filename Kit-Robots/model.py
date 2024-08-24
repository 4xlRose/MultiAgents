from mesa.model import Model
from agent import Box, Goal, Bot, Meta, ConvBeltIn, ConvBeltOut, Shelves, Package

from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import numpy as np



class Environment(Model):
    DEFAULT_MODEL_DESC = ['BBBBBBBBBBBBBBBBBBBB',
                        'B1F2FFFFFFFFFFFFFFFB',
                        'BFFFFFFFFFFFFFFFFFFB',
                        'BFFFFFFFFFFFFFFFGIIB',
                        'BFFFFFFFFFFFFFFFFIIB',
                        'BFFHHHHHHHHFFFFFFFFB',
                        'BFFHHHHHHHHFFFFFFFFB',
                        'BFFFGFFFFFFFFFHHFFHB',
                        'BFFFFFFFFFFFFFHHFFHB',
                        'BFFHHHHHHHHFFFHHFFHB',
                        'BFFHHHHHHHHFFFHHFFHB',
                        'BFFFFFFFFFFFFFFFFFHB',
                        'BFFFGFFFFFFFFFFF3FFB',
                        'BFFHHHHHHHHHFFFF4FFB',
                        'BFFHHHHHHHHHFFFFFFFB',
                        'BFFFFFFFFFFFFFFFFFFB',
                        'BFFFFFFFFFFFFFFFFFFB',
                        'BFFHHHHHHHHHFFGOOFFB',
                        'BFFFFFFFFFFFFFFOOFFB',
                        'BBBBBBBBBBBBBBBBBBBB']
    
    # DEFAULT_MODEL_DESC = [
    #     'BBBBBBBBBBBBBBBBBBBB',
    #     'B1FFFFFFFFFFFFFFFB2B',
    #     'BBFFFFFBBBBBBFFFFBFB',
    #     'BFFBBBBBFFFFBFFBBBFB',
    #     'BFFFFBBBFBBBBFFFBFFB',
    #     'BBBFBBFFFBBFFFFBBFFB',
    #     'BFFBBFFBFFFFBBFFBBFB',
    #     'BBFBFFFFBBBFFBBFFFFB',
    #     'BFBBFFFFFBFFBBFFFFFB',
    #     'BFFFGFFBBFFFFFBBBBBB',
    #     'BFFFFBBFFBBBBBBFFFFB',
    #     'BBFBBFFFFFBBFFFFBBBB',
    #     'BBFFFFBBFFFFBBFFBBFB',
    #     'BFFBBBFFFFFBFFFFFBFB',
    #     'BFBFBBFFFFFFFBBFFFFB',
    #     'BFBBFFBFFGFFBBFBBBBB',
    #     'BFFFFFBBFFFFBBFFFFFB',
    #     'BFBFBFBFBBFFFFFFBBFB',
    #     'B3FFFFFFFBBFFFFFFF4B',
    #     'BBBBBBBBBBBBBBBBBBBB'
    # ]

    def __init__(self, desc=None, q_file=None, train=False):
        super().__init__()
        self._q_file = q_file

        self.goal_states = []

        # Default environment description for the model
        self.train = train
        if desc is None:
            desc = self.DEFAULT_MODEL_DESC

        M, N = len(desc), len(desc[0])

        self.grid = SingleGrid(M, N, False)
        self.schedule = SimultaneousActivation(self)

        # Place agents in the environment
        self.place_agents(desc)

        self.states = {}
        self.rewards = {}
        for state, cell in enumerate(self.grid.coord_iter()):
            a, pos = cell

            # Define states for the environment
            self.states[pos] = state

            # Define rewards for the environment
            if isinstance(a, Goal):
                self.rewards[state] = 1
                self.goal_states.append(state)
            elif isinstance(a, Box):
                self.rewards[state] = -1
            elif isinstance(a, Shelves):
                self.rewards[state] = -1
            elif isinstance(a, ConvBeltIn):
                self.rewards[state] = -1
            elif isinstance(a, ConvBeltOut):
                self.rewards[state] = -1
            else:
                self.rewards[state] = 0

        reporters = {
            f"Bot{i+1}": lambda m, i=i: m.schedule.agents[i].total_return for i in range(len(self.schedule.agents))
        }
        # Data collector
        self.datacollector = DataCollector(
            model_reporters=reporters
        )
        #reporters["Enviroment"] = self.get_mapa
        
    #def get_mapa(self):
        

    def step(self):
        # Train the agents in the environment
        if self.train and self._q_file is not None:
            for agent in self.schedule.agents:
                agent.train()
                self.train = False

        self.datacollector.collect(self)

        self.schedule.step()

        self.running = True #not any([a.done for a in self.schedule.agents])

    def place_agents(self, desc: list):
        M, N = self.grid.height, self.grid.width
        for pos in self.grid.coord_iter():
            _, (x, y) = pos
            if desc[M - y - 1][x] == 'B':
                box = Box(int(f"1000{x}{y}"), self)
                self.grid.place_agent(box, (x, y))
            elif desc[M - y - 1][x] == 'G':
                meta = Goal(int(f"10{x}{y}"), self)
                self.grid.place_agent(meta, (x, y))
            elif desc[M - y - 1][x] == 'I': #Cinta transportadora/ConvBelt in
                convBeltIn = ConvBeltIn(int(f"{x}{y}") + 1, self)
                self.grid.place_agent(convBeltIn, (x, y))
            elif desc[M - y - 1][x] == 'O': #Cinta transportadora/ConvBelt out
                convBeltOut = ConvBeltOut(int(f"{x}{y}") + 1, self)
                self.grid.place_agent(convBeltOut, (x, y))
            elif desc[M - y - 1][x] == 'H': #shelve
                shelves = Shelves(int(f"{x}{y}") + 1, self)
                self.grid.place_agent(shelves, (x, y))
            else:
                try:
                    bot_num = int(desc[M - y - 1][x])
                    bot = Bot(int(f"{bot_num}"), self, self._q_file)
                    self.grid.place_agent(bot, (x, y))
                    self.schedule.add(bot)

                except ValueError:
                    pass
