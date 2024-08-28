from mesa.model import Model
from agent import Box, Goal, Bot, ConvBeltIn, ConvBeltOut, Shelves

from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import numpy as np


class Environment(Model):
    DEFAULT_MODEL_DESC =  ['BBBBBBBBBBBBBBBBBBBB',
                        'B1F2F3F4F5FFFFFFFFFB',
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
                        'BFFFGFFFFFFFFFFFFFFB',
                        'BFFHHHHHHHHHFFFFFFFB',
                        'BFFHHHHHHHHHFFFFFFFB',
                        'BFFFFFFFFFFFFFFFFFFB',
                        'BFFFFFFFFFFFFFFFFFFB',
                        'BFFHHHHHHHHHFFGOOFFB',
                        'BFFFFFFFFFFFFFFOOFFB',
                        'BBBBBBBBBBBBBBBBBBBB']

    def __init__(self, desc_file=None, **kwargs):
        super().__init__()
        self.goal_states = []

        self.enable_decay = kwargs.get("enable_decay", False)

        # Default environment description for the model
        if desc_file is None or desc_file == "None":
            desc = self.DEFAULT_MODEL_DESC
        else:
            desc = self.from_txt_to_desc(desc_file)


        # Get the dimensions of the environment
        M, N = len(desc), len(desc[0])

        # Get the number of bots in the environment from the description
        num_bots = 0
        for i in range(M):
            for j in range(N):
                if desc[i][j].isdigit():
                    num_bots += 1
        self.num_bots = num_bots
        self.bots = {}

        # Define if the agents will be trained and their training hyperparameters
        self.train_episodes = kwargs.get("train_episodes", 1000)
        self.alpha = kwargs.get("alpha", 0.1)
        self.gamma = kwargs.get("gamma", 0.9)
        self.epsilon = kwargs.get("epsilon", 0.1)

        # Get the Q-Files and enable values for the bots
        for i in range(self.num_bots):
            q_file = kwargs.get(f"q_file_bot{i+1}", None)
            setattr(self, f"q_file_bot{i+1}", q_file)

            train_bot = kwargs.get(f"train_bot{i+1}", True)
            setattr(self, f"train_bot{i+1}", train_bot)

        # Create the grid and schedule
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

        model_reporters = {
            f"Bot{i+1}": lambda m, i=i: m.bots[i+1].total_return/(m.bots[i+1].movements+1) for i in range(self.num_bots)
        }

        # Data collector
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            #agent_reporters=agent_reporters
        )

    def step(self):
        # Train the agents in the environment
        for bot_id, bot in self.bots.items():
            # print(f"Training bot {bot_id} {bot.unique_id}")
            if self.__getattribute__(f"train_bot{bot_id}"):
                bot.train(episodes=self.train_episodes, alpha=self.alpha, gamma=self.gamma)
                self.__setattr__(f"train_bot{bot_id}", False)

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
                    q_file = eval(f"self.q_file_bot{bot_num}")
                    bot = Bot(int(f"{bot_num}"), self, q_file, self.epsilon)
                    self.grid.place_agent(bot, (x, y))
                    self.schedule.add(bot)
                    self.bots[bot_num] = bot

                except ValueError:
                    pass

    @staticmethod
    def from_txt_to_desc(file_path):
        """
        Converts a maze text file to a list of strings.

        Args:
            file_path (str): Path to the text file.

        Returns:
            list: A list where each line of the file is an element.
        """
        try:
            with open("./mazes/" + file_path, 'r') as file:
                desc = [line.strip() for line in file.readlines()]  # Read lines, strip newlines
            return desc
        except Exception as e:
            print(f"Error reading the file: {e}")
            return None
