import os
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

import os

from mesa.agent import Agent
import numpy as np



class Bot(Agent):

    NUM_OF_ACTIONS = 4
    BASE_PATH = "q_files"

    # Define the movements (0: down, 1: right, 2: up, 3: left)
    MOVEMENTS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  


    def __init__(self, unique_id, model, q_file=None, epsilon=0.1):
        super().__init__(unique_id, model)
        self.q_values = None
        self.done = False
        self.state = None
        self.next_state = None
        self.action = None
        self.next_pos = None
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = 0.999
        self.min_epsilon = 0.01
        self.total_return = 0
        self.training_step = 0
        self.movements = 0
        self.pos_array = []
        self.data_array = []
        self.route = 1
        self.dimension = (41, 41)
        


        self.num_states = self.model.grid.width * self.model.grid.height

        if q_file is None:
            self.reset_q_values()
        else:
            self.load_q_values(q_file)

    def reset_q_values(self):
        self.q_values = {(state, action): np.random.uniform(0.01, 0.5)
                         for state in range(self.num_states)
                         for action in range(self.NUM_OF_ACTIONS)}

    def step(self) -> None:
        if self.state is None:
            self.state = self.model.states[self.pos]
        
        print(f"self state: {self.state}, model goal states: {self.model.goal_states}")
        if self.state in self.model.goal_states:
            self.plot_action_value_grid(self.q_values, self.dimension)
            self.route += 1
            print(f"unique id: {self.unique_id} Numero de ruta: {self.route}")
            self.load_q_values(f"qf_bot{self.unique_id}_r_{self.route}")
            print("Q_file loaded -------------------------------------------------")

        # Agent chooses an action from the policy
        self.action = self.eps_greedy_policy(self.state)

        # Get the next position
        self.next_pos = self.perform(self.pos, self.action)
        self.next_state = self.model.states[self.next_pos]

    def advance(self) -> None:
        # Check if the agent can move to the next position
        if self.model.grid.is_cell_empty(self.next_pos) or self.next_state in self.model.goal_states:
            if self.next_state in self.model.goal_states:
                # Remove the goal agent from the grid
                self.model.grid.remove_agent(self.model.grid.get_cell_list_contents(self.next_pos)[0])
                self.done = True

            # Move the agent to the next position and update everything
            self.model.grid.move_agent(self, self.next_pos)
            self.movements += 1

            # Update the state
            self.state = self.next_state
            
            

            # Get the reward
            reward = self.model.rewards[self.state]

        else:
            # If the agent cannot move to the next position, the reward is -2
            reward = -2
            
        # Update the q-values
        #self._update_q_values(self.state, self.action, reward, self.next_state)

        # Update the total return
        self.total_return += reward

        # Reduce epsilon for exploration-exploitation tradeoff for each 100 movements
        # if self.movements % 100 == 0 and self.model.enable_decay:
        #     self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            
        print(f"self.pos_array: {self.pos_array}") #---------------
            


    def add_data(self, id_value, x, y, route):
        # Crear un diccionario con los datos
        data_dict = {"id": id_value, "x": x, "y": y, "route": route}
        filename = f"robot_{id_value}.json"
        data_array = self.data_array
        # Añadir el diccionario a la lista
        data_array.append(data_dict)
        
        # Guardar los datos en el archivo JSON
        with open(filename, 'w') as file:
            json.dump(data_array, file, indent=4)

        #print(f"Datos añadidos y guardados en {filename}")


    def train(self, episodes=200, alpha=0.1, gamma=0.9, log_interval=10):
        """
        Train the agent using Q-learning.

        Args:
            episodes (int): The number of episodes to train the agent.
            alpha (float): The learning rate.
            gamma (float): The discount factor.
            log_interval (int): Log the progress every `log_interval` episodes.
        """
        initial_pos = self.pos
        initial_state = self.model.states[initial_pos]
        epsilon = self.epsilon

        # Main training loop
        for episode in range(episodes):
            pos = initial_pos
            state = initial_state
            total_return = 0
            done = False
            training_step = 0

            while not done:
                # Increment step counter
                training_step += 1

                # Choose an action using the epsilon-greedy policy
                action = self.random_policy(state) #eps_greedy_policy

                # Perform the action and get the next state
                next_pos = self.perform(pos, action)
                next_state = self.model.states[next_pos]

                # Receive the reward for the next state
                reward = self.model.rewards[next_state]

                # Check if the episode is done (goal state reached)
                if next_state in self.model.goal_states:
                    done = True

                # Update the Q-values using the Bellman equation
                self._update_q_values(state, action, reward, next_state, alpha, gamma)

                # Accumulate the total return for the episode
                total_return += reward

                # Update the agent's position and state if reward is non-negative
                if reward >= 0:
                    pos = next_pos
                    state = next_state

            # Log progress at regular intervals
            if episode % log_interval == 0:
                print(f"Episode {episode}, Step {training_step}, Total Return: {total_return:.2f}")

            # Optional: Decay epsilon for exploration-exploitation tradeoff
            if self.model.enable_decay:
                epsilon = max(self.min_epsilon, epsilon * self.decay_rate)

        # Save the trained Q-values after all episodes
        self.save_q_values()

    def save_q_values(self):
        # Convert tuple keys to strings for JSON serialization
        q_values_str_keys = {str(key): value for key, value in self.q_values.items()}

        # Create the directory if it does not exist
        if not os.path.exists(self.BASE_PATH):
            os.makedirs(self.BASE_PATH)

        # Create a timestamp for the file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the Q-values to a JSON file
        with open(f"{self.BASE_PATH}/qf_bot{self.unique_id}_{timestamp}.json", "w") as f:
            json.dump(q_values_str_keys, f)
        print(f"Q-values saved to q_values{self.unique_id}.json")

    def load_q_values(self, q_file):
        try:
            with open(f"{self.BASE_PATH}/{q_file}.json", "r") as f:
                q_values_str_keys = json.load(f)
            # Convert string keys back to tuples
            self.q_values = {eval(key): value for key, value in q_values_str_keys.items()}
            print(f"Q-values from {q_file} have been loaded.")
        except FileNotFoundError:
            self.reset_q_values()
            print("File not found. Q-values have been reset.")
        except Exception as e:
            self.reset_q_values()
            print(f"Failed to load Q-values: {e}. Q-values have been reset.")

    def perform(self, pos, action) -> tuple:
        x = pos[0] + self.MOVEMENTS[action][0]
        y = pos[1] + self.MOVEMENTS[action][1]
        next_pos = (x, y)
        self.pos_array = []
        self.pos_array.append({"id": self.unique_id, "x": pos[0], "y": pos[1]})
        #self.add_data(self.unique_id, pos[0], pos[1], self.route) #-------------------------------------------------------------------------------
        return next_pos
    
    def get_pos(self):
        unique_id = self.unique_id
        return unique_id, self.pos[0], self.pos[1]
    
    def random_policy(self):
        return np.random.randint(self.NUM_OF_ACTIONS)

    def eps_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            self.training_step += 1
            return self.random_policy()
        else:
            q_values = [self.q_values[state, action] for action in range(self.NUM_OF_ACTIONS)]
            return np.argmax(q_values)

    def _update_q_values(self, state, action, reward, next_state, alpha=0.1, gamma=0.9):
        q_values = [self.q_values[next_state, action] for action in range(self.NUM_OF_ACTIONS)]
        max_q_value = np.max(q_values)
        self.q_values[state, action] += alpha * (reward + gamma * max_q_value - self.q_values[state, action])
    
    
    @staticmethod
    def plot_action_value_grid(action_value_pairs, dimensions):
        #""" Plots the State_Value_Grid """
        def quatro_matrix(left, bottom, right, top, ax=None, triplotkw=None, tripcolorkw=None):
            triplotkw = {} if triplotkw is None else triplotkw
            tripcolorkw = {} if tripcolorkw is None else tripcolorkw
            if not ax: ax = plt.gca()
            n = left.shape[0]
            m = left.shape[1]

            a = np.array([[0, 0], [0, 1], [.5, .5], [1, 0], [1, 1]])
            tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])

            A = np.zeros((n * m * 5, 2))
            Tr = np.zeros((n * m * 4, 3))

            for i in range(n):
                for j in range(m):
                    k = i * m + j
                    A[k * 5:(k + 1) * 5, :] = np.c_[a[:, 0] + j, a[:, 1] + i]
                    Tr[k * 4:(k + 1) * 4, :] = tr + k * 5

            C = np.c_[left.flatten(), bottom.flatten(),
                    right.flatten(), top.flatten()].flatten()

            triplot = ax.triplot(A[:, 0], A[:, 1], Tr, **triplotkw)
            tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)
            return tripcolor
        
        num_rows, num_cols = dimensions

        state_values_grid = np.zeros((*dimensions, 4))
        for pair, value in action_value_pairs.items():
            state, action = pair
            row, col = np.unravel_index(state, dimensions)
            state_values_grid[row][col][action] = value

        left = state_values_grid[:, :, 0].T
        bottom = state_values_grid[:, :, 1].T
        right = state_values_grid[:, :, 2].T
        top = state_values_grid[:, :, 3].T  

        top_value_pos = [(x + .38, y + .25)
                        for y in range(num_rows) for x in range(num_cols)]
        right_value_pos = [(x + .65, y + .5)
                        for y in range(num_rows) for x in range(num_cols)]
        bottom_value_pos = [(x + .38, y + .8)
                            for y in range(num_rows) for x in range(num_cols)]
        left_value_pos = [(x + .05, y + .5)
                        for y in range(num_rows) for x in range(num_cols)]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_ylim(num_rows, 0)
        tripcolor = quatro_matrix(left, top, right, bottom,
                                ax=ax,
                                triplotkw={"color": "k", "lw": 1},
                                tripcolorkw={"cmap": "coolwarm"})
        ax.margins(0)
        ax.set_aspect("equal")
        # fig.colorbar(tripcolor)

        # for i, (xi, yi) in enumerate(top_value_pos):
        #     plt.text(xi, yi, round(top.flatten()[i], 2), size=11, color="k")
        # for i, (xi, yi) in enumerate(right_value_pos):
        #     plt.text(xi, yi, round(right.flatten()[i], 2), size=11, color="k")
        # for i, (xi, yi) in enumerate(left_value_pos):
        #     plt.text(xi, yi, round(left.flatten()[i], 2), size=11, color="k")
        # for i, (xi, yi) in enumerate(bottom_value_pos):
        #     plt.text(xi, yi, round(bottom.flatten()[i], 2), size=11, color="k")
        ax.axis(False)

        plt.show()


class Box(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class Goal(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class Meta(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
class ConvBeltIn(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class ConvBeltOut(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class Shelves(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.shelves_num = None