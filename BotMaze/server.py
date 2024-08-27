import mesa
from mesa.visualization.ModularVisualization import VisualizationElement
import matplotlib.pyplot as plt
import io
import base64
from model import Environment, Bot, Box, Goal
from agent import Meta, ConvBeltIn, ConvBeltOut, Shelves

import os

BOT_COLORS = ["#4169E1", "#DC143C", "#228B22", "#FFD700", "#FF4500", "#8A2BE2", "#FF1493", "#00FFFF", "#FF69B4",
              "#FFA500"]


def agent_portrayal(agent):
    if isinstance(agent, Bot):
        return {"Shape": "circle", "Filled": "false", "Color": BOT_COLORS[agent.unique_id - 1], "Layer": 1, "r": 1.0,
                "text": f"{agent.unique_id}", "text_color": "black"}
    elif isinstance(agent, Box):
        object_emoji = "📦"
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9, "text_color": "#2F4F4F",
                "Color": "rgba(112, 66, 20, 0.5)", "text": object_emoji}
    elif isinstance(agent, Goal):
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 1, "h": 1, "text_color": "#2F4F4F",
                "Color": "rgba(0, 255, 0, 0.3)", "text": "️⛳️"}
    elif isinstance(agent, Meta):
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9, "text_color": "Black",
        "Color": "#ccbeaf", "text": "🚩"}
    elif isinstance(agent, ConvBeltIn):
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9, "text_color": "Black",
                "Color": "#ccbeaf", "text": "🟩"}
    elif isinstance(agent, ConvBeltOut):
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9, "text_color": "Black",
                "Color": "#ccbeaf", "text": "🟥"}
    elif isinstance(agent, Shelves):
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9, "text_color": "Black",
                "Color": "#ccbeaf", "text": "🗄️"}
    else:
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9, "text_color": "Black",
                "Color": "white", "text": ""}



# A function to read the available Q-Tables files
def get_q_files():
    try:
        files = os.listdir("./q_files")
        files = [f.split(".")[0] for f in files if f.endswith(".json")]
        return ["None"] + sorted(files)
    except FileNotFoundError:
        # os.makedirs("q_files")
        return ["None"]


# A function to read the available maze files
def get_maze_files():
    try:
        files = os.listdir("./mazes")
        files = [f for f in files if f.endswith(".txt")]
        return ["None"] + files
    except FileNotFoundError:
        # os.makedirs("maze_files")
        return ["None"]


# A function to read the number of bots in the environment
def get_num_bots():
    dummy_model = Environment()
    return len(dummy_model.schedule.agents)


grid = mesa.visualization.CanvasGrid(
    agent_portrayal, 20, 20, 600, 600)

# Create a chart to track the battery of the robots
chart_charges = mesa.visualization.ChartModule(
    [
        {f"Label": f"Bot{i + 1}", "Color": BOT_COLORS[i], "label": f"Bot{i + 1} Moves"} for i in
        range(get_num_bots())
    ],
    data_collector_name='datacollector',
    canvas_height=150,
    canvas_width=600
)


# Define the model parameters
def model_params():
    params = {}

    params["desc_file"] = mesa.visualization.Choice(
        name="Maze",
        choices=get_maze_files(),
        value='None',
        description="Choose the maze file",
    )

    for i in range(get_num_bots()):
        params[f"train_bot{i + 1}"] = mesa.visualization.Checkbox(
            name="Train Bot" + str(i + 1),
            value=False,
            description="Train the agent",
        )

        params[f"q_file_bot{i + 1}"] = mesa.visualization.Choice(
            name="Model Bot" + str(i + 1),
            choices=get_q_files(),
            value='None',
            description="Choose the file with the Q-Table",
        )

    params["epsilon"] = mesa.visualization.Slider(
        name="Epsilon",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        description="Epsilon for the epsilon-greedy policy",
    )

    params["train_episodes"] = mesa.visualization.Slider(
        name="Train Episodes",
        min_value=1,
        max_value=10000,
        value=200,
        step=100,
        description="Number of training episodes",
    )

    params["alpha"] = mesa.visualization.Slider(
        name="Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        description="Learning rate",
    )

    params["gamma"] = mesa.visualization.Slider(
        name="Gamma",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.01,
        description="Discount factor",
    )

    params["enable_decay"] = mesa.visualization.Checkbox(
        name="Enable Epsilon Decay",
        value=True,
        description="Enable epsilon decay",
    )

    return params


server = mesa.visualization.ModularServer(
    Environment, [grid, chart_charges],
    "Bot Wars of the Ever-Shifting Maze!", model_params(), 6969
)

server.launch(open_browser=True)
