import mesa

from model import Environment, Bot, Box, Goal
from agent import Meta, ConvBeltIn, ConvBeltOut, Shelves, Package
import numpy as np

BOT_COLORS = ["Red", "Blue", "Olive", "Black", "Yellow"]


def agent_portrayal(agent):
    if isinstance(agent, Bot):
        return {"Shape": "circle", "Filled": "false", "Color": BOT_COLORS[agent.unique_id - 1], "Layer": 1, "r": 1.0,
                "text": "🤖", "text_color": "black"}
    elif isinstance(agent, Box):
        object_emoji = "🧱"  #np.random.choice(["📦", "🗿", "🪨", "📚"])
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9, "text_color": "Black",
                "Color": "Gray", "text": object_emoji}
    elif isinstance(agent, Goal):
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 1, "h": 1, "text_color": "Black",
                "Color": "white", "text": "️⛳️"}
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
    elif isinstance(agent, Package):
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9, "text_color": "Black",
            "Color": "#ccbeaf", "text": "📦"}
    else:
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9, "text_color": "Black",
                "Color": "white", "text": ""}
    


grid = mesa.visualization.CanvasGrid(
    agent_portrayal, 20, 20, 400, 400)

# Create a chart to track the battery of the robots
chart_charges = mesa.visualization.ChartModule(
    [
        {"Label": "Bot1", "Color": BOT_COLORS[0], "label": "Bot1 Movements"},
        {"Label": "Bot2", "Color": BOT_COLORS[1], "label": "Bot2 Movements"},
        {"Label": "Bot3", "Color": BOT_COLORS[2], "label": "Bot3 Movements"},
        {"Label": "Bot4", "Color": BOT_COLORS[3], "label": "Bot4 Movements"},
    ],
    data_collector_name='datacollector'
)

model_params = {
    "q_file": mesa.visualization.Choice(
        name="q_file",
        choices=['None', "q_values1.npy", "q_values2.npy", "q_values3.npy", "q_values4.npy"],
        value='None',
        description="Choose the file with the Q-Table",
    ),
    "train": mesa.visualization.Checkbox(
        name="train",
        value=False,
        description="Train the agents",
    ),
}

server = mesa.visualization.ModularServer(
    Environment, [grid, chart_charges],
    "Retito", model_params, 6969
)

server.launch(open_browser=True)
