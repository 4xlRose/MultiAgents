from fastapi import FastAPI, HTTPException
from typing import List, Dict
import json

app = FastAPI()

# Ruta del archivo JSON
json_file_path = "robot_2.json"

# Leer el archivo JSON
def read_json_file():
    try:
        with open(json_file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="JSON file not found.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding JSON file.")

@app.get("/data/", response_model=List[Dict[str, int]])
async def get_data():
    """
    Devuelve el contenido completo del archivo JSON.
    """
    return read_json_file()

@app.get("/data/{item_id}", response_model=List[Dict[str, int]])
async def get_data_by_id(item_id: int):
    """
    Devuelve todos los elementos del JSON que coincidan con el ID proporcionado.
    """
    data = read_json_file()
    filtered_data = [item for item in data if item["id"] == item_id]
    if not filtered_data:
        raise HTTPException(status_code=404, detail=f"No items found with id: {item_id}")
    return filtered_data
