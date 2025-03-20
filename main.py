from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import client
import os
from dotenv import load_dotenv


# FastAPI
app = FastAPI()

# initialization process
## loading .env
load_dotenv()

## initializing diffusers client
MODEL_FILE_PATH = os.getenv("MODEL_FILE_PATH")
client = client.DiffusersClient(model_file_path=MODEL_FILE_PATH, device="cuda", cache_dir="./cache")


class LoraInfo(BaseModel):
    path: str
    weights: float


# request body model
class GenerationRequest(BaseModel):
    prompt: str
    ng_prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 25
    scale: int = 5
    seed: int = 1
    clip_skip: int = 2
    file_name: str = "sample.png"
    lora_info: Union[LoraInfo, None] = None


# response body model
class GenerateResponse(BaseModel):
    message: str


@app.post("/image", response_model=GenerateResponse)
async def create_item(req: GenerationRequest):
    """
    Genedating images
    """
    images = client.generate(
        prompt=[req.prompt], 
        ng_prompt=[req.ng_prompt], 
        width=req.width, 
        height=req.height,
        steps=req.steps,
        scale=req.scale,
        seed=req.seed,
        clip_skip=req.clip_skip,
        lora_info=req.lora_info,
    )
    img = images[0]
    output_path = os.path.join(".", req.file_name)
    img.save(output_path)

    return { "message": output_path }
