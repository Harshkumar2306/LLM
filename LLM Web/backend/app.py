import os
import sys
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Ensure the LLM Model directory is in the Python path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
llm_model_path = os.path.join(repo_root, "LLM Model")
sys.path.append(llm_model_path)

from config.gpt_config import GPTConfig
from core.model import GPT
from data.tokenizer import Tokenizer

app = FastAPI()

# Allow CORS for the Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model, tokenizer
    
    print("Loading tokenizer...")
    # The exact 65 unique characters present in the TinyShakespeare dataset
    chars = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    tokenizer = Tokenizer(chars)
    
    print("Loading model...")
    # This must match our 17.6M parameter architecture
    config = GPTConfig(
        vocab_size=65,
        d_model=256,
        n_heads=8,
        n_layers=6,
        context_length=1024
    )
    model = GPT(config)
    
    # The checkpoint should be uploaded by the user to the root of the Hugging Face Space
    ckpt_path = "best.pt"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully from best.pt!")
    else:
        print(f"Warning: {ckpt_path} not found. Using untrained random weights for testing.")
        
    model.to(device)
    model.eval()

# Load it on startup
load_model()

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200

@app.post("/generate")
def generate(req: GenerateRequest):
    if not model or not tokenizer:
        return {"error": "Model not loaded properly"}
        
    encoded = tokenizer.encode(req.prompt)
    x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=req.max_new_tokens)
        
    response = tokenizer.decode(y[0].tolist())
    
    return {"response": response}

@app.get("/")
def health_check():
    return {"status": "ok", "message": "SwiGLU LLM API is running!"}
