from fastapi import FastAPI
from pydantic import BaseModel

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="Project NEXUS API")


# -----------------------------
# Device configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = "../nexus_lora2"


# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True
)


# -----------------------------
# Load base GPT-2
# -----------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)


# -----------------------------
# Attach LoRA adapter
# -----------------------------
model = PeftModel.from_pretrained(base_model, MODEL_PATH)

model.to(device)
model.eval()


# -----------------------------
# Request schema
# -----------------------------
class PromptRequest(BaseModel):
    prompt: str


# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/")
def home():
    return {"message": "Project NEXUS API running successfully"}


# -----------------------------
# Text generation endpoint
# -----------------------------
@app.post("/generate")
def generate_text(request: PromptRequest):

    prompt = f"""### Instruction:
{request.prompt}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.85,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return {"response": response}