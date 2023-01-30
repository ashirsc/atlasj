import torch
from torch.nn import CosineSimilarity
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
cossim = CosineSimilarity(dim=0, eps=1e-6)

def dist(v1, v2):
    return cossim(v1, v2)

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

models = [
    'openai/clip-vit-base-patch16',
    'openai/clip-vit-base-patch32',
    'openai/clip-vit-large-patch14',
]

model_id = models[1]

print("Loading tokenizer")
tokenizer = CLIPTokenizer.from_pretrained(model_id)
print("Loading text encoder")
text_encoder = CLIPTextModel.from_pretrained(model_id).to(torch_device)
print("Loading model")
model = CLIPModel.from_pretrained(model_id).to(torch_device)

prompts = [
    "[10/20/2022 1:18 PM] Josh Singer i think the new contract is a lot cleaner I essentially put that we won't charge him any consulting only a retainer fee.",
    "what did josh say about the contract?",
     "A dog", 
     "A labrador", 
     "A poodle",
      "A wolf",
       "A lion", 
       "A house",
] 

print("Tokenizing inputs")
text_inputs = tokenizer(
    prompts, 
    padding="max_length", 
    return_tensors="pt",
    ).to(torch_device)
print("Encoding text")
text_features = model.get_text_features(**text_inputs)
print("Encoding text")
text_embeddings = torch.flatten(text_encoder(text_inputs.input_ids.to(torch_device))['last_hidden_state'],1,-1)

print("\n\nusing text_features")
for i1, label1 in enumerate(prompts):
    for i2, label2 in enumerate(prompts):
        if (i2>=i1):
            print(f"{label1} <-> {label2} = {dist(text_features[i1], text_features[i2]):.4f}")

print("\n\nusing text_embeddings")
for i1, label1 in enumerate(prompts):
    for i2, label2 in enumerate(prompts):
        if (i2>=i1):
            print(f"{label1} <-> {label2} = {dist(text_embeddings[i1], text_embeddings[i2]):.4f}")
