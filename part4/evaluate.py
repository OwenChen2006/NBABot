import json
import torch
from transformers import AutoTokenizer, AutoModel
from backend.utils import ollama_embed
from backend.config import EMBED_MODEL
import numpy as np

def get_finetuned_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    # Load fine-tuned model
    model_path = "/app/part4/fine-tuned-model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    # Load test data (using the same data for simplicity)
    with open("/app/part4/dataset.json", "r") as f:
        dataset = json.load(f)

    # Use first 5 pairs for evaluation
    eval_data = dataset[:5]
    questions = [item['question'] for item in eval_data]
    contexts = [item['context'] for item in eval_data]

    # Get embeddings for all contexts
    finetuned_context_embeds = np.array([get_finetuned_embedding(c, tokenizer, model) for c in contexts])
    baseline_context_embeds = np.array([ollama_embed(EMBED_MODEL, c) for c in contexts])
    
    recall_at_1_finetuned = 0
    recall_at_1_baseline = 0

    for i, q in enumerate(questions):
        # Fine-tuned model
        q_embed_finetuned = get_finetuned_embedding(q, tokenizer, model)
        sims_finetuned = [cosine_similarity(q_embed_finetuned, c_embed) for c_embed in finetuned_context_embeds]
        if np.argmax(sims_finetuned) == i:
            recall_at_1_finetuned += 1

        # Baseline model
        q_embed_baseline = np.array(ollama_embed(EMBED_MODEL, q))
        sims_baseline = [cosine_similarity(q_embed_baseline, c_embed) for c_embed in baseline_context_embeds]
        if np.argmax(sims_baseline) == i:
            recall_at_1_baseline += 1
            
    recall_at_1_finetuned /= len(questions)
    recall_at_1_baseline /= len(questions)

    results = {
        "finetuned_model": {"Recall@1": recall_at_1_finetuned},
        "baseline_model": {"Recall@1": recall_at_1_baseline}
    }
    
    print("Evaluation Results:")
    print(json.dumps(results, indent=2))
    
    with open("/app/part4/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
