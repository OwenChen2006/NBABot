import json
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from torch.utils.data import Dataset
import torch

class QADataset(Dataset):
    def __init__(self, dataset_path, tokenizer):
        with open(dataset_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {"question": item['question'], "context": item['context']}

def main():
    model_name = "intfloat/e5-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    dataset = QADataset("/app/part4/dataset.json", tokenizer)

    training_args = TrainingArguments(
        output_dir="/app/part4/results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        remove_unused_columns=False,
        logging_dir='/app/part4/logs',
    )

    def collate_fn(batch):
        questions = [item['question'] for item in batch]
        contexts = [item['context'] for item in batch]

        tokenized_questions = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
        tokenized_contexts = tokenizer(contexts, return_tensors="pt", padding=True, truncation=True)

        return {
            'input_ids': tokenized_questions['input_ids'],
            'attention_mask': tokenized_questions['attention_mask'],
            'context_input_ids': tokenized_contexts['input_ids'],
            'context_attention_mask': tokenized_contexts['attention_mask'],
        }


    class ContrastiveTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            question_outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            context_outputs = model(input_ids=inputs['context_input_ids'], attention_mask=inputs['context_attention_mask'])
            
            q_embeddings = question_outputs.last_hidden_state.mean(dim=1)
            c_embeddings = context_outputs.last_hidden_state.mean(dim=1)
            
            # Simple contrastive loss
            cos = torch.nn.CosineSimilarity(dim=1)
            sim = cos(q_embeddings, c_embeddings)
            loss = torch.mean(1 - sim)
            
            return (loss, {"outputs": (q_embeddings, c_embeddings)}) if return_outputs else loss

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn
    )

    trainer.train()
    model.save_pretrained("/app/part4/fine-tuned-model")
    tokenizer.save_pretrained("/app/part4/fine-tuned-model")
    print("Fine-tuning complete. Model saved to /app/part4/fine-tuned-model")

if __name__ == "__main__":
    main()
