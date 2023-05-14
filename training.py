from dataset import *
import tqdm 


import torch
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def train(model, train_dataloader, val_dataloader, epochs, lr, seed, saved_model_dir):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(seed)

    # Set optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    best_val_acc = 0.0
    # Train loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training"): # batch is a tuple of (input_ids, attention_mask, labels)

            input_ids = batch[0].to(device) # input_ids is a tensor of shape (batch_size, max_seq_length)
            attention_mask = batch[1].to(device) # attention_mask is a tensor of shape (batch_size, max_seq_length)
            labels = batch[2].to(device) # labels is a tensor of shape (batch_size,)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            train_loss += loss.item()
            train_acc += torch.sum(preds == labels).item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} - Validation"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                val_loss += loss.item()
                val_acc += torch.sum(preds == labels).item()
        
        # Print results
        train_loss /= len(train_dataloader.dataset)
        val_loss /= len(val_dataloader.dataset)
        train_acc /= len(train_dataloader.dataset)
        val_acc /= len(val_dataloader.dataset)

        print(f"Epoch {epoch+1}:")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        torch.save(model.state_dict(), saved_model_dir)



if __name__ == "__main__":
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    online_dataset = OnlineDataset()
    api = API()
    dataset1 = online_dataset.extract_argument('arg_search_framework/data/essay/train.json')
    dataset2 = api.convert_text_to_lst("new_train_dataset_api.txt")
    merged_dataset = get_merged_dataset(dataset1, dataset2, 250)
    train_dataloader, test_dataloader = train_test_split(merged_dataset, 0.8, 64)
    # train(model, train_dataloader, test_dataloader, epochs=3, lr=2e-5)
    print(len(train_dataloader.dataset))