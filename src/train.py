import torch
from torch.utils.data import DataLoader
from .model.bert4rec import BERT4Rec
import torch.optim as optim
import torch.nn as nn
import os
from utils import get_device
from tqdm import tqdm
from evaluate import calculate_metrics_batch
import torch.optim as optim
import time
import torch.nn.functional as F
def compute_loss(logits, labels):
    # logits shape: [batch_size, seq_len, vocab_size]
    # labels shape: [batch_size, seq_len]
    
    # Reshape logits and labels
    logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
    labels = labels.view(-1)  # [batch_size * seq_len]
    
    # Create a mask to ignore padding tokens
    mask = (labels > 0).float()
    
    # Compute cross entropy loss
    loss = F.cross_entropy(logits, labels, reduction='none')
    
    # Apply mask and compute mean loss
    loss = (loss * mask).sum() / mask.sum()
    
    return loss
def train(model, train_dataset, test_dataset, device, 
          config):
    
    model.to(device)
    batch_size=config.batch_size
    num_epochs=config.num_epochs
    lr=config.learning_rate
    eval_steps=config.eval_step
    patience=config.patience
    k_values=config.k_values
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_hr = 0
    best_epoch = -1
    patience_counter = 0
    global_step = 0
    
    # Save initial model
    torch.save(model.state_dict(), 'initial_model.pth')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            interactions, labels = batch
            interactions, labels = interactions.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(interactions)
            loss = compute_loss(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            
            if global_step % eval_steps == 0:
                avg_loss = total_loss / eval_steps
                print(f"\nStep {global_step}, Average Loss: {avg_loss:.4f}")
                
                # Evaluate on test set
                hrs, ndcgs = calculate_metrics_batch(model, test_dataset, batch_size, k_values, device)
                
                print("Evaluation Results:")
                for k in k_values:
                    print(f"HR@{k}: {hrs[k]:.4f}, NDCG@{k}: {ndcgs[k]:.4f}")
                
                # Check for improvement
                if hrs[k_values[0]] > best_hr:
                    best_hr = hrs[k_values[0]]
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(model.state_dict(), 'best_model.pth')
                    print("New best model saved!")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"No improvement for {patience} evaluations. Early stopping.")
                    break
                
                total_loss = 0
                model.train()
        
        end_time = time.time()
        print(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds")
        
        if patience_counter >= patience:
            break
    
    print(f"Training completed. Best HR@{k_values[0]}: {best_hr:.4f} at epoch {best_epoch+1}")
    
    # Load best model if exists, otherwise keep the current model
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        print("Loaded the best model.")
    elif os.path.exists('initial_model.pth'):
        model.load_state_dict(torch.load('initial_model.pth'))
        print("No improvement during training. Loaded the initial model.")
    else:
        print("No saved model found. Returning the current model state.")
    
    return model
