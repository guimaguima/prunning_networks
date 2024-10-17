import torch
import torch.nn.utils.prune as prune
import numpy as np

def generate_mask(model):
    
    W_list = list(model.parameters())
    
    med_minimal_value = []
    for weight in W_list:
        
        if weight.dim() > 1:
            minimal = torch.min(weight)
            med_minimal_value.append(minimal.item())
    
    
    median_array = np.array(med_minimal_value)
    min_median = np.median(median_array)
    
    
    mask = []
    
    for weight in W_list:
        if weight.dim() > 1:
            mask.append(weight > min_median) 
        else:
            mask.append(torch.ones_like(weight))#always true, otherwise it will affect the bias term
    
    return mask

def apply_mask(model, mask):
    with torch.no_grad(): 
        for param, mask_tensor in zip(model.parameters(), mask):
            param.data *= mask_tensor 