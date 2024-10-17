import torch
import torch.nn.utils.prune as prune
import numpy as np
from train import *

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
            

def prunning_iterations(n,model,device,train_loader,test_loader,optimizer,model_delta):
    acc_list = []
    tl_list = []
    for _ in range(n):
        mask = generate_mask(model)
        
        model.load_state_dict(model_delta)
        
        apply_mask(model,mask)
        
        loss_items, final_epoch, acc_items = train(model,device,train_loader,optimizer)
        test_loss,acc = test(model,device,test_loader)
        
        acc_list.append(acc)
        tl_list.append(test_loss)
            
    prunning_plot(acc_list,n)
    prunning_plot(tl_list,n,name='Loss')
        
        
        
def prunning_plot(items,n,name='Accuracy'):
    x = list(range(1, n+1))
    plt.figure(figsize=(5, 2))
    plt.plot(x, items, label=name)
    plt.xlabel("Iterations")
    plt.ylabel("Acuuracy in %")
    plt.title('Model')
    plt.legend()
    plt.show()
    