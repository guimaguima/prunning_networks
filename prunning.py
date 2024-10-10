import torch


def generate_mask(model):
    W = list(model.parameters())
    i = 0

    mask = []
    max_minimal_value = -1e4
    for weight in W:
        j=0
        
        for neuron in weight:
            
            if i % 2 == 0:
                minimal = torch.min(neuron)
                max_minimal_value = minimal.item() if minimal>max_minimal_value else max_minimal_value 
                
            j+=1
                
        i+=1
        mask.append(torch.clone(weight).detach().ge(max_minimal_value))
        max_minimal_value = -1e4


    return mask