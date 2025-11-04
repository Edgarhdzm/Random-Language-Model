import torch
import torch.nn as nn
import torch.nn.functional as F


def whiten(tensor, eps):	# subtract meand and divide by std along the batch dimension
    """
    Remove the tensor mean and scale by std along the batch dimension.
    
    Returns:
        Whitened tensor.
    """
    wtensor = torch.clone(tensor)
    return (wtensor-wtensor.mean(dim=0,keepdim=True))/(eps+wtensor.std(dim=0,keepdim=True))


def test( model, criterion, dataloader, device):
    """
    Test the model on data from dataloader.
    
    Returns:
        Cross-entropy loss, Classification accuracy.
    """
    model.eval()

    correct = 0
    total = 0
    loss = 0.

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predictions = outputs.max(-1)

            loss += criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1)).item() * targets.numel()
            correct += predictions.eq(targets).sum().item()
            total += targets.numel()

    return loss / total, 1.0 * correct / total


def joint_children(M: torch.Tensor, prior) -> torch.Tensor:
    if not torch.is_tensor(prior):
        prior = torch.tensor(prior, dtype=M.dtype, device=M.device)
    prior = prior / prior.sum()
    Pxy = torch.einsum('n,nab->ab', prior, M)
    Pxy = Pxy / Pxy.sum()
    return Pxy

def entropy(p) -> torch.Tensor:
    if not torch.is_tensor(p):
        p = torch.tensor(p, dtype=torch.float64)
    q = p.clone().reshape(-1).to(torch.float64)
    s = q.sum()
    if torch.isclose(s, torch.tensor(0.0, dtype=q.dtype)):
        raise ValueError("distribución vacía")
    q = q / s
    mask = q > 0
    return -(q[mask] * torch.log(q[mask])).sum()

'''def conditional_entropy_from_joint(Pxy: torch.Tensor) -> torch.Tensor:
    Px = Pxy.sum(dim=1)
    return entropy(Pxy) - entropy(Px)'''


'''
def conditional_entropys(M: torch.Tensor,prior):
    if not torch.is_tensor(prior):
        prior = torch.tensor(prior, dtype=M.dtype, device=M.device)
    prior = prior / prior.sum()
    Pxy = torch.einsum('n,nab->ab', prior, M)
    Pxy = Pxy / Pxy.sum()
    Px = Pxy.sum(dim=1)
    return entropy(Pxy) - entropy(Px)
'''

def conditional_entropy(M: torch.Tensor,V, num_data,prior=None,device=None, generator=None):
    if prior is None:
        prior = torch.full((V,), 1.0 / V, dtype=M.dtype)
    else:
        p0 = prior.to(device=device, dtype=M.dtype)
        p0 = p0 / p0.sum()
        prior = torch.multinomial(p0, num_data, replacement=True, generator=generator)
    prior = prior / prior.sum()
    Pxy = torch.einsum('n,nab->ab', prior, M)
    Pxy = Pxy / Pxy.sum()
    Px = Pxy.sum(dim=1)
    
    return entropy(Pxy) - entropy(Px)


def marginal(M: torch.Tensor,V,num_data,prior=None,device=None, generator=None):
    if prior is None:
        prior = torch.full((V,), 1.0 / V, dtype=M.dtype)
    else:
        p0 = prior.to(device=device, dtype=M.dtype)
        p0 = p0 / p0.sum()
        prior = torch.multinomial(p0, num_data, replacement=True, generator=generator)
    prior = prior / prior.sum()
    marg = torch.sum(joint_children(M,prior))

    marg = entropy(joint_children(M,prior))

    return torch.sum(marg, dim=1)

