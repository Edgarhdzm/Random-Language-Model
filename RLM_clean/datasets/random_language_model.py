import math, torch, random,importlib
import init, measures
importlib.reload(init)
importlib.reload(measures)

def rlm_make_M(V, beta, seed, device=None, generator=None, dtype=torch.float32): #I change to float 16
    #code for generating Matrix with the distribution
    #It is expecting the vocabulary size and the Beta to generate this matrix. 
    random.seed(seed)
    device = device or torch.device('cpu') #Select the device 
    S = torch.randn((V, V, V), generator=generator, device=device, dtype=dtype) * math.sqrt(math.log(V))
    #S = torch.randn((V, V, V), generator=generator, device=device, dtype=dtype)
    S = S * beta
    return torch.softmax(S.view(V, -1), dim=1).view(V, V, V)

def rlm_sample_trees(num_data, L, M, seed, prior=None, device=None, generator=None):
    #It expects a M sized tensor.
    #L is the number of layers it would have. 
    #number of data is for the amount of trees you are going to have.
    random.seed(seed)
    device = device or M.device
    V = M.shape[0]
    trees = {}
    if prior is None:
        labels = torch.randint(V, (num_data,), device=device, generator=generator)
    else:
        p0 = prior.to(device=device, dtype=M.dtype)
        p0 = p0 / p0.sum()
        labels = torch.multinomial(p0, num_data, replacement=True, generator=generator)
    trees[0] = labels
    for l in range(1, L + 1):
        parents = trees[l-1].reshape(-1)
        probs = M[parents].reshape(parents.numel(), -1)
        idx = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        x = idx // V
        y = idx % V
        trees[l] = torch.stack([x, y], dim=1).reshape(num_data, -1)
    return trees



class RLM:
    """
    Implement the Random Language Model (RLM).
    """

    def __init__(
            self,
            v, # vocabulary size
            L, # number of layers
            beta,
            seed_rules,
            seed_samples,
            num_data,
            probs,
            transform=None
    ):
        self.vocab_size = v
        self.num_data= num_data
        self.M =rlm_make_M(self.vocab_size,beta,seed_rules, device=None)
        self.trees= rlm_sample_trees(num_data,L,self.M, seed_samples, device=None)
        self.entropy = init.measures.conditional_entropy(self.M,self.vocab_size,self.num_data)
        self.marginal = init.measures.marginal(self.M,self.vocab_size,self.num_data)        

        self.transform = transform
        

    def __len__(self):
        return len(self.trees[0])
