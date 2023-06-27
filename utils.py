def get_n_params(model):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    
    return n_params


def l2_normalize(t, p=2, dim=1):
    t = t / t.norm(p=p, dim=dim).unsqueeze(1)

    return t
