import torch
from torch import nn


cos_cri = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')


def cosine_sim_dissim_objective(embeds):
    te = embeds['text_emb']
    ae = embeds['audio_emb']
    ve = embeds['video_emb']
    bs = te.shape[0]
    bs_h = bs // 2
    pos_targets = torch.ones(bs).to(te.device)
    neg_targets = (torch.ones(bs_h) * -1).to(te.device)
    
    cos_loss = 0
    cos_loss += cos_cri(te, ae, pos_targets) + cos_cri(te, ve, pos_targets) + cos_cri(ae, ve, pos_targets)
    cos_loss += cos_cri(te[:bs_h, :], ae[bs_h:, :], neg_targets) + \
        cos_cri(te[bs_h:, :], ae[:bs_h, :], neg_targets)
    cos_loss += cos_cri(te[:bs_h, :], ve[bs_h:, :], neg_targets) + \
        cos_cri(te[bs_h:, :], ve[:bs_h, :], neg_targets)
    cos_loss += cos_cri(ae[:bs_h, :], ve[bs_h:, :], neg_targets) + \
        cos_cri(ae[bs_h:, :], ve[:bs_h, :], neg_targets)
    
    return cos_loss


def weighted_cosine_sim_dissim_objective(embeds):
    te = embeds['text_emb']
    ae = embeds['audio_emb']
    ve = embeds['video_emb']
    bs = te.shape[0]
    bs_h = bs // 2
    pos_targets = torch.ones(bs).to(te.device)
    neg_targets = (torch.ones(bs_h) * -1).to(te.device)
    
    cos_loss = 0
    cos_loss += 0.15*cos_cri(te, ae, pos_targets) + 0.15*cos_cri(te, ve, pos_targets) + 0.7*cos_cri(ae, ve, pos_targets)
    cos_loss += 0.15*(cos_cri(te[:bs_h, :], ae[bs_h:, :], neg_targets) + \
        cos_cri(te[bs_h:, :], ae[:bs_h, :], neg_targets))
    cos_loss += 0.15*(cos_cri(te[:bs_h, :], ve[bs_h:, :], neg_targets) + \
        cos_cri(te[bs_h:, :], ve[:bs_h, :], neg_targets))
    cos_loss += 0.7*(cos_cri(ae[:bs_h, :], ve[bs_h:, :], neg_targets) + \
        cos_cri(ae[bs_h:, :], ve[:bs_h, :], neg_targets))
    
    return cos_loss


def contrastive_loss(emb1, emb2):
    prods = emb1.matmul(emb2.T)
    labels = torch.arange(emb1.shape[0]).to(emb1.device)
    
    loss_func = nn.CrossEntropyLoss()
    loss1 = loss_func(prods, labels)
    loss2 = loss_func(prods.T, labels)
    loss = loss1 + loss2

    return loss


def trimodal_contrastive_objective(embeds):
    cl = contrastive_loss
    te = embeds['text_emb']
    ae = embeds['audio_emb']
    ve = embeds['video_emb']
    
    trimodal_loss = cl(te, ae) + cl(te, ve) + cl(ae, ve)
    
    return trimodal_loss


def trimodal_weighted_contrastive_objective(embeds):
    cl = contrastive_loss
    te = embeds['text_emb']
    ae = embeds['audio_emb']
    ve = embeds['video_emb']
    
    trimodal_loss = 0.15*cl(te, ae) + 0.15*cl(te, ve) + 0.70*cl(ae, ve)
    
    return trimodal_loss
