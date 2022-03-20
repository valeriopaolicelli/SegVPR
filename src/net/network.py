import torch
from torch import nn
import net.functional as func


class MultiScaleAttention(nn.Module):
    def __init__(self, encoder_dim):
        super(MultiScaleAttention, self).__init__()
        self.layer1 = nn.Conv2d(encoder_dim, 64, (3, 3))
        self.layer2 = nn.Conv2d(encoder_dim, 64, (5, 5))
        self.layer3 = nn.Conv2d(encoder_dim, 64, (7, 7))
        self.final_conv = nn.Conv2d(3*64, 1, (1, 1))
        self.act_func = nn.ReLU()
        self.act_func_final = nn.Softplus()

    def forward(self, features):
        x1 = self.act_func(self.layer1(features))
        x2 = self.act_func(self.layer2(features))
        x2 = func.interpol(x2, x1.size()[2:])
        x3 = self.act_func(self.layer3(features))
        x3 = func.interpol(x3, x1.size()[2:])
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.final_conv(x)
        x = self.act_func_final(x)
        return x


class SemVPR(nn.Module):
    def __init__(self, vpr_pooling, sem_branch, num_queries=0, num_negatives=0, DA=None):
        super(SemVPR, self).__init__()
        self.num_queries = num_queries
        self.num_negatives = num_negatives
        self.sem_branch = sem_branch
        self.vpr_pooling = vpr_pooling
        self.DA = DA
        self.attention_layers = MultiScaleAttention(int(self.sem_branch.encoder_dim/2)).cuda()

    def concat_feat(self, features):
        global_encoding = None
        for feat in features:
            if global_encoding is None:
                global_encoding = self.vpr_pooling(feat)
            else:
                global_encoding = torch.cat([global_encoding, self.vpr_pooling(feat)], 1)
        return global_encoding

    def forward(self, inputs, mode='semvpr'):
        if mode == 'only_embeddings':
            multiscale_feat = self.sem_branch(inputs, no_classifier=True)[0]
            att_scores = self.attention_layers(multiscale_feat[0])
            multiscale_feat[0] = func.interpol(att_scores, multiscale_feat[0].size()[2:]) * multiscale_feat[0]
            multiscale_feat[-1] = func.interpol(att_scores, multiscale_feat[-1].size()[2:]) * multiscale_feat[-1]
            return self.concat_feat(multiscale_feat)

        elif mode == 'only_classifier':
            return self.sem_branch(inputs, no_classifier=False)[-1]  # final_sem

        elif mode == 'full_net':
            multiscale_feat, final_sem = self.sem_branch(inputs, no_classifier=False)
            att_scores = self.attention_layers(multiscale_feat[0])
            multiscale_feat[0] = func.interpol(att_scores, multiscale_feat[0].size()[2:]) * multiscale_feat[0]
            multiscale_feat[-1] = func.interpol(att_scores, multiscale_feat[-1].size()[2:]) * multiscale_feat[-1]
            att_scores_up = func.interpol(att_scores.clone().detach(), (final_sem.shape[2], final_sem.shape[3]))
            final_sem = att_scores_up * final_sem
            return multiscale_feat, final_sem

        elif mode == 'att_scores':
            feat_enc = self.sem_branch(inputs, no_classifier=True)[0]
            return self.attention_layers(feat_enc[0])

        else:
            raise Exception(f'Invalid forward mode {mode}')

