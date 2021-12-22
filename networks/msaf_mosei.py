import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from MSAF import MSAF

from modules.transformer import TransformerEncoder

class BottleAttentionNet(nn.Module):
    def __init__(self):
        super(BottleAttentionNet, self).__init__()
        self.embed_dim = 32
        self.seq = 4
        self.layer_unimodal = 1
        self.layer_multimodal = 2 # cmu_mosei 
        self.audio_linear = nn.Linear(74, self.embed_dim)
        self.visual_linear = nn.Linear(35, self.embed_dim)
        self.transformer = TransformerEncoder(embed_dim=self.embed_dim, num_heads=8, layers=4, attn_mask=False)
    
    def forward(self, audio, visual):        
        audio = self.audio_linear(audio).permute(1, 0, 2)
        for i in range(self.layer_unimodal):
            audio = self.transformer(audio)

        visual = self.visual_linear(visual).permute(1, 0, 2)
        for i in range(self.layer_unimodal):
            visual = self.transformer(visual)

        fsn = torch.zeros(self.seq, audio.size(1), self.embed_dim).cuda()
        x = torch.cat([audio, fsn], dim = 0)

        for i in range(self.layer_multimodal):
            if i == 0:
                x = self.transformer(x)
                x = torch.cat([x[audio.size(0):,:,:], visual], dim = 0)
                x = self.transformer(x)
            else:
                x = self.transformer(x)

        return x


class MSAFLSTMNet(nn.Module):
    def __init__(self, model_param):
        super(MSAFLSTMNet, self).__init__()
        self.max_feature_layers = 1  # number of layers in unimodal models before classifier

        self.audio_visual_model = BottleAttentionNet()

        self.embed_dim = 32
        self.cross_transformer = TransformerEncoder(embed_dim=self.embed_dim, num_heads=8, layers=4, attn_dropout=0.4, attn_mask=False)
        self.classifcation = TransformerEncoder(embed_dim=self.embed_dim, num_heads=8, layers=4, attn_mask=False)

        if "bert" in model_param:
            text_model = model_param["bert"]["model"]
            self.text_model = nn.ModuleList([
                text_model.lstm1,
                text_model.lstm2
            ])
            self.text_id = model_param["bert"]["id"]

        self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.multimodal_classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, x):
        for i in range(self.max_feature_layers):
            if hasattr(self, "text_id"):
                x[self.text_id], _ = self.text_model[i](x[self.text_id])
                x[self.text_id], _ = self.text_model[i+1](x[self.text_id])
            
            audio_visual_feature = self.audio_visual_model(x[self.audio_id], x[self.visual_id])
            x[self.text_id] = self.layer_norm(x[self.text_id]).permute(1, 0, 2)

            result1 = self.classifcation(x[self.text_id])

            l_av = self.cross_transformer(audio_visual_feature, result1, result1)
            av_l = self.cross_transformer(result1, audio_visual_feature, audio_visual_feature)

            l_result = self.classifcation(av_l)
            av_result = self.classifcation(l_av)

            result1 = result1[-1]
            result2 = audio_visual_feature[-1]
            l_result = l_result[-1]
            av_result = av_result[-1]
        
        return self.multimodal_classifier(result1 + result2 + l_result + av_result)
