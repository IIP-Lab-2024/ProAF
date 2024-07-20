import torch
import torch.nn as nn
import math
import random
import torch.nn.functional as F
from transformers import BertModel, BertConfig, RobertaTokenizer, RobertaConfig,RobertaModel
import torch
import numpy as np
import timm
from myModels.MyDCT import DctCNN
from myModels.RobertaLMHead import RobertaLMHead
class NetShareFusion(nn.Module):
    def __init__(self, model_dim=256,
                 drop_and_BN='drop-BN',
                 dropout=0.5):
        super(NetShareFusion, self).__init__()
        self.model_dim = model_dim
        self.drop_and_BN = drop_and_BN

        ################################# 代码编写  ############################
        self.bert_config = RobertaConfig()
        self.roberta = RobertaModel.from_pretrained('workspace/roberta-base')
        self.masklm = RobertaLMHead(self.bert_config)
        self.vocab_size = self.bert_config.vocab_size

        self.learn_tokens = -1  # 可学习的部分
        self.img_tokens = -2  # 图片的部分
        self.mask_id = 50264  #

        self.image_encoder = timm.create_model('nf_resnet50', pretrained=False, num_classes=0)
        self.image_encoder.load_state_dict(torch.load('workspace/nf_resnet50/nf_resnet50_ra2-9f236009.pth'), False)
        self.img_linear = nn.Linear(2048,768)

        self.learnable_token_emb = nn.Embedding(  # 生成 [3,768]
            num_embeddings=3, embedding_dim=768)
        self.entity_conv1 = nn.Conv2d(1, 768, (3, 768))
        self.entity_gru = nn.GRU(
            input_size=768, hidden_size=768 // 2, bidirectional=True, batch_first=True, dropout=0.33)
        self.norm = nn.LayerNorm(normalized_shape=768)
        self.dropout = nn.Dropout(dropout)

        # dct图片的处理
        self.dct_img = DctCNN(model_dim,
                              dropout,
                              kernel_sizes=[3, 3, 3],
                              num_channels=[32, 64, 128],
                              in_channel=128,
                              branch1_channels=[64],
                              branch2_channels=[48, 64],
                              branch3_channels=[64, 96, 96],
                              branch4_channels=[32],
                              out_channels=64)
        self.dct_linear = nn.Linear(4096,768)

        self.loss_fct = nn.CrossEntropyLoss()
        self.logsoftmax = nn.LogSoftmax(-1)

        self.positive_weights = nn.Parameter(torch.rand(  # 针对每个标签数字生成10个可训练的随机数
            10), requires_grad=True)
        self.negative_weights = nn.Parameter(torch.rand(
           10), requires_grad=True)
        self.aph = nn.Parameter(torch.rand(1), requires_grad=True)


    def forward(self, promptT1_ids, attentionT1_mask, promptT2_ids, attentionT2_mask, dct_fea,img,
                pos_tokens,neg_tokens,enti_ids,label,flag):
        """
        'promptT1_ids': promptT1_ids,
        'attentionT1_mask': attentionT1_mask,
        'promptT2_ids': promptT2_ids,
        'attentionT2_mask': attentionT2_mask,
        'dct_fea': dct_fea,
        'img': img,
        """
        batch_size = dct_fea.shape[0]
        #################################  离散模板  ############################
        mask1_ids = (promptT1_ids == self.mask_id).nonzero(as_tuple=True)  # 得到掩码mask的索引位置
        # 找到图片的索引位置
        img1_ids = (promptT1_ids == self.img_tokens).nonzero(
            as_tuple=True)

        img_feature = self.image_encoder(img)   # [b,2048]
        img_feature = self.img_linear(img_feature)   # [b,768]
        promptT1_ids[img1_ids] = self.mask_id

        promptT1_emb = self.roberta.embeddings.word_embeddings(  # 转换成embedding形式
            promptT1_ids)
        promptT1_emb[img1_ids] = img_feature  #[b,512,768]


        #################################  连续模板  ############################
        mask2_ids = (promptT2_ids == self.mask_id).nonzero(as_tuple=True)
        img2_ids = (promptT2_ids == self.img_tokens).nonzero(
            as_tuple=True)
        learn2_ids = (promptT2_ids == self.learn_tokens).nonzero(
            as_tuple=True)
        promptT2_ids[img2_ids] = self.mask_id
        promptT2_ids[learn2_ids] = self.mask_id

        replace_embeds = self.learnable_token_emb(torch.arange(3).to(promptT1_emb.device)) # [3,768]
        replace_embeds = replace_embeds.unsqueeze(0).repeat(   # [b,3,768]
            batch_size, 1, 1)

        # 实体的处理
        entity_lens = (enti_ids != 1).sum(1) - 2  # ignore cls, eos, pad , 里面真实的值
        entity_emb = self.roberta.embeddings.word_embeddings(enti_ids)

        entity_reprs = self.entity_conv1(
            entity_emb.unsqueeze(1)).squeeze(-1).transpose(1, 2)
        entity_reprs = self.norm(entity_reprs)
        entity_reprs, _ = self.entity_gru(entity_reprs)
        entity_reprs = entity_reprs.transpose(1, 2)   # [b,768,510] GRU输出特征

        _, _, e_len = entity_reprs.size()  # e_len is the encoded entity seq's max length     510
        n2 = entity_lens.max()
        if n2 == 0:
            n2 = 1
        n1 = e_len / n2
        len_scale = (entity_lens * n1).long()

        eo1, eo2 = replace_embeds[:, 0], replace_embeds[:, 1]  # 得到模板中的两个可学习的参数 [b,768]
        e1, e2 = entity_reprs[:, :, 0], entity_reprs[torch.arange(batch_size), :, len_scale - 1]
        e1, e2 = e1 + eo1, e2 + eo2

        dct_out = self.dct_img(dct_fea)  # [b,4096]
        dct_out = F.relu(self.dct_linear(dct_out))  # [b,768]
        eo3 = replace_embeds[:, 2]  # [b,768]
        e3 =  dct_out + eo3  # [b,768]


        replace_embeds1 = torch.cat([e3.unsqueeze(1), e1.unsqueeze(1), e2.unsqueeze(1)], dim=1)
        replace_embeds1 = self.norm(replace_embeds1)  # [b,3,768]   实体部分的新向量

        promptT2_emb = self.roberta.embeddings.word_embeddings(  # 将文本变成编码形式  [b,512,768]
            promptT2_ids)
        promptT2_emb[learn2_ids] = replace_embeds1.view(-1, 768)
        promptT2_emb[img2_ids] = img_feature


        #################################  开始训练处理1 ############################
        promptT1_outputs = self.roberta(
            inputs_embeds=promptT1_emb, attention_mask=attentionT1_mask)
        sequence1_output = promptT1_outputs.last_hidden_state
        logits1 = self.masklm(sequence1_output) # [b,512,50265]
        _, _, vocab_size = logits1.size()

        mask1_logits = logits1[mask1_ids]  # batch_size, vocab_size   # 根据上面的索引找到mask部分的输出  [b,50265]
        mask1_logits = F.log_softmax(mask1_logits, dim=1)

        mask1_logits = mask1_logits.view(batch_size, -1, vocab_size)  # [b,1,50265]
        _, mask_num, _ = mask1_logits.size()

        mask1_logits = mask1_logits.sum(dim=1).squeeze(1)  # batch_size, vocab_size  [b,50265]

        positive1_logits = mask1_logits[:,pos_tokens]
        negative1_logits = mask1_logits[:,neg_tokens]


        positive1_logits = positive1_logits.sum(1).unsqueeze(1)   # batch_size, 1
        negative1_logits = negative1_logits.sum(1).unsqueeze(1)  # batch_size, 1
        final1_logits = torch.cat([positive1_logits, negative1_logits], dim=1)   # [b,2]

        #################################  开始训练处理2 ############################
        promptT2_outputs = self.roberta(
            inputs_embeds=promptT2_emb, attention_mask=attentionT2_mask)
        sequence2_output = promptT2_outputs.last_hidden_state
        logits2 = self.masklm(sequence2_output)  # [b,512,50265]
        _, _, vocab_size2 = logits2.size()

        mask2_logits = logits2[mask2_ids]  # batch_size, vocab_size   # 根据上面的索引找到mask部分的输出  [b,50265]
        mask2_logits = F.log_softmax(mask2_logits, dim=1)
        #mask2_logits = F.softmax(mask2_logits, dim=1)
        #mask2_logits = torch.sigmoid(mask2_logits)
        mask2_logits = mask2_logits.view(batch_size, -1, vocab_size2)  # [b,1,50265]
        _, mask_num, _ = mask2_logits.size()

        mask2_logits = mask2_logits.sum(dim=1).squeeze(1)  # batch_size, vocab_size  [b,50265]

        #    找出robert里面的相关词乘以权重得出这个词的概率
        positive2_logits = mask2_logits[:, pos_tokens] # [b,10]
        negative2_logits = mask2_logits[:, neg_tokens] # [b,10]

        #positive2_logits = positive2_logits * positive_weight
       # negative2_logits = negative_weight * negative2_logits

        positive2_logits = positive2_logits.sum(1).unsqueeze(1)  # batch_size, 1
        negative2_logits = negative2_logits.sum(1).unsqueeze(1)  # batch_size, 1
        final2_logits = torch.cat([positive2_logits, negative2_logits], dim=1)   # [b,2]

        #################################  结合处理 ############################

        logits_1 = self.logsoftmax(final1_logits) * self.aph  # Log prob of right polarity   # [b,2]
        logits_2 = self.logsoftmax(final2_logits) * (1-self.aph) # Log prob of right polarity   # [b,2]
        logits = (logits_1 + logits_2) / 2   # [b,2]
        loss = self.loss_fct(logits, label)
        return logits, loss


#####################################################################################################################################################
    #################################################################################################################################################
"""
model = NetShareFusion()
promptT1_ids = torch.randint(1,10,(8,512))
promptT2_ids = torch.randint(1,10,(8,512))
dct_fea = torch.randn((8,64,250))
img = torch.randn((8,3,256,256))
enti_ids = torch.randint(1,10,(8,512))
y = model(promptT1_ids,None,promptT2_ids,None,dct_fea,img,None,None,enti_ids,True)
"""

