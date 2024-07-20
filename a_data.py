from json import JSONDecodeError

import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import transforms
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
import json
from timm.data.transforms_factory import create_transform
from torchvision.transforms import Compose, Lambda
import re

class FakeNewsDataset(Dataset):
    def __init__(self, root_dir, img_train, tokenizer, MAX_LEN, train_true,image_transform):
        self.root_dir = root_dir  # "./dataset/gossip_train.csv"
        self.img_train = img_train     # './dataset/gossip_train/'
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN
        self.train_true = train_true
        self.image_transform = image_transform

        self.entity = {}
        if train_true:
            with open("./dataset/caption/gossip_train_caption.json",'r') as f:
                self.captions = json.load(f)

            data1 = np.load("./dataset/dct/gossip_train_dct.npz")  # 0-30000   ===
            self.dct_fea = data1["img_dcts"]

            with open("./dataset/link_entity/gossip_train_entity.txt",'r') as f:
                entity_total = f.readlines()
                for line in entity_total:
                    try:
                        item = line.strip()
                        item = re.sub(r'"([^"]*)"', lambda x: x.group(0).replace("'", '$'), item)
                        item = re.sub(r"'([^']*)'", lambda x: x.group(0).replace('"', '￥'), item)
                        items = item.replace("'", "\"").replace("$", "").replace(", None", "").replace("None,", "").replace("None", "")
                        items1 = items.replace("￥", "\'")
                        items = json.loads(items1)
                        self.entity.update(items)
                    except(JSONDecodeError) as f:
                        print(items1)
                        print(f)
                        print(123)


        else:
            with open("./dataset/caption/gossip_test_caption.json",'r') as f:
                self.captions = json.load(f)

            data1 = np.load("./dataset/dct/gossip_test_dct.npz")  # 0-30000   ===
            self.dct_fea = data1["img_dcts"]

            # 实体信息
            with open("./dataset/link_entity/gossip_test_entity.txt", 'r') as f:
                entity_total = f.readlines()
                for line in entity_total:
                    item = line.strip()
                    item = re.sub(r'"([^"]*)"', lambda x: x.group(0).replace("'", '$'), item)
                    item = re.sub(r"'([^']*)'", lambda x: x.group(0).replace('"', '￥'), item)
                    items = item.replace("'", "\"").replace("$", "").replace(", None", "").replace("None,", "").replace(
                        "None", "")
                    items1 = items.replace("￥", "\'")
                    items = json.loads(items1)
                    self.entity.update(items)

    def __len__(self):
        return self.root_dir.shape[0]


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.root_dir["image"][idx]
        raw = Image.open(self.img_train + img_path)
        raw = raw.convert('RGB') if raw.mode != 'RGB' else raw
        img = self.image_transform(raw).squeeze()     # [3,256,256]

        caption = self.captions[img_path]
        caption = caption.strip().replace("\"","'")
        caption_id = self.tokenizer(caption, padding=False, return_tensors="pt")
        caption_token_id = caption_id['input_ids'][0][1:-1]

        text = self.root_dir["content"][idx]
        text_token = self.tokenizer(text,padding=False, return_tensors="pt")  # 将起填充512
        text_token_id = text_token['input_ids'][0][1:-1]

        enti = ""
        entity_s = self.entity[img_path]
        entity_s = [entity for entity in entity_s if entity is not None]
        for index in entity_s:
            #if len(index['neighbours'])==0:
                #continue
            enti += " "
            enti += index["entity_name"]
            for ne_idx in index["neighbours"]:
                enti += " "
                enti += ne_idx
        enti_ids = self.tokenizer(enti,padding='max_length',max_length=512,truncation=True, return_tensors="pt")['input_ids'][0]

        prompt = "This is <mask> news"
        prompt_token = self.tokenizer(prompt, padding=False, return_tensors="pt")
        p_token_id = prompt_token['input_ids'][0]
        prompt_token_id = p_token_id[1:]    #   This is <mask> news </s>

        cls_id = torch.tensor([p_token_id[0]])  # cls的id  101
        eos_id = torch.tensor([prompt_token_id[-1]])  # sep的id  102
        learn_id = torch.tensor([-1])
        img_id = torch.tensor([-2])
        is_id = torch.tensor([self.tokenizer.convert_tokens_to_ids("is")])
        mask_id = torch.tensor([self.tokenizer.convert_tokens_to_ids("<mask>")])

        can_len = len(prompt_token_id) + len(caption_token_id) + 5
        txt_maxlen = 512 - can_len

        if len(text_token_id) > txt_maxlen :
            text_token_id = text_token_id[:txt_maxlen]
        attention_mask = torch.ones(can_len+len(text_token_id))  # 有效id的长度
        prompt1_id = torch.cat([cls_id,img_id,is_id,caption_token_id,eos_id,text_token_id,eos_id,prompt_token_id],dim=0)  # <=512
        promptT1_ids = torch.ones(512)
        promptT1_ids[:len(prompt1_id)] = prompt1_id
        attentionT1_mask = torch.zeros(512)
        attentionT1_mask[:len(attention_mask)] = attention_mask

        can2_len = len(caption_token_id) + 9
        txt_maxlen2 = 512 - can2_len     # 503

        if len(text_token_id) > txt_maxlen2 : # 文本长度大于txt_maxlen2
            text_token_id = text_token_id[:txt_maxlen2]
        attention_mask2 = torch.ones(can2_len + len(text_token_id))
        prompt2_id = torch.cat([cls_id,img_id,caption_token_id,learn_id,eos_id,mask_id,eos_id,learn_id,text_token_id,learn_id,eos_id],dim=0)

        promptT2_ids = torch.ones(512)
        promptT2_ids[:len(prompt2_id)] = prompt2_id
        attentionT2_mask = torch.zeros(512)
        attentionT2_mask[:len(attention_mask2)] = attention_mask2

        dct_fea = torch.from_numpy(self.dct_fea[idx, :, :]).float()  # [64,250]  tensor

        label = self.root_dir['label'][idx]
        label = torch.tensor(int(label))

        if label==0:
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)

        promptT1_ids = promptT1_ids.flatten().clone().detach().type(torch.LongTensor)
        attentionT1_mask = attentionT1_mask.flatten().clone().detach().type(torch.LongTensor)
        promptT2_ids = promptT2_ids.flatten().clone().detach().type(torch.LongTensor)
        attentionT2_mask = attentionT2_mask.flatten().clone().detach().type(torch.LongTensor)


        sample = {
            'label': label,
            'promptT1_ids' : promptT1_ids,
            'attentionT1_mask' : attentionT1_mask,
            'promptT2_ids': promptT2_ids,
            'attentionT2_mask': attentionT2_mask,
            'enti_ids' : enti_ids,
            'dct_fea': dct_fea,
            'img' : img,
            'len_enti' : len(enti)
        }
        return sample










