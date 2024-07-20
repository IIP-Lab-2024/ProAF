import pandas as pd
import torch
from sklearn.utils import compute_class_weight
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer
from torch.utils.data import DataLoader,  Subset
import transformers
from a_data import *
from a_train import *
from a_model import *
import os

df_train = pd.read_csv("./dataset/gossip_train.csv")
df_test = pd.read_csv("./dataset/gossip_test.csv")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# 实例化 BERT tokenizer
# 对英文进行分词处理，使用bert-base-uncased分词工具
tokenizer = RobertaTokenizer.from_pretrained('workspace/roberta-base')
MAX_LEN = 512
img_train = './dataset/Images/gossip_train/'
img_test = './dataset/Images/gossip_test/'

positive_words = ['true', 'real', 'actual', 'substantial', 'authentic', 'genuine', 'factual', 'correct', 'fact',
                          'truth']
negative_words = ['false', 'fake', 'unreal', 'misleading', 'artificial', 'bogus', 'virtual', 'incorrect',
                          'wrong', 'fault']
pos_tokens = tokenizer(" ".join(positive_words))['input_ids'][1:-1]
neg_tokens = tokenizer(" ".join(negative_words))['input_ids'][1:-1]

def get_image_transform():
    config = {
        'input_size': (3, 256, 256),
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'crop_pct': 0.94,
    }
    transform = create_transform(**config)
    transform.transforms.append(
        Lambda(lambda x: x.unsqueeze(0)),
    )
    return transform
image_transform = get_image_transform()

train_dataset = FakeNewsDataset(df_train,img_train,tokenizer, MAX_LEN, True,image_transform)
test_dataset = FakeNewsDataset(df_test,img_test, tokenizer, MAX_LEN, False,image_transform)
num = len(train_dataset)


def get_label_blance(data):
    random.seed(874)
    shot = 8

    ids = [i for i in range(len(data))]
    random.shuffle(ids)
    # to make sure label blanced and entity not-null
    train_ids_pool =  []  # type: ignore
    for i, idx in enumerate(ids):
        #res = data[idx]
        #if data[idx]["len_enti"] == 0:  # entities is null
            #continue
        if len(train_ids_pool) < shot:
            if len(train_ids_pool) == 0 or data[train_ids_pool[-1]]["label"] != data[idx]["label"]:
                train_ids_pool.append(idx)
            else:
                continue
    return train_ids_pool
train_ids_pool = get_label_blance(train_dataset)
#sampler = RandomSampler(train_dataset, replacement=False, num_samples=shot)
train_dataset = Subset(train_dataset, train_ids_pool)

train_dataloader = DataLoader(train_dataset, batch_size=8,
                            shuffle=True,
                             num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=8,shuffle=False, num_workers=0)

# 基本的设置
def set_seed(seed_value):
    # 设置种子
    #random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
EPOCHS=20
set_seed(2347)
final_model = NetShareFusion()
final_model = final_model.to(device)


param_optimizer = list(final_model.named_parameters())

no_decay = ["bias", "LayerNorm.weight", "bn_text.weight","bn_final","bn_gc0.weight","bn_entity.weight","bn_ela1.weight","bn_box.weight","bn_fus1.weight","multiatt1.layer_norm.weight",
            "multiatt2.layer_norm.weight","feedlinear1.layer_norm.weight","feedlinear2.layer_norm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01,"lr": 1e-4},
    {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,"lr": 1e-4},  # 在no_decay里面的权重不需要衰减
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=2e-5,weight_decay=1e-4)

# training steps总数   所有的batch次数
total_steps = len(train_dataloader) * EPOCHS

# 学习率衰减
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=round(total_steps*0.1),
                                            num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss()

# 开始
train(
    model=final_model,
    loss_fn=loss_fn,
    optimizer=optimizer, scheduler=scheduler,
    train_dataloader=train_dataloader, test_dataloader=test_dataloader,
    epochs=EPOCHS, evaluation=True,
    device=device,
    save_best=True, pos_tokens=pos_tokens, neg_tokens=neg_tokens
)

