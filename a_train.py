
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn import  metrics
import time
import warnings

warnings.filterwarnings("ignore")
def train(model, loss_fn, optimizer, scheduler, train_dataloader,
          test_dataloader=None, epochs=150, evaluation=False,
          device='cpu',save_best=False, pos_tokens=None, neg_tokens=None
          ):
    best_acc_val = 0
    best_f1_val=0
    best_macro_f1_val=0
    count3 = 0
    count1=0
    count2=0
    pos_tokens = torch.tensor(pos_tokens).to(device)
    if pos_tokens[0]>30000:
        print(123)
    neg_tokens = torch.tensor(neg_tokens).to(device)
    print("Start training...\n")
    for epoch_i in range(epochs):  # epoches = 100

        print(
            f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'test Loss':^10} | {'test Acc':^9} | {'Elapsed':^9} ")
        print("-" * 100)


        t0_epoch, t0_batch = time.time(), time.time()

        total_loss, batch_loss, batch_counts, batch_acc, total_acc = 0, 0, 0, 0, 0
        train_batch_loss = []
        train_batch_acc = []


        model.train()

        for step, batch in enumerate(train_dataloader):

            batch_counts += 1
            promptT1_ids, attentionT1_mask, promptT2_ids, attentionT2_mask, enti_ids, dct_fea,img,label = batch['promptT1_ids'], \
                                                                                                batch['attentionT1_mask'], \
                                                                                                batch['promptT2_ids'], \
                                                                                                batch['attentionT2_mask'], batch[
                                                                                                    'enti_ids'], \
                                                                                                batch['dct_fea'],batch['img'],batch[
                                                                                                    'label']
            dct_fea = dct_fea.to(device)
            promptT1_ids = promptT1_ids.to(device)
            attentionT1_mask = attentionT1_mask.to(device)
            promptT2_ids = promptT2_ids.to(device)
            attentionT2_mask = attentionT2_mask.to(device)
            enti_ids = enti_ids.to(device)
            img = img.to(device)
            label = label.to(device).to(torch.float32).long()


            model.zero_grad()

            y_pre, loss = model(promptT1_ids=promptT1_ids,
                                                   attentionT1_mask=attentionT1_mask,
                                                dct_fea=dct_fea, promptT2_ids=promptT2_ids,
                                                   attentionT2_mask=attentionT2_mask,enti_ids=enti_ids,
                                                            pos_tokens=pos_tokens,neg_tokens=neg_tokens,img=img,
                                                   label=label, flag=True)


            if step < 1:
                print('第', step, "个batch的loss:",  loss, y_pre[0, :])
            batch_loss += loss.item()
            total_loss += loss.item()

            # 计算这个batch的训练精度
            train_label = label.cpu().detach().numpy().tolist()
            pred_input = y_pre.cpu().detach().numpy().tolist()
            pred_flat = np.argmax(pred_input, axis=1)
            acc = np.sum(pred_flat == train_label) / len(train_label)
            batch_acc += acc
            total_acc += acc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()


            train_batch_loss.append(loss.detach().item())
            train_batch_acc.append(acc)
            if (step % 300 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                print(f"epoch{epoch_i + 1:^3} | batch{step:^4} | loss{batch_loss / batch_counts:^12.6f} | acc{batch_acc / batch_counts:^12.3f} | {'-':^10} | {'-':^9} | elapsed{time_elapsed:^9.5f}")

                batch_loss, batch_counts, batch_acc = 0, 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_acc = total_acc / len(train_dataloader)


        print("-" * 100)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:  # 传参中传了True

            test_epoch_loss, metrics_value = evaluate(model, loss_fn, test_dataloader, pos_tokens, neg_tokens, device)
            time_elapsed = time.time() - t0_epoch

            acc_test = metrics_value['accuracy']
            precision = metrics_value['bi_precision']
            recall = metrics_value['bi_recall']
            f1 = metrics_value['bi_f1']
            macro_f1 = metrics_value['macro_f1']

            print(
                f" {epoch_i + 1:^7} | {'-':^7} |loss= {avg_train_loss:^12.6f} |训练acc= {avg_train_acc:^10.6f} |测试acc= {acc_test:^9.5f} |loss= {test_epoch_loss:^9.5f} time= {time_elapsed:^9.2f} |pre= {precision:^6.5}|recall={recall:^6.5f}|f1={f1:^6.5f}")
            print("-" * 70)

            if save_best:
                if acc_test > best_acc_val:
                    best_acc_val = acc_test
                    count1 = epoch_i + 1
                if  f1 > best_f1_val:
                    best_f1_val = f1
                    count2 = epoch_i + 1
                if macro_f1 > best_macro_f1_val:
                    best_macro_f1_val = macro_f1
                    count3 = epoch_i + 1
                    mac_acc = acc_test
                    print('模型已保存')
            print('epoch:', count1, '当前模型的最佳acc:', best_acc_val)
            print('epoch:', count2, '当前模型的最佳f1:', best_f1_val)
            print('epoch:', count3, '当前模型的最佳macro_f1:', best_macro_f1_val,"此时的acc",mac_acc)


        print("\n")

    print("Training complete!")


def evaluate(model, loss_fn, test_dataloader,pos_tokens,neg_tokens, device):  # 只是验证测试，不影响后面的训练
    """
        在每个epoch训练完成后，测试模型的性能
    """
    model.eval()

    loss_detection_total = 0
    detection_count = 0

    detection_pre_label_all = []
    detection_label_all = []

    # 每个损失之后   text_token, entity_token,sen_score, dct_fea,box_feature,txt_txt_graph,img_txt_graph,label,flag
    for batch in test_dataloader:

        promptT1_ids, attentionT1_mask, promptT2_ids, attentionT2_mask, enti_ids, dct_fea, img, label = batch[
                                                                                    'promptT1_ids'], \
                                                                                    batch['attentionT1_mask'], \
                                                                                    batch['promptT2_ids'], \
                                                                                    batch['attentionT2_mask'], batch[
                                                                                    'enti_ids'], \
                                                                                    batch['dct_fea'], batch['img'], batch['label']
        dct_fea = dct_fea.to(device)
        promptT1_ids = promptT1_ids.to(device)
        attentionT1_mask = attentionT1_mask.to(device)
        promptT2_ids = promptT2_ids.to(device)
        attentionT2_mask = attentionT2_mask.to(device)
        enti_ids = enti_ids.to(device)
        img = img.to(device)
        label = label.to(device).to(torch.float32).long()

        batch_size = dct_fea.shape[0]

        with torch.no_grad():
            y_pre, loss = model(promptT1_ids=promptT1_ids,
                                attentionT1_mask=attentionT1_mask,
                                dct_fea=dct_fea, promptT2_ids=promptT2_ids,
                                attentionT2_mask=attentionT2_mask, enti_ids=enti_ids,
                                pos_tokens=pos_tokens, neg_tokens=neg_tokens, img=img,
                                label=label, flag=True)

        # 计算测试集精确度
        pred_flat = torch.softmax(y_pre, dim=1).argmax(1)
        loss_detection_total += loss.item() * batch_size
        detection_count += batch_size
        detection_pre_label_all.append(pred_flat.detach().cpu().numpy())
        detection_label_all.append(label.detach().cpu().numpy())

    loss_detection_test = loss_detection_total / detection_count

    detection_pre_label_all = np.concatenate(detection_pre_label_all, 0)
    detection_label_all = np.concatenate(detection_label_all, 0)
    metrics_value = compute_measures(detection_pre_label_all,detection_label_all)
    return loss_detection_test,metrics_value

def compute_measures(logit, y_gt):
    #predicts = torch.max(logit, 1)[1].cpu().numpy()
    predicts = logit
    accuracy = metrics.accuracy_score(y_true=y_gt, y_pred=predicts)

    # binary: set 1(fake news) as positive sample
    bi_precision = metrics.precision_score(y_true=y_gt, y_pred=predicts, average='binary', zero_division=0)
    bi_recall = metrics.recall_score(y_true=y_gt, y_pred=predicts, average='binary', zero_division=0)
    bi_f1 = metrics.f1_score(y_true=y_gt, y_pred=predicts, average='binary', zero_division=0)

    # micro
    micro_precision = metrics.precision_score(y_true=y_gt, y_pred=predicts, average='micro', zero_division=0)
    micro_recall = metrics.recall_score(y_true=y_gt, y_pred=predicts, average='micro', zero_division=0)
    micro_f1 = metrics.f1_score(y_true=y_gt, y_pred=predicts, average='micro', zero_division=0)

    # macro
    macro_precision = metrics.precision_score(y_true=y_gt, y_pred=predicts, average='macro', zero_division=0)
    macro_recall = metrics.recall_score(y_true=y_gt, y_pred=predicts, average='macro', zero_division=0)
    macro_f1 = metrics.f1_score(y_true=y_gt, y_pred=predicts, average='macro', zero_division=0)

    # weighted macro
    weighted_precision = metrics.precision_score(y_true=y_gt, y_pred=predicts, average='weighted', zero_division=0)
    weighted_recall = metrics.recall_score(y_true=y_gt, y_pred=predicts, average='weighted', zero_division=0)
    weighted_f1 = metrics.f1_score(y_true=y_gt, y_pred=predicts, average='weighted', zero_division=0)

    measures = {"accuracy":accuracy,
                "bi_precision": bi_precision, "bi_recall": bi_recall, "bi_f1": bi_f1,
                "micro_precision": micro_precision, "micro_recall": micro_recall, "micro_f1": micro_f1,
                "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1,
                "weighted_precision": weighted_precision, "weighted_recall": weighted_recall, "weighted_f1": weighted_f1
              }
    return measures
