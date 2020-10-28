import torch
from sklearn import metrics
import numpy as np

def _get_trainable_params(model):
    """Get Parameters with `requires.grad` set to `True`"""
    trainable_params = []
    for x in model.parameters():
        if x.requires_grad:
            trainable_params.append(x)
    return trainable_params

def _evaluate_model(model, val_loader, criterion, epoch, num_epochs, writer, current_lr, log_every=20):
    """Runs model over val dataset and returns auc and avg val loss"""

    # Set to eval mode
    model.eval()

    y_probs = [[],[],[]]
    y_gt = [[],[],[]]
    losses = []

    for i, (images, label) in enumerate(val_loader):

        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()

        output = model(images)

        loss = criterion(output, label)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(output)

        for j, x in enumerate(label[0]):
            y_gt[j].append(int(label[0][j].item()))
        
        for j, x in enumerate(probas[0]):
            y_probs[j].append(probas[0][j].item())

        aucs = []
        for j in range(3):
            try:
                aucs.append(metrics.roc_auc_score(y_gt[j], y_probs[j]))
            except:
                aucs.append(0.5)

        writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC', np.mean(aucs), epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC_abnormal', aucs[0], epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC_acl', aucs[1], epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC_meniscus', aucs[2], epoch * len(val_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} | Batch : {2} / {3} ]| Avg Val Loss {4} | Val AUC : {5} abnorm:{6} acl:{7} meni:{8} | lr : {9}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(val_loader),
                      np.round(np.mean(losses), 4),
                      np.round(np.mean(aucs), 4),
                      np.round(aucs[0], 4),
                      np.round(aucs[1], 4),
                      np.round(aucs[2], 4),
                      current_lr
                  )
                  )

    writer.add_scalar('Val/AUC_epoch', np.mean(aucs), epoch)
    writer.add_scalar('Val/AUC_epoch_abnormal', aucs[0], epoch)
    writer.add_scalar('Val/AUC_epoch_acl', aucs[1], epoch)
    writer.add_scalar('Val/AUC_epoch_meniscus', aucs[2], epoch)

    print('Epoch {} End Val Avg AUC : {} abnorm : {} acl : {} meni : {}'.format(epoch, 
                                                                            np.round(np.mean(aucs), 4), 
                                                                            np.round(aucs[0], 4),
                                                                            np.round(aucs[1], 4),
                                                                            np.round(aucs[2], 4),))
    

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(np.mean(aucs), 4)

    return val_loss_epoch, val_auc_epoch

def _train_model(model, train_loader, epoch, num_epochs, optimizer, criterion, writer, current_lr, log_every=100):
    
    # Set to train mode
    model.train()

    y_probs = [[],[],[]]
    y_gt = [[],[],[]]
    losses = []

    for i, (images, label) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()

        output = model(images)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(output)

        for j, x in enumerate(label[0]):
            y_gt[j].append(int(label[0][j].item()))
        
        for j, x in enumerate(probas[0]):
            y_probs[j].append(probas[0][j].item())

        aucs = []
        for j in range(3):
            try:
                aucs.append(metrics.roc_auc_score(y_gt[j], y_probs[j]))
            except:
                # print("nope")
                # print(y_gt, y_probs)
                aucs.append(0.5)
        

        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', np.mean(aucs), epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC_abnormal', aucs[0], epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC_acl', aucs[1], epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC_meniscus', aucs[2], epoch * len(train_loader) + i)


        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} | Batch : {2} / {3} ]| Avg Train Loss {4} | Train Avg AUC : {5} abnorm:{6} acl:{7} meni:{8} | lr : {9}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(np.mean(aucs), 4),
                      np.round(aucs[0], 4),
                      np.round(aucs[1], 4),
                      np.round(aucs[2], 4),
                      current_lr
                  )
                  )

    writer.add_scalar('Train/AUC_epoch', np.mean(aucs), epoch)
    writer.add_scalar('Train/AUC_epoch_abnormal', aucs[0], epoch)
    writer.add_scalar('Train/AUC_epoch_acl', aucs[1], epoch)
    writer.add_scalar('Train/AUC_epoch_meniscus', aucs[2], epoch)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(np.mean(aucs), 4)

    print('Epoch {} End Train Avg AUC : {} abnorm : {} acl : {} meni : {}'.format(epoch, 
                                                                            np.round(np.mean(aucs), 4), 
                                                                            np.round(aucs[0], 4),
                                                                            np.round(aucs[1], 4),
                                                                            np.round(aucs[2], 4),))

    return train_loss_epoch, train_auc_epoch

def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
