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

    y_probs = []
    y_gt = []
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

        y_gt.append(int(label.item()))
        y_probs.append(probas.item())

        try:
            auc = metrics.roc_auc_score(y_gt, y_probs)
        except:
            auc = 0.5

        writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} | Batch : {2} / {3} ]| Avg Val Loss {4} | Val AUC : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(val_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

    writer.add_scalar('Val/AUC_epoch', auc, epoch + i)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)

    return val_loss_epoch, val_auc_epoch

def _train_model(model, train_loader, epoch, num_epochs, optimizer, criterion, writer, current_lr, log_every=100):
    
    # Set to train mode
    model.train()

    y_probs = []
    y_gt = []
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

        y_gt.append(int(label.item()))
        y_probs.append(probas.item())

        try:
            auc = metrics.roc_auc_score(y_gt, y_probs)
        except:
            auc = 0.5

        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} | Batch : {2} / {3} ]| Avg Train Loss {4} | Train AUC : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

    writer.add_scalar('Train/AUC_epoch', auc, epoch + i)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)

    return train_loss_epoch, train_auc_epoch

def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
