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
    # List of probabilities obtained from the model
    y_probs = []
    # List of groundtruth labels
    y_gt = []
    # List of losses obtained
    losses = []

    # Iterate over the validation dataset
    for i, (images, label) in enumerate(val_loader):
        # If GPU is available, load the images and label
        # on GPU
        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()

        # Obtain the model output by passing the images as input
        output = model(images)
        # Evaluate the loss by comparing the output and groundtruth label
        loss = criterion(output, label)
        # Add loss to the list of losses
        loss_value = loss.item()
        losses.append(loss_value)
        # Find probability for each class by applying
        # sigmoid function on model output
        probas = torch.sigmoid(output)
        # Add the groundtruth to the list of groundtruths
        y_gt.append(int(label.item()))
        # Add predicted probability to the list
        y_probs.append(probas.item())

        try:
            # Evaluate area under ROC curve based on the groundtruth label
            # and predicted probability
            auc = metrics.roc_auc_score(y_gt, y_probs)
        except:
            # Default area under ROC curve
            auc = 0.5
        # Add information to the writer about validation loss and Area under ROC curve
        writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)

        if (i % log_every == 0) & (i > 0):
            # Display the information about average validation loss and area under ROC curve
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
    # Add information to the writer about total epochs and Area under ROC curve
    writer.add_scalar('Val/AUC_epoch', auc, epoch + i)
    # Find mean area under ROC curve and validation loss
    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)

    return val_loss_epoch, val_auc_epoch

def _train_model(model, train_loader, epoch, num_epochs, optimizer, criterion, writer, current_lr, log_every=100):
    
    # Set to train mode
    model.train()

    # Initialize the predicted probabilities
    y_probs = []
    # Initialize the groundtruth labels
    y_gt = []
    # Initialize the loss between the groundtruth label
    # and the predicted probability
    losses = []

    # Iterate over the training dataset
    for i, (images, label) in enumerate(train_loader):
        # Reset the gradient by zeroing it
        optimizer.zero_grad()
        
        # If GPU is available, transfer the images and label
        # to the GPU
        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()

        # Obtain the prediction using the model
        output = model(images)

        # Evaluate the loss by comparing the prediction
        # and groundtruth label
        loss = criterion(output, label)
        # Perform a backward propagation
        loss.backward()
        # Modify the weights based on the error gradient
        optimizer.step()

        # Add current loss to the list of losses
        loss_value = loss.item()
        losses.append(loss_value)

        # Find probabilities from output using sigmoid function
        probas = torch.sigmoid(output)

        # Add current groundtruth label to the list of groundtruths
        y_gt.append(int(label.item()))
        # Add current probabilities to the list of probabilities
        y_probs.append(probas.item())

        try:
            # Try finding the area under ROC curve
            auc = metrics.roc_auc_score(y_gt, y_probs)
        except:
            # Use default value of area under ROC curve as 0.5
            auc = 0.5
        
        # Add information to the writer about training loss and Area under ROC curve
        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            # Display the information about average training loss and area under ROC curve
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
    # Add information to the writer about total epochs and Area under ROC curve
    writer.add_scalar('Train/AUC_epoch', auc, epoch + i)

    # Find mean area under ROC curve and training loss
    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)

    return train_loss_epoch, train_auc_epoch

def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
