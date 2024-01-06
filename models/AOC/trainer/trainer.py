import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from merlion.evaluate.anomaly import ScoreType
from models import ad_predict
from models.reasonable_metric import tsad_reasonable
from models.reasonable_metric import reasonable_accumulator
from .early_stopping import EarlyStopping

sys.path.append("../../COCA")
def Trainer(model, model_optimizer, train_dl, val_dl, test_dl, device, config, idx):
    # Start training
    print("Training started ....")

    save_path = "./best_network/" + config.dataset
    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping(save_path, idx)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    all_epoch_train_loss, all_epoch_test_loss = [], []
    center = torch.zeros(config.project_channels, device=device)
    center = center_c(train_dl, model, device, center, config, eps=config.center_eps)
    length = torch.tensor(0, device=device)  # radius R initialized with 0 by default.

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_target, train_score, train_loss, length = model_train(model, model_optimizer, train_dl, center,
                                                                    length, config, device, epoch)
        val_target, val_score_origin, val_loss = model_evaluate(model, val_dl, center, length, config,
                                                                            device, epoch)
        test_target, test_score_origin, test_loss = model_evaluate(model, test_dl, center, length,
                                                                                   config, device, epoch)

        if epoch < config.change_center_epoch:
            center = center_c(train_dl, model, device, center, config, eps=config.center_eps)
        scheduler.step(train_loss)
        print(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \n'
                     f'Valid Loss     : {val_loss:.4f}\t  | \n'
                     f'Test Loss     : {test_loss:.4f}\t  | \n'
                     )
        all_epoch_train_loss.append(train_loss.item())
        all_epoch_test_loss.append(val_loss.item())
        if config.dataset == 'UCR':
            val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                       config.detect_nu)
            test_affiliation, test_score, _, _, predict = ad_predict(test_target, test_score_origin,
                                                               config.threshold_determine, config.detect_nu)
            score_reasonable = tsad_reasonable(test_target, predict, config.time_step)
            test_f1 = test_score.f1(ScoreType.RevisedPointAdjusted)
            test_precision = test_score.precision(ScoreType.RevisedPointAdjusted)
            test_recall = test_score.recall(ScoreType.RevisedPointAdjusted)
            print("Test accuracy metrics")
            print(
                f'Test accuracy: {score_reasonable.correct_num:2.4f}\n')
            early_stopping(score_reasonable, test_affiliation, test_score, model)
            print("Test affiliation-metrics")
            print(
                f'Test precision: {test_affiliation["precision"]:2.4f}  | \tTest recall: {test_affiliation["recall"]:2.4f}\n')
            print("Test RAP F1")
            print(
                f'Test F1: {test_f1:2.4f}  | \tTest precision: {test_precision:2.4f}  | \tTest recall: {test_recall:2.4f}\n')
            if early_stopping.early_stop:
                print("Early stopping")
                break

    print("\n################## Training is Done! #########################")
    # according to scores to create predicting labels
    if config.dataset == 'UCR':
        score_reasonable = early_stopping.best_score
        test_affiliation = early_stopping.best_affiliation
        test_score = early_stopping.best_rpa_score
        print("Test accuracy metrics")
        print(
            f'Test accuracy: {score_reasonable.correct_num:2.4f}\n')
    else:
        val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                   config.detect_nu)
        test_affiliation, test_score, _, _, predict = ad_predict(test_target, test_score_origin, config.threshold_determine,
                                                           config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
    val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
    val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
    val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
    print("Valid affiliation-metrics")
    print(
        f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')
    print("Valid RAP F1")
    print(f'Valid F1: {val_f1:2.4f}  | \tValid precision: {val_precision:2.4f}  | \tValid recall: {val_recall:2.4f}\n')

    test_f1 = test_score.f1(ScoreType.RevisedPointAdjusted)
    test_precision = test_score.precision(ScoreType.RevisedPointAdjusted)
    test_recall = test_score.recall(ScoreType.RevisedPointAdjusted)
    print("Test affiliation-metrics")
    print(
        f'Test precision: {test_affiliation["precision"]:2.4f}  | \tTest recall: {test_affiliation["recall"]:2.4f}\n')
    print("Test RAP F1")
    print(f'Test F1: {test_f1:2.4f}  | \tTest precision: {test_precision:2.4f}  | \tTest recall: {test_recall:2.4f}\n')

    return test_score_origin, test_affiliation, test_score, score_reasonable, predict


def model_train(model, model_optimizer, train_loader, center, length, config, device, epoch):

    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []

    model.train()
    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, (data, target, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, target = data.float().to(device), target.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        # all_data = torch.cat((data, aug1, aug2), dim=0)
        all_data = data
        all_data = all_data.permute(0, 2, 1)
        # optimizer
        model_optimizer.zero_grad()
        output, hidden = model(all_data)
        loss, score = train(all_data, output, hidden, center, length, epoch, config, device)
        # Update hypersphere radius R on mini-batch distances
        if (config.objective == 'soft-boundary') and (epoch >= config.freeze_length_epoch):
            length = torch.tensor(get_radius(score, config.nu), device=device)
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        target = target.reshape(-1)

        predict = score.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        all_target.extend(target)
        all_predict.extend(predict)

    total_loss = torch.tensor(total_loss).mean()

    return all_target, all_predict, total_loss, length


def model_evaluate(model, test_dl, center, length, config, device, epoch):
    model.eval()
    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []
    with torch.no_grad():
        for data, target, aug1, aug2 in test_dl:
            data, target = data.float().to(device), target.long().to(device)
            data = data.permute(0, 2, 1)
            output, hidden = model(data, training=False)
            loss, score = train(data, output, hidden, center, length, epoch, config, device)
            total_loss.append(loss.item())
            predict = score.detach().cpu().numpy()
            target = target.reshape(-1)
            all_target.extend(target.detach().cpu().numpy())
            all_predict.extend(predict)

    total_loss = torch.tensor(total_loss).mean()  # average loss
    all_target = np.array(all_target)

    return all_target, all_predict, total_loss


def train(data, output, hidden, center, length, epoch, config, device):
    # normalize feature vectors
    loss_func = torch.nn.MSELoss(reduction="none")
    loss_ae = loss_func(output, data)
    loss_ae = loss_ae.squeeze(2)
    center = center.unsqueeze(0)
    # center = F.normalize(center, dim=1)
    # hidden = F.normalize(hidden, dim=1)
    # loss_oc = 1 - F.cosine_similarity(hidden, center, eps=1e-6)
    loss_oc = (hidden - center) ** 2

    score = config.omega1 * torch.mean(loss_ae, dim=1) + config.omega2 * torch.mean(loss_oc, dim=1)
    if config.objective == 'soft-boundary':
        diff = score - length
        loss = length + (1 / config.nu) * torch.mean(torch.max(torch.zeros_like(diff), diff))
    else:
        loss = torch.mean(score)
    return loss, score

def center_c(train_loader, model, device, center, config, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = center
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            data, target, aug1, aug2 = data
            data = data.float().to(device)
            aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
            # all_data = torch.cat((data, aug1, aug2), dim=0)
            all_data = data
            all_data = all_data.permute(0, 2, 1)
            # optimizer
            output, hidden = model(all_data)
            n_samples += hidden.shape[0]
            all_feature = hidden
            c += torch.sum(all_feature, dim=0)
    c /= n_samples
    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    # return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    dist = dist.reshape(-1)
    return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)



