from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.auxiliaries import utils
import numpy as np

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def knn_monitor(net, memory_data_loader, test_data_loader, k=200, t=0.1, device="cpu", verbose=True):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    feature_labels = []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=not verbose):
            feature = net(data.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target.to(device))
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=device)
        feature_labels = torch.cat(feature_labels, dim=0).contiguous()

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=not verbose)
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
    return total_top1 / total_num * 100


class CLTrainer(GeneralTorchTrainer):
    def _hook_on_batch_forward(self, ctx):
        x, label = [utils.move_to(_, ctx.device) for _ in ctx.data_batch]
#         print(len(x), x[0].size(), x[1].size(), label.size())
        x1, x2 = x[0], x[1]
        z1, z2 = ctx.model(x1, x2)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch = ctx.criterion(z1, z2)
        ctx.y_true = label
        ctx.y_prob = z1, z2

        ctx.batch_size = len(label)
        
    def _hook_on_batch_end(self, ctx):
        # update statistics
        setattr(
            ctx, "loss_batch_total_{}".format(ctx.cur_data_split),
            ctx.get("loss_batch_total_{}".format(ctx.cur_data_split)) +
            ctx.loss_batch.item() * ctx.batch_size)

        if ctx.get("loss_regular", None) is None or ctx.loss_regular == 0:
            loss_regular = 0.
        else:
            loss_regular = ctx.loss_regular.item()
        setattr(
            ctx, "loss_regular_total_{}".format(ctx.cur_data_split),
            ctx.get("loss_regular_total_{}".format(ctx.cur_data_split)) +
            loss_regular)
        setattr(
            ctx, "num_samples_{}".format(ctx.cur_data_split),
            ctx.get("num_samples_{}".format(ctx.cur_data_split)) +
            ctx.batch_size)

        # cache label for evaluate
        ctx.get("{}_y_true".format(ctx.cur_data_split)).append(
            ctx.y_true.detach().cpu().numpy())

#         print(len(ctx.y_prob), ctx.y_prob[0].size(), ctx.y_prob[1].size())
        ctx.get("{}_y_prob".format(ctx.cur_data_split)).append(
            ctx.y_prob[0].detach().cpu().numpy())

        # clean temp ctx
        ctx.data_batch = None
        ctx.batch_size = None
        ctx.loss_task = None
        ctx.loss_batch = None
        ctx.loss_regular = None
        ctx.y_true = None
        ctx.y_prob = None
        
    def _hook_on_fit_end(self, ctx):
        """Evaluate metrics.

        """
        setattr(
            ctx, "{}_y_true".format(ctx.cur_data_split),
            np.concatenate(ctx.get("{}_y_true".format(ctx.cur_data_split))))
        setattr(
            ctx, "{}_y_prob".format(ctx.cur_data_split),
            np.concatenate(ctx.get("{}_y_prob".format(ctx.cur_data_split))))
        results = self.metric_calculator.eval(ctx)
        setattr(ctx, 'eval_metrics', results)
        
class LPTrainer(GeneralTorchTrainer):
    pass
    
def call_cl_trainer(trainer_type):
    if trainer_type == 'cltrainer':
        trainer_builder = CLTrainer
        return trainer_builder
    elif trainer_type == 'lptrainer':
        trainer_builder = LPTrainer
        return trainer_builder 


register_trainer('cltrainer', call_cl_trainer)
