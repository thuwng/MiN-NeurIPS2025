import math
import random

import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
from utils.inc_net import MiNbaseNet
from torch.utils.data import WeightedRandomSampler
from utils.toolkit import tensor2numpy, count_parameters
import os
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler
from utils.toolkit import calculate_class_metrics, calculate_task_metrics
from trainer.BaseTrainer import compute_fisher

EPSILON = 1e-8


class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.logger = loger
        self._network = MiNbaseNet(args)
        self.device = args['device']
        self.num_workers = args["num_workers"]

        self.init_epochs = args["init_epochs"]
        self.init_lr = args["init_lr"]
        self.init_weight_decay = args["init_weight_decay"]
        self.init_batch_size = args["init_batch_size"]

        self.lr = args["lr"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.epochs = args["epochs"]

        self.init_class = args["init_class"]
        self.increment = args["increment"]

        self.buffer_size = args["buffer_size"]
        self.buffer_batch = args["buffer_batch"]
        self.gamma = args['gamma']
        self.fit_epoch = args["fit_epochs"]

        self.known_class = 0
        self.cur_task = -1
        self.total_acc = []
        self.class_acc = []
        self.task_acc = []

    def after_train(self, data_manger):
        if self.cur_task == 0:
            self.known_class = self.init_class
        else:
            self.known_class += self.increment

        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
                                 num_workers=self.num_workers)
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        self.logger.info('task_confusion_metrix:\n{}'.format(eval_res['task_confusion']))
        print('total acc: {}'.format(self.total_acc))
        print('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        del test_set

    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

    def compute_test_acc(self, test_loader):
        model = self._network.eval()
        correct, total = 0, 0
        device = self.device
        for i, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                logits = outputs["logits"]
            predicts = torch.max(logits, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    @staticmethod
    def cat2order(targets, datamanger):
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets

    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(0)
        self.logger.info("task_list: {}".format(train_list_name))
        self.logger.info("task_order: {}".format(train_list))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
                                 num_workers=self.num_workers)

        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        self.run(train_loader)
        fisher = self.compute_fisher_safe(train_loader)
        self.pass_fisher_to_backbone(fisher)
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.update_task_prototype(prototype)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)
        self.fit_fc(train_loader, test_loader)

        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader, test_loader)

        del train_set
        del test_set

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info("task_list: {}".format(train_list_name))
        self.logger.info("task_order: {}".format(train_list))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)

        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.fit_fc(train_loader, test_loader)

        new_total_class = self.known_class + self.increment
        #print("FC updated to:", new_total_class)
        #print("known_class:", self.known_class)
        self._network.update_fc(new_total_class)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers)
        self._network.update_noise()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        self.run(train_loader)
        fisher = compute_fisher(self._network, train_loader)
        self.pass_fisher_to_backbone(fisher)
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.update_task_prototype(prototype)

        del train_set

        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                    num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                    num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader, test_loader)

        del train_set
        del test_set

    def fit_fc(self, train_loader, test_loader):
        print("=== BEFORE FIT_FC ===")
        print("Backbone training mode:", self._network.backbone.training)

        total_grad = sum(p.requires_grad for p in self._network.backbone.parameters())
        print("Backbone params requiring grad:", total_grad)

        self._network.eval()
        self._network.to(self.device)
        self._network.backbone.eval()

        prog_bar = tqdm(range(self.fit_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.to(self.device)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = torch.nn.functional.one_hot(targets)
                self._network.fit(inputs, targets)
            
            info = "Task {} --> Update Analytical Classifier!".format(
                self.cur_task,
            )
            self.logger.info(info)
            prog_bar.set_description(info)
            print("Memory allocated:",
                torch.cuda.memory_allocated() / 1024**3, "GB")

    def re_fit(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        prog_bar = tqdm(train_loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = torch.nn.functional.one_hot(targets)
            self._network.fit(inputs, targets)

            info = "Task {} --> Reupdate Analytical Classifier!".format(
                self.cur_task,
            )
            
            self.logger.info(info)
            prog_bar.set_description(info)

    def compute_fisher_safe(self, dataloader):


        for p in self._network.parameters():
            p.requires_grad = False

        for p in self._network.normal_fc.parameters():
            p.requires_grad = True

        fisher = compute_fisher(self._network, dataloader)

        for p in self._network.parameters():
            p.requires_grad = False

        return fisher

    def run(self, train_loader):

        if self.cur_task == 0:
            epochs = self.init_epochs
            lr = self.init_lr
            weight_decay = self.init_weight_decay
        else:
            epochs = self.epochs
            lr = self.lr
            weight_decay = self.weight_decay

        # ---- Freeze all ----
        for param in self._network.parameters():
            param.requires_grad = False

        # ---- Enable classifier ----
        for param in self._network.normal_fc.parameters():
            param.requires_grad = True

        # ---- Enable noise ----
        if self.cur_task == 0:
            self._network.init_unfreeze()
        else:
            self._network.unfreeze_noise()

        params = filter(lambda p: p.requires_grad, self._network.parameters())
        optimizer = get_optimizer(
            self.args['optimizer_type'], params, lr, weight_decay
        )
        scheduler = get_scheduler(
            self.args['scheduler_type'], optimizer, epochs
        )

        # 🔴 VERY IMPORTANT
        prog_bar = tqdm(range(epochs))
        self._network.train()
        self._network.backbone.eval()
        self._network.to(self.device)

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # ===== Forward =====
                if self.cur_task > 0:
                    with torch.no_grad():
                        old_out = self._network(inputs, new_forward=False)
                        old_logits = old_out['logits']

                    new_out = self._network.forward_normal_fc(
                        inputs, new_forward=False
                    )
                    new_logits = new_out['logits']

                    logits = new_logits + old_logits
                else:
                    out = self._network.forward_normal_fc(
                        inputs, new_forward=False
                    )
                    logits = out['logits']

                # ===== CE Loss =====
                loss = F.cross_entropy(logits, targets.long())

                # ===== Fisher-guided noise regularization =====
                fisher_loss = 0.0
                noise_maker = self._network.backbone.noise_maker

                if hasattr(noise_maker[0], "fisher_importance"):
                    fisher_vec = noise_maker[0].fisher_importance

                    if fisher_vec is not None:

                        fisher_flat = fisher_vec.view(-1)

                        for noise_layer in noise_maker:
                            fisher_vec = noise_layer.fisher_importance
                            if fisher_vec is None:
                                continue

                            fisher_flat = fisher_vec.view(-1)

                            for mu_layer in noise_layer.mu:
                                w = mu_layer.weight.view(-1)
                                min_len = min(w.numel(), fisher_flat.numel())
                                fisher_loss += (w[:min_len].pow(2) * fisher_flat[:min_len]).mean()

                        loss = loss + 0.1 * fisher_loss

                # ===== Backward =====
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += targets.size(0)

            scheduler.step()
            train_acc = 100. * correct / total

            info = (
                f"Task {self.cur_task} --> Learning Beneficial Noise!: "
                f"Epoch {epoch + 1}/{epochs} "
                f"=> Loss {losses / len(train_loader):.3f}, "
                f"train_accy {train_acc:.2f}"
            )

            self.logger.info(info)
            prog_bar.set_description(info)

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        for i, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)
            logits = outputs["logits"]
            predicts = torch.max(logits, dim=1)[1]
            pred.extend([int(predicts[i].cpu().numpy()) for i in range(predicts.shape[0])])
            label.extend(int(targets[i].cpu().numpy()) for i in range(targets.shape[0]))
        class_info = calculate_class_metrics(pred, label)
        task_info = calculate_task_metrics(pred, label, self.init_class, self.increment)
        return {
            "all_class_accy": class_info['all_accy'],
            "class_accy": class_info['class_accy'],
            "class_confusion": class_info['class_confusion_matrices'],
            "task_accy": task_info['all_accy'],
            "task_confusion": task_info['task_confusion_matrices'],
            "all_task_accy": task_info['task_accy'],
        }

    def get_task_prototype(self, model, train_loader):

        model = model.eval()
        model.to(self.device)

        proto_sum = None
        count = 0

        for _, inputs, targets in train_loader:
            inputs = inputs.to(self.device)

            with torch.no_grad():
                feature = model.extract_feature(inputs)

            batch_mean = feature.mean(dim=0)

            if proto_sum is None:
                proto_sum = batch_mean
            else:
                proto_sum += batch_mean

            count += 1

            del feature


        prototype = proto_sum / count
        return prototype
            
    def pass_fisher_to_backbone(self, fisher_vec):

        backbone = self._network.backbone


        for noise_layer in backbone.noise_maker:
            noise_layer.set_fisher(fisher_vec)

