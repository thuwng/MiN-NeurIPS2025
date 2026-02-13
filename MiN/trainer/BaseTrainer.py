import sys
import logging
import torch
import torch.nn.functional as F
from utils.factory import get_model
from data_process.data_manger import DataManger
import os
import datetime
import json


def train(args):
    _train(args)


def _train(args):
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logs_name = "logs/{}/{}/{}_{}/{}".format(args["dataset"], args["model"], args["init_class"], args["increment"],
                                             now_time)
    workdir = os.path.join(logs_name, 'work_dir')
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    with open(os.path.join(workdir, 'configs.json'), 'w', encoding='utf-8') as json_file:
        json.dump(args, json_file, indent=2)

    logfilename = "logs/{}/{}/{}_{}/{}/{}_{}_{}_{}_{}".format(
        args["dataset"],
        args["model"],
        args["init_class"],
        args["increment"],
        now_time,
        args["dataset"],
        args["model"],
        args["init_class"],
        args["increment"],
        args["backbone_type"]
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
        ],
    )

    # _set_random()
    _set_device(args)
    print_args(args)

    datamanger = DataManger(args['dataset'], args['device'], args)

    model = get_model(args, logging)

    model.init_train(data_manger=datamanger)
    model.after_train(data_manger=datamanger)

    loader = datamanger.get_train_loader()   
    fisher = compute_fisher(model, loader)
    model.fisher_list.append(fisher)
    model.pass_fisher_to_backbone(fisher)

    save_path = os.path.join(workdir, 'task_0_check_point.pth')
    if args["save_all_checkpoint"] is True:
        model.save_check_point(save_path)

    for i in range(datamanger.task_size):
        save_path = os.path.join(workdir, 'task_{}_check_point.pth'.format(i+1))
        model.increment_train(data_manger=datamanger)
        model.after_train(data_manger=datamanger)

        loader = datamanger.get_train_loader()
        fisher = compute_fisher(model, loader)
        model.fisher_list.append(fisher)
        model.pass_fisher_to_backbone(fisher)

        if args["save_all_checkpoint"] is True:
            model.save_check_point(save_path)
    save_path = os.path.join(logs_name, 'Last_check_point.pth')
    model.save_check_point(save_path)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)

    args["device"] = gpus[0]


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def _set_random(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def compute_fisher(model, dataloader):

    device = next(model.parameters()).device
    model.eval()

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    if len(params) == 0:
        raise RuntimeError("No parameters require grad for Fisher")

    fisher = {
        name: torch.zeros_like(p, device="cpu")
        for name, p in params
    }

    for batch in dataloader:

        if len(batch) == 3:
            _, images, labels = batch
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model.forward_normal_fc(images, new_forward=False)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)

        grads = torch.autograd.grad(
            loss,
            [p for _, p in params],
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )

        for (name, _), g in zip(params, grads):
            if g is not None:
                fisher[name] += (g.detach().cpu() ** 2)

        del images, labels, outputs, logits, loss, grads
        torch.cuda.empty_cache()

    for name in fisher:
        fisher[name] /= len(dataloader)

    torch.cuda.empty_cache()
    return fisher
