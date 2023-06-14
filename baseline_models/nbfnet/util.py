import os
import sys
import ast
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch_geometric.data import Data
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR, Planetoid

from baseline_models.nbfnet import models, datasets


logger = logging.getLogger(__file__)


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", default = '../baseline_models/nbfnet/data_config/cora.yaml')
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=999)
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--lr', type=float, default=5.0e-3)
    parser.add_argument('--input_dim', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.1) 
    parser.add_argument('--hidden_dims',  nargs='+', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='output_test')

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    
    ######
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    # ####
    # vars = dict()
    # vars['gpus'] = '[2]'
    #####
    return args, vars


def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    # if file:
    #     handler = logging.FileHandler("log.txt")
    #     format = logging.Formatter(format, datefmt)
    #     handler.setFormatter(format)
    #     logger.addHandler(handler)

    return logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(cfg):
    if cfg.train.gpus:
        device = torch.device(cfg.train.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = get_world_size()
    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    # working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
    #                            cfg.model["class"], cfg.dataset["class"], time.strftime("%Y-%m-%d-%H-%M-%S"))
    
    # working_dir = os.path.join(cfg.output_dir, time.strftime("%Y-%m-%d-%H-%M-%S"))
    working_dir = cfg.output_dir
    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def build_dataset(cfg, data_struct=None):
    cls = cfg.dataset.pop("class")
    if cls == "FB15k-237":
        dataset = RelLinkPredDataset(name=cls, **cfg.dataset)
        data = dataset.data
        train_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                          target_edge_index=data.train_edge_index, target_edge_type=data.train_edge_type)
        valid_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                          target_edge_index=data.valid_edge_index, target_edge_type=data.valid_edge_type)
        test_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                         target_edge_index=data.test_edge_index, target_edge_type=data.test_edge_type)
        dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    elif cls == "WN18RR":
        dataset = WordNet18RR(**cfg.dataset)
        # convert wn18rr into the same format as fb15k-237
        data = dataset.data
        num_nodes = int(data.edge_index.max()) + 1
        num_relations = int(data.edge_type.max()) + 1
        edge_index = data.edge_index[:, data.train_mask]
        edge_type = data.edge_type[data.train_mask]
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
        edge_type = torch.cat([edge_type, edge_type + num_relations])
        train_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                          target_edge_index=data.edge_index[:, data.train_mask],
                          target_edge_type=data.edge_type[data.train_mask])
        valid_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                          target_edge_index=data.edge_index[:, data.val_mask],
                          target_edge_type=data.edge_type[data.val_mask])
        test_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                         target_edge_index=data.edge_index[:, data.test_mask],
                         target_edge_type=data.edge_type[data.test_mask])
        dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
        dataset.num_relations = num_relations * 2
    elif cls.startswith("Ind"):
        dataset = datasets.IndRelLinkPredDataset(name=cls[3:], **cfg.dataset)
    elif cls in ["cora", "citeseer", "pubmed"]:
        dataset = datasets.build_citation_dataset(cls,data_struct )
    elif cls in ["ogbl-collab", "ogbl-ppa", "ogbl-ddi", "ogbl-citation2"]:
        dataset = datasets.build_ogb_dataset(cls )

    else:
        raise ValueError("Unknown dataset `%s`" % cls)

    if get_rank() == 0:
        logger.warning("%s dataset" % cls)
        logger.warning("#train: %d, #valid: %d, #test: %d" %
                       (dataset[0].target_edge_index.shape[1], dataset[1].target_edge_index.shape[1],
                        dataset[2].target_edge_index.shape[1]))

    return dataset


def build_model(cfg):
    cls = cfg.model.pop("class")
    

    if cls == "NBFNet":
        model = models.NBFNet(**cfg.model)
    else:
        raise ValueError("Unknown model `%s`" % cls)
    if "checkpoint" in cfg:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    return model


