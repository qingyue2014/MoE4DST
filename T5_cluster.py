# Copyright (c) Facebook, Inc. and its affiliates
import os, random

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, AutoModelForSeq2SeqLM)
from data_loader import prepare_data, load_dist_weight, prepare_test_data, prepare_multidata, EXPERIMENT_DOMAINS, load_dist_domain
from config import get_args
import copy
from transformers import AdapterConfig
from utils.model_utils import set_frozen, load_single_adapter, save_adapter, prepare_model
from utils.eval_util import evaluate_model, evaluate_sgd
from generate import generate_ensemble_output, generate_ensemble_param
from generate_sgd import generate_ensemble_output_sgd, generate_ensemble_param_sgd
import logging

def get_logger(file_log, fh_mode="w"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename=file_log, encoding='utf-8', mode=fh_mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(("%(asctime)s - %(levelname)s - %(message)s"))
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class DST_Seq2Seq(pl.LightningModule):
    def __init__(self, args, tokenizer, model, adapter_name=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint_file)
        if args.adapter and adapter_name:
            config = AdapterConfig.load(args.adapter_config)
            self.model.add_adapter(adapter_name, config)
            self.model.train_adapter(adapter_name)
            self.model.set_active_adapters(adapter_name)
            self.args.logger.info("Set adapter Successfully!")

        self.lr = args.lr


    def training_step(self, batch, batch_idx):
        self.model.train()
        (loss) = self.model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"]).loss

        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss}}
        # return result

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        (loss) = self.model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"]).loss
        self.log("val_loss", loss)
        return {'val_loss': loss, 'log': {'val_loss': loss}}
        # return result

    def validation_step_end(self, batch_parts):
        # print(batch_parts)
        losses = batch_parts["val_loss"]
        average_loss = losses
        return {"val_loss": average_loss, "log": {"val_loss": average_loss}}

    def validation_epoch_end(self, outputs):
        #print(outputs)
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_loss in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        return results

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

def train(args):
    seed_everything(args.seed)
    args.logger = get_logger('{}/train_{}.log'.format(args.saving_dir, args.model_name), "a")
    model, tokenizer = prepare_model(args)

    args.base_adapter_name = "single"

    task = DST_Seq2Seq(args, tokenizer, model, args.base_adapter_name)
    set_frozen(args, task, args.base_adapter_name)
    train_loader, val_loader, test_loader, ALL_SLOTS, train_data, \
        test_data, all_data = prepare_data(args, task.tokenizer)

    trainer = Trainer(
        default_root_dir=args.model_checkpoint,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_norm,
        max_epochs=args.n_epochs,
        #callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00, patience=6, verbose=False, mode='min')],
        gpus=args.GPU,
        deterministic=True,
        num_nodes=1,
        # precision=16,
        accelerator="cuda"
    )
    trainer.fit(task, train_loader, val_loader)
    if args.adapter:
        save_adapter(args, task.model, args.base_adapter_name)
    else:
        task.model.save_pretrained(args.saving_dir)
        task.tokenizer.save_pretrained(args.saving_dir)
    if "mwz" in args.data_dir:
        _ = evaluate_model(args, task.tokenizer, task.model, test_loader, ALL_SLOTS)
    elif "sgd" in args.data_dir:
        _ = evaluate_sgd(args, task.tokenizer, task.model, test_loader, ALL_SLOTS, all_data)


def train_multi(args):
    seed_everything(args.seed)
    args.logger = get_logger('{}/{}_{}.log'.format(args.saving_dir, args.model_name, args.class_id), "a")
    args.logger.info(args)

    model, tokenizer = prepare_model(args)
    train_loader_list, dev_loader, test_loader, ALL_SLOTS, data_train, \
    data_test, all_data = prepare_multidata(args, tokenizer)

    for sid in range(args.class_id):
        adapter_name = f"class{args.class_id}_sub{str(sid)}"
        args.logger.info(f"Train {args.except_domain}'s {adapter_name}.pt!")
        task = DST_Seq2Seq(args, tokenizer, model, adapter_name)
        set_frozen(args, task, adapter_name)
        trainer = Trainer(
            default_root_dir=args.model_checkpoint,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gradient_clip_val=args.max_norm,
            max_epochs=args.n_epochs,
            callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00, patience=4, verbose=False, mode='min')],
            gpus=args.GPU,
            deterministic=True,
            num_nodes=1,
            accelerator="cuda"
        )
        trainer.fit(task, train_loader_list[sid], dev_loader)
        save_adapter(args, task.model, adapter_name)
        args.logger.info("Test start...")
        if args.dataset == "mwz":
            _ = evaluate_model(args, task.tokenizer, task.model, test_loader, ALL_SLOTS, prefix=f"class{sid}")
        elif args.dataset == "sgd":
            _ = evaluate_sgd(args, task.tokenizer, task.model, test_loader, ALL_SLOTS, all_data)

    args.logger.info(f"Finish {args.except_domain}'s adapters!")
    ensemble_param(args, log=True)
    ensemble_output(args, log=True)

def evaluate(args):
    seed_everything(args.seed)
    model, tokenizer = prepare_model(args)
    if args.class_id > 0:
        log_name = '{}/eval_{}_{}.log'.format(args.saving_dir, args.model_name, args.class_id)
        adapter_name = f"class{args.class_id}_sub{args.sub_id}"
        model_file = os.path.join(args.saving_dir, f"class{args.class_id}_sub{args.sub_id}.pt")
    else:
        log_name = '{}/eval_{}.log'.format(args.saving_dir, args.model_name)
        adapter_name = "single"
        model_file = os.path.join(args.saving_dir, f"single.pt")

    args.logger = get_logger(log_name, fh_mode="a")
    task = DST_Seq2Seq(args, tokenizer, model, args.mode)
    test_loader, test_data_raw, ALL_SLOTS, all_data = prepare_test_data(args, tokenizer)
    adapter_dict = load_single_adapter(adapter_name, args.mode, torch.load(model_file))
    task.model.load_state_dict(adapter_dict, strict=False)
    args.logger.info(f"load {model_file} successfully!")
    if args.dataset == "mwz":
        _ = evaluate_model(args, task.tokenizer, task.model, test_loader, ALL_SLOTS, prefix="eval")
    elif args.dataset == 'sgd':
        _ = evaluate_sgd(args, task.tokenizer, task.model, test_loader, ALL_SLOTS, all_data)



def ensemble_param(args, log=False):
    seed_everything(args.seed)
    model, tokenizer = prepare_model(args)

    if log is False:
        args.logger = get_logger('{}/enparam_{}_{}.log'.
                                 format(args.saving_dir, args.model_name, args.class_id), fh_mode="a")
        args.logger.info(args)

    args.base_adapter_name = "en_param"
    task = DST_Seq2Seq(args, tokenizer, model, args.base_adapter_name)
    test_loader, test_data_raw, ALL_SLOTS, all_data = prepare_test_data(args, tokenizer)

    state_dicts = []
    for i in range(args.class_id):
        model_path = os.path.join(args.saving_dir, f"class{args.class_id}_sub{i}.pt")
        state_dicts.append({"name": f"class{args.class_id}_sub{i}", "model": torch.load(model_path)})

    args.logger.info(f"Load class{args.class_id}_sub*.pt successfully !")
    args.logger.info(f"dist_way:{args.dist_way},T:{args.T}")
    weights = load_dist_weight(args)
    if args.dataset == "mwz":
        generate_ensemble_param(args, tokenizer, task, test_loader, weights,
                                ALL_SLOTS, state_dicts, "ensemble_param")
    elif args.dataset == "sgd":
        generate_ensemble_param_sgd(args, tokenizer, task, test_loader, weights,
                                    ALL_SLOTS, state_dicts, all_data)

    ave_weight = torch.tensor([1 / args.class_id for i in range(args.class_id)]). \
        repeat(int(len(test_data_raw) / len(ALL_SLOTS)), 1)
    args.logger.info(f"dist_way: average")
    if args.dataset == "mwz":
        generate_ensemble_param(args, tokenizer, task, test_loader, ave_weight,
                            ALL_SLOTS, state_dicts, "ensemble_param")
    elif args.dataset == "sgd":
        generate_ensemble_param_sgd(args, tokenizer, task, test_loader, weights,
                                    ALL_SLOTS, state_dicts, all_data)

def ensemble_output(args, log=False):
    seed_everything(args.seed)

    if log is False:
        args.logger = get_logger('{}/enout_{}_{}.log'.
                                 format(args.saving_dir, args.model_name, args.class_id), fh_mode= "a")
    args.logger.info("Try ensemble multiple outputs ...")

    args.base_adapter_name = "ensemble_out"
    model, tokenizer = prepare_model(args)
    dst_model = DST_Seq2Seq(args, tokenizer, model, args.base_adapter_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    models = []
    for i in range(args.class_id):
        if args.adapter:
            model_path = os.path.join(args.saving_dir, f"class{args.class_id}_sub{i}.pt")
            state_dict = load_single_adapter(f"class{args.class_id}_sub{i}",
                                                 args.base_adapter_name, torch.load(model_path))
        else:
            model_path = os.path.join(args.saving_dir, f"model{i}.pt")
            state_dict = load_single_adapter(f"class{args.class_id}_sub{i}",
                                                     args.base_adapter_name, torch.load(model_path))
        model_copy = copy.deepcopy(dst_model)
        model_copy.model.load_state_dict(state_dict, strict=False)
        model_copy.to(device)
        model_copy.eval()
        models.append(model_copy)

    args.logger.info(f"Load #{len(models)} models successfully!")
    test_loader, test_data_raw, ALL_SLOTS, all_data = prepare_test_data(args, tokenizer)

    if args.dist_format == 'one-hot':
        for dist_way in ["inner", "euc"]:
            weights = load_dist_weight(args)
            args.logger.info(f"dist_way:{dist_way}")
            args.dist_way = dist_way
            generate_ensemble_output(args, models, test_loader, tokenizer, weights, ALL_SLOTS, prefix="ensemble_out")
    elif args.dist_format =='soft':
        for t in np.arange(1, 4, 1):
            for dist_way in ["inner", "euc"]:
                args.logger.info(f"dist_way:{dist_way},T:{t}")
                args.dist_way, args.T = dist_way, t
                weights = load_dist_weight(args)
                if args.dataset == "mwz":
                    generate_ensemble_output(args, models, test_loader, tokenizer, weights, ALL_SLOTS, prefix="ensemble_out")
                elif args.dataset == "sgd":
                    generate_ensemble_output_sgd(args, models, test_loader, tokenizer, weights, ALL_SLOTS, all_data)
    elif args.dist_format == 'average':
        ave_weight = torch.tensor([1 / args.class_id for i in range(args.class_id)]). \
            repeat(int(len(test_data_raw) / len(ALL_SLOTS)), 1)
        args.logger.info(f"dist_way: average")
        if args.dataset == "mwz":
            generate_ensemble_output(args, models, test_loader, tokenizer, ave_weight, ALL_SLOTS, prefix="ensemble_out")
        elif args.dataset == "sgd":
            generate_ensemble_output_sgd(args, models, test_loader, tokenizer, ave_weight, ALL_SLOTS, all_data)


def ensemble_out_domain(args, log=False):
    seed_everything(args.seed)
    if log is False:
        args.logger = get_logger('SaveOnlyDomain/{}/ensemble_out.log'.format(args.except_domain, args.except_domain))
    args.logger.info("Try ensemble multiple outputs ...")
    args.base_adapter_name = "ensemble_out"
    model, tokenizer = prepare_model(args)
    dst_model = DST_Seq2Seq(args, tokenizer, model, args.base_adapter_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.saving_dir = os.path.join("SaveOnlyDomain", args.except_domain)
    models = []
    for domain in EXPERIMENT_DOMAINS:
        if args.except_domain != domain:
            args.logger.info(f"Loading {domain}.pt ...")
            model_path = os.path.join("SaveOnlyDomain", domain, f"{domain}.pt")
            adapter_state_dict = load_single_adapter(f"single", args.base_adapter_name, torch.load(model_path))
            model_copy = copy.deepcopy(dst_model)
            model_copy.model.load_state_dict(adapter_state_dict, strict=False)
            model_copy.to(device)
            model_copy.eval()
            models.append(model_copy)

    args.logger.info(f"Load #{len(models)} models successfully!")
    test_loader, test_data_raw, ALL_SLOTS, all_data = prepare_test_data(args, tokenizer)

    for t in np.arange(1, 4, 1):
        for dist_way in ["inner", "euc"]:
            args.logger.info(f"dist_way:{dist_way},T:{t}")
            args.dist_way, args.T = dist_way, t
            weights = load_dist_domain(args)
            generate_ensemble_output(args, models, test_loader, tokenizer, weights, ALL_SLOTS,
                                         prefix="ensemble_out")

def ensemble_param_domain(args, log=False):
    seed_everything(args.seed)
    model, tokenizer = prepare_model(args)

    if log is False:
        args.logger = get_logger('SaveOnlyDomain/{}/enparam.log'.
                                 format(args.except_domain), fh_mode="a")
        args.logger.info(args)

    args.base_adapter_name = "ensemble_param"
    args.saving_dir = os.path.join("SaveOnlyDomain", args.except_domain)
    task = DST_Seq2Seq(args, tokenizer, model, args.base_adapter_name)
    test_loader, test_data_raw, ALL_SLOTS, all_data = prepare_test_data(args, tokenizer)

    state_dicts = []
    for domain in EXPERIMENT_DOMAINS:
        if args.except_domain != domain:
            args.logger.info(f"Loading {domain}.pt ...")
            model_path = os.path.join("SaveOnlyDomain", domain, f"{domain}.pt")
            state_dicts.append({"name": "single", "model": torch.load(model_path)})
            args.logger.info(f"Load {domain}.pt successfully !")

    for t in np.arange(0.1, 0.5, 0.1):
        for dist_way in ["inner", "euc"]:
            args.logger.info(f"dist_way:{dist_way},T:{t}")
            args.dist_way, args.T = dist_way, t
            weights = load_dist_domain(args)
            if args.dataset == "mwz":
                generate_ensemble_param(args, tokenizer, task, test_loader, weights,
                                    ALL_SLOTS, state_dicts, "ensemble_param")
            elif args.dataset == "sgd":
                generate_ensemble_param_sgd(args, tokenizer, task, test_loader, weights,
                                        ALL_SLOTS, state_dicts, all_data)

    ave_weight = torch.tensor([1 / args.class_id for i in range(args.class_id)]). \
        repeat(int(len(test_data_raw) / len(ALL_SLOTS)), 1)
    args.logger.info(f"dist_way: average")
    if args.dataset == "mwz":
        generate_ensemble_param(args, tokenizer, task, test_loader, ave_weight,
                            ALL_SLOTS, state_dicts, "ensemble_param")
    elif args.dataset == "sgd":
        generate_ensemble_param_sgd(args, tokenizer, task, test_loader, weights,
                                    ALL_SLOTS, state_dicts, all_data)

if __name__ == "__main__":
    args = get_args()

    args.saving_dir = os.path.join(args.saving_dir, args.except_domain, args.clu_encoder)
    if not os.path.exists(args.saving_dir):
        os.makedirs(args.saving_dir)

    args.cluster_dir = os.path.join(args.cluster_dir, args.except_domain, args.clu_encoder)
    args.model_checkpoint_file = f"../pretrained-model/{args.model_name}"

    if args.mode == "train":
        train(args)
    elif args.mode == "train_multi":
        train_multi(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "ensemble_param":
        ensemble_param(args)
    elif args.mode == "ensemble_output":
        ensemble_output(args)
    elif args.mode == "ensemble_domain":
        ensemble_param_domain(args)
        ensemble_out_domain(args)

