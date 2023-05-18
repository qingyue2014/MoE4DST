import os.path
from transformers import (T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, AutoModelForSeq2SeqLM)
import torch
import collections

def load_adapter_params(device, state_dicts, base_adapter_name, proportions):
    param_lst = collections.defaultdict(list)
    for state_dict, prop in zip(state_dicts, proportions):
        name = state_dict["name"]
        model = state_dict["model"]
        for k, p in model.items():
            p = p.to(device)
            if f"adapters.{name}." in k:
                rk = k.replace(f".{name}.", f".{base_adapter_name}.")
                param_lst[rk].append(p * prop)
    avg_dict = {
        k: torch.sum(torch.stack(vs, dim=0), dim=0)
        for k, vs in param_lst.items()
    }
    return avg_dict

def load_single_adapter(adapter_name, base_adapter_name, model):
    adapter_dict = {}
    flag = 0
    for k, p in model.items():
        if f"adapters.{adapter_name}." in k:
            rk = k.replace(f".{adapter_name}.", f".{base_adapter_name}.")
            adapter_dict[rk] = p
            flag = 1
    if flag == 1:
        print("Load adapter successfully!")
    else:
        print("Load adapter fail!")
    return adapter_dict


def set_frozen(args, model, adapter_name):
    total = frozen = 0
    for k, p in model.named_parameters():
        total += 1
        if freeze(args, k, adapter_name):
            p.requires_grad = False
            frozen += 1
        else:
            p.requires_grad = True
    print(f"froze {frozen}/{total} parameters")

def freeze(args, k, adapter_name):
    if 'adapter' and adapter_name in k:
        return args.freeze_adapter
    return args.freeze_transformer

def save_adapter(args, model, adapter_name):
    params = dict(list(model.named_parameters()))
    state_dict = collections.OrderedDict()
    for k, v in model.state_dict().items():
        if k in params and params[k].requires_grad:
            state_dict[k] = v

    save_file = os.path.join(args.saving_dir, f"{adapter_name}.pt")
    args.logger.info(
        f"saving model to {save_file} "
        f"({len(state_dict)}/{len(params)} parameters)"
    )
    torch.save(state_dict, save_file)

def prepare_model(args):
    args.model_checkpoint = f"../pretrained-model/{args.model_name}"
    if "t5" in args.model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
        tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint, sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    elif "bart" in args.model_name:
        model = BartForConditionalGeneration.from_pretrained(args.model_checkpoint)
        tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    return model, tokenizer



