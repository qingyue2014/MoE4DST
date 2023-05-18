# Copyright (c) Facebook, Inc. and its affiliates

import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import random
from functools import partial
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
#SGD dataset
SGD_UNSEEN_DOMAINS = ["Alarm", "Alarm_1", "Payment","Paymemt_1", "Messaging", "Messaging_1", "Trains", "Trains_1"]

random.seed(577)
HISTORY_MAX_LEN = 450
GPT_MAX_LEN = 1024

class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        if self.args["slot_lang"] == "value":
            random.shuffle(item_info["value_list"])
            item_info["intput_text"] += " is " + " or ".join(item_info["value_list"]) + " or none?"
        return item_info

    def __len__(self):
        return len(self.data)

def read_data(args, path_name, tokenizer, ALL_SLOTS, dataset=None):
    domain_counter = {}
    context_data = []
    with open(path_name) as f:
        dials = json.load(f)
        if dataset== "train" and args["fewshot"]>0:
            random.Random(args["seed"]).shuffle(dials)
            dials = dials[:int(len(dials)*args["fewshot"])]

        for idx, dial_dict in enumerate(tqdm(dials)):
            dial_id = dial_dict["dial_id"]
            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "none" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "none" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
            (args["except_domain"] != "none" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]):
                continue

            # Reading data
            dial_dict_turns = dial_dict["turns"]
            for ti, turn in enumerate(dial_dict_turns):
                if ti == 0:
                    history_uttr = []
                    last_uttr = ""
                    last_turn_belief = {}
                    for slot in ALL_SLOTS:
                        last_turn_belief[slot] = "none"

                slot_values = turn["state"]["slot_values"]

                turn_belief_list = {}
                for slot in ALL_SLOTS:
                    turn_belief_list[slot] = "none"
                for k, v in slot_values.items():
                    if v == "dontcare":
                        slot_values[k] = "do not care"
                    turn_belief_list[str(k)] = str(slot_values[k])

                turn_only_label = {}  # turn label
                for slot in ALL_SLOTS:
                    if last_turn_belief[slot] != turn_belief_list[slot]:
                        d, s = slot.split("-")
                        turn_only_label[d] = s + "-" + turn_belief_list[slot]

                history_uttr.append(last_uttr)
                text_a = (" System: " + turn["system"] + " User: " + turn["user"])
                text_b = ' '.join(history_uttr[-args["num_turn"]:])
                last_uttr = text_a
                # accumulate dialogue utterances
                diag_1 = tokenizer.tokenize(text_a)
                diag_2 = tokenizer.tokenize(text_b)
                avail_length = args["max_length"] - 2 - len(diag_1)
                if avail_length <= 0:
                    diag_2 = []
                elif len(diag_2) > avail_length:
                    avail_length = len(diag_2) - avail_length
                    diag_2 = diag_2[avail_length:]

                if 'bert' in args["pretrained_model"] or 'BERT' in args["pretrained_model"]:
                    diag = [tokenizer.cls_token] + diag_2 + [tokenizer.sep_token] +  diag_1
                else:
                    diag = diag_2 + [tokenizer.sep_token] + diag_1
                input_id = tokenizer.convert_tokens_to_ids(diag)
                input_mask = [1] * len(input_id)
                if 'bert' in args["pretrained_model"] or 'BERT' in args["pretrained_model"]:
                    segment_id = [0] * len([tokenizer.cls_token] + diag_2) + [1] * len([tokenizer.sep_token] +  diag_1)
                else:
                    segment_id = [0] * len(diag_2) + [1] * len(diag_1)
                # input: dialogue history + slot
                # output: value
                # baseline gpt have different preprocessing, e.g., output: (slot1-value1, slot2-value2, slot3-value3, ...)
                data = {
                    "dial_id": dial_id,
                    "turn_id": ti,
                    "diag": " ".join(diag),
                    "input_ids": input_id,
                    "input_mask": input_mask,
                    "segment_id": segment_id,
                    "turn_only_label": turn_only_label
                }
                context_data.append(data)
                last_turn_belief = turn_belief_list

    print("domain_counter", domain_counter)
    print("context_number", len(context_data))

    return context_data


def read_aug_data(args, path_name, tokenizer, ALL_SLOTS, dataset=None):
    domain_counter = {}
    context_data = []
    with open(path_name) as f:
        dials = json.load(f)
        if dataset== "train" and args["fewshot"]>0:
            random.Random(args["seed"]).shuffle(dials)
            dials = dials[:int(len(dials)*args["fewshot"])]

        for idx, dial_dict in enumerate(tqdm(dials)):
            #wqy
            dial_id = dial_dict["dialogue_idx"]
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "none" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "none" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
            (args["except_domain"] != "none" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]):
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                if ti == 0:
                    history_uttr = []
                    last_uttr = ""
                    last_turn_belief = {}
                    for slot in ALL_SLOTS:
                        last_turn_belief[slot] = "none"

                history_uttr.append(last_uttr)
                text_a = (" System: " + turn["system_transcript"] + " User: " + turn["transcript"])
                text_b = ' '.join(history_uttr[-args["num_turn"]:])
                last_uttr = text_a
                # accumulate dialogue utterances
                diag_1 = tokenizer.tokenize(text_a)
                diag_2 = tokenizer.tokenize(text_b)
                avail_length = args["max_length"] - 2 - len(diag_1)
                if avail_length <= 0:
                    diag_2 = []
                elif len(diag_2) > avail_length:
                    avail_length = len(diag_2) - avail_length
                    diag_2 = diag_2[avail_length:]

                if 'bert' in args["pretrained_model"] or 'BERT' in args["pretrained_model"]:
                    diag = [tokenizer.cls_token] + diag_2 + [tokenizer.sep_token] +  diag_1
                else:
                    diag = diag_2 + [tokenizer.sep_token] + diag_1
                input_id = tokenizer.convert_tokens_to_ids(diag)
                input_mask = [1] * len(input_id)
                if 'bert' in args["pretrained_model"] or 'BERT' in args["pretrained_model"]:
                    segment_id = [0] * len([tokenizer.cls_token] + diag_2) + [1] * len([tokenizer.sep_token] +  diag_1)
                else:
                    segment_id = [0] * len(diag_2) + [1] * len(diag_1)
                # input: dialogue history + slot
                # output: value
                # baseline gpt have different preprocessing, e.g., output: (slot1-value1, slot2-value2, slot3-value3, ...)
                data = {
                    "dial_id": dial_id,
                    "turn_id": ti,
                    "diag": " ".join(diag),
                    "input_id": input_id,
                    "input_mask": input_mask,
                    "segment_id": segment_id,
                }
                context_data.append(data)

    print("domain_counter", domain_counter)
    print("context_number", len(context_data))

    return context_data

def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]

    return SLOTS

def prepare_cluster_data(args, tokenizer):
    data_dir = args["data_dir"]
    if "mwz" in data_dir:
        path_train = f'{data_dir}/train_dials.json'
        path_test = f'{data_dir}/test_dials.json'
        ontology = json.load(open("../data/MULTIWOZ2.1/ontology.json", 'r'))
        ALL_SLOTS = get_slot_information(ontology)
        if "aug" in data_dir:
            data_train = read_aug_data(args, path_train, tokenizer, ALL_SLOTS, "train")
            data_test = read_aug_data(args, path_test, tokenizer, ALL_SLOTS, "test")
        else:
            data_train = read_data(args, path_train, tokenizer, ALL_SLOTS, "train")
            data_test = read_data(args, path_test, tokenizer, ALL_SLOTS, "test")
    elif "sgd" in data_dir:
        path_train = f'{data_dir}/train'
        path_test = f'{data_dir}/test'
        data_train, *_ = read_SGD(args, path_train, tokenizer, "train")
        data_test, all_data, ALL_SLOTS = read_SGD(args, path_test, tokenizer, "test")

    train_dataset = DSTDataset(data_train, args)
    test_dataset = DSTDataset(data_test, args)

    train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"],
                              collate_fn=partial(cluster_collate_fn, args=args, tokenizer=tokenizer), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"],
                             collate_fn=partial(cluster_collate_fn, args=args,tokenizer=tokenizer), shuffle=False, num_workers=0)
    return train_loader, test_loader, data_train, data_test


def cluster_collate_fn(data, args, tokenizer):
    def truncate(seq, max_length):
        if len(seq) <= max_length:
            return seq
        else:
            return seq[len(seq) - max_length:]

    def padding(list1, list2, list3, pad_token):
        max_len = max([len(i) for i in list1])  # utter-len
        result1 = torch.ones(len(list1), max_len).long() * pad_token
        result2 = torch.ones(len(list2), max_len).long() * pad_token
        result3 = torch.ones(len(list3), max_len).long() * pad_token
        for i in range(len(list1)):
            result1[i, :len(list1[i])] = list1[i]
            result2[i, :len(list2[i])] = list2[i]
            result2[i, :len(list3[i])] = list3[i]

        return result1, result2, result3

    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    for i, d in enumerate(data):
        input_ids = truncate(d["input_ids"], max_length=args["max_length"])
        input_ids_list.append(torch.LongTensor(input_ids))
        input_mask_list.append(torch.LongTensor(d["input_mask"]))
        segment_ids_list.append(torch.LongTensor(d["segment_id"]))

    batch_data["encoder_input"], batch_data["attention_mask"], batch_data["segment_id"] = padding(input_ids_list, input_mask_list, segment_ids_list,
                                                                        tokenizer.pad_token_id)

    return batch_data

def get_input(args, history, curr, tokenizer):
    diag_1 = tokenizer.tokenize(curr)
    diag_2 = tokenizer.tokenize(history)
    avail_length = args["max_length"] - len(diag_1) - 2
    if avail_length <= 0:
        diag_2 = []
    elif len(diag_2) > avail_length:
        avail_length = len(diag_2) - avail_length
        diag_2 = diag_2[avail_length:]

    diag = diag_2 + diag_1
    input_ids = tokenizer.convert_tokens_to_ids(diag)
    return input_ids, diag

def read_SGD(args, path_name, tokenizer, dataset):

    # read test set
    all_data = []
    # read from original data
    for filename in os.listdir(path_name):
        if filename.startswith("dialogues_"):
            with open(os.path.join(path_name, filename)) as f:
                data = json.load(f)
                all_data += data

    with open(os.path.join(path_name, "schema.json")) as f:
        data = json.load(f)
        check_list = ["what", "how", "whether", "which"]
        schema = {}
        for service in data:
            schema[service["service_name"]] = {}
            # collect required_slots and optional_slots
            slot_collection = []
            for intent in service["intents"]:
                for slot in intent["required_slots"]:
                    slot_collection.append(slot)
                for slot in intent["optional_slots"].keys():
                    slot_collection.append(slot)

            for slot in service["slots"]:
                description = slot["description"].lower()
                if any(c_l in description for c_l in check_list):
                    description = f"{description}?"
                else:
                    description = f"what is the {description}?"

                if slot["name"] in slot_collection:
                    schema[service["service_name"]][slot["name"]] = (description, slot["possible_values"])

    #schema = adjust_sgd_questions(schema)
    all_slot = []
    if dataset != "train":
        for k, value in schema.items():
            if k.split("_")[0] == args["except_domain"]:
                for v in value.keys():
                    all_slot.append(v)
    p_data = []
    # read dialogues

    for ID, dial in enumerate(tqdm(all_data, desc=dataset)):
        for idx, turn in enumerate(dial["turns"]):
            if idx == 0:
                history_uttr = []
                system = "none"
                last_uttr = ""
            utterance = turn["utterance"]
            utterance = fix_number(utterance)
            # User start the conversation
            if turn["speaker"] == "USER":
                assert idx % 2 == 0
                utterance = "System: " + system + " User: " + utterance
                history_uttr.append(last_uttr)
                last_uttr = utterance
                dialog_history = ' '.join(history_uttr[-args["num_turn"]:])
                input_ids, input_text = get_input(args, dialog_history, utterance, tokenizer)
                if dataset == "train":
                    data_detail = {
                        "ID": ID,
                        "dial_id": dial["dialogue_id"],
                        "turn_id": idx,
                        "input_text":input_text,
                        "input_ids": input_ids,
                        "input_mask": [1] * len(input_ids),
                        "segment_id": [0] * len(input_ids),
                    }
                    p_data.append(data_detail)
                if dataset != "train":
                    for fid, frame in enumerate(turn["frames"]):
                        # only select zero-shot domain
                        if frame["service"] in SGD_UNSEEN_DOMAINS:
                            data_detail = {
                                "ID": ID,
                                "dial_id": dial["dialogue_id"],
                                "domains": frame["service"],
                                "turn_id": idx,
                                "frame_id": fid,
                                "input_text": input_text,
                                "input_ids": input_ids,
                                "input_mask": [1] * len(input_ids),
                                "segment_id": [0] * len(input_ids),
                            }
                            p_data.append(data_detail)
            else:
                assert idx % 2 == 1
                system = utterance

    return p_data, all_data, all_slot

def fix_number(text):
    number_mapper = {"one": "1", "two": "2", "three":"3", "four":"4", "five":"5", "six":"6", "seven":"7", "eight":"8", "nine":"9", "ten":"10", "eleven":"11", "twelve":"12"}
    for fromx, tox in number_mapper.items():
        text = ' ' + text + ' '
        text = text.replace(f" {fromx} ", f" {tox} ")[1:-1]
    return text
