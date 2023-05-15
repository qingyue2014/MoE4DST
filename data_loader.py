# Copyright (c) Facebook, Inc. and its affiliates

import json
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ast
from tqdm import tqdm
import os
import numpy as np
import random
from functools import partial
from utils.fix_label import fix_general_label_error
from collections import OrderedDict

#MWZ dataset
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
#SGD dataset
UNSEEN_DOMAINS = ["Alarm", "Alarm_1", "Buses", "Buses_3", "Payment",
                  "Paymemt_1", "Messaging", "Messaging_1", "Trains", "Trains_1"]
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
        if self.args.slot_lang == "value":
            random.shuffle(item_info["value_list"])
            item_info["intput_text"] += " is " + " or ".join(item_info["value_list"]) + " or none?"
        return item_info

    def __len__(self):
        return len(self.data)

def read_data(args, path_name, SLOTS, tokenizer, description, dataset=None, file_set=None):
    data = []
    domain_counter = {}
    data_set = {i: [] for i in range(args.class_id)}
    if file_set is not None:
        label_dict = pickle.load(open(file_set, "rb"))

    with open(path_name) as f:
        dials = json.load(f)
        for idx, dial_dict in enumerate(tqdm(dials)):
            #wqy
            dialog_history = ""
            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args.only_domain != "none" and args.only_domain not in dial_dict["domains"]:
                continue
            if (args.except_domain != "none" and dataset == "test" and args.except_domain not in dial_dict["domains"]) or \
            (args.except_domain != "none" and dataset != "test" and [args.except_domain] == dial_dict["domains"]):
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["turns"]):
                turn_id = ti

                dialog_history +=  (" System: " + turn["system"] + " User: " + turn["user"])

                # accumulate dialogue utterances
                if args.fix_label:
                    slot_values = fix_general_label_error(turn["state"]["slot_values"], SLOTS)
                else:
                    slot_values = turn["state"]["slot_values"]
                # input: dialogue history + slot
                # output: value

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset != "test":
                    if args.except_domain != "none":
                        slot_temp = [k for k in SLOTS if args.except_domain not in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args.except_domain not in k])
                    elif args.only_domain != "none":
                        slot_temp = [k for k in SLOTS if args.only_domain in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args.only_domain in k])
                else:
                    if args.except_domain != "none":
                        slot_temp = [k for k in SLOTS if args.except_domain in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args.except_domain in k])
                    elif args.only_domain != "none":
                        slot_temp = [k for k in SLOTS if args.only_domain in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args.only_domain in k])

                turn_belief_list = []
                for k, v in slot_values.items():
                    if v == "dontcare":
                        slot_values[k] = "do not care"
                    if v != "none":
                        turn_belief_list.append(str(k) + '-' + str(slot_values[k]))

                for slot in slot_temp:
                    # skip unrelevant slots for out of domain setting
                    if args.except_domain != "none" and dataset != "test":
                        if slot.split("-")[0] not in dial_dict["domains"]:
                            continue

                    output_text = slot_values.get(slot, 'none').strip() + f" {tokenizer.eos_token}"
                    slot_text = slot
                    value_text = slot_values.get(slot, 'none').strip()
                    if args.slot_lang == "human":
                        slot_lang = description[slot]["description_human"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                    elif args.slot_lang == "naive":
                        slot_lang = description[slot]["naive"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                    elif args.slot_lang == "value":
                        slot_lang = description[slot]["naive"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                    elif args.slot_lang == "question":
                        slot_lang = description[slot]["question"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                    elif args.slot_lang == "slottype":
                        slot_lang = description[slot]["slottype"]
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                    else:
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot}"

                    data_detail = {
                        "ID": dial_dict["dial_id"],
                        "domains": dial_dict["domains"],
                        "turn_id": turn_id,
                        "dialog_history": dialog_history,
                        "turn_belief": turn_belief_list,
                        "input_text": input_text,
                        "output_text": output_text,
                        "slot_text": slot_text,
                        "value_text": value_text,
                        "value_list": description[slot]["values"]
                    }
                    data.append(data_detail)

                if file_set is not None:
                    label = label_dict[str(dial_dict["dial_id"]) + "-" + str(turn_id)]
                    data_set[label].extend(data)
                    data = []

    args.logger.info(f"{dataset}_domain_counter:{domain_counter}")
    if file_set is not None:
        for i in range(args.class_id):
            args.logger.info(f"class{i}: {len(data_set[i])}")
        return data_set, slot_temp, dials
    else:
        return data, slot_temp, dials

def read_aug_data(args, path_name, SLOTS, tokenizer, description, dataset=None, file_set=None):
    data = []
    domain_counter = {}
    data_set = {i: [] for i in range(args.class_id)}
    if file_set is not None:
        label_dict = pickle.load(open(file_set, "rb"))

    with open(path_name) as f:
        dials = json.load(f)
        for idx, dial_dict in enumerate(tqdm(dials)):
            #wqy
            dialog_history = ""
            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args.only_domain != "none" and args.only_domain not in dial_dict["domains"]:
                continue
            if (args.except_domain != "none" and dataset == "test" and args.except_domain not in dial_dict["domains"]) or \
            (args.except_domain != "none" and dataset != "test" and [args.except_domain] == dial_dict["domains"]):
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_id = ti

                dialog_history +=  (" System: " + turn["system_transcript"] + " User: " + turn["transcript"])

                # accumulate dialogue utterances
                if args.fix_label:
                    slot_values = fix_general_label_error(turn["belief_state"]["slot_values"], SLOTS)
                else:
                    slot_values = {}
                    for item in turn["belief_state"]:
                        name, value = item["slots"][0][0], item["slots"][0][1]
                        slot_values[name] = value
                # input: dialogue history + slot
                # output: value

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset != "test":
                    if args.except_domain != "none":
                        slot_temp = [k for k in SLOTS if args.except_domain not in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args.except_domain not in k])
                    elif args.only_domain != "none":
                        slot_temp = [k for k in SLOTS if args.only_domain in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args.only_domain in k])
                else:
                    if args.except_domain != "none":
                        slot_temp = [k for k in SLOTS if args.except_domain in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args.except_domain in k])
                    elif args.only_domain != "none":
                        slot_temp = [k for k in SLOTS if args.only_domain in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args.only_domain in k])

                turn_belief_list = []
                for k, v in slot_values.items():
                    if v == "dontcare":
                        slot_values[k] = "do not care"
                    if v != "none":
                        turn_belief_list.append(str(k) + '-' + str(slot_values[k]))
                # baseline gpt have different preprocessing, e.g., output: (slot1-value1, slot2-value2, slot3-value3, ...)
                if "gpt" in args.model_name:
                    turn_slots = []
                    turn_slot_values = []
                    if len(dialog_history.split())>800:
                        continue
                    for slot in slot_temp:
                        # skip unrelevant slots for out of domain setting
                        if args.except_domain != "none" and dataset !="test":
                            if slot.split("-")[0] not in dial_dict["domains"]:
                                continue
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot}" + " " + tokenizer.bos_token
                        output_text = input_text+ " " + turn["state"]["slot_values"].get(slot, 'none').strip() + " " + tokenizer.eos_token
                        slot_text = slot
                        value_text = turn["state"]["slot_values"].get(slot, 'none').strip()

                        data_detail = {
                            "ID":dial_dict["dialogue_idx"],
                            "domains":dial_dict["domains"],
                            "turn_id":turn_id,
                            "dialog_history":dialog_history,
                            "turn_belief":turn_belief_list,
                            "intput_text":input_text,
                            "output_text":output_text,
                            "slot_text":slot_text,
                            "value_text":value_text
                            }
                        data.append(data_detail)

                else:
                    for slot in slot_temp:
                        # skip unrelevant slots for out of domain setting
                        if args.except_domain != "none" and dataset !="test":
                            if slot.split("-")[0] not in dial_dict["domains"]:
                                continue

                        output_text = slot_values.get(slot, 'none').strip() + f" {tokenizer.eos_token}"
                        slot_text = slot
                        value_text = slot_values.get(slot, 'none').strip()
                        if args.slot_lang == "human":
                            slot_lang = description[slot]["description_human"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                        elif args.slot_lang == "naive":
                            slot_lang = description[slot]["naive"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                        elif args.slot_lang == "value":
                            slot_lang = description[slot]["naive"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                        elif args.slot_lang == "question":
                            slot_lang = description[slot]["question"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                        elif args.slot_lang == "slottype":
                            slot_lang = description[slot]["slottype"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                        else:
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot}"

                        data_detail = {
                            "ID": dial_dict["dialogue_idx"],
                            "domains":dial_dict["domains"],
                            "turn_id":turn_id,
                            "dialog_history":dialog_history,
                            "turn_belief": turn_belief_list,
                            "input_text":input_text,
                            "output_text":output_text,
                            "slot_text":slot_text,
                            "value_text":value_text,
                            "value_list":description[slot]["values"]
                            }
                        data.append(data_detail)

                    if file_set is not None:
                        label = label_dict[str(dial_dict["dialogue_idx"]) + "-" + str(turn_id)]
                        data_set[label].extend(data)
                        data = []

    args.logger.info(f"{dataset}_domain_counter:{domain_counter}")
    if file_set is not None:
        for i in range(args.class_id):
            args.logger.info(f"class{i}: {len(data_set[i])}")
        return data_set, slot_temp, dials
    else:
        return data, slot_temp, dials

def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]

    return SLOTS

def collate_fn(data, args, tokenizer):
    def truncate(seq, max_length):
        if len(seq) <= max_length:
            return seq
        else:
            return seq[len(seq) - max_length:]

    def padding(list1, list2, pad_token):
        max_len = max([len(i) for i in list1])  # utter-len
        result1 = torch.ones(len(list1), max_len).long() * pad_token
        result2 = torch.ones(len(list2), max_len).long() * pad_token
        for i in range(len(list1)):
            result1[i, :len(list1[i])] = list1[i]
            result2[i, :len(list2[i])] = list2[i]

        return result1, result2

    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_ids_list = []
    input_mask_list = []
    for i, d in enumerate(data):
        input_ids = tokenizer.encode(d["input_text"], add_special_tokens=False, verbose=False)
        input_ids = truncate(input_ids, max_length=args.max_length)
        input_ids_list.append(torch.LongTensor(input_ids))
        input_masks = [1] * len(input_ids)
        input_mask_list.append(torch.LongTensor(input_masks))

    batch_data["encoder_input"], batch_data["attention_mask"] = padding(input_ids_list, input_mask_list,
                                                                        tokenizer.pad_token_id)
    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt",
                             add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids'] == tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    return batch_data

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b

def load_dist_domain(args):
    test_vec = pickle.load(open(f"{args.saving_dir}/test_vec.pkl", "rb"))
    cent_vec = pickle.load(open(f"{args.saving_dir}/cent_vec_4.pkl", "rb"))
    if args.dist_way == "inner":
        new_cent_pred = np.transpose(cent_vec)
        dist_array = np.dot(test_vec, new_cent_pred)
        weight = torch.softmax(torch.tensor(dist_array) / args.T, dim=-1)
    elif args.dist_way == "euc":
        new_test_vecs = np.expand_dims(test_vec, axis=1)
        new_cent_pred = np.expand_dims(cent_vec, axis=0)
        dist_array = - np.sqrt(np.sum((new_test_vecs - new_cent_pred) ** 2, axis=-1))
        weight = torch.softmax(torch.tensor(dist_array) / args.T, dim=-1)
    return weight

def load_dist_weight(args):
    test_vec = pickle.load(open(f"{args.cluster_dir}/test_vec.pkl", "rb"))
    cent_vec = pickle.load(open(f"{args.cluster_dir}/{args.clu_algorithm}_cent_vec_{args.class_id}.pkl", "rb"))

    if args.dist_way == "inner":
        new_cent_pred = np.transpose(cent_vec)
        dist_array = np.dot(test_vec, new_cent_pred)
        weight = torch.softmax(torch.tensor(dist_array) / args.T, dim=-1)
    elif args.dist_way == "euc":
        new_test_vecs = np.expand_dims(test_vec, axis=1)
        new_cent_pred = np.expand_dims(cent_vec, axis=0)
        dist_array = - np.sqrt(np.sum((new_test_vecs - new_cent_pred) ** 2, axis=-1))
        weight = torch.softmax(torch.tensor(dist_array) / args.T, dim=-1)
    elif args.dist_way == 'mix':
        ave_weight = torch.tensor([1 / args.class_id for i in range(args.class_id)]).repeat(len(test_vec), 1)
        new_test_vecs = np.expand_dims(test_vec, axis=1)
        new_cent_pred = np.expand_dims(cent_vec, axis=0)
        dist_array = - np.sqrt(np.sum((new_test_vecs - new_cent_pred) ** 2, axis=-1))
        euc_weight = torch.softmax(torch.tensor(dist_array) / args.T, dim=-1)
        weight = ave_weight * 0.5 + euc_weight * 0.5

    if args.dist_format == "one-hot":
        weight = torch.tensor(props_to_onehot(weight))
    return weight

def prepare_test_data(args, tokenizer):
    if "mwz" in args.data_dir:
        ontology = json.load(open("data/MULTIWOZ2.1/ontology.json", 'r'))
        ALL_SLOTS = get_slot_information(ontology)
        description = json.load(open("utils/slot_description.json", 'r'))
        if "aug" in args.data_dir:
            data_test, ALL_SLOTS, all_data = read_aug_data(args, f'{args.data_dir}/test_dials.json', ALL_SLOTS,
                                                       tokenizer, description, "test")
        else:
            data_test, ALL_SLOTS, all_data = read_data(args, f'{args.data_dir}/test_dials.json', ALL_SLOTS,
                                                   tokenizer, description, "test")
    elif args.dataset == "sgd":
        data_test, ALL_SLOTS, all_data = read_SGD(args=args, path_name="data/sgd/test", dataset="test", tokenizer=tokenizer)

    test_dataset = DSTDataset(data_test, args)
    test_loader = DataLoader(test_dataset, batch_size=len(ALL_SLOTS), shuffle=False,
                             collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer), num_workers=0)
    return test_loader, data_test, ALL_SLOTS, all_data

def fix_number(text):
    number_mapper = {"one": "1", "two": "2", "three":"3", "four":"4", "five":"5", "six":"6", "seven":"7", "eight":"8", "nine":"9", "ten":"10", "eleven":"11", "twelve":"12"}
    for fromx, tox in number_mapper.items():
        text = ' ' + text + ' '
        text = text.replace(f" {fromx} ", f" {tox} ")[1:-1]
    return text

def read_SGD(args, path_name, tokenizer, dataset, file_set=None):
    all_data = []
    # read from original data
    for filename in os.listdir(os.path.join(path_name, dataset)):
        if filename.startswith("dialogues_"):
            with open(os.path.join(path_name,dataset,filename)) as f:
                data = json.load(f)
                all_data += data

    if file_set is not None:
        label_dict = pickle.load(open(file_set, "rb"))

    data_set = {i: [] for i in range(args.class_id)}
    with open(os.path.join(path_name, dataset, "schema.json")) as f:
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
            if k.split("_")[0] == args.except_domain:
                all_slot = value.keys()
    p_data = []
    # read dialogues

    for ID, dial in enumerate(tqdm(all_data, desc=dataset)):
        #print(ID)
        if args.do_test and len(p_data) > 1:
            break
        for idx, turn in enumerate(dial["turns"]):
            if idx == 0:
                history_uttr = []
                system = "none"
                last_uttr = ""
                last_turn_belief = []
            utterance = turn["utterance"]
            utterance = fix_number(utterance)
            # User start the conversation
            if turn["speaker"] == "USER":
                assert idx%2==0
                utterance = "System: "+ system + " User: " + utterance
                history_uttr.append(last_uttr)
                last_uttr = utterance
                dialog_history = ' '.join(history_uttr)
                for fid, frame in enumerate(turn["frames"]):
                    # read slot values
                    turn_belief_list = []
                    for k in schema[frame["service"]]:
                        value_text = frame["state"]["slot_values"].get(k, ['none'])[0].strip().lower()
                        if value_text == 'dontcare':
                            value_text = 'do not care'
                        if value_text != 'none':
                             turn_belief_list.append(str(frame["service"][:-2]) + "-" + str(k)+'*'+str(value_text))

                    for k in schema[frame["service"]]:
                        value_text = frame["state"]["slot_values"].get(k, ['none'])[0].strip().lower()
                        slot_lang = schema[frame["service"]][k][0]
                        input_text = dialog_history + f" {utterance} {tokenizer.sep_token} {slot_lang}"
                        #only select zero-shot domain
                        if dataset == "train":
                            data_detail = {
                                "ID": ID,
                                "dial_id": dial["dialogue_id"],
                                "domains": frame["service"][:-2],
                                "turn_id": idx,
                                "frame_id": fid,
                                "slot_text": k,
                                "slot_lang": slot_lang,
                                "dialog_history": dialog_history,
                                "curr_uttr":utterance,
                                "input_text": input_text,
                                "value_text": value_text,
                                "output_text": value_text + f" {tokenizer.eos_token}",
                                "turn_belief_list": turn_belief_list,
                                "last_turn_belief": last_turn_belief,
                            }
                            p_data.append(data_detail)
                        elif dataset != "train":
                            if args.except_domain in frame["service"]:
                                data_detail = {
                                    "ID": ID,
                                    "dial_id": dial["dialogue_id"],
                                    "turn_id": idx,
                                    "frame_id": fid,
                                    "dialog_history": dialog_history,
                                    "curr_uttr": utterance,
                                    "input_text": input_text,
                                    "slot_lang": slot_lang,
                                    "slot_text": k,
                                    "value_text": value_text,
                                    "output_text": value_text + f" {tokenizer.eos_token}",
                                }
                                p_data.append(data_detail)
                    if file_set is not None:
                        label = label_dict[str(dial["dialogue_id"]) + "-" + str(idx)]
                        data_set[label].extend(p_data)
                        p_data = []
                last_turn_belief = turn_belief_list.copy()
            # system turn
            else:
                assert idx%2==1
                system = utterance

    if file_set is not None:
        for key, value in data_set.items():
            args.logger.info(f"#data{key}:{len(value)}")
        args.logger.info("Example 0:")
        input0, out0 = data_set[0][0]["input_text"][-100], data_set[0][0]["output_text"]
        args.logger.info(f"input:{input0} output:{out0}")
        return data_set, all_slot, all_data
    else:
        args.logger.info(f"#data: {len(p_data)}")
        return p_data, all_slot, all_data

def prepare_multidata(args, tokenizer):
    file_set = os.path.join(args.cluster_dir, f"{args.clu_algorithm}_dial2label_{args.class_id}.pkl")
    if "mwz" in args.data_dir:
        path_train = f'{args.data_dir}/train_dials.json'
        path_dev = f'{args.data_dir}/dev_dials.json'
        path_test = f'{args.data_dir}/test_dials.json'
        ontology = json.load(open("data/MULTIWOZ2.1/ontology.json", 'r'))
        ALL_SLOTS = get_slot_information(ontology)
        description = json.load(open("utils/slot_description.json", 'r'))
        if "aug" in args.data_dir:
            data_train, *_ = read_aug_data(args, path_train, ALL_SLOTS, tokenizer, description, "train", file_set=file_set)
            data_dev, *_ = read_aug_data(args, path_dev, ALL_SLOTS, tokenizer, description, "dev")
            data_test, ALL_SLOTS, all_data = read_aug_data(args, path_test, ALL_SLOTS, tokenizer, description, "test")
        else:
            data_train, *_ = read_data(args, path_train, ALL_SLOTS, tokenizer, description, "train", file_set=file_set)
            data_dev, *_ = read_data(args, path_dev, ALL_SLOTS, tokenizer, description, "dev")
            data_test, ALL_SLOTS, all_data = read_data(args, path_test, ALL_SLOTS, tokenizer, description, "test")
    elif "sgd" in args.data_dir:
        data_train, *_ = read_SGD(args=args, path_name="data/sgd/", dataset="train", tokenizer=tokenizer, file_set=file_set)
        data_dev, *_ = read_SGD(args=args, path_name="data/sgd/", dataset="test", tokenizer=tokenizer)
        data_test, ALL_SLOTS, all_data = read_SGD(args=args, path_name="data/sgd/", dataset="test", tokenizer=tokenizer)

    train_loader_list = []
    for idx in data_train:
        train_dataset = DSTDataset(data_train[idx], args)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args),
                                  num_workers=args.num_workers)
        train_loader_list.append(train_loader)

    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args),
                             num_workers=args.num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args),
                            num_workers=args.num_workers)

    return train_loader_list, dev_loader, test_loader, ALL_SLOTS, data_train, data_test, all_data


def prepare_data(args, tokenizer):
    if 'mwz' in args.data_dir:
        path_train = 'data/mwz2.1/train_dials.json'
        path_dev = 'data/mwz2.1/dev_dials.json'
        path_test = 'data/mwz2.1/test_dials.json'
        ontology = json.load(open("data/mwz2.1/ontology.json", 'r'))
        ALL_SLOTS = get_slot_information(ontology)
        description = json.load(open("utils/slot_description.json", 'r'))
        if "aug" in args.data_dir:
            data_train, *_ = read_aug_data(args, path_train, ALL_SLOTS, tokenizer, description, dataset="train")
            data_dev, *_ = read_aug_data(args, path_dev, ALL_SLOTS, tokenizer, description, dataset="dev")
            data_test, ALL_SLOTS, all_data = read_aug_data(args, path_test, ALL_SLOTS, tokenizer, description,
                                                       dataset="test")
        else:
            data_train, *_ = read_data(args, path_train, ALL_SLOTS, tokenizer, description, dataset="train")
            data_dev, *_ = read_data(args, path_dev, ALL_SLOTS, tokenizer, description, dataset="dev")
            data_test, ALL_SLOTS, all_data = read_data(args, path_test, ALL_SLOTS, tokenizer, description, dataset="test")
    else:
        data_train, *_ = read_SGD(args=args, path_name = "data/sgd/train", dataset="train", tokenizer=tokenizer)
        data_dev, *_ = read_SGD(args=args, path_name="data/sgd/dev", dataset="dev", tokenizer=tokenizer)
        data_test, ALL_SLOTS, all_data = read_SGD(args=args, path_name="data/sgd/test", dataset="test", tokenizer=tokenizer)

    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args),
                              num_workers=args.num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args),
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args),
                             num_workers=args.num_workers)

    return train_loader, dev_loader, test_loader, ALL_SLOTS, data_train, data_test, all_data
