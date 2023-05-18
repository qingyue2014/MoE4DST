from tqdm import tqdm
import torch
import json
import os
from generate_sgd import print_sgd_metrics

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
UNSEEN_DOMAINS = ["Alarm", "Alarm_1", "Buses", "Buses_3", "Payment",
                  "Paymemt_1", "Messaging", "Messaging_1", "Trains", "Trains_1"]
def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC

def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
    else:
        if len(pred)==0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count

def evaluate_metrics(all_prediction, SLOT_LIST):
    total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for idx, dial in all_prediction.items():
        for k, cv in dial["turns"].items():
            if set(cv["turn_belief"]) == set(cv["pred_belief"]):
                joint_acc += 1
            total += 1

            # Compute prediction slot accuracy
            temp_acc = compute_acc(set(cv["turn_belief"]), set(cv["pred_belief"]), SLOT_LIST)
            turn_acc += temp_acc

            # Compute prediction joint F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(set(cv["turn_belief"]), set(cv["pred_belief"]))
            F1_pred += temp_f1
            F1_count += count

    joint_acc_score = joint_acc / float(total) if total!=0 else 0
    turn_acc_score = turn_acc / float(total) if total!=0 else 0
    F1_score = F1_pred / float(F1_count) if F1_count!=0 else 0
    return joint_acc_score, F1_score, turn_acc_score

def evaluate_model(args, tokenizer, model, test_loader, ALL_SLOTS, prefix="zeroshot"):
    predictions = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    slot_logger = {slot_name:[0,0,0] for slot_name in ALL_SLOTS}
    for batch in tqdm(test_loader):
        dst_outputs = model.generate(input_ids=batch["encoder_input"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=15)

        value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)

        for idx, value in enumerate(value_batch):
            dial_id = batch["ID"][idx]
            if dial_id not in predictions:
                predictions[dial_id] = {}
                predictions[dial_id]["domain"] = batch["domains"][idx][0]
                predictions[dial_id]["turns"] = {}
            if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
                predictions[dial_id]["turns"][batch["turn_id"][idx]] = {"turn_belief": batch["turn_belief"][idx], "pred_belief":[]}

            if value != "none":
                predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(str(batch["slot_text"][idx])+'-'+str(value))

            # analyze slot acc:
            if str(value)==str(batch["value_text"][idx]):
                slot_logger[str(batch["slot_text"][idx])][1]+=1 # hit
            slot_logger[str(batch["slot_text"][idx])][0]+=1 # total

    for slot_log in slot_logger.values():
        slot_log[2] = slot_log[1]/slot_log[0]

    with open(os.path.join(args.saving_dir, f"{prefix}_slot_acc.json"), 'w') as f:
        json.dump(slot_logger,f, indent=4)

    with open(os.path.join(args.saving_dir, f"{prefix}_prediction.json"), 'w') as f:
        json.dump(predictions,f, indent=4)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ALL_SLOTS)

    evaluation_metrics = {"Joint Acc":joint_acc_score, "Turn Acc":turn_acc_score, "Joint F1": F1_score}
    args.logger.info(f"evaluation_metrics:{evaluation_metrics}")
    args.logger.info(f"slot acc:{slot_logger}")

    with open(os.path.join(args.saving_dir, f"{prefix}_result.json"), 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)

    return joint_acc_score

def evaluate_sgd(args, tokenizer, dst_model, test_loader, ALL_SLOTS, sgd_data):
    # to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    dst_model.to(device)
    dst_model.eval()

    # delete all the gold slot values for testing
    for dial in sgd_data:
        for turn in dial["turns"]:
            if turn["speaker"] == "USER":
                for frame in turn["frames"]:
                    frame["state"]["slot_values"] = {}

    with torch.no_grad():
        for batch in tqdm(test_loader):
            dst_outputs = dst_model.generate(input_ids=batch["encoder_input"].to(device),
                                             attention_mask=batch["attention_mask"].to(device),
                                             eos_token_id=tokenizer.eos_token_id,
                                             max_length=10)

            value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)
            for idx, value in enumerate(value_batch):
                dial_id = batch["ID"][idx]
                turn_id = batch["turn_id"][idx]
                frame_id = batch["frame_id"][idx]
                slot_key = batch["slot_text"][idx]
                # double check:
                assert sgd_data[dial_id]["dialogue_id"] == batch["dial_id"][idx]
                if value != "none":
                    sgd_data[dial_id]["turns"][turn_id]["frames"][frame_id]["state"]["slot_values"][slot_key] = [value]

    with open(os.path.join(args.saving_dir, "sgd_output.json"), 'w') as fout:
        json.dump(sgd_data, fout, indent=4)

    print_sgd_metrics(args, os.path.join(args.saving_dir, "sgd_output.json"))
