import torch
from tqdm import tqdm
import os
from utils.eval_util import evaluate_metrics
import json
from utils.model_utils import load_adapter_params
import EnsembleModel

def generate_ensemble_output(args, models, test_loader, tokenizer, weights, ALL_SLOTS, prefix="zeroshot"):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble_model = EnsembleModel(args, models)
    ensemble_model.to(args.device)
    ensemble_model.eval()
    slot_logger = {slot_name: [0, 0, 0] for slot_name in ALL_SLOTS}
    predictions = {}
    wid = 0
    for batch in tqdm(test_loader):
        w = weights[wid, :].to(args.device)
        wid = wid + 1
        input_ids = batch["encoder_input"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        output = ensemble_model.greedy_search(input_ids, attention_mask, w)

        value_batch = tokenizer.batch_decode(output, skip_special_tokens=True)
        for idx, value in enumerate(value_batch):
            dial_id = batch["ID"][idx]
            if dial_id not in predictions:
                predictions[dial_id] = {}
                predictions[dial_id]["domain"] = batch["domains"][idx][0]
                predictions[dial_id]["turns"] = {}
            if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
                predictions[dial_id]["turns"][batch["turn_id"][idx]] = {"turn_belief": batch["turn_belief"][idx],
                                                                        "pred_belief": []}

            if value != "none":
                predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(
                    str(batch["slot_text"][idx]) + '-' + str(value))

            # analyze slot acc:
            if str(value) == str(batch["value_text"][idx]):
                slot_logger[str(batch["slot_text"][idx])][1] += 1  # hit
            slot_logger[str(batch["slot_text"][idx])][0] += 1  # total

    for slot_log in slot_logger.values():
        slot_log[2] = slot_log[1] / slot_log[0]

    with open(os.path.join(args.saving_dir, f"{prefix}_slot_acc.json"), 'w') as f:
        json.dump(slot_logger, f, indent=4)

    with open(os.path.join(args.saving_dir, f"{prefix}_prediction.json"), 'w') as f:
        json.dump(predictions, f, indent=4)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ALL_SLOTS)

    evaluation_metrics = {"Joint Acc": joint_acc_score, "Turn Acc": turn_acc_score, "Joint F1": F1_score}
    args.logger.info(f"evaluation_metrics:{evaluation_metrics}")
    args.logger.info(slot_logger)
    with open(os.path.join(args.saving_dir, f"{prefix}_result.json"), 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)


def generate_ensemble_param(args, tokenizer, task, test_loader, weight, ALL_SLOTS, state_dicts, prefix="zeroshot"):
    predictions = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task.model.to(device)
    task.model.eval()
    slot_logger = {slot_name: [0, 0, 0] for slot_name in ALL_SLOTS}
    wid = 0
    for batch in tqdm(test_loader):
        adapter_state_dict = load_adapter_params(device, state_dicts, args.base_adapter_name, weight[wid, :])
        task.model.load_state_dict(adapter_state_dict, strict=False)
        dst_outputs = task.model.generate(input_ids=batch["encoder_input"].to(device),
                                          attention_mask=batch["attention_mask"].to(device),
                                          eos_token_id=tokenizer.eos_token_id,
                                          max_length=15)

        value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)

        wid = wid + 1

        for idx, value in enumerate(value_batch):
            dial_id = batch["ID"][idx]
            if dial_id not in predictions:
                predictions[dial_id] = {}
                predictions[dial_id]["domain"] = batch["domains"][idx][0]
                predictions[dial_id]["turns"] = {}
            if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
                predictions[dial_id]["turns"][batch["turn_id"][idx]] = {"turn_belief": batch["turn_belief"][idx],
                                                                        "pred_belief": []}

            if value != "none":
                predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(
                    str(batch["slot_text"][idx]) + '-' + str(value))

            # analyze slot acc:
            if str(value) == str(batch["value_text"][idx]):
                slot_logger[str(batch["slot_text"][idx])][1] += 1  # hit
            slot_logger[str(batch["slot_text"][idx])][0] += 1  # total

    for slot_log in slot_logger.values():
        slot_log[2] = slot_log[1] / slot_log[0]

    with open(os.path.join(args.saving_dir, f"{prefix}_slot_acc.json"), 'w') as f:
        json.dump(slot_logger, f, indent=4)

    with open(os.path.join(args.saving_dir, f"{prefix}_prediction.json"), 'w') as f:
        json.dump(predictions, f, indent=4)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ALL_SLOTS)

    evaluation_metrics = {"Joint Acc": joint_acc_score, "Turn Acc": turn_acc_score, "Joint F1": F1_score}
    args.logger.info(f"evaluation_metrics:{evaluation_metrics}")
    args.logger.info(slot_logger)

    with open(os.path.join(args.saving_dir, f"{prefix}_result.json"), 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)

    return predictions




