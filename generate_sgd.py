import torch
from tqdm import tqdm
import collections
from utils import metrics
import numpy as np
import os
import json
import EnsembleModel
from utils.model_utils import load_adapter_params
ALL_SERVICES = "#ALL_SERVICES"
SEEN_SERVICES = "#SEEN_SERVICES"
UNSEEN_SERVICES = "#UNSEEN_SERVICES"

# Name of the file containing all predictions and their corresponding frame
# metrics.

def get_in_domain_services(schema_path_1, schema_path_2):
  """Get the set of common services between two schemas."""
  return get_service_set(schema_path_1) & get_service_set(schema_path_2)

def get_service_set(schema_path):
  """Get the set of all services present in a schema."""
  service_set = set()
  with open(schema_path,"r") as f:
    schema = json.load(f)
    for service in schema:
      service_set.add(service["service_name"])
  return service_set

PER_FRAME_OUTPUT_FILENAME = "dialogues_and_metrics.json"


def generate_ensemble_output_sgd(args, models, test_loader, tokenizer, weights, ALL_SLOTS, sgd_data):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble_model = EnsembleModel(args, models)
    ensemble_model.to(args.device)
    ensemble_model.eval()
    dataset_ref = {}
    wid = 0
    for dial in sgd_data:
        dataset_ref[dial["dialogue_id"]] = dial.copy()
    # delete all the gold slot values for testing
    for dial in sgd_data:
        for turn in dial["turns"]:
            if turn["speaker"] == "USER":
                for frame in turn["frames"]:
                    frame["state"]["slot_values"] = {}

    for batch in tqdm(test_loader):
        w = weights[wid, :].to(args.device)
        wid = wid + 1
        input_ids = batch["encoder_input"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        output = ensemble_model.greedy_search(input_ids, attention_mask, w)

        value_batch = tokenizer.batch_decode(output, skip_special_tokens=True)
        for idx, value in enumerate(value_batch):
            dial_id = batch["ID"][idx]
            turn_id = batch["turn_id"][idx]
            frame_id = batch["frame_id"][idx]
            slot_key = batch["slot_text"][idx]
            # double check
            assert sgd_data[dial_id]["dialogue_id"] == batch["dial_id"][idx]
            if value != "none":
                sgd_data[dial_id]["turns"][turn_id]["frames"][frame_id]["state"]["slot_values"][slot_key] = [value]

    with open(os.path.join(args.saving_dir, f"sgd_output.json"), 'w') as fout:
        json.dump(sgd_data, fout, indent=4)

    print_sgd_metrics(args, os.path.join(args.saving_dir, f"sgd_output.json"))


def generate_ensemble_param_sgd(args, tokenizer, task, test_loader, weight, ALL_SLOTS, state_dicts, sgd_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task.model.to(device)
    task.model.eval()
    wid = 0
    dataset_ref = {}
    for dial in sgd_data:
        dataset_ref[dial["dialogue_id"]] = dial.copy()
    # delete all the gold slot values for testing
    for dial in sgd_data:
        for turn in dial["turns"]:
            if turn["speaker"] == "USER":
                for frame in turn["frames"]:
                    frame["state"]["slot_values"] = {}

    for batch in tqdm(test_loader):
        adapter_state_dict = load_adapter_params(device, state_dicts, args.base_adapter_name, weight[wid, :])
        task.model.load_state_dict(adapter_state_dict, strict=False)
        dst_outputs = task.model.generate(input_ids=batch["encoder_input"].to(device),
                                          attention_mask=batch["attention_mask"].to(device),
                                          eos_token_id=tokenizer.eos_token_id,
                                          max_length=10)

        value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)
        wid = wid + 1
        for idx, value in enumerate(value_batch):
            dial_id = batch["ID"][idx]
            turn_id = batch["turn_id"][idx]
            frame_id = batch["frame_id"][idx]
            slot_key = batch["slot_text"][idx]
            assert sgd_data[dial_id]["dialogue_id"] == batch["dial_id"][idx]
            if value != "none":
                sgd_data[dial_id]["turns"][turn_id]["frames"][frame_id]["state"]["slot_values"][slot_key] = [value]

    with open(os.path.join(args.saving_dir, "sgd_output.json"), 'w') as fout:
        json.dump(sgd_data, fout, indent=4)

    print_sgd_metrics(args, os.path.join(args.saving_dir, "sgd_output.json"))

def load_json_file(file_path):
    dataset_dict = {}
    data = json.load(open(file_path,"r"))
    if isinstance(data, list):
        for dial in data:
            dataset_dict[dial["dialogue_id"]] = dial
    return dataset_dict

def get_metrics(dataset_ref, dataset_hyp, service_schemas, in_domain_services):
  """Calculate the DSTC8 metrics.

  Args:
    dataset_ref: The ground truth dataset represented as a dict mapping dialogue
      id to the corresponding dialogue.
    dataset_hyp: The predictions in the same format as `dataset_ref`.
    service_schemas: A dict mapping service name to the schema for the service.
    in_domain_services: The set of services which are present in the training
      set.

  Returns:
    A dict mapping a metric collection name to a dict containing the values
    for various metrics. Each metric collection aggregates the metrics across
    a specific set of frames in the dialogues.
  """
  metric_collections = collections.defaultdict(
      lambda: collections.defaultdict(list))

  # Ensure the dialogs in dataset_hyp also occur in dataset_ref.
  assert set(dataset_hyp.keys()).issubset(set(dataset_ref.keys()))

  # Store metrics for every frame for debugging.
  per_frame_metric = {}
  for dial_id, dial_hyp in dataset_hyp.items():
    dial_ref = dataset_ref[dial_id]

    if set(dial_ref["services"]) != set(dial_hyp["services"]):
      raise ValueError(
          "Set of services present in ground truth and predictions don't match "
          "for dialogue with id {}".format(dial_id))
    joint_metrics = [
        metrics.JOINT_GOAL_ACCURACY, metrics.JOINT_CAT_ACCURACY,
        metrics.JOINT_NONCAT_ACCURACY
    ]
    for turn_id, (turn_ref, turn_hyp) in enumerate(
        zip(dial_ref["turns"], dial_hyp["turns"])):
      metric_collections_per_turn = collections.defaultdict(
          lambda: collections.defaultdict(lambda: 1.0))
      if turn_ref["speaker"] != turn_hyp["speaker"]:
        raise ValueError(
            "Speakers don't match in dialogue with id {}".format(dial_id))

      # Skip system turns because metrics are only computed for user turns.
      if turn_ref["speaker"] != "USER":
        continue

      if turn_ref["utterance"] != turn_hyp["utterance"]:
        raise ValueError(
            "Utterances don't match for dialogue with id {}".format(dial_id))

      hyp_frames_by_service = {
          frame["service"]: frame for frame in turn_hyp["frames"]
      }

      # Calculate metrics for each frame in each user turn.
      for frame_ref in turn_ref["frames"]:
        service_name = frame_ref["service"]
        if service_name not in hyp_frames_by_service:
          raise ValueError(
              "Frame for service {} not found in dialogue with id {}".format(
                  service_name, dial_id))
        service = service_schemas[service_name]
        frame_hyp = hyp_frames_by_service[service_name]

        active_intent_acc = metrics.get_active_intent_accuracy(
            frame_ref, frame_hyp)
        slot_tagging_f1_scores = metrics.get_slot_tagging_f1(
            frame_ref, frame_hyp, turn_ref["utterance"], service)
        requested_slots_f1_scores = metrics.get_requested_slots_f1(
            frame_ref, frame_hyp)
        goal_accuracy_dict = metrics.get_average_and_joint_goal_accuracy(
            frame_ref, frame_hyp, service, True)
        #True: use_fuzzy_match
        frame_metric = {
            metrics.ACTIVE_INTENT_ACCURACY:
                active_intent_acc,
            metrics.REQUESTED_SLOTS_F1:
                requested_slots_f1_scores.f1,
            metrics.REQUESTED_SLOTS_PRECISION:
                requested_slots_f1_scores.precision,
            metrics.REQUESTED_SLOTS_RECALL:
                requested_slots_f1_scores.recall
        }
        if slot_tagging_f1_scores is not None:
          frame_metric[metrics.SLOT_TAGGING_F1] = slot_tagging_f1_scores.f1
          frame_metric[metrics.SLOT_TAGGING_PRECISION] = (
              slot_tagging_f1_scores.precision)
          frame_metric[
              metrics.SLOT_TAGGING_RECALL] = slot_tagging_f1_scores.recall
        frame_metric.update(goal_accuracy_dict)

        frame_id = "{:s}-{:03d}-{:s}".format(dial_id, turn_id,
                                             frame_hyp["service"])
        per_frame_metric[frame_id] = frame_metric
        # Add the frame-level metric result back to dialogues.
        frame_hyp["metrics"] = frame_metric

        # Get the domain name of the service.
        domain_name = frame_hyp["service"].split("_")[0]
        domain_keys = [ALL_SERVICES, frame_hyp["service"], domain_name]
        if frame_hyp["service"] in in_domain_services:
          domain_keys.append(SEEN_SERVICES)
        else:
          domain_keys.append(UNSEEN_SERVICES)
        for domain_key in domain_keys:
          for metric_key, metric_value in frame_metric.items():
            if metric_value != metrics.NAN_VAL:
                metric_collections[domain_key][metric_key].append(metric_value)
  all_metric_aggregate = {}
  for domain_key, domain_metric_vals in metric_collections.items():
    domain_metric_aggregate = {}
    for metric_key, value_list in domain_metric_vals.items():
      if value_list:
        # Metrics are macro-averaged across all frames.
        domain_metric_aggregate[metric_key] = float(np.mean(value_list))
      else:
        domain_metric_aggregate[metric_key] = metrics.NAN_VAL
    all_metric_aggregate[domain_key] = domain_metric_aggregate
  return all_metric_aggregate, per_frame_metric

def print_sgd_metrics(args, json_file):
    dataset_ref = {}
    all_data = []
    with open(os.path.join("data/sgd/test", "schema.json")) as f:
        eval_services = {}
        list_services = json.load(f)
        for service in list_services:
            eval_services[service["service_name"]] = service

    in_domain_services = get_in_domain_services(
        os.path.join("data/sgd/test", "schema.json"),
        os.path.join("data/sgd/train", "schema.json"))

    for filename in os.listdir("data/sgd/test"):
        if filename.startswith("dialogues_"):
            with open(os.path.join("data/sgd/test", filename)) as f:
                data = json.load(f)
                all_data += data
    for dial in all_data:
        dataset_ref[dial["dialogue_id"]] = dial.copy()
    dataset_hyp = load_json_file(json_file)
    all_metric_aggregate, _ = get_metrics(dataset_ref, dataset_hyp, eval_services,
                                          in_domain_services)
    args.logger.info("Dialog metrics: %s", str(all_metric_aggregate[args.except_domain]))

