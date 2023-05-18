"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import argparse
import os
import random
import torch
from sklearn.mixture import GaussianMixture
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import Birch, KMeans, AgglomerativeClustering, MiniBatchKMeans
from tqdm import tqdm
import pickle
from sklearn import metrics
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
from cluster_data_loader import prepare_cluster_data

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def batch_to_device(batch, device):
    batch_on_device = []
    for element in batch:
        if isinstance(element, dict):
            batch_on_device.append({k: v.to(device) for k, v in element.items()})
        else:
            batch_on_device.append(element.to(device))
    return tuple(batch_on_device)

def get_text_vecs(args, device, model, dataloader):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids, input_mask = batch["encoder_input"].to(device), batch["attention_mask"].to(device),
            segment_ids = batch["segment_id"].to(device)
            if 'bert-base' in args["pretrained_model"]:
                outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
                pooler_output = outputs.pooler_output  # [CLS] token
            elif "roberta" in args["pretrained_model"]:
                outputs = model(input_ids=input_ids, attention_mask=input_mask)
                pooler_output = outputs.pooler_output  # [CLS] token
                #pooler_output = torch.mean(outputs.last_hidden_state, dim=1)
            elif "t5" in args["pretrained_model"]:
                outputs = model.encoder(input_ids=input_ids, attention_mask=input_mask, return_dict=True)
                pooler_output = torch.mean(outputs.last_hidden_state, dim=1)  # [CLS] token
            elif "bart" in args["pretrained_model"]:
                outputs = model.encoder(input_ids=input_ids, attention_mask=input_mask, return_dict=True)
                pooler_output = torch.mean(outputs.last_hidden_state, dim=1)  # [CLS] token
            if step == 0:
                text_vecs = pooler_output
            else:
                text_vecs = torch.cat([text_vecs, pooler_output], dim=0)

            if args["do_test"]:
                if step > 2:
                    break

    text_vecs = text_vecs.cpu().detach().numpy()
    return text_vecs

def BERT_clustering(args, train_vecs):
    labels_list = []
    cent_list = []
    for k in range(args["min_k"], args["max_k"] + 1):
        if args["clustering"] == "birch":
            cluster_model = Birch(n_clusters=k)
        elif args["clustering"] == "kmeans":
            cluster_model = KMeans(n_clusters=k, random_state=args["random_seed"])
        elif args["clustering"] == "agg":
            cluster_model = AgglomerativeClustering(n_clusters=k)
        elif args["clustering"] == "mini_kmeans":
            cluster_model = MiniBatchKMeans(n_clusters=k,init='k-means++',batch_size=100,
                                           verbose=0,compute_labels=True,random_state=None,\
                                           tol=0.0,max_no_improvement=10,init_size=None)
        elif args["clustering"] == "gmm":
            cluster_model = GaussianMixture(n_components=k)
        else:
            logger.error("Clustering model not supported!")
        labels_pred = cluster_model.fit_predict(train_vecs)
        if "kmeans" in args["clustering"]:
            cent_pred = cluster_model.cluster_centers_
            cent_list.append(cent_pred)
        else:
            cent_list.append(None)
        labels_list.append(labels_pred)
    return labels_list, cent_list

def sample_node(text_vecs, ratio=0.2, labels_pred=None):
    vis_n = int(ratio * len(text_vecs))
    rand_idx = random.sample(range(0, len(text_vecs)), vis_n)
    new_text_vecs = text_vecs[rand_idx, :]
    print("sample number:", len(new_text_vecs))
    if labels_pred is not None:
        new_labels = labels_pred[rand_idx]
        return new_text_vecs, new_labels
    return new_text_vecs

def visual_cluster_result(cluster, model_name, save_path, labels_pred, cent_pred, text_vecs, eval_vecs, k):
    # sample part training nodes

    tsne2d = TSNE(n_components=2)
    #text_vecs, labels_pred = sample_node(text_vecs, 0.5, labels_pred)
    print("#total nodes:", text_vecs.shape[0])

    vecs2d = tsne2d.fit_transform(text_vecs)
    plt.figure(dpi=600)
    plt.title(f"K={k}, model:{model_name}", fontdict={"size":20})
    my_x_ticks = np.arange(-40, 60, 20)
    plt.xticks(my_x_ticks)
    plt.yticks(my_x_ticks)
    plt.tick_params(labelsize=20)
    plt.scatter(vecs2d[:, 0], vecs2d[:, 1], c=labels_pred, marker="o", s=np.ones(len(text_vecs)))
    #plt.scatter(eval_node[:, 0], eval_node[:, 1], c='r', marker="*", s=np.ones(len(eval_node)) * 2)
    #plt.scatter(cent_node[:, 0], cent_node[:, 1], c='k', marker='x')
    print(f"{model_name}_{k}_{cluster}.jpg saved!")
    plt.savefig(f"{save_path}/{model_name}_{k}_{cluster}.pdf")
    plt.savefig(f"{save_path}/{model_name}_{k}_{cluster}.jpg")
    plt.close()

    '''tsne3d = TSNE(n_components=3)
    text_vecs, labels_pred = sample_node(text_vecs, 0.1, labels_pred)
    eval_vecs = sample_node(eval_vecs, 0.5)
    nodes = np.concatenate((text_vecs, eval_vecs, cent_pred), axis=0)
    print("#total nodes:", nodes.shape[0])
    vecs3d = tsne3d.fit_transform(nodes)
    sp1, sp2 = len(text_vecs), len(text_vecs) + len(eval_vecs)
    text_node, eval_node, cent_node = vecs3d[:sp1, :], vecs3d[sp1:sp2, :], vecs3d[sp2:, :]
    # new_centre = tsne.fit_transform(centroids)
    # plt.figure(figsize=(12, 6))
    with open(f"text_node_{k}.pkl", 'wb') as f:
        pickle.dump(text_node, f)
    with open("eval_node.pkl", 'wb') as f:
        pickle.dump(eval_node, f)
    with open(f"cent_node_{k}.pkl", 'wb') as f:
        pickle.dump(cent_node, f)
    with open(f"labels_pred_{k}.pkl", 'wb') as f:
        pickle.dump(labels_pred, f)'''

    #draw_3d_fig(k, model_name)

def draw_3d_fig(k, model_name):
    text_node = pickle.load(open(f"text_node_{k}.pkl", "rb"))
    eval_node = pickle.load(open(f"eval_node.pkl", "rb"))
    cent_node = pickle.load(open(f"cent_node_{k}.pkl", "rb"))
    labels_pred = pickle.load(open(f"labels_pred_{k}.pkl", "rb"))

    plt.figure(dpi=600)
    ax = plt.subplot(projection='3d')
    ax.set_title(f"K={k}, model={model_name}")
    ax.scatter(text_node[:, 0], text_node[:, 1], text_node[:, 2], c=labels_pred, s=np.ones(len(text_node)))
    plt.scatter(eval_node[:, 0], eval_node[:, 1], c='r', marker="*", s=np.ones(len(eval_node)))
    ax.scatter(cent_node[:, 0], cent_node[:, 1], cent_node[:, 2], c='k', marker='x')
    plt.draw()
    plt.show()
    #plt.savefig(f"{save_path}/vis3d_c{k}.jpg")
    plt.close()


def draw_distortions(distortions):
    plt.plot(len(distortions), distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    '''if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)'''

def main(args):

    # TODO: multi GPU
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)

    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])

    # logger
    logger_file_name = args["save_dir"].split('/')[1]
    fileHandler = logging.FileHandler(os.path.join(args["save_dir"], "%s.txt" % (logger_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)

    # cuda setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("device: {}".format(device))

    # set random seed
    np.random.seed(args["random_seed"])
    random.seed(args["random_seed"])
    torch.manual_seed(args["random_seed"])
    if device == "cuda":
        torch.cuda.manual_seed(args["random_seed"])
        torch.cuda.manual_seed_all(args["random_seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # ******************************************************
    # load data
    # ******************************************************

    except_domain = args["except_domain"]
    save_dir = args["save_dir"]
    args["result_path"] = f"{save_dir}/{except_domain}"
    if not os.path.exists(args["result_path"]):
        os.makedirs(args["result_path"])

    for model_name in ["t5-base"]:
        args["pretrained_model"] = os.path.join("../../pretrained-model/", model_name)
        tokenizer = AutoTokenizer.from_pretrained(args["pretrained_model"], sep_token="[sep]")
        model = AutoModel.from_pretrained(args["pretrained_model"]).to(device)
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        train_loader, test_loader, train_data, test_data = prepare_cluster_data(args, tokenizer)
        train_vecs = get_text_vecs(args, device, model, train_loader)
        test_vecs = get_text_vecs(args, device, model, test_loader)

        save_path = os.path.join(args["result_path"], model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        test_vecs_file = os.path.join(save_path, 'test_vec.pkl')
        with open(test_vecs_file, 'wb') as f:
            pickle.dump(test_vecs, f)

        if args["do_random"]:
            for idx, k in enumerate(range(args["min_k"], args["max_k"] + 1)):
                random_dir = os.path.join(f"../RandomData/{except_domain}", model_name)
                if not os.path.exists(random_dir):
                    os.makedirs(random_dir)
                test_vecs_file = os.path.join(random_dir, 'test_vec.pkl')
                with open(test_vecs_file, 'wb') as f:
                    pickle.dump(test_vecs, f)
                save_random_result(random_dir, model_name, train_data, train_vecs, k)

        if args["do_cluster"]:
            for cluster in ["kmeans"]:
                args["clustering"] = cluster
                print("current setting:", model_name, args["clustering"])
                labels_list, cent_list = BERT_clustering(args, train_vecs)
                for idx, k in enumerate(range(args["min_k"], args["max_k"] + 1)):
                    print(f"Clustering {k}...")
                    compute_score(train_vecs, labels_list[idx])
                    if args["dataset"] == "mwz":
                        visual_cluster_result(cluster, model_name, save_path, labels_list[idx],
                                              cent_list[idx], train_vecs, test_vecs, k)
                    save_cluster_result(cluster, save_path, labels_list[idx], cent_list[idx], train_data, train_vecs, k)

    print(f"Clustering Finish!")

def compute_score(text_vecs, labels_pred):
    silhouette_score = metrics.silhouette_score(text_vecs,
                                                labels_pred.reshape(-1, 1),
                                                metric='euclidean')
    print(f"Silhouette Coefficient: {silhouette_score:.3f}")
    ch = metrics.calinski_harabasz_score(text_vecs, labels_pred.reshape(-1, 1))
    print(f"CH: {ch:.3f}")

def save_random_result(save_path, model_name, train_data, train_vecs, K):
    print(f"Divide randomly...")
    new_train_data = train_data.copy()
    random.shuffle(new_train_data)
    sub0 = int(len(new_train_data) / K) + 1
    label = 0
    dial_label = {}
    label_dial = {i: [] for i in range(K)}
    for idx, item in enumerate(new_train_data):
        line = item["dial_id"] + "-" + str(item["turn_id"])
        dial_label[line] = label
        label_dial[label].append(line)
        if len(dial_label) % sub0 == 0:
            label = label + 1

    dial2label = os.path.join(save_path, "dial2label_{}.pkl".format(K))
    with open(dial2label, 'wb') as f:
        pickle.dump(dial_label, f)

    dim = len(train_vecs[0, :])
    cent_pred = np.zeros([K, dim])
    for idx, (item, vec) in enumerate(zip(train_data, train_vecs)):
        line = item["dial_id"] + "-" + str(item["turn_id"])
        cent_pred[dial_label[line], :] = cent_pred[dial_label[line], :] + vec

    for i in range(K):
        cent_pred[i, :] = cent_pred[i, :] / len(label_dial[i])

    cent_file = os.path.join(save_path, 'cent_vec_{}.pkl'.format(K))
    with open(cent_file, 'wb') as f:
        pickle.dump(cent_pred, f)

def save_cluster_result(cluster, save_path, labels_pred, cent_pred, train_data, train_vecs, K):
    # Divide train and dev data for each class
    label_dial = {i:[] for i in range(K)}
    dial_label ={}
    for idx, (label, item) in enumerate(tqdm(zip(labels_pred, train_data))):
        line = item["dial_id"] + "-" + str(item["turn_id"])
        label_dial[label].append(line)
        dial_label[line] = label

    label2dial = os.path.join(save_path, "{}_label2dial_{}.pkl".format(cluster, K))
    with open(label2dial, 'wb') as f:
        pickle.dump(label_dial, f)

    dial2label = os.path.join(save_path, "{}_dial2label_{}.pkl".format(cluster, K))
    with open(dial2label, 'wb') as f:
        pickle.dump(dial_label, f)

    if 'kmeans' in cluster:
        cent_file = os.path.join(save_path, '{}_cent_vec_{}.pkl'.format(cluster, K))
        with open(cent_file, 'wb') as f:
            pickle.dump(cent_pred, f)
    else:
        dim = len(train_vecs[0, :])
        cent_pred = np.zeros([K, dim])
        for idx, (item, vec) in enumerate(zip(train_data, train_vecs)):
            line = item["dial_id"] + "-" + str(item["turn_id"])
            cent_pred[dial_label[line], :] = cent_pred[dial_label[line], :] + vec

        for i in range(K):
            cent_pred[i, :] = cent_pred[i, :] / len(label_dial[i])

        cent_file = os.path.join(save_path, '{}_cent_vec_{}.pkl'.format(cluster, K))
        with open(cent_file, 'wb') as f:
            pickle.dump(cent_pred, f)

def trainTestSplit(X, dev_ratio=0.2):
    X_num = len(X)
    train_idx = [i for i in range(X_num)]
    test_num = int(X_num * dev_ratio)
    train_data, test_data = [], []
    test_idx = random.sample(train_idx, test_num)
    for i in range(X_num):
        if i not in test_idx:
            train_data.append(X[i])
        else:
            test_data.append(X[i])
    return train_data, test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--pretrained_model", default='../../pretrained-model/t5-small', type=str)
    parser.add_argument("--clustering", default='mini_kmeans', type=str)
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--word_dropout", default=0, type=float)
    parser.add_argument("--except_domain", default="none", type=str)
    parser.add_argument("--only_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--model_name", type=str, default="t5", help="use t5 or bart?")
    parser.add_argument("--num_turn", default=1, type=int)
    parser.add_argument("--min_k", default=2, type=int)
    parser.add_argument("--max_k", default=4, type=int)
    parser.add_argument("--num_history", default=20, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size for test")
    parser.add_argument("--do_random", action="store_true", help="do random")
    parser.add_argument("--do_cluster", action="store_true", help="do cluster")
    parser.add_argument("--data_dir",type=str, default="../data/sgd", help="../data/mwz, ../data/aug_mwz or ../data/sgd ")
    parser.add_argument("--save_dir", type=str, default="../ClusterData", help="The location of saving cluster data")

    args = vars(parser.parse_args())
    main(args)

