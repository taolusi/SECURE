import argparse
import itertools
import json
import logging
import os
import random
import sys
import traceback
import _pickle as cPickle
from tqdm import tqdm, trange


import neptune
import numpy as np
import networkx as nx
import torch
import torch.optim as optim
from graphviz import Graph
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from coref_scorer import f1, muc, b_cubed, ceafe, LEA, lea
from coref_bcubed_scorer import bcubed
from nearest_neighbor import nn_generate_mention_pairs, dataset_to_docs, get_summary
from candidate_generator import EncoderCosineRankerBase, EncoderCosineRankerSecure
from pairwise_classifier import (
    CoreferenceCrossEncoderBase,
    CoreferenceCrossEncoderSecure,
    tokenize_and_map_pair,
    tokenize_and_map_concat_pair,
)

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

# from classes import *

parser = argparse.ArgumentParser(description="Training a classifier")


parser.add_argument(
    "--model_type",
    default="secure",
    choices=["secure", "base"],
    type=str,
    help=(
        "Which type of model to use: "
        "the secure model with LLM summary or the base model without it."
    ),
)
parser.add_argument(
    "--summary_type",
    type=str,
    choices=["elaboration", "elaboration-entityCoref_date", "paraphrase"],
    default="elaboration-entityCoref_date",
    help=(
        "When the model type is secure, specify the type of LLM summary to use: "
        "elaboration (step 1), elaboration-entityCoref_date (step 2) or paraphrase (alternative verification))."
    ),
)
parser.add_argument(
    "--dataset_type",
    default="ecb+",
    choices=["ecb+", "gvc", "fcc"],
    type=str,
    help="Which dataset to use.",
)
parser.add_argument(
    "--model_name_or_path",
    default="./ckpts/roberta-large",
    type=str,
    help="Backbone pre-trained language model",
)
parser.add_argument(
    "--eval", action="store_true", default=False, help="Whether to train or evaluate."
)
parser.add_argument("--out_dir", type=str, help=" The directory to the output folder.")
parser.add_argument(
    "--window_size",
    type=int,
    default=3,
    help="The number of sentences surrounding each mention.",
)
parser.add_argument(
    "--candidate_generator_dir",
    type=str,
    help="The directory to the output folder of the candidate generator.",
)
parser.add_argument(
    "--train_neighbor_size",
    type=int,
    default=15,
    help="Neighbor size for the training samples.",
)
parser.add_argument(
    "--eval_neighbor_size",
    type=int,
    default=5,
    help="Neighbor size for the evaluation samples.",
)
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=2,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=4,
    help="Batch size (per device) for the evaluation dataloader.",
)
parser.add_argument(
    "--accumulated_batch_size",
    type=int,
    default=8,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--avg_accumulated_loss",
    action="store_true",
    default=False,
    help="Whether to average the accumulated loss",
)
parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer used.")
parser.add_argument(
    "--lr",
    type=float,
    default=1e-5,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="Weight decay to use.",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.0,
    help="Momentum.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--warmup_proportion",
    type=float,
    default=0.1,
    help="Proportion for the warmup in the lr scheduler.",
)
parser.add_argument(
    "--early_stop_patience",
    type=int,
    default=5,
    help="Patience for early stop.",
)
parser.add_argument(
    "--max_grad_norm",
    type=float,
    default=1.0,
    help="Norm for gradient clipping.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=5,
    help="A seed for reproducible training.",
)
parser.add_argument(
    "--remove_singletons",
    action="store_true",
    default=False,
    help="Whether to remove singletons for additional evaluation setting.",
)

args = parser.parse_args()

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

config_dict = vars(args)
if not args.eval:
    config_file_name = "pairwise_classification_train_config.json"
else:
    config_file_name = "pairwise_classification_eval_config.json"
with open(os.path.join(args.out_dir, config_file_name), "w") as js_file:
    json.dump(config_dict, js_file, indent=4)

if not config_dict["eval"]:
    log_file_name = "pairwise_classification_train.log"
else:
    log_file_name = "pairwise_classification_eval.log"
file_handler = logging.FileHandler(os.path.join(args.out_dir, log_file_name), mode="w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

run = neptune.init_run(mode="offline")  # your credentials
run["config"] = config_dict

# Fix the random seeds
seed = config_dict["seed"]
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not config_dict["eval"]:
        logger.info("Training with CUDA")
    else:
        logger.info("Testing with CUDA")


def get_sents(sentences, sentence_id, window=config_dict["window_size"]):
    if window > 0:
        lookback = max(0, sentence_id - window)
        lookforward = min(sentence_id + window, max(sentences.keys())) + 1
        return (
            [sentences[_id] for _id in range(lookback, lookforward)],
            sentence_id - lookback,
        )
    elif window == 0:
        return (
            [sentences[sentence_id]],
            0,
        )
    else:
        print(f"Bad window size: {window}")


def structure_pair(mention_1, mention_2, doc_dict, tokenizer, window):
    try:
        sents_1, sent_id_1 = get_sents(
            doc_dict[mention_1.doc_id].sentences, mention_1.sent_id, window
        )
        sents_2, sent_id_2 = get_sents(
            doc_dict[mention_2.doc_id].sentences, mention_2.sent_id, window
        )
        tokens, token_map, offset_1, offset_2 = tokenize_and_map_pair(
            sents_1, sents_2, sent_id_1, sent_id_2, tokenizer
        )
        start_piece_1 = token_map[offset_1 + mention_1.start_offset][0]
        if offset_1 + mention_1.end_offset in token_map:
            end_piece_1 = token_map[offset_1 + mention_1.end_offset][-1]
        else:
            end_piece_1 = token_map[offset_1 + mention_1.start_offset][-1]
        start_piece_2 = token_map[offset_2 + mention_2.start_offset][0]
        if offset_2 + mention_2.end_offset in token_map:
            end_piece_2 = token_map[offset_2 + mention_2.end_offset][-1]
        else:
            end_piece_2 = token_map[offset_2 + mention_2.start_offset][-1]
        label = [1.0] if mention_1.gold_tag == mention_2.gold_tag else [0.0]
        record = {
            "sentence": tokens,
            "label": label,
            "start_piece_1": [start_piece_1],
            "end_piece_1": [end_piece_1],
            "start_piece_2": [start_piece_2],
            "end_piece_2": [end_piece_2],
        }
    except Exception:
        if window > 0:
            return structure_pair(mention_1, mention_2, doc_dict, window - 1)
        else:
            traceback.print_exc()
            sys.exit()
    return record


def structure_concat_pair(
    mention_1,
    mention_2,
    doc_dict,
    tokenizer,
    window,
):
    summary_type = config_dict["summary_type"]

    summary_1 = get_summary(summary_type, mention_1)
    summary_1 = summary_1.split()
    start_piece_sum_1 = summary_1.index("#")
    summary_1.pop(start_piece_sum_1)
    end_piece_sum_1 = summary_1.index("#") - 1
    summary_1.pop(end_piece_sum_1 + 1)

    summary_2 = get_summary(summary_type, mention_2)
    summary_2 = summary_2.split()
    start_piece_sum_2 = summary_2.index("#")
    summary_2.pop(start_piece_sum_2)
    end_piece_sum_2 = summary_2.index("#") - 1
    summary_2.pop(end_piece_sum_2 + 1)

    dynamic_window = window
    while dynamic_window >= 0:
        sents_1, sent_id_1 = get_sents(
            doc_dict[mention_1.doc_id].sentences, mention_1.sent_id, dynamic_window
        )
        sents_1 = [summary_1] + sents_1
        sent_id_1 += 1

        sents_2, sent_id_2 = get_sents(
            doc_dict[mention_2.doc_id].sentences, mention_2.sent_id, dynamic_window
        )
        sents_2 = [summary_2] + sents_2
        sent_id_2 += 1

        (
            tokens,
            token_map,
            offset_1,
            offset_2,
            offset_sum_1,
            offset_sum_2,
        ) = tokenize_and_map_concat_pair(
            sents_1, sents_2, sent_id_1, sent_id_2, tokenizer
        )
        try:
            start_piece_1 = token_map[offset_1 + mention_1.start_offset][0]
            if offset_1 + mention_1.end_offset in token_map:
                end_piece_1 = token_map[offset_1 + mention_1.end_offset][-1]
            else:
                end_piece_1 = token_map[offset_1 + mention_1.start_offset][-1]
            start_piece_2 = token_map[offset_2 + mention_2.start_offset][0]
            if offset_2 + mention_2.end_offset in token_map:
                end_piece_2 = token_map[offset_2 + mention_2.end_offset][-1]
            else:
                end_piece_2 = token_map[offset_2 + mention_2.start_offset][-1]
            start_piece_sum_1 = token_map[offset_sum_1 + start_piece_sum_1][0]
            if offset_sum_1 + end_piece_sum_1 in token_map:
                end_piece_sum_1 = token_map[offset_sum_1 + end_piece_sum_1][-1]
            else:
                end_piece_sum_1 = token_map[offset_sum_1 + start_piece_sum_1][-1]
            start_piece_sum_2 = token_map[offset_sum_2 + start_piece_sum_2][0]
            if offset_sum_2 + end_piece_sum_2 in token_map:
                end_piece_sum_2 = token_map[offset_sum_2 + end_piece_sum_2][-1]
            else:
                end_piece_sum_2 = token_map[offset_sum_2 + start_piece_sum_2][-1]
        except Exception:
            if dynamic_window == 0:
                if offset_1 + mention_1.start_offset not in token_map:
                    start_piece_1 = start_piece_sum_1
                    end_piece_1 = end_piece_sum_1
                    print(
                        "Forcefully cut off the window1, causing the original trigger1 to become the trigger1 in the summary"
                    )
                if offset_2 + mention_2.start_offset not in token_map:
                    start_piece_2 = start_piece_sum_2
                    end_piece_2 = end_piece_sum_2
                    print(
                        "Forcefully cut off the window2, causing the original trigger2 to become the trigger2 in the summary"
                    )
                label = [1.0] if mention_1.gold_tag == mention_2.gold_tag else [0.0]
                record = {
                    "sentence": tokens,
                    "label": label,
                    "start_piece_1": [start_piece_1],
                    "end_piece_1": [end_piece_1],
                    "start_piece_2": [start_piece_2],
                    "end_piece_2": [end_piece_2],
                    "start_piece_sum_1": [start_piece_sum_1],
                    "end_piece_sum_1": [end_piece_sum_1],
                    "start_piece_sum_2": [start_piece_sum_2],
                    "end_piece_sum_2": [end_piece_sum_2],
                }
            else:
                pass
        else:
            label = [1.0] if mention_1.gold_tag == mention_2.gold_tag else [0.0]
            record = {
                "sentence": tokens,
                "label": label,
                "start_piece_1": [start_piece_1],
                "end_piece_1": [end_piece_1],
                "start_piece_2": [start_piece_2],
                "end_piece_2": [end_piece_2],
                "start_piece_sum_1": [start_piece_sum_1],
                "end_piece_sum_1": [end_piece_sum_1],
                "start_piece_sum_2": [start_piece_sum_2],
                "end_piece_sum_2": [end_piece_sum_2],
            }
            break
        dynamic_window -= 1
    return record


def structure_dataset(data_set, tokenizer, candidate_generator, is_train=False):
    processed_dataset = []
    doc_dict = {
        key: document
        for topic in data_set.topics.values()
        for key, document in topic.docs.items()
    }
    docs = dataset_to_docs(data_set)
    pairs = nn_generate_mention_pairs(
        docs,
        candidate_generator,
        tokenizer,
        config_dict["model_type"],
        config_dict["summary_type"],
        config_dict["eval_neighbor_size"],
        is_train,
    )
    pairs = list(pairs)
    for mention_1, mention_2 in pairs:
        if config_dict["model_type"] == "secure":
            record = structure_concat_pair(
                mention_1, mention_2, doc_dict, tokenizer, config_dict["window_size"]
            )
        elif config_dict["model_type"] == "base":
            record = structure_pair(
                mention_1, mention_2, doc_dict, tokenizer, config_dict["window_size"]
            )
        processed_dataset.append(record)
    sentences = torch.tensor([record["sentence"] for record in processed_dataset])
    labels = torch.tensor([record["label"] for record in processed_dataset])
    start_pieces_1 = torch.tensor(
        [record["start_piece_1"] for record in processed_dataset]
    )
    end_pieces_1 = torch.tensor([record["end_piece_1"] for record in processed_dataset])
    start_pieces_2 = torch.tensor(
        [record["start_piece_2"] for record in processed_dataset]
    )
    end_pieces_2 = torch.tensor([record["end_piece_2"] for record in processed_dataset])
    logger.info(labels.sum() / float(labels.shape[0]))
    if config_dict["model_type"] == "secure":
        start_pieces_sum_1 = torch.tensor(
            [record["start_piece_sum_1"] for record in processed_dataset]
        )
        end_pieces_sum_1 = torch.tensor(
            [record["end_piece_sum_1"] for record in processed_dataset]
        )
        start_pieces_sum_2 = torch.tensor(
            [record["start_piece_sum_2"] for record in processed_dataset]
        )
        end_pieces_sum_2 = torch.tensor(
            [record["end_piece_sum_2"] for record in processed_dataset]
        )
        return (
            TensorDataset(
                sentences,
                start_pieces_1,
                end_pieces_1,
                start_pieces_2,
                end_pieces_2,
                start_pieces_sum_1,
                end_pieces_sum_1,
                start_pieces_sum_2,
                end_pieces_sum_2,
                labels,
            ),
            pairs,
            doc_dict,
        )
    elif config_dict["model_type"] == "base":
        return (
            TensorDataset(
                sentences,
                start_pieces_1,
                end_pieces_1,
                start_pieces_2,
                end_pieces_2,
                labels,
            ),
            pairs,
            doc_dict,
        )


def get_optimizer(model):
    lr = config_dict["lr"]
    optimizer = None
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if config_dict["optimizer"] == "adadelta":
        optimizer = optim.Adadelta(
            parameters, lr=lr, weight_decay=config_dict["weight_decay"]
        )
    elif config_dict["optimizer"] == "adam":
        optimizer = optim.Adam(
            parameters, lr=lr, weight_decay=config_dict["weight_decay"]
        )
    elif config_dict["optimizer"] == "sgd":
        optimizer = optim.SGD(
            parameters, lr=lr, momentum=config_dict["momentum"], nesterov=True
        )

    assert optimizer is not None, "Config error, check the optimizer field"

    return optimizer


def get_scheduler(optimizer, len_train_data):
    batch_size = config_dict["accumulated_batch_size"]
    epochs = config_dict["epochs"]

    num_train_steps = int(len_train_data / batch_size) * epochs
    num_warmup_steps = int(num_train_steps * config_dict["warmup_proportion"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_train_steps
    )
    return scheduler


def find_cluster_key(node, clusters):
    if node in clusters:
        return node
    for key, value in clusters.items():
        if node in value:
            return key
    return None


def is_cluster_merge(cluster_1, cluster_2, mentions, model, doc_dict, tokenizer):
    score = 0.0
    sample_size = 100
    if len(cluster_1) > sample_size:
        c_1 = random.sample(cluster_1, sample_size)
    else:
        c_1 = cluster_1
    if len(cluster_2) > sample_size:
        c_2 = random.sample(cluster_2, sample_size)
    else:
        c_2 = cluster_2
    for mention_id_1 in c_1:
        records = []
        mention_1 = mentions[mention_id_1]
        for mention_id_2 in c_2:
            # comparison_set = comparison_set | set(
            #     [frozenset([mention_id_1, mention_id_2])]
            # )
            mention_2 = mentions[mention_id_2]
            if config_dict["model_type"] == "secure":
                record = structure_concat_pair(
                    mention_1,
                    mention_2,
                    doc_dict,
                    tokenizer,
                    config_dict["window_size"],
                )
            elif config_dict["model_type"] == "base":
                record = structure_pair(
                    mention_1,
                    mention_2,
                    doc_dict,
                    tokenizer,
                    config_dict["window_size"],
                )

            records.append(record)
        sentences = torch.tensor([record["sentence"] for record in records]).to(
            model.device
        )
        labels = torch.tensor([record["label"] for record in records]).to(model.device)
        start_pieces_1 = torch.tensor(
            [record["start_piece_1"] for record in records]
        ).to(model.device)
        end_pieces_1 = torch.tensor([record["end_piece_1"] for record in records]).to(
            model.device
        )
        start_pieces_2 = torch.tensor(
            [record["start_piece_2"] for record in records]
        ).to(model.device)
        end_pieces_2 = torch.tensor([record["end_piece_2"] for record in records]).to(
            model.device
        )
        if config_dict["model_type"] == "secure":
            start_pieces_sum_1 = torch.tensor(
                [record["start_piece_sum_1"] for record in records]
            ).to(model.device)
            end_pieces_sum_1 = torch.tensor(
                [record["end_piece_sum_1"] for record in records]
            ).to(model.device)
            start_pieces_sum_2 = torch.tensor(
                [record["start_piece_sum_2"] for record in records]
            ).to(model.device)
            end_pieces_sum_2 = torch.tensor(
                [record["end_piece_sum_2"] for record in records]
            ).to(model.device)
        with torch.no_grad():
            if config_dict["model_type"] == "secure":
                out_dict = model(
                    sentences,
                    start_pieces_1,
                    end_pieces_1,
                    start_pieces_2,
                    end_pieces_2,
                    start_pieces_sum_1,
                    end_pieces_sum_1,
                    start_pieces_sum_2,
                    end_pieces_sum_2,
                    labels,
                )
            elif config_dict["model_type"] == "base":
                out_dict = model(
                    sentences,
                    start_pieces_1,
                    end_pieces_1,
                    start_pieces_2,
                    end_pieces_2,
                    labels,
                )
            mean_prob = torch.mean(out_dict["probabilities"]).item()
            score += mean_prob
    return (score / len(cluster_1)) >= 0.5


def transitive_closure_merge(
    edges, mentions, model, doc_dict, graph, graph_render, tokenizer
):
    clusters = {}
    inv_clusters = {}
    mentions = {mention.mention_id: mention for mention in mentions}
    for edge in tqdm(edges):
        cluster_key = find_cluster_key(edge[0], clusters)
        alt_key = find_cluster_key(edge[1], clusters)
        if cluster_key == None and alt_key == None:
            cluster_key = edge[0]
            clusters[cluster_key] = set()
        elif cluster_key == None and alt_key != None:
            cluster_key = alt_key
            alt_key = None
        elif cluster_key == alt_key:
            alt_key = None
        # If alt_key exists, merge clusters
        perform_merge = True
        if alt_key:
            perform_merge = is_cluster_merge(
                clusters[cluster_key],
                clusters[alt_key],
                mentions,
                model,
                doc_dict,
                tokenizer,
            )
        elif clusters[cluster_key] != set():
            new_elements = set([edge[0], edge[1]]) - clusters[cluster_key]
            if len(new_elements) > 0:
                perform_merge = is_cluster_merge(
                    clusters[cluster_key],
                    new_elements,
                    mentions,
                    model,
                    doc_dict,
                    tokenizer,
                )
        if alt_key and perform_merge:
            clusters[cluster_key] = clusters[cluster_key] | clusters[alt_key]
            for node in clusters[alt_key]:
                inv_clusters[node] = cluster_key
            del clusters[alt_key]
        if perform_merge:
            if not (
                graph.has_edge(edge[0], edge[1]) or graph.has_edge(edge[1], edge[0])
            ):
                graph.add_edge(edge[0], edge[1])
                color = "black"
                if edge[2] != 1.0:
                    color = "red"
                graph_render.edge(edge[0], edge[1], color=color, label=str(edge[3]))
            cluster = clusters[cluster_key]
            cluster.add(edge[0])
            cluster.add(edge[1])
            inv_clusters[edge[0]] = cluster_key
            inv_clusters[edge[1]] = cluster_key
    return clusters, inv_clusters


def evaluate(
    model,
    eval_dataloader,
    eval_pairs,
    doc_dict,
    epoch_num,
    eval_set,
    tokenizer,
    best_score=None,
    patience=None,
):
    model = model.eval()
    offset = 0
    edges = set()
    saved_edges = []
    best_edges = {}
    mentions = set()
    acc_sum = 0.0
    all_probs = []
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Eval Batch")):
        batch = tuple(t.to(model.device) for t in batch)
        if config_dict["model_type"] == "secure":
            (
                sentences,
                start_pieces_1,
                end_pieces_1,
                start_pieces_2,
                end_pieces_2,
                start_pieces_sum_1,
                end_pieces_sum_1,
                start_pieces_sum_2,
                end_pieces_sum_2,
                labels,
            ) = batch
            with torch.no_grad():
                out_dict = model(
                    sentences,
                    start_pieces_1,
                    end_pieces_1,
                    start_pieces_2,
                    end_pieces_2,
                    start_pieces_sum_1,
                    end_pieces_sum_1,
                    start_pieces_sum_2,
                    end_pieces_sum_2,
                    labels,
                )
        elif config_dict["model_type"] == "base":
            (
                sentences,
                start_pieces_1,
                end_pieces_1,
                start_pieces_2,
                end_pieces_2,
                labels,
            ) = batch
            with torch.no_grad():
                out_dict = model(
                    sentences,
                    start_pieces_1,
                    end_pieces_1,
                    start_pieces_2,
                    end_pieces_2,
                    labels,
                )
        acc_sum += out_dict["accuracy"]
        predictions = out_dict["predictions"].detach().cpu().tolist()
        probs = out_dict["probabilities"].detach().cpu().tolist()
        for p_index in range(len(predictions)):
            pair_0, pair_1 = eval_pairs[offset + p_index]
            prediction = predictions[p_index]
            mentions.add(pair_0)
            mentions.add(pair_1)
            # comparison_set = comparison_set | set(
            #     [frozenset([pair_0.mention_id, pair_1.mention_id])]
            # )
            if probs[p_index][0] > 0.5:
                if pair_0.mention_id not in best_edges or (
                    probs[p_index][0] > best_edges[pair_0.mention_id][3]
                ):
                    best_edges[pair_0.mention_id] = (
                        pair_0.mention_id,
                        pair_1.mention_id,
                        labels[p_index][0],
                        probs[p_index][0],
                    )
                edges.add(
                    (
                        pair_0.mention_id,
                        pair_1.mention_id,
                        labels[p_index][0],
                        probs[p_index][0],
                    )
                )
            saved_edges.append(
                (
                    pair_0,
                    pair_1,
                    labels[p_index][0].detach().cpu().tolist(),
                    probs[p_index][0],
                )
            )
        # for item in best_edges:
        #    edges.add(best_edges[item])

        offset += len(predictions)

    logger.info("Epoch {} {}:".format(epoch_num, eval_set))
    logger.info(
        "Pairwise Accuracy: {:.6f}".format(acc_sum / float(len(eval_dataloader)))
    )
    run[f"{eval_set}/epoch/accuracy"].log(
        acc_sum / float(len(eval_dataloader)), step=epoch_num
    )
    clusters = form_clusters(mentions, doc_dict, edges, model, tokenizer)
    score = eval_coref(clusters, epoch_num, eval_set)
    assert len(saved_edges) >= len(edges)
    if eval_set == "test":
        with open(os.path.join(args.out_dir, "test_edges"), "wb") as f:
            cPickle.dump(saved_edges, f)
        with open(os.path.join(args.out_dir, "test_clusters.json"), "w") as f:
            json.dump(clusters, f)
        return
    if best_score is None or score > best_score:
        best_score = score
        patience = 0
        logger.info("F1 Improved Saving Model")
        torch.save(
            model.state_dict(),
            os.path.join(args.out_dir, "pairwise_classifier_best_model"),
        )
        with open(os.path.join(args.out_dir, "dev_edges"), "wb") as f:
            cPickle.dump(saved_edges, f)
        with open(os.path.join(args.out_dir, "dev_clusters.json"), "w") as f:
            json.dump(clusters, f)
    else:
        patience += 1
        if patience > config_dict["early_stop_patience"]:
            logger.info("Early Stopped")
            sys.exit()
    return best_score, patience


def form_clusters(mentions, doc_dict, edges, model, tokenizer):
    logger.info(len(mentions))
    dot = Graph(comment="Cross Doc Co-ref")
    G = nx.Graph()
    edges = sorted(edges, key=lambda x: -1 * x[3])
    for mention in mentions:
        G.add_node(mention.mention_id)
        dot.node(
            mention.mention_id,
            label=str(
                (
                    str(mention),
                    doc_dict[mention.doc_id]
                    .sentences[mention.sent_id]
                    .get_raw_sentence(),
                )
            ),
        )
    bridges = list(nx.bridges(G))
    articulation_points = list(nx.articulation_points(G))
    # edges = [edge for edge in edges if edge not in bridges]
    clusters, inv_clusters = transitive_closure_merge(
        edges,
        mentions,
        model,
        doc_dict,
        G,
        dot,
        tokenizer,
    )

    # Find Transitive Closure Clusters
    gold_sets = []
    model_sets = []
    ids = []
    model_map = {}
    gold_map = {}
    for mention in mentions:
        ids.append(mention.mention_id)
        gold_sets.append(mention.gold_tag)
        gold_map[mention.mention_id] = mention.gold_tag
        if mention.mention_id in inv_clusters:
            model_map[mention.mention_id] = inv_clusters[mention.mention_id]
            model_sets.append(inv_clusters[mention.mention_id])
        else:
            model_map[mention.mention_id] = mention.mention_id
            model_sets.append(mention.mention_id)
    model_clusters = [
        [thing[0] for thing in group[1]]
        for group in itertools.groupby(
            sorted(zip(ids, model_sets), key=lambda x: x[1]), lambda x: x[1]
        )
    ]
    gold_clusters = [
        [thing[0] for thing in group[1]]
        for group in itertools.groupby(
            sorted(zip(ids, gold_sets), key=lambda x: x[1]), lambda x: x[1]
        )
    ]
    clusters = {
        "model_map": model_map,
        "model_clusters": model_clusters,
        "model_sets": model_sets,
        "gold_map": gold_map,
        "gold_clusters": gold_clusters,
        "gold_sets": gold_sets,
    }
    return clusters


def eval_coref(
    clusters,
    epoch_num,
    eval_set,
):
    model_clusters = clusters["model_clusters"]
    model_map = clusters["model_map"]
    model_sets = clusters["model_sets"]
    gold_clusters = clusters["gold_clusters"]
    gold_map = clusters["gold_map"]
    gold_sets = clusters["gold_sets"]
    logger.info("muc:")
    m_pn, m_pd = muc(model_clusters, gold_map)
    m_rn, m_rd = muc(gold_clusters, model_map)
    m_f1 = f1(m_pn, m_pd, m_rn, m_rd, beta=1)
    logger.info(
        "Recall: {:.6f} Precision: {:.6f} F1: {:.6f}".format(
            m_rn / m_rd, m_pn / m_pd, m_f1
        )
    )
    logger.info("b_cubed:")
    b_pn, b_pd = b_cubed(model_clusters, gold_map)
    b_rn, b_rd = b_cubed(gold_clusters, model_map)
    alt_b_f1 = f1(b_pn, b_pd, b_rn, b_rd, beta=1)
    logger.info(
        "Alternate = Recall: {:.6f} Precision: {:.6f}".format(
            b_rn / b_rd, b_pn / b_pd, alt_b_f1
        )
    )
    b_p, b_r, b_f1 = bcubed(gold_sets, model_sets)
    run[f"{eval_set}/epoch/precision"].log(b_p, step=epoch_num)
    run[f"{eval_set}/epoch/recall"].log(b_r, step=epoch_num)
    run[f"{eval_set}/epoch/f1"].log(b_f1, step=epoch_num)
    logger.info("Recall: {:.6f} Precision: {:.6f} F1: {:.6f}".format(b_r, b_p, b_f1))
    logger.info("ceafe:")
    # alt_c_pn, alt_c_pd, alt_c_rn, alt_c_rd = CEAFE(model_clusters, gold_clusters)
    # alt_c_f1 = f1(alt_c_pn, alt_c_pd, alt_c_rn, alt_c_rd, beta=1)
    # logger.info("Alternate = Recall: {:.6f} Precision: {:.6f} F1: {:.6f}".format(alt_c_rn / alt_c_rd, alt_c_pn / alt_c_pd, alt_c_f1))
    c_pn, c_pd, c_rn, c_rd = ceafe(model_clusters, gold_clusters)
    c_f1 = f1(c_pn, c_pd, c_rn, c_rd, beta=1)
    logger.info(
        "Recall: {:.6f} Precision: {:.6f} F1: {:.6f}".format(
            c_rn / c_rd, c_pn / c_pd, c_f1
        )
    )
    logger.info("conll:")
    # conll_f1 = (m_f1 + alt_b_f1 + c_f1) / float(3)
    conll_f1 = (m_f1 + b_f1 + c_f1) / float(3)
    logger.info("F1: {:.6f}".format(conll_f1))
    logger.info("lea:")
    L_pn, L_pd = LEA(model_clusters, gold_clusters, gold_map)
    L_rn, L_rd = LEA(gold_clusters, model_clusters, model_map)
    L_f1 = f1(L_pn, L_pd, L_rn, L_rd, beta=1)
    logger.info(
        "Alternate = Recall: {:.6f} Precision: {:.6f} F1: {:.6f}".format(
            L_rn / L_rd, L_pn / L_pd, L_f1
        )
    )
    l_pn, l_pd = lea(model_clusters, gold_map)
    l_rn, l_rd = lea(gold_clusters, model_map)
    l_f1 = f1(l_pn, l_pd, l_rn, l_rd, beta=1)
    logger.info(
        "Recall: {:.6f} Precision: {:.6f} F1: {:.6f}".format(
            l_rn / l_rd, l_pn / l_pd, l_f1
        )
    )
    return b_f1


def train_model(train_data, dev_data, test_data, tokenizer, candidate_generator, model):
    train_event_pairs, _, _ = structure_dataset(
        train_data,
        tokenizer,
        candidate_generator,
        is_train=True,
    )
    dev_event_pairs, dev_pairs, dev_docs = structure_dataset(
        dev_data,
        tokenizer,
        candidate_generator,
    )
    test_event_pairs, test_pairs, test_docs = structure_dataset(
        test_data,
        tokenizer,
        candidate_generator,
    )
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, len(train_event_pairs))
    run["config/optimizer"] = type(optimizer).__name__
    run["config/scheduler"] = type(scheduler).__name__

    train_sampler = RandomSampler(train_event_pairs)
    train_dataloader = DataLoader(
        train_event_pairs,
        sampler=train_sampler,
        batch_size=config_dict["train_batch_size"],
    )
    dev_sampler = SequentialSampler(dev_event_pairs)
    dev_dataloader = DataLoader(
        dev_event_pairs, sampler=dev_sampler, batch_size=config_dict["eval_batch_size"]
    )
    test_sampler = SequentialSampler(test_event_pairs)
    test_dataloader = DataLoader(
        test_event_pairs,
        sampler=test_sampler,
        batch_size=config_dict["eval_batch_size"],
    )

    best_score = None
    patience = 0

    for epoch_idx in trange(int(config_dict["epochs"]), desc="Epoch", leave=True):
        model = model.train()
        tr_loss = 0.0
        tr_p = 0.0
        tr_a = 0.0
        batcher = tqdm(train_dataloader, desc="Batch")
        for step, batch in enumerate(batcher):
            batch = tuple(t.to(model.device) for t in batch)
            if config_dict["model_type"] == "secure":
                (
                    sentences,
                    start_pieces_1,
                    end_pieces_1,
                    start_pieces_2,
                    end_pieces_2,
                    start_pieces_sum_1,
                    end_pieces_sum_1,
                    start_pieces_sum_2,
                    end_pieces_sum_2,
                    labels,
                ) = batch
                out_dict = model(
                    sentences,
                    start_pieces_1,
                    end_pieces_1,
                    start_pieces_2,
                    end_pieces_2,
                    start_pieces_sum_1,
                    end_pieces_sum_1,
                    start_pieces_sum_2,
                    end_pieces_sum_2,
                    labels,
                )
            elif config_dict["model_type"] == "base":
                (
                    sentences,
                    start_pieces_1,
                    end_pieces_1,
                    start_pieces_2,
                    end_pieces_2,
                    labels,
                ) = batch
                out_dict = model(
                    sentences,
                    start_pieces_1,
                    end_pieces_1,
                    start_pieces_2,
                    end_pieces_2,
                    labels,
                )
            loss = out_dict["loss"]
            precision = out_dict["precision"]
            accuracy = out_dict["accuracy"]
            if config_dict["avg_accumulated_loss"]:
                loss = loss / (
                    config_dict["accumulated_batch_size"]
                    / config_dict["train_batch_size"]
                )
            loss.backward()
            tr_loss += loss.item()
            tr_p += precision.item()
            tr_a += accuracy.item()

            if ((step + 1) * config_dict["train_batch_size"]) % config_dict[
                "accumulated_batch_size"
            ] == 0:
                batcher.set_description(
                    "Batch (average loss: {:.6f} precision: {:.6f} accuracy: {:.6f})".format(
                        tr_loss / float(step + 1),
                        tr_p / float(step + 1),
                        tr_a / float(step + 1),
                    )
                )
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config_dict["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                log_step = step + int(len(train_dataloader)) * epoch_idx
                run["training/batch/loss"].log(tr_loss / float(step + 1), step=log_step)
                run["training/batch/precision"].log(
                    tr_p / float(step + 1), step=log_step
                )
                run["training/batch/accuracy"].log(
                    tr_a / float(step + 1), step=log_step
                )

        if (tr_a / float(step + 1)) > 0.0:
            best_score, patience = evaluate(
                model,
                dev_dataloader,
                dev_pairs,
                dev_docs,
                epoch_idx,
                "dev",
                best_score,
                patience,
            )
            evaluate(
                model,
                test_dataloader,
                test_pairs,
                test_docs,
                epoch_idx,
                "test",
            )
        else:
            logger.info(
                f"Accuracy {tr_a / float(step + 1)} < 0.0, so no evaluation at this epoch."
            )


def evaluate_model(eval_data, eval_set, tokenizer, candidate_generator, model):
    logger.info(f"Loading pairwise classifier model {config_dict['model_type']}...")
    with open(os.path.join(args.out_dir, "pairwise_classifier_best_model"), "rb") as f:
        params = torch.load(f)
        model.load_state_dict(params)
        model = model.eval()
    logger.info(f"Pairwise classifier model {config_dict['model_type']} loaded.")

    saved_edges_path = os.path.join(args.out_dir, f"{eval_set}_edges")
    saved_clusters_path = os.path.join(args.out_dir, f"{eval_set}_clusters.json")
    if not os.path.exists(saved_clusters_path):
        if not os.path.exists(saved_edges_path):
            eval_event_pairs, eval_pairs, eval_docs = structure_dataset(
                eval_data,
                tokenizer,
                candidate_generator,
            )
            eval_sampler = SequentialSampler(eval_event_pairs)
            eval_dataloader = DataLoader(
                eval_event_pairs,
                sampler=eval_sampler,
                batch_size=config_dict["eval_batch_size"],
            )
            evaluate(
                model, eval_dataloader, eval_pairs, eval_docs, 0, eval_set, tokenizer
            )
        else:
            doc_dict = {
                key: document
                for topic in eval_data.topics.values()
                for key, document in topic.docs.items()
            }
            with open(saved_edges_path, "rb") as js_file:
                saved_edges = cPickle.load(js_file)
            edges = set()
            mentions = set()
            for edge in saved_edges:
                mentions.add(edge[0])
                mentions.add(edge[1])
                if edge[3] > 0.5:
                    new_edge = (
                        edge[0].mention_id,
                        edge[1].mention_id,
                        torch.tensor(edge[2], device=model.device),
                        edge[3],
                    )
                    edges.add(new_edge)
            clusters = form_clusters(mentions, doc_dict, edges, model, tokenizer)
            score = eval_coref(clusters, 0, eval_set)
    else:
        with open(saved_clusters_path, "r") as js_file:
            clusters = json.load(js_file)

        if config_dict["remove_singletons"]:
            model_clusters = [
                mentions for mentions in clusters["model_clusters"] if len(mentions) > 1
            ]
            gold_clusters = [
                mentions for mentions in clusters["gold_clusters"] if len(mentions) > 1
            ]

        score = eval_coref(clusters, 0, eval_set)


def integrate_llm_summary(original_data, original_file, summary_type, llm="gpt4"):
    """
    Integrates the specified type of summary into the original feature set of the data.
    """
    original_dir, original_name = os.path.split(original_file)
    parent_dir = os.path.dirname(original_dir)
    num_missing_mentions = 0
    with open(f"{parent_dir}/{llm}/{summary_type}_{original_name}.json", "r") as f:
        summary_data = json.load(f)
        docs = [
            document
            for topic in original_data.topics.values()
            for document in topic.docs.values()
        ]
        for doc in docs:
            sentences = doc.get_sentences()
            for sentence_id in sentences:
                sentence = sentences[sentence_id]
                sentence_mentions = sentence.gold_event_mentions
                if len(sentence_mentions) == 0:
                    continue
                for mention in sentence_mentions:
                    # Store the LLM generated summary in the mention attribute of span_rep
                    if mention.span_rep is None:
                        mention.span_rep = {}
                    if mention.mention_id in summary_data:
                        mention.span_rep[summary_type] = summary_data[
                            mention.mention_id
                        ][summary_type]["content"]
                    else:
                        num_missing_mentions += 1
                        logger.info(
                            f"Missing mention {mention.mention_str} ({mention.mention_id}) in {original_name}."
                        )
    logger.info(
        f"A total of {num_missing_mentions} missing mentions in {original_name}."
    )
    return original_data


def main():
    logger.info("Loading original data...")
    train_file = f"data/{config_dict['dataset_type']}/processed/training_data"
    with open(train_file, "rb") as f:
        training_data = cPickle.load(f)
    dev_file = f"data/{config_dict['dataset_type']}/processed/dev_data"
    with open(dev_file, "rb") as f:
        dev_data = cPickle.load(f)
    test_file = f"data/{config_dict['dataset_type']}/processed/test_data"
    with open(test_file, "rb") as f:
        test_data = cPickle.load(f)
    logger.info("Original data loaded.")

    if config_dict["model_type"] == "secure":
        logger.info("Loading llm generated summaries...")
        training_data = integrate_llm_summary(
            training_data, train_file, config_dict["summary_type"], llm="gpt4"
        )
        dev_data = integrate_llm_summary(
            dev_data, dev_file, config_dict["summary_type"], llm="gpt4"
        )
        test_data = integrate_llm_summary(
            test_data, test_file, config_dict["summary_type"], llm="gpt4"
        )
        logger.info("LLM generated summaries loaded.")

    # # Use a toy dataset to debug.
    # training_data.topics = {"3_ecb": training_data.topics["3_ecb"]}
    # dev_data.topics = {"23_ecb": dev_data.topics["23_ecb"]}
    # test_data.topics = {"37_ecb": test_data.topics["37_ecb"]}

    logger.info(f"Loading candidate generator model {config_dict['model_type']}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config_dict["model_type"] == "secure":
        candidate_generator = EncoderCosineRankerSecure(
            config_dict["model_name_or_path"], device
        )
    elif config_dict["model_type"] == "base":
        candidate_generator = EncoderCosineRankerBase(
            config_dict["model_name_or_path"], device
        )
    else:
        sys.exit()
    candidate_generator.to(device)
    candidate_generator_model_file = os.path.join(
        config_dict["candidate_generator_dir"], "candidate_generator_best_model"
    )
    with open(candidate_generator_model_file, "rb") as f:
        params = torch.load(f)
        candidate_generator.load_state_dict(params)
        candidate_generator.eval()
    logger.info(f"Candidate generator model {config_dict['model_type']} loaded.")

    if config_dict["model_type"] == "secure":
        model = CoreferenceCrossEncoderSecure(config_dict["model_name_or_path"], device)
    elif config_dict["model_type"] == "base":
        model = CoreferenceCrossEncoderBase(config_dict["model_name_or_path"], device)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config_dict["model_name_or_path"])

    if not config_dict["eval"]:
        train_model(
            training_data, dev_data, test_data, tokenizer, candidate_generator, model
        )
    else:
        evaluate_model(test_data, "test", tokenizer, candidate_generator, model)


if __name__ == "__main__":
    main()
