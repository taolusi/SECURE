import argparse
import json
import logging
import os
import random
import sys
import traceback
import numpy as np
import _pickle as cPickle
from tqdm import tqdm, trange

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import RobertaTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from candidate_generator import (
    EncoderCosineRankerBase,
    EncoderCosineRankerSecure,
    tokenize_and_map,
    tokenize_and_map_concat,
)
from nearest_neighbor import nn_eval, get_summary

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))
# from classes import *
# from eval_utils import *
# from scorer import *

parser = argparse.ArgumentParser(description="Training a regressor")

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
    "--train_neighbor_size",
    type=int,
    default=15,
    help="Neighbor size for the training mentions.",
)
parser.add_argument(
    "--eval_neighbor_size",
    type=int,
    default=5,
    help="Neighbor size for the evaluation mentions.",
)
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=16,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=32,
    help="Batch size (per device) for the evaluation dataloader.",
)
parser.add_argument(
    "--accumulated_batch_size",
    type=int,
    default=16,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--avg_accumulated_loss",
    action="store_true",
    default=False,
    help="Whether to average the accumulated loss",
)
parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use.")
parser.add_argument(
    "--lr",
    type=float,
    default=1e-5,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--weight_decay", type=float, default=0.0, help="Weight decay to use."
)
parser.add_argument("--momentum", type=float, default=0.0, help="Momentum to use.")
parser.add_argument(
    "--epochs", type=int, default=50, help="Total number of training epochs to perform."
)
parser.add_argument(
    "--warmup_proportion",
    type=float,
    default=0.1,
    help="Proportion for the warmup in the lr scheduler.",
)
parser.add_argument(
    "--early_stop_patience", type=int, default=10, help="Patience for early stop."
)
parser.add_argument(
    "--max_grad_norm", type=float, default=1.0, help="Norm for gradient clipping."
)
parser.add_argument(
    "--seed", type=int, default=5, help="A seed for reproducible training."
)

args = parser.parse_args()

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

config_dict = vars(args)
if not args.eval:
    config_file_name = "candidate_generation_train_config.json"
else:
    config_file_name = "candidate_generation_eval_config.json"
with open(os.path.join(args.out_dir, config_file_name), "w") as js_file:
    json.dump(config_dict, js_file, indent=4)

if not config_dict["eval"]:
    log_file_name = "candidate_generation_train.log"
else:
    log_file_name = "candidate_generation_eval.log"
file_handler = logging.FileHandler(os.path.join(args.out_dir, log_file_name), mode="w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

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


def generate_records_for_sent(
    sentence_id, sentences, labels_triple, tokenizer, window=5
):
    labels_to_ids, label_sets, label_vocab_size = labels_triple
    sentence = sentences[sentence_id]
    sentence_mentions = sentence.gold_event_mentions
    if len(sentence_mentions) == 0:
        return ([], labels_triple)
    try:
        lookback = max(0, sentence_id - window)
        lookforward = min(sentence_id + window, max(sentences.keys())) + 1
        tokenization_input = (
            [sentences[_id] for _id in range(lookback, lookforward)],
            sentence_id - lookback,
        )
        tokenized_sentence, tokenization_mapping, sent_offset = tokenize_and_map(
            tokenization_input[0], tokenizer, tokenization_input[1]
        )
        sentence_records = []
        # logger.info(sentence_mentions[0].get_tokens())
        for mention in sentence_mentions:
            if mention.gold_tag not in labels_to_ids:
                labels_to_ids[mention.gold_tag] = label_vocab_size
                label_sets[label_vocab_size] = []
                label_vocab_size += 1
            label_id = labels_to_ids[mention.gold_tag]
            start_piece = tokenization_mapping[sent_offset + mention.start_offset][0]
            end_piece = tokenization_mapping[sent_offset + mention.end_offset][-1]
            record = {
                "sentence": tokenized_sentence,
                "label": [label_id],
                "start_piece": [start_piece],
                "end_piece": [end_piece],
            }
            sentence_records.append(record)
        return (sentence_records, (labels_to_ids, label_sets, label_vocab_size))
    except Exception:
        if window > 0:
            return generate_records_for_sent(
                sentence_id,
                sentences,
                labels_triple,
                window=window - 1,
            )
        else:
            traceback.print_exc()
            sys.exit()


def generate_records_for_sent_concat_sum(
    sentence_id,
    sentences,
    labels_triple,
    tokenizer,
    window=5,
):
    labels_to_ids, label_sets, label_vocab_size = labels_triple
    sentence = sentences[sentence_id]
    sentence_mentions = sentence.gold_event_mentions
    if len(sentence_mentions) == 0:
        return ([], labels_triple)
    records = []
    for mention in sentence_mentions:
        if mention.gold_tag not in labels_to_ids:
            labels_to_ids[mention.gold_tag] = label_vocab_size
            label_sets[label_vocab_size] = []
            label_vocab_size += 1
        label_id = labels_to_ids[mention.gold_tag]
        summary = get_summary(config_dict["summary_type"], mention)
        summary = summary.split()
        start_piece_sum = summary.index("#")
        summary.pop(start_piece_sum)
        end_piece_sum = summary.index("#") - 1
        summary.pop(end_piece_sum + 1)

        dynamic_window = window
        while dynamic_window > 0:
            lookback = max(0, sentence_id - dynamic_window)
            lookforward = min(sentence_id + dynamic_window, max(sentences.keys())) + 1
            tokenization_input = [summary]
            for _id in range(lookback, lookforward):
                tokenization_input.append(sentences[_id])
            (
                tokenized_sentence,
                tokenization_mapping,
                sent_offset,
            ) = tokenize_and_map_concat(
                tokenization_input, tokenizer, sentence_id - lookback + 1
            )
            start_piece_sum = tokenization_mapping[start_piece_sum][0]
            end_piece_sum = tokenization_mapping[end_piece_sum][-1]
            try:
                start_piece = tokenization_mapping[sent_offset + mention.start_offset][
                    0
                ]
                end_piece = tokenization_mapping[sent_offset + mention.end_offset][-1]
            except Exception:
                pass
            else:
                record = {
                    "sentence": tokenized_sentence,
                    "label": [label_id],
                    "start_piece": [start_piece],
                    "end_piece": [end_piece],
                    "start_piece_sum": [start_piece_sum],
                    "end_piece_sum": [end_piece_sum],
                }
                records.append(record)
                break
            dynamic_window -= 1
    return (records, (labels_to_ids, label_sets, label_vocab_size))


def structure_dataset(data_set, tokenizer):
    processed_dataset = []
    labels_to_ids = {}
    label_sets = {}
    label_vocab_size = 0
    labels_triple = (labels_to_ids, label_sets, label_vocab_size)
    docs = [
        document
        for topic in data_set.topics.values()
        for document in topic.docs.values()
    ]
    for doc in docs:
        sentences = doc.get_sentences()
        for sentence_id in sentences:
            if config_dict["model_type"] == "base":
                sentence_records, labels_triple = generate_records_for_sent(
                    sentence_id,
                    sentences,
                    labels_triple,
                    tokenizer,
                )
            elif config_dict["model_type"] == "secure":
                sentence_records, labels_triple = generate_records_for_sent_concat_sum(
                    sentence_id,
                    sentences,
                    labels_triple,
                    tokenizer,
                )

            for record in sentence_records:
                label_id = record["label"][0]
                processed_dataset.append(record)
                label_sets[label_id].append(record)
    labels_to_ids, label_sets, label_vocab_size = labels_triple
    sentences = torch.tensor([record["sentence"] for record in processed_dataset])
    labels = torch.tensor([record["label"] for record in processed_dataset])
    start_pieces = torch.tensor([record["start_piece"] for record in processed_dataset])
    end_pieces = torch.tensor([record["end_piece"] for record in processed_dataset])

    if config_dict["model_type"] == "secure":
        start_pieces_sum = torch.tensor(
            [record["start_piece_sum"] for record in processed_dataset]
        )
        end_pieces_sum = torch.tensor(
            [record["end_piece_sum"] for record in processed_dataset]
        )
        return (
            TensorDataset(
                sentences,
                start_pieces,
                end_pieces,
                labels,
                start_pieces_sum,
                end_pieces_sum,
            ),
            label_sets,
        )
    elif config_dict["model_type"] == "base":
        return (
            TensorDataset(
                sentences,
                start_pieces,
                end_pieces,
                labels,
            ),
            label_sets,
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


def evaluate(
    model,
    eval_raw,
    tokenizer,
    eval_set,
    epoch_num=0,
    eval_data=None,
    eval_event_gold=None,
    best_score=None,
    patience=0,
):
    model.eval()
    recall, mrr, maP, p_at_k = nn_eval(
        eval_raw,
        model,
        config_dict["eval_neighbor_size"],
        config_dict["model_type"],
        config_dict["summary_type"],
        tokenizer,
    )
    logger.info(eval_set)
    logger.info(
        "Epoch {} - Recall: {:.6f} - MRR: {:.6f} - MAP: {:.6f}".format(
            epoch_num, recall, mrr, maP
        )
    )
    if eval_set == "test":
        return
    loss_based = False
    if not loss_based:
        if best_score is None or recall > best_score:
            patience = 0
            logger.info("Model saved: Recall improved, saving the current best model.")
            best_score = recall
            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, "candidate_generator_best_model"),
            )
        else:
            patience += 1
            if patience >= config_dict["early_stop_patience"]:
                logger.info("Early Stopping")
                sys.exit()
    else:
        with torch.no_grad():
            model.update_cluster_lookup(eval_event_gold, dev=True)
            tr_loss = 0
            tr_accuracy = 0
            examples = 0
            for step, batch in enumerate(tqdm(eval_data, desc="Evaluation")):
                batch = tuple(t.to(model.device) for t in batch)
                sentences, start_pieces, end_pieces, labels = batch
                out_dict = model(sentences, start_pieces, end_pieces, labels, dev=True)
                loss = out_dict["loss"]
                accuracy = out_dict["accuracy"]
                examples += len(batch)
                tr_loss += loss.item()
                tr_accuracy += accuracy.item()
            if best_score is None or tr_loss < best_score:
                tqdm.write(
                    "Model saved: Loss decreased, saving the current best model."
                )
                best_score = tr_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(args.out_dir, "candidate_generator_best_model"),
                )
            logger.debug(
                "Epoch {} - Dev Loss: {:.6f} - Dev Precision: {:.6f}".format(
                    epoch_num, tr_loss / float(examples), tr_accuracy / len(eval_data)
                )
            )
            tqdm.write(
                "Epoch {} - Dev Loss: {:.6f} - Dev Precision: {:.6f}".format(
                    epoch_num, tr_loss / float(examples), tr_accuracy / len(eval_data)
                )
            )
    return best_score


def train_model(train_set, dev_set, test_set, tokenizer, model):

    train_event_mentions, train_event_gold = structure_dataset(
        train_set,
        tokenizer,
    )
    dev_event_mentions, dev_event_gold = structure_dataset(
        dev_set,
        tokenizer,
    )
    test_event_mentions, test_event_gold = structure_dataset(
        test_set,
        tokenizer,
    )
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, len(train_event_mentions))
    train_sampler = RandomSampler(train_event_mentions)
    train_dataloader = DataLoader(
        train_event_mentions,
        sampler=train_sampler,
        batch_size=config_dict["train_batch_size"],
    )
    dev_sampler = SequentialSampler(dev_event_mentions)
    dev_dataloader = DataLoader(
        dev_event_mentions,
        sampler=dev_sampler,
        batch_size=config_dict["eval_batch_size"],
    )
    test_sampler = SequentialSampler(test_event_mentions)
    test_dataloader = DataLoader(
        test_event_mentions,
        sampler=test_sampler,
        batch_size=config_dict["eval_batch_size"],
    )

    best_score = None
    patience = 0
    for epoch_idx in trange(int(config_dict["epochs"]), desc="Epoch"):
        model.train()
        tr_loss = 0.0
        model.update_cluster_lookup(train_event_gold)
        batcher = tqdm(train_dataloader, desc="Batch")
        for step, batch in enumerate(batcher):
            batch = tuple(t.to(model.device) for t in batch)
            if config_dict["model_type"] == "secure":
                (
                    sentences,
                    start_pieces,
                    end_pieces,
                    labels,
                    start_pieces_sum,
                    end_pieces_sum,
                ) = batch
                out_dict = model(
                    sentences,
                    start_pieces,
                    end_pieces,
                    labels,
                    start_pieces_sum,
                    end_pieces_sum,
                )
            elif config_dict["model_type"] == "base":
                sentences, start_pieces, end_pieces, labels = batch
                out_dict = model(sentences, start_pieces, end_pieces, labels)
            loss = out_dict["loss"]
            if config_dict["avg_accumulated_loss"]:
                loss = loss / (
                    config_dict["accumulated_batch_size"]
                    / config_dict["train_batch_size"]
                )
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) * config_dict["train_batch_size"] % config_dict[
                "accumulated_batch_size"
            ] == 0:
                batcher.set_description(
                    "Batch {} - epoch {} average loss: {:.6f}".format(
                        step,
                        epoch_idx,
                        tr_loss / float(config_dict["accumulated_batch_size"]),
                    )
                )
                tr_loss = 0
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config_dict["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        best_score = evaluate(
            model,
            dev_set,
            tokenizer,
            "dev",
            epoch_idx,
            dev_dataloader,
            dev_event_gold,
            best_score,
            patience,
        )
        evaluate(
            model,
            test_set,
            tokenizer,
            "test",
            epoch_idx,
        )


def evaluate_model(test_set, tokenizer, model):
    model_file = os.path.join(config_dict["out_dir"], "candidate_generator_best_model")
    with open(model_file, "rb") as f:
        params = torch.load(f)
        model.load_state_dict(params)

    evaluate(model, test_set, tokenizer, "test")


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

    logger.info(f"Loading model {config_dict['model_type']}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config_dict["model_type"] == "secure":
        model = EncoderCosineRankerSecure(config_dict["model_name_or_path"], device)
    elif config_dict["model_type"] == "base":
        model = EncoderCosineRankerBase(config_dict["model_name_or_path"], device)
    else:
        sys.exit()
    model.to(device)
    logger.info(f"Model {config_dict['model_type']} loaded.")

    tokenizer = RobertaTokenizer.from_pretrained(config_dict["model_name_or_path"])

    if not config_dict["eval"]:
        train_model(training_data, dev_data, test_data, tokenizer, model)
    else:
        evaluate_model(test_data, tokenizer, model)


if __name__ == "__main__":
    main()
