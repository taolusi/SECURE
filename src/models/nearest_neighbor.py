import random
import re

import faiss
import numpy as np
import torch
from thefuzz import fuzz

# for pack in os.listdir("src"):
#     sys.path.append(os.path.join("src", pack))
# from classes import *
from candidate_generator import tokenize_and_map, tokenize_and_map_concat

summary_name_map = {
    "elaboration": "Elaboration",
    "elaboration-entityCoref_date": "Elaboration",
    "paraphrase_concat": "Paraphrase",
}


def dataset_to_docs(dataset):
    docs = [
        document
        for topic in dataset.topics.values()
        for document in topic.docs.values()
    ]
    return docs


def generate_singleton_set(docs):
    clusters = {}
    for doc in docs:
        sentences = doc.get_sentences()
        for sentence_id in sentences:
            for mention in sentences[sentence_id].gold_event_mentions:
                if mention.gold_tag not in clusters:
                    clusters[mention.gold_tag] = [mention.mention_id]
                else:
                    clusters[mention.gold_tag].append(mention.mention_id)
    singletons = [
        mention
        for cluster_id in clusters
        for mention in clusters[cluster_id]
        if len(clusters[cluster_id]) == 1
    ]
    return set(singletons)


def build_mention_reps(docs, model, tokenizer):
    processed_dataset = []
    labels = []
    mentions = []
    label_vocab_size = 0
    for doc in docs:
        sentences = doc.get_sentences()
        for sentence_id in sentences:
            sentence = sentences[sentence_id]
            sentence_mentions = sentence.gold_event_mentions
            if len(sentence_mentions) == 0:
                continue
            lookback = max(0, sentence_id - 5)
            lookforward = min(sentence_id + 5, max(sentences.keys())) + 1
            tokenization_input = (
                [sentences[_id] for _id in range(lookback, lookforward)],
                sentence_id - lookback,
            )
            tokenized_sentence, tokenization_mapping, sent_offset = tokenize_and_map(
                tokenization_input[0], tokenizer, tokenization_input[1]
            )
            sentence_vec = model.get_sentence_vecs(
                torch.tensor([tokenized_sentence]).to(model.device)
            )
            for mention in sentence_mentions:
                start_piece = torch.tensor(
                    [[tokenization_mapping[sent_offset + mention.start_offset][0]]]
                )
                end_piece = torch.tensor(
                    [[tokenization_mapping[sent_offset + mention.end_offset][-1]]]
                )
                mention_rep = model.get_mention_rep(
                    sentence_vec,
                    start_piece.to(model.device),
                    end_piece.to(model.device),
                )
                processed_dataset.append(mention_rep.detach().cpu().numpy()[0])
                labels.append((mention.mention_str, mention.gold_tag))
                mentions.append(mention)

    return np.concatenate(processed_dataset, axis=0), labels, mentions


def build_mention_concat_reps(
    docs,
    model,
    summary_type,
    tokenizer,
):
    processed_dataset = []
    labels = []
    mentions = []
    label_vocab_size = 0
    for doc in docs:
        sentences = doc.get_sentences()
        for sentence_id in sentences:
            sentence = sentences[sentence_id]
            sentence_mentions = sentence.gold_event_mentions
            if len(sentence_mentions) == 0:
                continue
            for mention in sentence_mentions:
                summary = get_summary(summary_type, mention)
                summary = summary.split()
                start_piece_sum = summary.index("#")
                summary.pop(start_piece_sum)
                end_piece_sum = summary.index("#") - 1
                summary.pop(end_piece_sum + 1)

                dynamic_window = 5
                while dynamic_window > 0:
                    lookback = max(0, sentence_id - dynamic_window)
                    lookforward = (
                        min(sentence_id + dynamic_window, max(sentences.keys())) + 1
                    )
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
                        start_piece = tokenization_mapping[
                            sent_offset + mention.start_offset
                        ][0]
                        end_piece = tokenization_mapping[
                            sent_offset + mention.end_offset
                        ][-1]
                    except Exception:
                        pass
                    else:
                        break
                    dynamic_window -= 1

                sentence_vec = model.get_sentence_vecs(
                    torch.tensor([tokenized_sentence]).to(model.device)
                )
                start_piece = torch.tensor([[start_piece]])
                end_piece = torch.tensor([[end_piece]])
                start_piece_sum = torch.tensor([[start_piece_sum]])
                end_piece_sum = torch.tensor([[end_piece_sum]])

                mention_rep = model.get_mention_rep(
                    sentence_vec,
                    start_piece.to(model.device),
                    end_piece.to(model.device),
                    start_piece_sum.to(model.device),
                    end_piece_sum.to(model.device),
                )

                processed_dataset.append(mention_rep.detach().cpu().numpy()[0])
                labels.append((mention.mention_str, mention.gold_tag))
                mentions.append(mention)

    return np.concatenate(processed_dataset, axis=0), labels, mentions


def get_original_context(context_type, sentence_id, sentences, window):
    if context_type.startswith("window"):
        lookback = max(0, sentence_id - window)
        lookforward = min(sentence_id + window, max(sentences.keys())) + 1
        raw_strings = []
        for i in range(lookback, lookforward):
            sentence = sentences[i]
            raw_strings.append(
                " ".join(
                    [tok.replace(" ", "") for tok in sentence.get_tokens_strings()]
                )
            )
        return " ".join(raw_strings)


def get_summary(summary_type, mention):
    summary = f"\n{summary_name_map[summary_type]}: {mention.span_rep[summary_type]}"
    # r'(?<=[^\s])#(?=[^\s])' 用于匹配一个 # 符号，但是要求它前后都不是空白字符（空格、制表符等）
    summary = re.sub(r"(?<=[^\s])#(?=[^\s])", " # ", summary)
    # r'(?<=\S)#(?=\s)' 用于匹配一个 # 符号，但要求它前面是非空白字符（\S 表示非空白字符），后面是空白字符（\s 表示空白字符）
    summary = re.sub(r"(?<=\S)#(?=\s)", " #", summary)
    # r'(?<=\S)#(?=$)' 用于匹配一个 # 符号，但要求它前面是非空白字符（\S 表示非空白字符），后面是字符串结尾
    summary = re.sub(r"(?<=\S)#(?=$)", " #", summary)
    # r'(?<=\s)#(?=\S)' 用于匹配一个 # 符号，但要求它前面是空白字符（\s 表示空白字符），后面是非空白字符（\S 表示非空白字符）
    summary = re.sub(r"(?<=\s)#(?=\S)", "# ", summary)

    return summary


def find_similar_word(input_string, given_word):
    matches = re.findall(r"#(.*?)#", input_string)
    most_similar = None
    highest_similarity = -1

    for match in matches:
        similarity = fuzz.ratio(match, given_word)
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar = match

    for match in matches:
        if most_similar != match:
            input_string = input_string.replace(f"#{match}#", match)

    return input_string


def reconstruct_mention_reps(docs, model):
    labels = []
    mentions = []
    singleton_set = set()
    if remove_singletons:
        singleton_set = generate_singleton_set(docs)
    for doc in docs:
        sentences = doc.get_sentences()
        for sentence_id in sentences:
            sentence = sentences[sentence_id]
            sentence_mentions = sentence.gold_event_mentions
            if len(sentence_mentions) == 0:
                continue
            for mention in sentence_mentions:
                labels.append((mention.mention_str, mention.gold_tag))
                mentions.append(mention)
    # assert len(model) == len(mentions)
    mention_vectors = {mention.mention_id: vector for mention, vector in model.items()}
    vectors = []
    for mention in mentions:
        vectors.append(mention_vectors[mention.mention_id])

    return np.array(vectors), labels, mentions


def build_cluster_rep(cluster, model, docs):
    cluster_rep = []
    for mention in cluster.mentions.values():
        sentence = docs[mention.doc_id].get_sentences()[mention.sent_id]
        tokenized_sentence, tokenization_mapping, sent_offset = tokenize_and_map(
            [sentence], tokenizer, 0
        )
        sent_rep = model.get_sentence_vecs(
            torch.tensor([tokenized_sentence]).to(model.device)
        )
        start_piece = torch.tensor([[tokenization_mapping[mention.start_offset][0]]])
        end_piece = torch.tensor([[tokenization_mapping[mention.end_offset][-1]]])
        mention_rep = model.get_mention_rep(
            sent_rep, start_piece.to(model.device), end_piece.to(model.device)
        )
        cluster_rep.append(mention_rep)
    return torch.cat(cluster_rep, dim=0).mean(dim=0).detach().cpu().numpy()


def build_cluster_reps(clusters, model, docs):
    cluster_reps = []
    sent_reps = {}
    for cluster in clusters:
        cluster_rep = build_cluster_rep(cluster, model, docs)
        cluster_reps.append(cluster_rep)
    return np.concatenate(cluster_reps, axis=0)


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("Relevance score length < k")
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.0
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])


def nn_cluster_pairs(clusters, model, docs, k=10):
    with torch.no_grad():
        vectors = build_cluster_reps(clusters, model, docs)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    D, I = index.search(vectors, k + 1)
    pairs = []
    for i, cluster in enumerate(clusters):
        nearest_neighbor_indexes = I[i][1:]
        nearest_neighbors = [(cluster, clusters[j]) for j in nearest_neighbor_indexes]
        pairs.extend(nearest_neighbors)
    return pairs


def create_cluster_index(clusters, model, docs):
    with torch.no_grad():
        vectors = build_cluster_reps(clusters, model, docs)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    return index


def create_mention_index(docs, model):
    with torch.no_grad():
        vectors = build_mention_reps(docs, model)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    return index


def nn_combine_pairs(
    data,
    model,
    k=10,
    remove_singletons=False,
    cross_validation=False,
    context_type="window",
    concat_sum=False,
    dual_encoder=False,
    is_train=True,
):
    labels = []
    mentions = []
    label_vocab_size = 0
    singleton_set = set()
    if remove_singletons:
        singleton_set = generate_singleton_set(data)
    # print(len(singleton_set))
    for doc in data:
        sentences = doc.get_sentences()
        for sentence_id in sentences:
            sentence = sentences[sentence_id]
            sentence_mentions = sentence.gold_event_mentions
            if len(sentence_mentions) == 0:
                continue
            for mention in sentence_mentions:
                if remove_singletons and mention.mention_id in singleton_set:
                    continue
                labels.append((mention.mention_str, mention.gold_tag))
                mentions.append(mention)

    pairs = list()
    positive_pairs, negative_pairs = list(), list()
    for i, mention_1 in enumerate(mentions):
        nearest_neighbors = list()
        if is_train:
            for j, mention_2 in enumerate(mentions):
                if i != j:
                    nearest_neighbors.append([mention_1, mention_2])
                    if mention_1.gold_tag == mention_2.gold_tag:
                        positive_pairs.append([mention_1, mention_2])
                    else:
                        negative_pairs.append([mention_1, mention_2])
        else:
            for j, mention_2 in enumerate(mentions[i:]):
                if i != j:
                    nearest_neighbors.append([mention_1, mention_2])
        pairs.extend(nearest_neighbors)
    if is_train:
        positive_ratio = len(positive_pairs) / len(pairs)
        selected_negative_pairs = [
            neg for neg in negative_pairs if random.random() < positive_ratio * 20
        ]
        final_pairs = positive_pairs + selected_negative_pairs
        random.shuffle(final_pairs)
        pairs = final_pairs

    return pairs, mentions


def nn_generate_mention_pairs(
    data,
    model,
    tokenizer,
    model_type,
    summary_type,
    k=10,
    is_train=False,
):
    if model_type == "secure":
        vectors, labels, mentions = build_mention_concat_reps(
            data, model, summary_type, tokenizer
        )
        index = faiss.IndexFlatIP(
            model.mention_model.embeddings.word_embeddings.embedding_dim * 4
        )
    elif model_type == "base":
        vectors, labels, mentions = build_mention_reps(
            data,
            model,
            tokenizer,
        )
        index = faiss.IndexFlatIP(
            model.mention_model.embeddings.word_embeddings.embedding_dim * 2
        )

    index.add(vectors)
    D, I = index.search(vectors, k + 1)
    if is_train:
        pairs = list()
        for i, mention in enumerate(mentions):
            nearest_neighbor_indexes = I[i]
            nearest_neighbors = list()
            for nn_index, j in enumerate(nearest_neighbor_indexes):
                if mention.mention_id != mentions[j].mention_id:
                    nearest_neighbors.append([mention, mentions[j]])
            pairs.extend(nearest_neighbors)
    else:
        pairs = set()
        for i, mention in enumerate(mentions):
            nearest_neighbor_indexes = I[i]
            nearest_neighbors = set()
            for nn_index, j in enumerate(nearest_neighbor_indexes):
                if mention.mention_id != mentions[j].mention_id:
                    nearest_neighbors.add(frozenset([mention, mentions[j]]))
            pairs = pairs | nearest_neighbors
    return pairs


def nn_eval(
    eval_data,
    model,
    k,
    model_type,
    summary_type,
    tokenizer,
):
    if model_type == "secure":
        vectors, labels, _ = build_mention_concat_reps(
            dataset_to_docs(eval_data),
            model,
            summary_type,
            tokenizer,
        )
        index = faiss.IndexFlatIP(
            model.mention_model.embeddings.word_embeddings.embedding_dim * 4
        )
    elif model_type == "base":
        vectors, labels, _ = build_mention_reps(
            dataset_to_docs(eval_data),
            model,
            tokenizer,
        )
        index = faiss.IndexFlatIP(
            model.mention_model.embeddings.word_embeddings.embedding_dim * 2
        )

    index.add(vectors)
    # Add 1 since the first will be identity
    D, I = index.search(vectors, k + 1)
    relevance_matrix = []
    tp = 0
    precision = 0
    singletons = 0
    for results in [[labels[i] for i in row] for row in I]:
        original_str, true_label = results[0]
        if "Singleton" in true_label:
            singletons += 1
            continue
        matches = results[1:]
        relevance = [label == true_label for _, label in matches]
        num_correct = np.sum(relevance)
        precision += num_correct / k
        if num_correct >= 1:
            tp += 1
        relevance_matrix.append(relevance)
    return (
        tp / float(len(I) - singletons),
        mean_reciprocal_rank(relevance_matrix),
        mean_average_precision(relevance_matrix),
        precision / float(len(I) - singletons),
    )
