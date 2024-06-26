import torch
import torch.nn as nn
import faiss
from transformers import AutoModel
from tqdm import tqdm


def tokenize_and_map(sentences, tokenizer, mention_sentence=0):
    max_seq_length = tokenizer.model_max_length
    mapping = {}
    raw_strings = []
    offset = 0
    for i, sentence in enumerate(sentences):
        raw_strings.append(
            " ".join([tok.replace(" ", "") for tok in sentence.get_tokens_strings()])
        )
        if i == mention_sentence:
            mention_offset = offset
        for _ in sentence.get_tokens_strings():
            mapping[offset] = []
            offset += 1
    embeddings = tokenizer(
        " ".join(raw_strings),
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
    )["input_ids"]
    counter = 0
    for i, token in enumerate(tokenizer.convert_ids_to_tokens(embeddings)):
        if token == "<s>" or token == "</s>" or token == "<pad>":
            continue
        elif token[0] == "Ġ":
            counter += 1
            mapping[counter].append(i)
        else:
            mapping[counter].append(i)
            continue
    return embeddings, mapping, mention_offset


def tokenize_and_map_concat(sentences, tokenizer, mention_sentence=1):
    max_seq_length = tokenizer.model_max_length
    raw_strings = []
    offset = 0
    mapping = {}
    for i, sentence in enumerate(sentences):
        if i == 0:
            raw_strings.append(" ".join(sentence))
        else:
            raw_strings.append(
                " ".join(
                    [tok.replace(" ", "") for tok in sentence.get_tokens_strings()]
                )
            )
        if i == mention_sentence:
            mention_offset = offset
        if i == 0:
            for _ in sentence:
                mapping[offset] = []
                offset += 1
        else:
            for _ in sentence.get_tokens_strings():
                mapping[offset] = []
                offset += 1
    embeddings = tokenizer(
        " ".join(raw_strings),
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
    )["input_ids"]
    counter = 0
    for i, token in enumerate(tokenizer.convert_ids_to_tokens(embeddings)):
        if token == "<s>" or token == "</s>" or token == "<pad>":
            continue
        elif token[0] == "Ġ":
            counter += 1
            mapping[counter].append(i)
        else:
            mapping[counter].append(i)
            continue
    return embeddings, mapping, mention_offset


class EncoderCosineRankerBase(nn.Module):
    def __init__(self, pretrained_model, device):
        super(EncoderCosineRankerBase, self).__init__()
        self.device = device
        self.model_type = "EncoderCosineRankerBase"
        self.mention_model = AutoModel.from_pretrained(
            pretrained_model, return_dict=True
        )
        self.word_embedding_dim = (
            self.mention_model.embeddings.word_embeddings.embedding_dim
        )
        self.cluster_lookup = {}
        self.faiss_index = {}

    def update_cluster_lookup(self, label_sets, dev=False):
        cluster_lookup = {}
        del self.cluster_lookup
        self.cluster_lookup = {}
        index = faiss.IndexFlatIP(self.word_embedding_dim * 2)
        with torch.no_grad():
            for label_id, label_records in tqdm(
                label_sets.items(), desc="Exemplar Reps"
            ):
                label_records = label_records[:50]
                assert label_records[0]["label"] == [label_id]
                sentences = torch.tensor(
                    [record["sentence"] for record in label_records]
                ).to(self.device)
                start = torch.tensor(
                    [record["start_piece"] for record in label_records]
                ).to(self.device)
                end = torch.tensor(
                    [record["end_piece"] for record in label_records]
                ).to(self.device)
                # sentences = torch.tensor(sentences).to(self.device)
                transformer_output = self.get_sentence_vecs(sentences)
                # start = torch.tensor(start_pieces).to(self.device)
                # end = torch.tensor(end_pieces).to(self.device)
                mention_rep = self.get_mention_rep(transformer_output, start, end)
                mention_rep = mention_rep.mean(dim=0)
                cluster_lookup[label_id] = mention_rep
                index.add(mention_rep.cpu().numpy())
        self.cluster_lookup = cluster_lookup
        self.faiss_index = index

    def get_sentence_vecs(self, sentences):
        expected_transformer_input = self.to_transformer_input(sentences)
        transformer_output = self.mention_model(
            **expected_transformer_input
        ).last_hidden_state
        return transformer_output

    def get_mention_rep(self, transformer_output, start_pieces, end_pieces):
        start_pieces = start_pieces.repeat(1, self.word_embedding_dim).view(
            -1, 1, self.word_embedding_dim
        )
        start_piece_vec = torch.gather(transformer_output, 1, start_pieces)
        end_piece_vec = torch.gather(
            transformer_output,
            1,
            end_pieces.repeat(1, self.word_embedding_dim).view(
                -1, 1, self.word_embedding_dim
            ),
        )

        mention_rep = torch.cat([start_piece_vec, end_piece_vec], dim=2)
        return mention_rep

    def to_transformer_input(self, sentence_tokens):
        segment_idx = sentence_tokens * 0
        mask = sentence_tokens != 1
        return {
            "input_ids": sentence_tokens,
            "token_type_ids": segment_idx,
            "attention_mask": mask,
        }

    def convert_labels_to_reps(self, label):
        return self.cluster_lookup[label]

    def get_hard_cases(self, mention_reps, labels):
        mention_reps = mention_reps.squeeze(1).detach().cpu().numpy()
        _, hard_case_lists = self.faiss_index.search(mention_reps, 10)
        hard_cases = []
        for i, h_list in enumerate(hard_case_lists):
            for j, hard_case in enumerate(h_list):
                # if labels[i] == hard_case:
                #    tqdm.write(str(j))
                #    break
                hard_cases.append(hard_case)
        return torch.tensor(hard_cases).to(self.device).unsqueeze(1)

    def forward(self, sentences, start_pieces, end_pieces, labels, dev=False):
        transformer_output = self.get_sentence_vecs(sentences)
        mention_reps = self.get_mention_rep(
            transformer_output, start_pieces, end_pieces
        )
        hard_cases = self.get_hard_cases(mention_reps, labels)
        labels_with_hard_neg = torch.cat([labels, hard_cases], dim=0)
        unique_clusters, local_labels = labels_with_hard_neg.unique(return_inverse=True)
        local_labels = local_labels[: len(labels)].squeeze(1)
        exemplars = list(
            map(self.convert_labels_to_reps, unique_clusters.cpu().tolist())
        )
        exemplars = torch.cat(exemplars, dim=0)
        exemplars = exemplars
        mention_reps = mention_reps.squeeze(1)
        scores = torch.mm(mention_reps, exemplars.t())
        if not self.training:
            return {"logits": scores}

        predictions = scores.argmax(dim=1)
        correct = torch.sum(predictions == local_labels)
        total = float(predictions.shape[0])
        acc = correct / total
        loss_fct = nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fct(scores, local_labels)
        return {"logits": scores, "loss": loss, "accuracy": acc}


class EncoderCosineRankerSecure(nn.Module):
    def __init__(self, pretrained_model, device):
        super(EncoderCosineRankerSecure, self).__init__()
        self.device = device
        self.model_type = "EncoderCosineRankerSecure"
        self.mention_model = AutoModel.from_pretrained(
            pretrained_model, return_dict=True
        )
        self.word_embedding_dim = (
            self.mention_model.embeddings.word_embeddings.embedding_dim
        )
        self.cluster_lookup = {}
        self.faiss_index = {}
        self.mention_dim = self.word_embedding_dim * 2
        self.input_dim = int(self.mention_dim * 2)

        self.dropout = nn.Dropout(p=0.5)
        self.hidden_layer_1 = nn.Linear(self.input_dim, self.mention_dim)
        self.hidden_layer_2 = nn.Linear(self.mention_dim, self.mention_dim)
        self.out_layer = nn.Linear(self.mention_dim, self.mention_dim)

    def update_cluster_lookup(self, label_sets, dev=False):
        cluster_lookup = {}
        del self.cluster_lookup
        self.cluster_lookup = {}
        index = faiss.IndexFlatIP(self.word_embedding_dim * 4)
        with torch.no_grad():
            for label_id, label_records in tqdm(
                label_sets.items(), desc="Exemplar Reps"
            ):
                label_records = label_records[:50]
                assert label_records[0]["label"] == [label_id]
                sentences = torch.tensor(
                    [record["sentence"] for record in label_records]
                ).to(self.device)
                start = torch.tensor(
                    [record["start_piece"] for record in label_records]
                ).to(self.device)
                end = torch.tensor(
                    [record["end_piece"] for record in label_records]
                ).to(self.device)
                start_sum = torch.tensor(
                    [record["start_piece_sum"] for record in label_records]
                ).to(self.device)
                end_sum = torch.tensor(
                    [record["end_piece_sum"] for record in label_records]
                ).to(self.device)
                # sentences = torch.tensor(sentences).to(self.device)
                transformer_output = self.get_sentence_vecs(sentences)
                # start = torch.tensor(start_pieces).to(self.device)
                # end = torch.tensor(end_pieces).to(self.device)
                mention_rep = self.get_mention_rep(
                    transformer_output, start, end, start_sum, end_sum
                )
                mention_rep = mention_rep.mean(dim=0)
                cluster_lookup[label_id] = mention_rep
                index.add(mention_rep.cpu().numpy())
        self.cluster_lookup = cluster_lookup
        self.faiss_index = index

    def get_sentence_vecs(self, sentences):
        expected_transformer_input = self.to_transformer_input(sentences)
        transformer_output = self.mention_model(
            **expected_transformer_input
        ).last_hidden_state
        return transformer_output

    def get_mention_sep_rep(self, transformer_output, start_pieces, end_pieces):
        start_pieces = start_pieces.repeat(1, self.word_embedding_dim).view(
            -1, 1, self.word_embedding_dim
        )
        start_piece_vec = torch.gather(transformer_output, 1, start_pieces)
        end_piece_vec = torch.gather(
            transformer_output,
            1,
            end_pieces.repeat(1, self.word_embedding_dim).view(
                -1, 1, self.word_embedding_dim
            ),
        )

        mention_rep = torch.cat([start_piece_vec, end_piece_vec], dim=2)
        return mention_rep

    def get_mention_rep(
        self,
        transformer_output,
        start_pieces,
        end_pieces,
        start_pieces_sum,
        end_pieces_sum,
    ):
        mention_reps = self.get_mention_sep_rep(
            transformer_output, start_pieces, end_pieces
        )
        mention_sum_reps = self.get_mention_sep_rep(
            transformer_output, start_pieces_sum, end_pieces_sum
        )
        combined_rep = torch.cat([mention_reps, mention_sum_reps], dim=2)
        return combined_rep

    def to_transformer_input(self, sentence_tokens):
        segment_idx = sentence_tokens * 0
        mask = sentence_tokens != 1
        return {
            "input_ids": sentence_tokens,
            "token_type_ids": segment_idx,
            "attention_mask": mask,
        }

    def convert_labels_to_reps(self, label):
        return self.cluster_lookup[label]

    def get_hard_cases(self, mention_reps, labels):
        mention_reps = mention_reps.squeeze(1).detach().cpu().numpy()
        _, hard_case_lists = self.faiss_index.search(mention_reps, 10)
        hard_cases = []
        for i, h_list in enumerate(hard_case_lists):
            for j, hard_case in enumerate(h_list):
                # if labels[i] == hard_case:
                #    tqdm.write(str(j))
                #    break
                hard_cases.append(hard_case)
        return torch.tensor(hard_cases).to(self.device).unsqueeze(1)

    def forward(
        self,
        sentences,
        start_pieces,
        end_pieces,
        labels,
        start_pieces_sum=None,
        end_pieces_sum=None,
        dev=False,
    ):
        transformer_output = self.get_sentence_vecs(sentences)
        mention_reps = self.get_mention_rep(
            transformer_output,
            start_pieces,
            end_pieces,
            start_pieces_sum,
            end_pieces_sum,
        )
        hard_cases = self.get_hard_cases(mention_reps, labels)
        labels_with_hard_neg = torch.cat([labels, hard_cases], dim=0)
        unique_clusters, local_labels = labels_with_hard_neg.unique(return_inverse=True)
        local_labels = local_labels[: len(labels)].squeeze(1)
        exemplars = list(
            map(self.convert_labels_to_reps, unique_clusters.cpu().tolist())
        )
        exemplars = torch.cat(exemplars, dim=0)
        exemplars = exemplars
        mention_reps = mention_reps.squeeze(1)
        scores = torch.mm(mention_reps, exemplars.t())
        if not self.training:
            return {"logits": scores}

        predictions = scores.argmax(dim=1)
        correct = torch.sum(predictions == local_labels)
        total = float(predictions.shape[0])
        acc = correct / total
        loss_fct = nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fct(scores, local_labels)
        return {"logits": scores, "loss": loss, "accuracy": acc}
