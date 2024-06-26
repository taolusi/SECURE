import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def get_raw_strings(sentences, mention_sentence, mapping=None):
    raw_strings = []
    offset = 0
    if mapping:
        offset = max(mapping.keys()) + 1
    else:
        mapping = {}
    for i, sentence in enumerate(sentences):
        blacklist = []
        raw_strings.append(
            " ".join(
                [
                    (
                        tok.get_token().replace(" ", "")
                        if (int(tok.token_id) not in blacklist)
                        else "[MASK]"
                    )
                    for tok in sentence.get_tokens()
                ]
            )
        )
        if i == mention_sentence:
            mention_offset = offset
        for _ in sentence.get_tokens_strings():
            mapping[offset] = []
            offset += 1
    return raw_strings, mention_offset, mapping


def get_raw_concat_strings(sentences, mention_sentence, mapping=None):
    raw_strings = []
    offset = 0
    if mapping:
        offset = max(mapping.keys()) + 1
    else:
        mapping = {}
    for i, sentence in enumerate(sentences):
        blacklist = []
        if i == 0:
            raw_strings.append(" ".join(sentence))
        else:
            raw_strings.append(
                " ".join(
                    [
                        (
                            tok.get_token().replace(" ", "")
                            if (int(tok.token_id) not in blacklist)
                            else "[MASK]"
                        )
                        for tok in sentence.get_tokens()
                    ]
                )
            )
        if i == mention_sentence:
            mention_offset = offset
        if i == 0:
            mention_sum_offset = offset
        if i == 0:
            for _ in sentence:
                mapping[offset] = []
                offset += 1
        else:
            for _ in sentence.get_tokens_strings():
                mapping[offset] = []
                offset += 1
    return raw_strings, mention_offset, mapping, mention_sum_offset


def tokenize_and_map_pair(
    sentences_1, sentences_2, mention_sentence_1, mention_sentence_2, tokenizer
):
    max_seq_length = tokenizer.model_max_length
    raw_strings_1, mention_offset_1, mapping = get_raw_strings(
        sentences_1, mention_sentence_1
    )
    raw_strings_2, mention_offset_2, mapping = get_raw_strings(
        sentences_2, mention_sentence_2, mapping
    )
    embeddings = tokenizer(
        " ".join(raw_strings_1),
        " ".join(raw_strings_2),
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
    )["input_ids"]
    counter = 0
    new_tokens = tokenizer.convert_ids_to_tokens(embeddings)
    for i, token in enumerate(new_tokens):
        if (
            ((i + 1) < len(new_tokens) - 1)
            and (new_tokens[i] == "</s>")
            and (new_tokens[i + 1] == "</s>")
        ):  # 两个text分割的位置
            counter = len(" ".join(raw_strings_1).split(" ")) - 1
        else:
            pass
        if token == "<s>" or token == "</s>" or token == "<pad>":
            continue
        elif token[0] == "Ġ" or new_tokens[i - 1] == "</s>":
            counter += 1
            mapping[counter].append(i)
        else:
            mapping[counter].append(i)
            continue
    return embeddings, mapping, mention_offset_1, mention_offset_2


def tokenize_and_map_concat_pair(
    sentences_1, sentences_2, mention_sentence_1, mention_sentence_2, tokenizer
):
    max_seq_length = tokenizer.model_max_length
    raw_strings_1, mention_offset_1, mapping, mention_sum_offset_1 = (
        get_raw_concat_strings(sentences_1, mention_sentence_1)
    )
    raw_strings_2, mention_offset_2, mapping, mention_sum_offset_2 = (
        get_raw_concat_strings(sentences_2, mention_sentence_2, mapping)
    )
    embeddings = tokenizer(
        " ".join(raw_strings_1),
        " ".join(raw_strings_2),
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )["input_ids"]
    counter = 0
    new_tokens = tokenizer.convert_ids_to_tokens(embeddings)
    for i, token in enumerate(new_tokens):
        if (
            ((i + 1) < len(new_tokens) - 1)
            and (new_tokens[i] == "</s>")
            and (new_tokens[i + 1] == "</s>")
        ):  # 两个text分割的位置
            counter = len(" ".join(raw_strings_1).split(" ")) - 1
        else:
            pass
        if token == "<s>" or token == "</s>" or token == "<pad>":
            continue
        elif token[0] == "Ġ" or new_tokens[i - 1] == "</s>":
            counter += 1
            mapping[counter].append(i)
        else:
            mapping[counter].append(i)
            continue
    return (
        embeddings,
        mapping,
        mention_offset_1,
        mention_offset_2,
        mention_sum_offset_1,
        mention_sum_offset_2,
    )


class CoreferenceCrossEncoderBase(nn.Module):
    def __init__(self, pretrained_model, device):
        super(CoreferenceCrossEncoderBase, self).__init__()
        self.device = device
        self.pos_weight = torch.tensor([0.1]).to(device)
        self.model_type = "CoreferenceCrossEncoderBase"
        self.mention_model = AutoModel.from_pretrained(
            pretrained_model, return_dict=True
        )
        self.word_embedding_dim = (
            self.mention_model.embeddings.word_embeddings.embedding_dim
        )
        self.mention_dim = self.word_embedding_dim * 2
        self.input_dim = int(self.mention_dim * 3)
        self.out_dim = 1

        self.dropout = nn.Dropout(p=0.5)
        self.hidden_layer_1 = nn.Linear(self.input_dim, self.mention_dim)
        self.hidden_layer_2 = nn.Linear(self.mention_dim, self.mention_dim)
        self.out_layer = nn.Linear(self.mention_dim, self.out_dim)

        self.threshold = 0.5

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
        mention_rep = torch.cat([start_piece_vec, end_piece_vec], dim=2).squeeze(1)
        return mention_rep

    def to_transformer_input(self, sentence_tokens):
        segment_idx = sentence_tokens * 0
        mask = sentence_tokens != 1
        return {
            "input_ids": sentence_tokens,
            "token_type_ids": segment_idx,
            "attention_mask": mask,
        }

    def forward(
        self,
        sentences,
        start_pieces_1,
        end_pieces_1,
        start_pieces_2,
        end_pieces_2,
        labels=None,
    ):
        transformer_output = self.get_sentence_vecs(sentences)
        mention_reps_1 = self.get_mention_rep(
            transformer_output, start_pieces_1, end_pieces_1
        )
        mention_reps_2 = self.get_mention_rep(
            transformer_output, start_pieces_2, end_pieces_2
        )
        combined_rep = torch.cat(
            [mention_reps_1, mention_reps_2, mention_reps_1 * mention_reps_2], dim=1
        )
        combined_rep = combined_rep
        first_hidden = F.relu(self.hidden_layer_1(combined_rep))
        second_hidden = F.relu(self.hidden_layer_2(first_hidden))
        out = self.out_layer(second_hidden)
        probs = F.sigmoid(out)
        predictions = torch.where(probs > self.threshold, 1.0, 0.0)
        output_dict = {"probabilities": probs, "predictions": predictions}
        if labels is not None:
            correct = torch.sum(predictions == labels)
            total = float(predictions.shape[0])
            acc = correct / total
            if torch.sum(predictions).item() != 0:
                precision = torch.sum(
                    (predictions == labels).float() * predictions == 1
                ) / (torch.sum(predictions) + sys.float_info.epsilon)
            else:
                precision = torch.tensor(1.0).to(self.device)
            output_dict["accuracy"] = acc
            output_dict["precision"] = precision
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out, labels)
            """
            loss_fct = nn.BCELoss()
            probs_pos = probs * labels
            probs_pos_cut = torch.where(probs_pos > (self.threshold + self.margin), 1.0, probs_pos.double())
            probs_neg = probs * (1 - labels)
            probs_neg_cut = torch.where(probs_neg < self.threshold, 0.0, probs_neg.double())
            new_probs = probs_pos_cut + probs_neg_cut
            loss = loss_fct(new_probs.float(), labels)
            """
            output_dict["loss"] = loss
        return output_dict


class CoreferenceCrossEncoderSecure(nn.Module):
    def __init__(self, pretrained_model, device):
        super(CoreferenceCrossEncoderSecure, self).__init__()
        self.device = device
        self.pos_weight = torch.tensor([0.1]).to(device)
        self.model_type = "CoreferenceCrossEncoderSecure"
        self.mention_model = AutoModel.from_pretrained(
            pretrained_model, return_dict=True
        )
        self.word_embedding_dim = (
            self.mention_model.embeddings.word_embeddings.embedding_dim
        )
        self.mention_dim = self.word_embedding_dim * 4
        self.input_dim = int(self.mention_dim * 3)
        self.out_dim = 1

        self.dropout = nn.Dropout(p=0.5)
        self.hidden_layer_1 = nn.Linear(self.input_dim, self.mention_dim)
        self.hidden_layer_2 = nn.Linear(self.mention_dim, self.mention_dim)
        self.out_layer = nn.Linear(self.mention_dim, self.out_dim)

        self.threshold = 0.5

        self.input_dim1 = self.mention_dim * 2
        self.hidden_layer_3 = nn.Linear(self.input_dim1, self.mention_dim)
        self.hidden_layer_4 = nn.Linear(self.mention_dim, self.mention_dim)
        self.out_layer1 = nn.Linear(self.mention_dim, self.mention_dim)

    def get_sentence_vecs(self, sentences):
        expected_transformer_input = self.to_transformer_input(sentences)
        transformer_output = self.mention_model(
            **expected_transformer_input
        ).last_hidden_state
        return transformer_output

    def get_mention_rep1(self, transformer_output, start_pieces, end_pieces):
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
        mention_rep = torch.cat([start_piece_vec, end_piece_vec], dim=2).squeeze(1)
        return mention_rep

    def get_mention_rep(
        self,
        transformer_output,
        start_pieces,
        end_pieces,
        start_pieces_sum,
        end_pieces_sum,
    ):
        mention_reps = self.get_mention_rep1(
            transformer_output, start_pieces, end_pieces
        )
        mention_sum_reps = self.get_mention_rep1(
            transformer_output, start_pieces_sum, end_pieces_sum
        )
        combined_rep = torch.cat([mention_reps, mention_sum_reps], dim=1)
        return combined_rep
        first_hidden = F.relu(self.hidden_layer_3(combined_rep))
        # first_hidden = self.dropout(first_hidden)
        second_hidden = F.relu(self.hidden_layer_4(first_hidden))
        # second_hidden = self.dropout(second_hidden)
        out = self.out_layer1(second_hidden)
        return out

    def to_transformer_input(self, sentence_tokens):
        segment_idx = sentence_tokens * 0
        mask = sentence_tokens != 1
        return {
            "input_ids": sentence_tokens,
            "token_type_ids": segment_idx,
            "attention_mask": mask,
        }

    def forward(
        self,
        sentences,
        start_pieces_1,
        end_pieces_1,
        start_pieces_2,
        end_pieces_2,
        start_pieces_sum_1,
        end_pieces_sum_1,
        start_pieces_sum_2,
        end_pieces_sum_2,
        labels=None,
    ):
        transformer_output = self.get_sentence_vecs(sentences)
        mention_reps_1 = self.get_mention_rep(
            transformer_output,
            start_pieces_1,
            end_pieces_1,
            start_pieces_sum_1,
            end_pieces_sum_1,
        )
        mention_reps_2 = self.get_mention_rep(
            transformer_output,
            start_pieces_2,
            end_pieces_2,
            start_pieces_sum_2,
            end_pieces_sum_2,
        )
        combined_rep = torch.cat(
            [mention_reps_1, mention_reps_2, mention_reps_1 * mention_reps_2], dim=1
        )
        combined_rep = combined_rep
        first_hidden = F.relu(self.hidden_layer_1(combined_rep))
        second_hidden = F.relu(self.hidden_layer_2(first_hidden))
        out = self.out_layer(second_hidden)
        probs = F.sigmoid(out)
        predictions = torch.where(probs > self.threshold, 1.0, 0.0)
        output_dict = {"probabilities": probs, "predictions": predictions}
        if labels is not None:
            correct = torch.sum(predictions == labels)
            total = float(predictions.shape[0])
            acc = correct / total
            if torch.sum(predictions).item() != 0:
                precision = torch.sum(
                    (predictions == labels).float() * predictions == 1
                ) / (torch.sum(predictions) + sys.float_info.epsilon)
            else:
                precision = torch.tensor(1.0).to(self.device)
            output_dict["accuracy"] = acc
            output_dict["precision"] = precision
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out, labels)
            """
            loss_fct = nn.BCELoss()
            probs_pos = probs * labels
            probs_pos_cut = torch.where(probs_pos > (self.threshold + self.margin), 1.0, probs_pos.double())
            probs_neg = probs * (1 - labels)
            probs_neg_cut = torch.where(probs_neg < self.threshold, 0.0, probs_neg.double())
            new_probs = probs_pos_cut + probs_neg_cut
            loss = loss_fct(new_probs.float(), labels)
            """
            output_dict["loss"] = loss
        return output_dict
