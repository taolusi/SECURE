import numpy as np
from collections import Counter

# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment

"""
Mostly borrowed from <https://github.com/clarkkev/deep-coref/blob/master/evaluation.py>
"""


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class Evaluator:
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, document, is_event):
        pred_clusters = (
            document.predicted_event_clusters
            if is_event
            else document.predicted_entity_clusters
        )
        gold_clusters = (
            document.gold_event_clusters if is_event else document.gold_entity_clusters
        )
        mention_to_gold = (
            document.event_mention_to_gold_cluster
            if is_event
            else document.entity_mention_to_gold_cluster
        )
        mention_to_predicted = (
            document.event_mention_to_predicted_cluster
            if is_event
            else document.entity_mention_to_predicted_cluster
        )
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(pred_clusters, gold_clusters)
        else:
            pn, pd = self.metric(pred_clusters, mention_to_gold)
            rn, rd = self.metric(gold_clusters, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, is_event, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document, is_event)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        # if len(c) == 1:
        #     continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            # if len(c2) != 1:
            correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(tuple(mention_to_gold[m]))
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    # clusters = [c for c in clusters if len(c) != 1]
    clusters = [c for c in clusters]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    # matching = linear_assignment(-scores)
    # similarity = sum(scores[matching[:, 0], matching[:, 1]])
    row_ind, col_ind = linear_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()
    return similarity, len(clusters), similarity, len(gold_clusters)


def CEAFE(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    clusters = [c for c in clusters]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    # matching = linear_assignment(-scores)
    # similarity = sum(scores[matching[:, 0], matching[:, 1]])
    row_ind, col_ind = linear_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1 :]:
                    if (
                        m2 in mention_to_gold
                        and mention_to_gold[m] == mention_to_gold[m2]
                    ):
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem


def check_the_singleton(m_id, output_clusters):
    for cl in output_clusters:
        if m_id in cl:
            return cl
        else:
            pass
            # continue
    return None
    return []


def LEA(input_clusters, output_clusters, mention_to_gold):
    num, den = 0, 0

    for c in input_clusters:
        if len(c) == 1:
            all_links = 1
            # if c[0] in mention_to_gold and len(check_the_singleton(mention_to_gold[c[0]],output_clusters))==1:
            if (
                c[0] in mention_to_gold
                and len(check_the_singleton(c[0], output_clusters)) == 1
            ):
                common_links = 1
            else:
                common_links = 0
        else:
            common_links = 0
            all_links = len(c) * (len(c) - 1) / 2.0
            for i, m in enumerate(c):
                if m in mention_to_gold:
                    for m2 in c[i + 1 :]:
                        if (
                            m2 in mention_to_gold
                            and mention_to_gold[m] == mention_to_gold[m2]
                        ):
                            common_links += 1
                        # else:
                        #    print('!! ', m2, '--', m2.get_span(), ' ',
                        #           m2.min_spans, ' ', mention_to_gold[m], ' ',
                        #           mention_to_gold[m2], ' ' ,
                        #           [str(s) for s in output_clusters[
                        #               mention_to_gold[m]]], ' -- ',
                        #           [str(s) for s in output_clusters[
                        #               mention_to_gold[m2]]])

        num += len(c) * common_links / float(all_links)
        den += len(c)

    return num, den
