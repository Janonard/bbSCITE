from lib2to3.pgen2 import token
from math import inf, log, exp
from pathlib import Path
from typing import List
import networkx as nx
import random
import numpy as np
from typing import List, Tuple, Optional
import lark


class MutationTree(nx.DiGraph):
    def __init__(self, attachments, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

        self.attachments = attachments

        # Assert that this is indeed a tree
        assert nx.is_tree(self)

        # Assert that every attachment point exists
        for attachment in self.attachments:
            assert attachment in self.nodes

        assert self.root in self.nodes

        # Assert that every node is reachable from the root (i.e. that the root is the root)
        for i in self.nodes:
            assert nx.has_path(self, self.root, i)

    @property
    def n_genes(self):
        return len(self.nodes) - 1

    @property
    def n_cells(self):
        return len(self.attachments)

    @property
    def root(self):
        return len(self.nodes) - 1

    def get_mutation_matrix(self) -> np.ndarray:
        # Evaluate which cells have which mutations
        mutations = [
            # Get all nodes on the the path from the root to the attachment and collect them in a set.
            set(nx.shortest_path(self, self.root, self.attachments[cell_i]))
            for cell_i in range(self.n_cells)
        ]

        # Produce the true, unaltered mutation matrix
        true_mutation_matrix = np.array([
            [1 if gene_i in mutations[cell_i]
                else 0 for cell_i in range(self.n_cells)]
            for gene_i in range(self.n_genes)
        ], np.int8)

        return true_mutation_matrix

    def get_newick_code(self, parent: Optional[int] = None):
        if parent is None:
            parent = self.root

        if self.out_degree(parent) > 0:
            return f"({','.join(self.get_newick_code(node) for node in self.adj[parent])}){parent}"
        else:
            return f"{parent}"

    def get_log_likelihood(self, mutation_matrix: np.array, prob_false_positives: float, prob_false_negatives: float):
        true_mutation_matrix = self.get_mutation_matrix()
        assert mutation_matrix.shape == (
            self.n_genes, self.n_cells)

        occurrences = [[0, 0], [0, 0]]
        for gene_i in range(self.n_genes):
            for cell_i in range(self.n_cells):
                observed_state = mutation_matrix[gene_i][cell_i]
                if observed_state < 2:
                    true_state = true_mutation_matrix[gene_i][cell_i]
                    occurrences[observed_state][true_state] += 1

        log_likelihood = log(1.0 - prob_false_positives)*occurrences[0][0]
        log_likelihood += log(prob_false_positives)*occurrences[0][1]
        log_likelihood += log(prob_false_negatives)*occurrences[1][0]
        log_likelihood += log(1.0 - prob_false_negatives)*occurrences[1][1]
        return log_likelihood


def random_mutation_tree(n_genes, n_cells):
    tree = nx.random_tree(n_genes+1)
    tree = nx.bfs_tree(tree, n_genes)

    attachments = [random.randrange(0, n_genes+1) for i in range(n_cells)]

    return MutationTree(attachments, incoming_graph_data=tree)


newick_grammar = """
    tree        : NUMBER                    -> leaf
                | "(" branches ")" NUMBER   -> internal
    branches    : branches "," tree         -> branch_list
                | tree                      -> first_branch

    NUMBER      : /[0-9]+/

    %import common.WS
    %ignore WS
"""
newick_parser = lark.Lark(newick_grammar, start="tree")


class NewickTransformer(lark.Transformer):

    def leaf(self, tree) -> Tuple[nx.DiGraph, int]:
        mutation_tree = nx.DiGraph()
        node = tree[0]
        mutation_tree.add_node(node)
        return (mutation_tree, node)

    def internal(self, tree) -> Tuple[nx.DiGraph, int]:
        mutation_tree = nx.DiGraph()
        children = []
        for (subtree, subtree_root) in tree[0]:
            mutation_tree.update(subtree)
            children.append(subtree_root)

        root = tree[1]
        mutation_tree.add_node(root)
        for child in children:
            mutation_tree.add_edge(root, child)

        return mutation_tree, root

    def branch_list(self, tree) -> List[Tuple[nx.DiGraph, int]]:
        subtrees = tree[0]
        subtrees.append(tree[1])
        return subtrees

    def first_branch(self, subtrees) -> List[Tuple[nx.DiGraph, int]]:
        return subtrees

    def NUMBER(self, tree) -> int:
        return int(tree.value)


def parse_newick_code(newick_code: str) -> Tuple[nx.DiGraph, int]:
    grammar_tree = newick_parser.parse(newick_code)
    tree = NewickTransformer().transform(grammar_tree)
    return tree


def get_most_likely_attachment(
        mutation_tree: nx.DiGraph,
        mutation_matrix: np.ndarray,
        cell_i: int,
        prob_false_positives: float,
        prob_false_negatives: float) -> int:
    max_likely_attachment_i = None
    max_log_likelihood = -inf
    root = len(mutation_tree.nodes) - 1

    for attachment_i in mutation_tree.nodes:
        occurrences = [[0, 0], [0, 0]]
        mutations = set(nx.shortest_path(mutation_tree, root, attachment_i))
        for gene_i in range(len(mutation_tree.nodes)-1):
            posterior = mutation_matrix[gene_i][cell_i]
            if posterior < 2:
                prior = 1 if gene_i in mutations else 0
                occurrences[posterior][prior] += 1
        log_likelihood = log(1.0 - prob_false_positives) * occurrences[0][0]
        log_likelihood += log(prob_false_positives) * occurrences[1][0]
        log_likelihood += log(prob_false_negatives) * occurrences[0][1]
        log_likelihood += log(1.0 - prob_false_negatives) * occurrences[1][1]
        if max_log_likelihood < log_likelihood:
            max_likely_attachment_i = attachment_i
            max_log_likelihood = log_likelihood

    return max_likely_attachment_i


def apply_sequencing_noise(mutation_matrix: np.ndarray, prob_false_positives: float, prob_false_negatives: float, prob_missing: float) -> np.ndarray:
    n_genes = mutation_matrix.shape[0]
    n_cells = mutation_matrix.shape[1]

    def state_filter(state):
        # Introduce false positives or false negatives
        if state == 0:
            state = 1 if random.random() <= prob_false_positives else 0
        else:
            state = 0 if random.random() <= prob_false_negatives else 1

        # Introduce missing data
        if random.random() <= prob_missing:
            state = 3

        return state

    # Introduce false positives and false negatives to the mutation matrix
    noisy_mutation_matrix = np.array([
        [state_filter(mutation_matrix[gene_i][cell_i])
         for cell_i in range(n_cells)]
        for gene_i in range(n_genes)
    ], np.int8)

    return noisy_mutation_matrix


def write_mutation_matrix(mutation_matrix: np.ndarray, path: Path):
    with open(path, mode="w") as out_file:
        for gene_i in range(mutation_matrix.shape[0]):
            print(" ".join(str(entry)
                  for entry in mutation_matrix[gene_i]), file=out_file)


def read_mutation_matrix(path: Path) -> np.ndarray:
    with open(path, mode="r") as in_file:
        matrix = [
            [int(entry) for entry in line.split(" ")]
            for line in in_file.readlines()
        ]
        return np.array(matrix, dtype=np.int8)
