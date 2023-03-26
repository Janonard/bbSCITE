from math import inf, log
from pathlib import Path
import re
from statistics import mean
from typing import Any, List, Tuple, Optional, List, Union, Dict
import networkx as nx
import random
import numpy as np
import lark
import enum
from datetime import timedelta, datetime
import sys


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
    tree, _ = NewickTransformer().transform(grammar_tree)

    # Check if the nodes in the tree are consecutively labeled
    is_consecutive = True
    for i in range(len(tree.nodes)):
        if i not in tree.nodes:
            is_consecutive = False
            break

    # If this is not the case, fix it.
    if not is_consecutive:
        nodes = list(tree.nodes)
        nodes.sort()
        tree = nx.relabel.relabel_nodes(
            tree, {nodes[i]: i for i in range(len(nodes))})

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


move_generation_re = re.compile(r"Move generation makespan: ([0-9.]+) s")
board_power_re = re.compile(r"Total board power \(W\): ([0-9.]+)")
instant_re = re.compile(r"At instant (.+)")
smartvid_disabled_re = re.compile(
    r"MMD INFO : Disabling SmartVID \(fix\) polling")
smartvid_enabled_re = re.compile(
    r"MMD INFO : Enabling SmartVID \(fix\) polling")
kernel_finished_re = re.compile(r"Time elapsed: ([0-9.]+) ms")


class LineType(enum.IntEnum):
    MOVE_GENERATION = enum.auto()
    BOARD_POWER = enum.auto()
    INSTANT = enum.auto()
    SMARTVID_DISABLED = enum.auto()
    SMARTVID_ENABLED = enum.auto()
    KERNEL_FINISHED = enum.auto()

    @classmethod
    def parse_line(cls, line: str) -> Tuple["LineType", Union[timedelta, float, datetime, None]]:
        if (m := move_generation_re.match(line)) is not None:
            return cls.MOVE_GENERATION, timedelta(seconds=float(m[1]))
        elif (m := board_power_re.match(line)) is not None:
            return cls.BOARD_POWER, float(m[1])
        elif (m := instant_re.match(line)) is not None:
            return cls.INSTANT, datetime.fromisoformat(m[1])
        elif (m := smartvid_disabled_re.match(line)) is not None:
            return cls.SMARTVID_DISABLED, None
        elif (m := smartvid_enabled_re.match(line)) is not None:
            return cls.SMARTVID_ENABLED, None
        elif (m := kernel_finished_re.match(line)) is not None:
            return cls.KERNEL_FINISHED, timedelta(milliseconds=float(m[1]))
        else:
            return None


def analyze_log(log: List[Tuple[LineType, Any]]):
    makespans = []
    powers = []

    power_readings: List[Tuple[float, datetime]] = []
    for line_type, param in log:
        if line_type == LineType.BOARD_POWER:
            # Add a new element to the list of power readings
            power_readings += [(param, None)]

        elif line_type == LineType.INSTANT:
            # Set the instant of the last power reading
            assert len(
                power_readings) > 0, "Illegal log file: Instant with no power reading"
            power_readings[-1] = (power_readings[-1][0], param)

        elif line_type == LineType.SMARTVID_ENABLED:
            # Reset the power readings. The FPGA has just been programmed
            # and we have only measured the power consumption of the idle
            # bit stream.
            power_readings = []

        elif line_type == LineType.KERNEL_FINISHED:
            # Append the makespan of this kernel invocation
            makespan = param / timedelta(seconds=1)
            makespans += [makespan]

            # If the kernel finished message has occurred between a power reading and an instant,
            # the last power reading will contain no instant. In this case, move the power reading
            # to the next kernel event.
            next_power_readings = []
            if len(power_readings) > 0 and power_readings[-1][1] is None:
                next_power_readings = [power_readings[-1]]
                power_readings = power_readings[:-1]
            else:
                next_power_readings = []

            if len(power_readings) > 1:
                energy = 0.0
                for i in range(len(power_readings)-1):
                    this_power, this_instant = power_readings[i]
                    next_power, next_instant = power_readings[i+1]
                    base_power = min(this_power, next_power)
                    triangle_power = abs(this_power - next_power)/2.0
                    delta_time = (this_instant - next_instant).seconds
                    energy += (base_power + triangle_power) * delta_time
                delta_time = power_readings[-1][1] - power_readings[0][1]
                mean_power = energy / delta_time.seconds
            elif len(power_readings) == 1:
                mean_power = power_readings[0][0]
            else:
                mean_power = None

            if mean_power is not None:
                powers += [mean_power]

            power_readings = next_power_readings

    return mean(makespans), (mean(powers) if len(powers) > 0 else None)


def load_performance_data(base_dir: Path) -> Dict:
    data = dict()

    for cell_dir in base_dir.iterdir():
        n_cells = int(cell_dir.name)

        for program in "ffSCITE64", "ffSCITE96", "SCITE":
            program_dir = cell_dir / Path(program)
            if not program_dir.is_dir():
                continue

            if program not in data:
                data[program] = dict()

            for log_file in program_dir.glob("*.log"):
                n_chains, n_steps = map(
                    lambda x: int(x), log_file.stem.split("_"))
                lines = (LineType.parse_line(line) for line in open(
                    log_file, mode="r").readlines())
                lines = list(filter(lambda x: x is not None, lines))
                data[program][(n_cells, n_chains, n_steps)] = analyze_log(
                    lines)

    return data


def calc_expected_runtime(n_words: int, n_chains: int, n_steps: int, f: float = 252.5e6, occupancy: float = 1.0) -> float:
    return (n_words * n_chains * n_steps) / occupancy / f

def print_table(table: List[List[str]], style: str, out_path = None):
    if out_path is None:
        out_file = sys.stdout
    else:
        out_file = open(out_path, mode="w")

    if style == "markdown":
        lead = "| "
        sep = " | "
        end = " |"
    elif style == "latex":
        lead = ""
        sep = " & "
        end = " \\\\"
    elif style == "csv":
        lead = ""
        sep = ","
        end = ""

    if style == "latex":
        table[0] = ["\\textbf{" + text + "}" for text in table[0]]

    for i, line in enumerate(table):
        print(lead + sep.join(line) + end, file=out_file)
        if i == 0:
            if style == "markdown":
                print("|" + "|".join(["-"] * len(line)) + "|", file=out_file)
            elif style == "latex":
                print("\\hline", file=out_file)
