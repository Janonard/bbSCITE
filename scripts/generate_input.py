#!/usr/bin/env python3
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import random
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Generate noisy inputs to test SCITE applications with")
parser.add_argument("-n", "--genes", nargs=1, required=True, type=int)
parser.add_argument("-m", "--cells", nargs=1, required=True, type=int)
parser.add_argument("-a", "--alpha", nargs=1, required=True, type=float)
parser.add_argument("-b", "--beta", nargs=1, required=True, type=float)
parser.add_argument("-e", "--missing", nargs=1, required=True, type=float)
parser.add_argument("-s", "--seed", nargs=1, required=False, type=int)
parser.add_argument("-o", "--out-base", nargs=1, required=False, type=str)
args = parser.parse_args()

n_genes = args.genes[0]
n_cells = args.cells[0]
prob_false_positives = args.alpha[0]
prob_false_negatives = args.beta[0]
prob_missing = args.missing[0]

if args.out_base is not None:
    out_base = Path(args.out_base[0])
else:
    out_base = Path("./random")

# Seeding the RNG
if args.seed is not None:
    random.seed(args.seed[0])
else:
    random.seed()

# Generate a random tree and turn it into a directed tree using a BFS.
tree = nx.random_tree(n_genes+1)
tree = nx.bfs_tree(tree, n_genes)

# Emit the tree as a graphviz file.
write_dot(tree, out_base.with_name(out_base.name + "_true_tree.gv"))


def to_newick(parent=n_genes):
    if tree.out_degree(parent) > 0:
        return f"({','.join(to_newick(node) for node in tree.adj[parent])}){parent}"
    else:
        return f"{parent}"


# Emit the tree's newick code.
with open(out_base.with_name(out_base.name + "_true_tree.newick"), mode="w") as out_file:
    print(to_newick(), file=out_file)

# Randomly attach cells to nodes
attachments = [random.randrange(0, n_genes+1) for i in range(n_cells)]

# Evaluate which cells have which mutations
mutations = [
    # Get all nodes on the the path from the root to the attachment and collect them in a set.
    set(nx.shortest_path(tree, n_genes, attachments[cell_i]))
    for cell_i in range(n_cells)
]

# Produce the true, unaltered mutation matrix
true_mutation_matrix = [
    [1 if gene_i in mutations[cell_i] else 0 for cell_i in range(n_cells)]
    for gene_i in range(n_genes)
]


def false_state_filter(true_state):
    if true_state == 0:
        return 1 if random.random() <= prob_false_positives else 0
    else:
        return 0 if random.random() <= prob_false_negatives else 1


# Introduce false positives and false negatives to the mutation matrix
false_state_mutation_matrix = [
    [false_state_filter(true_mutation_matrix[gene_i][cell_i])
     for cell_i in range(n_cells)]
    for gene_i in range(n_genes)
]


def missing_data_filter(state):
    return 3 if random.random() <= prob_missing else state


# Introduce missing data to the mutation matrix
noisy_mutation_matrix = [
    [missing_data_filter(false_state_mutation_matrix[gene_i][cell_i])
     for cell_i in range(n_cells)]
    for gene_i in range(n_genes)
]

# Emit the noisy mutation matrix
with open(out_base.with_name(out_base.name + "_input.csv"), mode="w") as out_file:
    for gene_i in range(n_genes):
        print(" ".join(str(entry)
              for entry in noisy_mutation_matrix[gene_i]), file=out_file)
