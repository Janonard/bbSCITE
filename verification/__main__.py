from verification.lib import *
from networkx.drawing.nx_pydot import write_dot
import random
import argparse
from pathlib import Path


def generate(args: argparse.Namespace):
    # Seeding the RNG
    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed()

    # Generate the true tree
    tree = random_mutation_tree(args.genes, args.cells)

    # Emit it for later inspection
    write_dot(tree, args.out_dir / Path("true_tree.gv"))
    with open(args.out_dir / Path("true_tree.newick"), mode="w") as out_file:
        print(tree.get_newick_code(), file=out_file)

    # Produce the true, unaltered mutation matrix
    true_mutation_matrix = tree.get_mutation_matrix()

    # Apply noise
    noisy_mutation_matrix = apply_sequencing_noise(
        true_mutation_matrix, args.alpha, args.beta, args.missing)

    # Emit the noisy mutation matrix
    write_mutation_matrix(noisy_mutation_matrix,
                          args.out_dir / Path("input.csv"))

    # Emit attachments
    with open(args.out_dir / Path("true_attachments.csv"), mode="w") as out_file:
        print(" ".join(str(tree.attachments[i])
              for i in range(args.cells)), file=out_file)

    # Compute and output the likelihood of the true tree.
    true_tree_log_likelihood = tree.get_log_likelihood(
        noisy_mutation_matrix, args.alpha, args.beta)
    if args.compact:
        print(true_tree_log_likelihood)
    else:
        print(
            f"True mutation tree likelihood: {exp(true_tree_log_likelihood)} = exp({true_tree_log_likelihood})")


def score_tree(args: argparse.Namespace):
    # Load the mutation matrix
    mutation_matrix = read_mutation_matrix(args.matrix)

    # Load the mutation tree
    with open(args.tree, mode="r") as tree_file:
        newick_code = tree_file.readline()
    mutation_tree, _ = parse_newick_code(newick_code)

    # Check if the nodes in the tree are consecutively labeled
    is_consecutive = True
    for i in range(len(mutation_tree.nodes)):
        if i not in mutation_tree.nodes:
            is_consecutive = False
            break

    # If this is not the case, fix it.
    if not is_consecutive:
        nodes = list(mutation_tree.nodes)
        nodes.sort()
        mutation_tree = nx.relabel.relabel_nodes(
            mutation_tree, {nodes[i]: i for i in range(len(nodes))})

    # Find the most-likely attachments for every cell.
    attachments = [
        get_most_likely_attachment(
            mutation_tree, mutation_matrix, cell_i, args.alpha, args.beta)
        for cell_i in range(mutation_matrix.shape[1])
    ]

    # Construct the mutation tree object.
    mutation_tree = MutationTree(
        attachments, incoming_graph_data=mutation_tree)

    # Compute the log-likelihood
    log_likelihood = mutation_tree.get_log_likelihood(
        mutation_matrix, args.alpha, args.beta)

    # Emit the likelihood
    if args.compact:
        print(log_likelihood)
    else:
        print(
            f"Mutation tree likelihood: {exp(log_likelihood)} = log({log_likelihood})")


parser = argparse.ArgumentParser(
    description="Tool for various tasks to test SCITE implementations.")

subparsers = parser.add_subparsers(dest="subcommand", required=True)

generate_parser = subparsers.add_parser(
    "generate", help="Generate input data for SCITE applications.")

generate_parser.add_argument("-o", "--out-dir", required=False, type=Path,
                             default=Path("."), help="The directory for the input data.")
generate_parser.add_argument("-n", "--genes", required=True,
                             type=int, help="The number of genes.")
generate_parser.add_argument("-m", "--cells", required=True,
                             type=int, help="The number of cells.")
generate_parser.add_argument("-a", "--alpha", required=True, type=float,
                             help="The probability of false positives.")
generate_parser.add_argument("-b", "--beta", required=True, type=float,
                             help="The probability of false negatives.")
generate_parser.add_argument("-e", "--missing", required=True,
                             type=float, help="The probability of missing data.")
generate_parser.add_argument("-s", "--seed", required=False,
                             type=int, help="The seed for the RNG.")
generate_parser.add_argument("-x", "--compact", action="store_true", default=False, required=False,
                             help="Only emit log-likelihood, no explanatory text.")

score_parser = subparsers.add_parser(
    "score", help="Calculate the likelihood of a tree.")

score_parser.add_argument("-a", "--alpha", required=True, type=float,
                          help="The probability of false positives.")
score_parser.add_argument("-b", "--beta", required=True, type=float,
                          help="The probability of false negatives.")
score_parser.add_argument("-t", "--tree", required=True,
                          type=Path, help="The path to the mutation tree.")
score_parser.add_argument("-m", "--matrix", required=True,
                          type=Path, help="The path to the mutation matrix.")
score_parser.add_argument("-x", "--compact", default=False, required=False,
                          action="store_true", help="Only emit log-likelihood, no explanatory text.")

args = parser.parse_args()

if args.subcommand == "generate":
    generate(args)
elif args.subcommand == "score":
    score_tree(args)
