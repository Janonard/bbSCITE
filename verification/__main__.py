from verification.lib import *
from networkx.drawing.nx_pydot import write_dot
import random
import argparse
from pathlib import Path


def generate(args: argparse.Namespace):
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

    # Compute and output the likelihood of the true tree.
    true_tree_log_likelihood = tree.get_log_likelihood(
        noisy_mutation_matrix, args.alpha, args.beta)
    print(
        f"True mutation tree likelihood: {exp(true_tree_log_likelihood)} = exp({true_tree_log_likelihood})")


parser = argparse.ArgumentParser(
    description="Generate noisy inputs to test SCITE applications with")

parser.add_argument("-n", "--genes", required=True,
                    type=int, help="The number of genes.")
parser.add_argument("-m", "--cells", required=True,
                    type=int, help="The number of cells.")
parser.add_argument("-a", "--alpha", required=True, type=float,
                    help="The probability of false positives.")
parser.add_argument("-b", "--beta", required=True, type=float,
                    help="The probability of false negatives.")
parser.add_argument("-e", "--missing", required=True,
                    type=float, help="The probability of missing data.")
parser.add_argument("-s", "--seed", required=False,
                    type=int, help="The seed for the RNG.")

subparsers = parser.add_subparsers(dest="subcommand", required=True)

generate_parser = subparsers.add_parser(
    "generate", help="Generate input data for SCITE applications.")
generate_parser.add_argument("-o", "--out-dir", required=False, type=Path,
                    default=Path("."), help="The directory for the input data.")

args = parser.parse_args()

# Seeding the RNG
if args.seed is not None:
    random.seed(args.seed)
else:
    random.seed()

if args.subcommand == "generate":
    generate(args)
