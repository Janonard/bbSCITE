from email.quoprimime import unquote
import sys
from verification.lib import *
from networkx.drawing.nx_pydot import write_dot
import random
import argparse
from pathlib import Path
from scipy.stats import ttest_1samp
from statistics import mean
from matplotlib import pyplot
import re


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
    mutation_tree = parse_newick_code(newick_code)

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

    # Write the true matrix:
    if args.out_matrix is not None:
        write_mutation_matrix(
            mutation_tree.get_mutation_matrix(), args.out_matrix)


def quality_test(args: argparse.Namespace):
    def get_likelihood(dir: Path) -> float:
        """
        Get the best likelihood obtained by one executable for one input.
        """
        path = dir / Path("likelihood.txt")
        with open(path, mode="r") as file:
            return float(file.readline())

    # For every input, store the difference in the likelihood produced by ffSCITE and SCITE for that input.
    differences = [
        get_likelihood(input_dir / Path("ffSCITE")) -
        get_likelihood(input_dir / Path("SCITE"))
        for input_dir in args.basedir.iterdir()
    ]

    max_bit_flip_difference = max(abs(log(args.alpha) - log(1 - args.beta)),
                                  abs(log(args.beta) - log(1 - args.alpha)))
    print(
        f"Mean differences: {mean(differences):.2f} ≌ {abs(mean(differences)) / max_bit_flip_difference:.2f} bit flips")

    epsilon = args.n_bits * max_bit_flip_difference

    _, lower_p_value = ttest_1samp(
        differences, -epsilon, alternative="greater")
    _, upper_p_value = ttest_1samp(
        differences, epsilon, alternative="less")

    p_value_bound = (1.0 - args.confidence_level) / 2.0

    print(f"H0: µ <= -{epsilon:.2f} or µ >= {epsilon:.2f}")
    print(f"H1: -{epsilon:.2f} < µ < {epsilon:.2f}")
    print(f"p-value for lower part of H0: {lower_p_value}")
    print(f"p-value for upper part of H0: {upper_p_value}")
    print(f"maximal p-value to reject H0 parts: {p_value_bound:.4f}")

    if lower_p_value < p_value_bound and upper_p_value < p_value_bound:
        print(f"ffSCITE is non-inferior to SCITE!")
        return_value = 0
    else:
        print(f"ffSCITE may not be non-inferior to SCITE!")
        return_value = 1

    print()
    print("# Additional statistics:")
    print(f"max difference (ffSCITE better): {max(differences):.2f} ≌ {max(differences) / max_bit_flip_difference:.2f} bit flips")
    print(f"min difference (SCITE better): {min(differences):.2f} ≌ {min(differences) / max_bit_flip_difference:.2f} bit flips")
    print(f"no. inputs ffSCITE better: {len([diff for diff in differences if diff > 0])}")
    print(f"no. inputs both equal: {len([diff for diff in differences if diff == 0])}")
    print(f"no. inputs SCITE better: {len([diff for diff in differences if diff < 0])}")

    diff_counts = dict()
    for diff in differences:
        diff = round(diff / max_bit_flip_difference)
        if diff not in diff_counts:
            diff_counts[diff] = 1
        else:
            diff_counts[diff] += 1
    unique_diffs = list(diff_counts.keys())
    unique_diffs.sort()
    diff_counts = [diff_counts[diff] for diff in unique_diffs]

    pyplot.bar(unique_diffs, diff_counts)
    pyplot.xlabel("Log-likelihood difference, in bit-flips")
    pyplot.ylabel("Number of outputs with the given difference")
    pyplot.savefig("differences.png", dpi=750)

    exit(return_value)

def performance_analysis(args: argparse.Namespace):
    if args.out_file is None:
        out_file = sys.stdout
    else:
        out_file = open(args.out_file, mode="w")

    print("| no. of cells | no. of genes | no. of chains | no. of steps per chain | ffSCITE: mean total makespan | ffSCITE: throughput | SCITE: mean total makespan | SCITE: throughput | speedup |", file=out_file)
    print("|-|-|-|-|-|-|-|-|-|", file=out_file)

    n_cell_dirs = list(args.basedir.iterdir())
    n_cell_dirs.sort(key=lambda dir: int(dir.name))

    for n_cell_dir in n_cell_dirs:
        n_cells = int(n_cell_dir.name)
        n_genes = n_cells - 1
        makespan_re = re.compile("Time elapsed: ([0-9]+(\.[0-9]+)?)")

        def analyze_makespans(out_dir):
            all_makespans = dict()
            for logfile_path in Path(out_dir).glob("*.log"):
                parts = logfile_path.stem.split("_")
                n_chains = int(parts[0])
                n_steps = int(parts[1])

                with open(logfile_path, mode="r") as logfile:
                    lines = (makespan_re.match(line) for line in logfile.readlines())
                    makespans = [float(match[1]) for match in lines if match is not None]
                    if len(makespans) > 0:
                        all_makespans[(n_chains, n_steps)] = mean(makespans)
            return all_makespans

        ffscite_makespans = analyze_makespans(n_cell_dir / Path("ffSCITE"))
        scite_makespans = analyze_makespans(n_cell_dir / Path("SCITE"))

        assert set(ffscite_makespans.keys()) == set(scite_makespans.keys())
        keys = list(ffscite_makespans.keys())
        keys.sort()

        for key in keys:
            if key in ffscite_makespans:
                ffscite_m = f"{ffscite_makespans[key] * 1e-3:.2f} s"
                ffscite_throughput = f"{(key[0] * key[1]) / ffscite_makespans[key]:.2f} ksteps/s"
            else:
                ffscite_m = "n/a"
                ffscite_per_step = "n/a"

            if key in scite_makespans:
                scite_m = f"{scite_makespans[key] * 1e-3:.2f} s"
                scite_throughput = f"{(key[0] * key[1]) / scite_makespans[key]:.2f} ksteps/s"
            else:
                scite_m = "n/a"
                scite_per_step = "n/a"
            
            if key in ffscite_makespans and key in scite_makespans:
                ratio = f"{scite_makespans[key] / ffscite_makespans[key]:.2f}"
            else:
                ratio = "n/a"
            
            print(f"| {n_cells} | {n_genes} | {key[0]} | {key[1]} | {ffscite_m} | {ffscite_throughput} | {scite_m} | {scite_throughput} | {ratio} |", file=out_file)


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
score_parser.add_argument("-o", "--out-matrix", required=False,
                          type=Path, help="The path to store the tree's mutation matrix at.")
score_parser.add_argument("-x", "--compact", default=False, required=False,
                          action="store_true", help="Only emit log-likelihood, no explanatory text.")

test_parser = subparsers.add_parser(
    "tost", help="Execute a hypothesis test to show the non-inferiority of ffSCITE.")

test_parser.add_argument("-d", "--basedir", default=Path(
    "./quality_benchmark.out"), type=Path, help="Base path of the collected data.")
test_parser.add_argument("-n", "--n-bits", default=1, type=int,
                         help="The number of allowed/negligible bit flips.")
test_parser.add_argument("-a", "--alpha", default=1e-6, type=float,
                         help="The probability of false positives.")
test_parser.add_argument("-b", "--beta", default=0.25, type=float,
                         help="The probability of false negatives.")
test_parser.add_argument("-c", "--confidence-level", default=0.95, type=float,
                         help="The confidence level used for the test.")

performance_parser = subparsers.add_parser("performance", help="Analyze the outputs of the performance benchmark")

performance_parser.add_argument("-d", "--basedir", default=Path("./performance_benchmark.out"), type=Path, help="Base path of the collected data.")
performance_parser.add_argument("-o", "--out-file", default=None, type=Path, help="Path to the output file. Stdout if not given")

args = parser.parse_args()

if args.subcommand == "generate":
    generate(args)
elif args.subcommand == "score":
    score_tree(args)
elif args.subcommand == "tost":
    quality_test(args)
elif args.subcommand == "performance":
    performance_analysis(args)
