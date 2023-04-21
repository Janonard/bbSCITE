import sys
from verification.lib import *
from networkx.drawing.nx_pydot import write_dot
import random
import argparse
from pathlib import Path
from scipy.stats import ttest_1samp
from statistics import mean
from matplotlib import pyplot
from math import exp


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
    print(
        f"max difference (ffSCITE better): {max(differences):.2f} ≌ {max(differences) / max_bit_flip_difference:.2f} bit flips")
    print(
        f"min difference (SCITE better): {min(differences):.2f} ≌ {min(differences) / max_bit_flip_difference:.2f} bit flips")
    print(
        f"no. inputs ffSCITE better: {len([diff for diff in differences if diff > 0])}")
    print(
        f"no. inputs both equal: {len([diff for diff in differences if diff == 0])}")
    print(
        f"no. inputs SCITE better: {len([diff for diff in differences if diff < 0])}")

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


def quickperf(args: argparse.Namespace):
    ffscite_perf_data, scite_perf_data = load_performance_data(
        args.basedir, verify_coverage=True)

    for n_cells, cell_data in ffscite_perf_data.items():
        throughputs = list()
        speedups = list()
        for n_chains, chain_data in cell_data.items():
            throughputs += [(n_chains * n_steps) /
                            makespan for (n_steps, makespan) in chain_data.items()]
            speedups += [scite_makespan / ffscite_makespan
                         for (ffscite_makespan, scite_makespan)
                         in zip(chain_data.values(), scite_perf_data[n_cells][n_chains].values())]
        print(f"# {n_cells} cells:")
        print(f"ffSCITE mean throughput: {mean(throughputs):.2f} ksteps/s")
        print(
            f"mean speedup: {mean(speedups):.2f}, min speedup: {min(speedups):.2f}, max speedup: {max(speedups):.2f}")
        print()


def performance_table(args: argparse.Namespace):
    raw_perf_data = load_performance_data(args.basedir)

    perf_data = dict()
    all_n_cells = set()
    for binary, data in raw_perf_data.items():
        throughput = dict()
        for (n_cells, n_chains, n_steps), (runtime, power) in data.items():
            if n_cells not in throughput:
                throughput[n_cells] = []
                all_n_cells.add(n_cells)
            throughput[n_cells] += [(n_chains * n_steps) / runtime]

        perf_data[binary] = {n_cells: mean(
            throughput_list) for n_cells, throughput_list in throughput.items()}

    all_n_cells = list(all_n_cells)
    all_n_cells.sort()

    table = [
        ["input size", "throughput (SCITE)", "throughput (96 bit)", "throughput (64 bit)"]]

    for n_cells in all_n_cells:
        n_genes = n_cells - 1

        row = [f"{n_cells} cells x {n_genes} genes"]

        if n_cells in perf_data["SCITE"]:
            scite_makespan = perf_data["SCITE"][n_cells]
            row += [f"{scite_makespan * 1e-3:.2f} ksteps/s"]
        else:
            scite_makespan = None
            row += ["n/a"]

        for binary in ["ffSCITE96", "ffSCITE64"]:
            if n_cells in perf_data[binary]:
                ffscite_makespan = perf_data[binary][n_cells]
                ffscite_speedup = ffscite_makespan / scite_makespan
                row += [
                    f"{ffscite_makespan*1e-3:.2f} ksteps/s (x{round(ffscite_speedup)} speedup)"]
            else:
                row += ["n/a"]

        table += [row]

    print_table(table, args.style, args.out_file)


def power_table(args: argparse.Namespace):
    raw_perf_data = load_performance_data(args.basedir)

    perf_data = dict()
    for (n_cells, n_chains, n_steps), (runtime, power) in raw_perf_data["ffSCITE96"].items():
        if n_cells not in perf_data:
            # power of ffSCITE96, energy of ffSCITE96, power of ffSCITE64, energy of ffSCITE64
            perf_data[n_cells] = [[], [], [], []]
        perf_data[n_cells][0].append(power)
        perf_data[n_cells][1].append((runtime * power) / (n_chains * n_steps))

    for (n_cells, n_chains, n_steps), (runtime, power) in raw_perf_data["ffSCITE64"].items():
        if n_cells not in perf_data:
            # power of ffSCITE96, energy of ffSCITE96, power of ffSCITE64, energy of ffSCITE64
            perf_data[n_cells] = [[], [], [], []]
        perf_data[n_cells][2].append(power)
        perf_data[n_cells][3].append((runtime * power) / (n_chains * n_steps))

    table = [["input size", "mean power draw (96 bit)", "mean energy per step (96 bit)", "mean power draw (64 bit)", "mean energy per step (64 bit)"]]
    for n_cells, (power96, energy96, power64, energy64) in perf_data.items():
        row = [f"{n_cells} cell x {n_cells+1} genes"]
        if len(power96) > 0:
            row.append(f"{mean(power96):.2f} W")
        else:
            row.append("n/a")

        if len(energy96) > 0:
            row.append(f"{mean(energy96)*1e6:.2f} µWs")
        else:
            row.append("n/a")

        if len(power64) > 0:
            row.append(f"{mean(power64):.2f} W")
        else:
            row.append("n/a")

        if len(energy64) > 0:
            row.append(f"{mean(energy64)*1e6:.2f} µWs")
        else:
            row.append("n/a")

        table.append(row)

    print_table(table, args.style, args.out_file)


def performance_graph(args: argparse.Namespace):
    ffscite_perf_data, scite_perf_data = load_performance_data(
        args.basedir, verify_coverage=True)

    # Rearrange the performance data so that the index order is n_chains, cells, n_steps
    def invert_perf_data_dict(perf_data):
        n_perf_data = dict()
        for (n_cells, data) in perf_data.items():
            for n_chains in data.keys():
                if n_chains not in n_perf_data:
                    n_perf_data[n_chains] = dict()
                if n_cells not in n_perf_data[n_chains]:
                    n_perf_data[n_chains][n_cells] = dict()
                n_perf_data[n_chains][n_cells].update(data[n_chains])
        return n_perf_data
    ffscite_perf_data = invert_perf_data_dict(ffscite_perf_data)
    scite_perf_data = invert_perf_data_dict(scite_perf_data)

    all_n_chains = list(ffscite_perf_data.keys())
    all_n_chains.sort()

    fig = pyplot.figure(figsize=(6 * len(all_n_chains), 5 * 3))

    ax_makespan = [fig.add_subplot(3, len(all_n_chains), i + 1)
                   for i in range(len(all_n_chains))]
    ax_throughput = [fig.add_subplot(3, len(all_n_chains), len(
        all_n_chains) + i + 1) for i in range(len(all_n_chains))]
    ax_speedup = [fig.add_subplot(3, len(
        all_n_chains), 2 * len(all_n_chains) + i + 1) for i in range(len(all_n_chains))]

    max_makespan = 0
    max_throughput = 0
    max_speedup = 0

    for (i_n_chains, n_chains) in enumerate(all_n_chains):

        all_n_cells = list(ffscite_perf_data[n_chains].keys())
        all_n_cells.sort()

        for i_n_cells, n_cells in enumerate(all_n_cells):
            n_genes = n_cells - 1

            ffscite_m = ffscite_perf_data[n_chains][n_cells]
            scite_m = scite_perf_data[n_chains][n_cells]

            all_n_steps = list(ffscite_m.keys())
            all_n_steps.sort()

            dashed_linestyle = (
                5 * i_n_cells, (4, 5 * (len(all_n_cells) - 1) + 1))
            color = f"C{i_n_cells}"
            scite_label = f"{n_cells} x {n_genes} (SCITE)"
            ffscite_label = f"{n_cells} x {n_genes}(ffSCITE)"

            scite_makespan_axis = [scite_m[n_steps]
                                   * 1e-3 for n_steps in all_n_steps]
            ax_makespan[i_n_chains].plot(
                all_n_steps, scite_makespan_axis, c=color, label=scite_label)

            ffscite_makespan_axis = [ffscite_m[n_steps]
                                     * 1e-3 for n_steps in all_n_steps]
            ax_makespan[i_n_chains].plot(
                all_n_steps, ffscite_makespan_axis, linestyle=dashed_linestyle, c=color, label=ffscite_label)

            max_makespan = max(
                [max_makespan] + ffscite_makespan_axis + scite_makespan_axis)

            scite_throughput_axis = [
                (n_chains * n_steps) / scite_m[n_steps] for n_steps in all_n_steps]
            ax_throughput[i_n_chains].plot(
                all_n_steps, scite_throughput_axis, c=color)

            ffscite_throughput_axis = [
                (n_chains * n_steps) / ffscite_m[n_steps] for n_steps in all_n_steps]
            ax_throughput[i_n_chains].plot(
                all_n_steps, ffscite_throughput_axis, linestyle=dashed_linestyle, c=color)

            max_throughput = max(
                [max_throughput] + ffscite_throughput_axis + scite_throughput_axis)

            speedup_axis = [scite_m[n_steps] / ffscite_m[n_steps]
                            for n_steps in all_n_steps]
            ax_speedup[i_n_chains].plot(
                all_n_steps, speedup_axis, linestyle="dashdot", c=color, label=f"{n_cells} x {n_genes}")

            max_speedup = max([max_speedup] + speedup_axis)

        ax_makespan[i_n_chains].set_title(f"Makespan, {n_chains} chains")
        ax_makespan[i_n_chains].set_xlabel("Number of chain steps per chain")
        ax_makespan[i_n_chains].set_ylabel("Makespan, in s")
        ax_makespan[i_n_chains].legend()
        ax_makespan[i_n_chains].grid(which="both")

        ax_throughput[i_n_chains].set_title(f"Throughput, {n_chains} chains")
        ax_throughput[i_n_chains].set_xlabel("Number of chain steps per chain")
        ax_throughput[i_n_chains].set_ylabel("Throughput in ksteps/s")
        ax_throughput[i_n_chains].grid(which="both")

        ax_speedup[i_n_chains].set_title(f"Speedup, {n_chains} chains")
        ax_speedup[i_n_chains].set_xlabel("Numer of chain steps per chain")
        ax_speedup[i_n_chains].set_ylabel(
            "Speedup (Throughput SCITE / Throughput ffSCITE)")
        ax_speedup[i_n_chains].legend()
        ax_speedup[i_n_chains].grid(which="both")

    for i in range(len(all_n_chains)):
        ax_makespan[i].set_ylim(bottom=0, top=1.1 * max_makespan)
        ax_throughput[i].set_ylim(bottom=0, top=1.1 * max_throughput)
        ax_speedup[i].set_ylim(bottom=1, top=1.1 * max_speedup)

    if args.out_file is None:
        pyplot.show()
    else:
        fig.savefig(args.out_file, dpi=args.resolution, bbox_inches="tight")


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

quickperf_parser = subparsers.add_parser(
    "quickperf", help="Print quick information about the performance benchmark")

quickperf_parser.add_argument("-d", "--basedir", default=Path(
    "./performance_benchmark.out"), type=Path, help="Base path of the collected data")

perftable_parser = subparsers.add_parser(
    "perftable", help="Analyze the outputs of the performance benchmark and print a table")

perftable_parser.add_argument("-d", "--basedir", default=Path(
    "./performance_benchmark.out"), type=Path, help="Base path of the collected data")
perftable_parser.add_argument("-o", "--out-file", default=None,
                              type=Path, help="Path to the output file. Stdout if not given")
perftable_parser.add_argument("-s", "--style", choices=[
                              "latex", "markdown", "csv"], default="markdown", help="Style of the output.")

powertable_parser = subparsers.add_parser(
    "powertable", help="Analyze the power draw of the performance benchmark and print a table")

powertable_parser.add_argument("-d", "--basedir", default=Path(
    "./performance_benchmark.out"), type=Path, help="Base path of the collected data")
powertable_parser.add_argument("-o", "--out-file", default=None,
                               type=Path, help="Path to the output file. Stdout if not given")
powertable_parser.add_argument("-s", "--style", choices=[
    "latex", "markdown", "csv"], default="markdown", help="Style of the output.")

perfgraph_parser = subparsers.add_parser(
    "perfgraph", help="Analyze the outputs of the performance benchmark and plot a graph")

perfgraph_parser.add_argument("-d", "--basedir", default=Path(
    "./performance_benchmark.out"), type=Path, help="Base path of the collected data")
perfgraph_parser.add_argument("-o", "--out-file", default=None, type=Path,
                              help="Path to the output file. Show the graph if not given")
perfgraph_parser.add_argument("-r", "--resolution", default=500,
                              type=float, help="Resolution of the output file in DPI")

args = parser.parse_args()

if args.subcommand == "generate":
    generate(args)
elif args.subcommand == "score":
    score_tree(args)
elif args.subcommand == "tost":
    quality_test(args)
elif args.subcommand == "quickperf":
    quickperf(args)
elif args.subcommand == "perftable":
    performance_table(args)
elif args.subcommand == "powertable":
    power_table(args)
elif args.subcommand == "perfgraph":
    performance_graph(args)
