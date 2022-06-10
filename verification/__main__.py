from verification import *
from networkx.drawing.nx_pydot import write_dot
import random
import argparse
from pathlib import Path
from math import log

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

# Generate the true tree
tree = random_mutation_tree(n_genes, n_cells)

# Emit it for later inspection
write_dot(tree, out_base.with_name(out_base.name + "_true_tree.gv"))
with open(out_base.with_name(out_base.name + "_true_tree.newick"), mode="w") as out_file:
    print(to_newick_code(tree, n_genes), file=out_file)

# Produce the true, unaltered mutation matrix
true_mutation_matrix = tree.get_mutation_matrix()

# Apply noise
noisy_mutation_matrix = apply_sequencing_noise(true_mutation_matrix, prob_false_positives, prob_false_positives, prob_missing)

write_mutation_matrix(noisy_mutation_matrix, out_base.with_name(out_base.name + "_input.csv"))

true_tree_likelihood = score_mutation_matrix(tree, noisy_mutation_matrix, prob_false_positives, prob_false_negatives)
print(f"True mutation tree likelihood: {true_tree_likelihood} = exp({log(true_tree_likelihood)})")
