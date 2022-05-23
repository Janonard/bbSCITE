# Initial Talk "Accelerating Single-Cell Inference for Tumor Evolution with FPGAs"

* Welcome to my initial talk of my bachelor's thesis!
* As you might know, I've been working together with Tobias to evaluate and explore the oneAPI workflow for FPGAs for almost three years
* Therefore, it was more or less obvious that I should also do my bachelor's thesis in this general area.
* The initial idea was that I continue to extend StencilStream, or current project.
* However, this was deemed inappropriate since it would make it hard to distinguish paid work as a student assistant and unpaid work for the bachelor's thesis.
* Therefore, Tobias came up with a more traditional topic: Efficiently Implementing the SCITE algorithm with FPGAs.

* First, some words about the problem at hand:
* As you might now, cancer tumors are created when a normal body cell mutates in a way that makes it more reproductive and let's it evade natural body defenses.
* These mutations are passed on to clones of the originally mutated cell, so-called subclones.
* When the tumor starts to grow, more mutations may appear which are passed down to the next subclones, accumulating all mutations in their lineage.
* These subclones may have vastly different behavior depending on their mutation combinations, and may need different treatments.
* There is therefore an incentive to analyze a tumor's cells and identify which mutation combinations are present in the tumor.
* It is already relatively easy to sample a bulk of cells to identify the most common mutations.
  * However, it's only a snapshot and identifies the most common mutation combination.
* If a more aggressive subclone has just started to emerge when the tumor was extracted and it is still rather small, it is missed by bulk sequencing.
* Single-Cell genome sequencing required, but it is very error-prone.
  * For example, false negatives are quoted with rates over 10% and over 50% of the datapoints may be missing due to errors in the sequencing process. (Quote to SCITE)
* Therefore, an algorithm is needed that can find the most likely mutation lineage for the given, noisy input data.

* SCITE is one such algorithm, and it belongs to the class of Monte-Carlo-Markov-Chain Algorithms.
* Monte-Carlo Algorithms basically repeat a random experiment that yields a possible solution to a problem and records the best encountered solution.
* Monte-Carlo-Markov-Chain algorithms therefore simulate a markov chain where every solution is a randomly modified version of the previous solution.
  * This has the advantage that new solutions don't need to be devised from scratch and possibly good characteristics of a solution may be preserved.
  * However, an algorithm designer has to take extra care that the chain actually converges on good solutions.

* The original implementation by Katharina Jahn, ETH Zürich, is not that efficient.
* Dominik Ernst, Uni Erlangen, et al. have written an unpublished report on how they optimized it.
  * Tobias got this report and saw that there might be some potential to also optimize it for FPGAs.
  * I also found this algorithm interesting and I have some personal connection to the topic, so it was a good choice for me.

* So, let's have a look at the actual, abstracted problem.
* First of all, the problem model in the paper and in the application differ quite a bit, so we will first have a look at the simpler model from the paper.
* The input is a matrix that contains an entry for every cell and genome position.
* The entries of the matrix denote whether the given genome position of a cell has been observed as mutated, unmutated, or missing.
* This input matrix is noisy, which means that the entries may or may not reflect the true state of the cells.
* We assume that every matrix entry is independent from the others, and that we have (Formula for error probabilities)

* The output of the algorithm is a mutation tree and a mapping of every cell onto one of the nodes in the tree.
* Every node in the mutation tree except the root represents a genome position. (Slide with mutation tree)
* If a gene A is the child of another gene B, then we assume that every cell that has a mutation at A also has a mutation at B, and so forth up to the root. (Example from one node in the tree)
* We assume that every gene mutates only once and that it does not mutate back, otherwise this tree would not make sense.
* Now, when we place a node on the tree, we have a statement of the true state of the cell and can compute the posterior likelihood that this statement over the cell's mutation status is correct.

* The goal now obviously is to find a tree and a cell-node mapping that maximizes this likelihood.
* This is done by starting of with a random tree, applying a random modification to it and computing it's likelihood.
  * If the likelihood of the new tree is better than the best we know, it is stored as the new, best solution.
  * The quality of the solution is contained by using rejection sampling, which rejects less likely trees with a higher probability than those with higher likelihoods.
* Then, these steps of proposing a new tree, computing it's likelihood and evaluating the new state is repeated multiple thousand times.

* However, As I said before, the actual application does things differently.
* First of all, it differentiates between homozygous and heterozygous mutations.
  * The first one means that the mutation has been observed on both chromosomes of a cell, while the second one means that the mutation was only observed on one of the two chromosomes.
  * This requires different error probabilities.
* Next, it tries to also learn the beta error rate, i.e. the probability that a mutated gene is observed in it's common form.
  * It starts of from an assumed mean value, tries different error probabilities and sticks with the most likely one.
* Lastly, it always attaches cells to the most likely tree node
  * I.e. it attaches every cell to every node in the tree, computes the likelihood that the attachment is correct and picks the most likely one.
* But the general idea of proposing a modification and evaluating is still the same.

* This algorithm offers multiple opportunities for exploitable parallelism.
* First of all, there is little feedback from one chain step to the next
  * Actually only the current and best states of the chain and their likelihoods.
* The bounds of the loop are also very easy to predict
* And you also have execute multiple independent chains from different starting points
* Therefore, this loop should benefit greatly from pipelining and unrolling.
* Lastly, there are also many internal loops that can be unrolled and pipelined, for example in the computation of the tree likelihood.
* I therefore assume that my implementation will perform quite well.
  * As least as fast as the original implementation by Katharina Jahn, but maybe even faster than the optimized version by Dominik Ernst if I'm able to obtain this version or at least performance figures.

* So, this is what I'm planning to do. Are there any questions?