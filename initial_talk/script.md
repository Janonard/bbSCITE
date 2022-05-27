# Initial Talk "Accelerating Single-Cell Inference for Tumor Evolution with FPGAs"

* Welcome to the initial talk of my bachelor's thesis!
* As you might know, I've been working together with Tobias to evaluate and explore the oneAPI workflow for FPGAs for almost three years
* Therefore, it was more or less obvious that I should also do my bachelor's thesis in this general area.
* The initial idea was that I continue to extend StencilStream, or current project.
* However, this was deemed inappropriate since it would make it hard to distinguish paid work as a student assistant and unpaid work for the bachelor's thesis.
* Therefore, Tobias suggested that I should do some classic performance engineering as a bachelor's thesis instead.
* I will therefore implement an application for single-cell inference of tumor evolution data as efficiently as possible, using Intel FPGAs and oneAPI.

* It is based on the SCITE algorithm by Katharina Jahn et al. from ETH Zürich, which was published in 2016.
* The original implementation however is not that efficient.
  * For example it spends 65% of it's runtime in `free` and `new` calls.
* We know this from an unpublished report by Dominik Ernst et al. from the FAU, who optimized the application for CPUs.
  * Tobias got this report and saw that there might also be some potential to also optimize it for FPGAs.
* He suggested this topic, and I also found this algorithm interesting.
* Additionally, I have some personal connection to the topic, so it was indeed a good choice for me.

* So let me finally present the problem at hand:
* As you might now, cancer tumors are created when a normal body cell mutates in a way that makes it more reproductive and let's it evade natural body defenses.
* These mutations are passed on to clones of the originally mutated cell, so-called subclones.
* When the tumor starts to grow, some cells may mutate again and these mutations are added to the genomes of their subclones.
* These newly mutated subclones may have different behavior depending on their combined mutations, and may require different treatments.
* There is therefore an incentive to analyze a tumor's cells and identify which mutation combinations are present in the tumor.
* It is already relatively common to sample a bulk of cells to identify the most common mutations.
  * However, this creates only a snapshot of the most common mutation combination.
* If a more aggressive subclone has just started to emerge and is still rather small, it is missed by bulk sequencing.
* If we want to identify this new subclone, we need to sequence the genomes of individual cells.
* This works, but it is very error-prone: 
  * For example, false negatives are quoted with rates over 10% and over 50% of the datapoints may be missing due to errors in the sequencing process.
* Therefore, an algorithm is needed that can find the most likely mutation lineage for the given, noisy input data.

* SCITE is one such algorithm, and it belongs to the class of Monte-Carlo-Markov-Chain Algorithms.
* Monte-Carlo Algorithms basically repeat a random experiment that yields a possible solution to a problem and records the best encountered solution.
* Monte-Carlo-Markov-Chain algorithms therefore simulate a markov chain where every solution is a randomly modified version of the previous solution.
  * This has the advantage that new solutions don't need to be devised from scratch and possibly good characteristics of a solution may be preserved.
  * However, an algorithm designer has to take extra care that the chain actually converges on good solutions.

* When we take a look at the abstracted problem, we see that the model in the paper and in the application differ by some details
  * It appears that the authors have tried different approaches in the application but only published a fraction of it
  * So I will focus on the slightly simpler model from the paper.
* The input is a matrix that contains an entry for every cell and genome position.
* The entries of the matrix denote whether the given genome position of a cell has been observed as mutated, unmutated, or missing.
* This input matrix is noisy, which means that the entries may or may not reflect the true state of the cells.
* We assume that every matrix entry is independent from the others, and that we have (Formula for error probabilities)
* Note that there are no probabilities given for missing data; The algorithm simply ignores them.

* The output of the algorithm is a mutation tree and a mapping of every cell onto one of the nodes in the tree.
* Every node in the mutation tree except the root represents a genome position. (Slide with mutation tree)
* If a gene A is the child of another gene B, then we assume that every cell that has a mutation at A also has a mutation at B, and so forth up to the root. (Example from one node in the tree)
* We assume that every gene mutates only once and that it does not mutate back, otherwise this tree would not make sense.
* Now, when we place a node on the tree, we have a statement of the true state of the cell and can compute the likelihood that the made statement over the cell's mutation status is correct.

* The goal now obviously is to find a tree and a cell-node mapping that maximizes this likelihood.
* This is done by starting of with a random tree, applying a random modification to it and computing it's likelihood.
  * If the likelihood of the new tree is better than the best we know, it is stored as the new, best solution.
  * The quality of the solution is contained by using rejection sampling, which rejects less likely trees with a higher probability than those with higher likelihoods.
* Then, these steps of proposing a new tree, computing it's likelihood and evaluating the new state are repeated multiple thousand times.

* This algorithm offers multiple opportunities for exploitable parallelism.
* First of all, there is little feedback from one chain step to the next
  * Actually only the current and best states of the chain and their likelihoods.
* And you also have execute multiple independent chains from different starting points
* The bounds of the loop are also deterministic since users simply request the number of repetititons
* Therefore, this loop should benefit greatly from pipelining and unrolling.
* Lastly, there are also many internal loops that can be unrolled and pipelined, for example in the computation of the tree likelihood.
* I therefore assume that my implementation will perform quite well.
  * As least as fast as the original implementation by Katharina Jahn, but maybe even faster than the optimized version by Dominik Ernst if I'm able to obtain this version or at least performance figures.

* So, this is what I'm planning to do. Are there any questions?