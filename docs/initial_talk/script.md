# Initial Talk "Accelerating Single-Cell Inference for Tumor Evolution with FPGAs"

* Welcome to the initial talk of my bachelor's thesis!
* As you might know, I've been working together with Tobias to evaluate and explore the oneAPI workflow for FPGAs for almost three years
* Therefore, it was more or less obvious that I should also do my bachelor's thesis in this general area.
* The initial idea was that I continue to extend StencilStream, our current project.
* However, this was deemed inappropriate since it would make it hard to distinguish paid work as a student assistant and unpaid work for the bachelor's thesis.
* Therefore, Tobias suggested that I should do some classic performance engineering as a bachelor's thesis instead.
* I will therefore implement an application for single-cell inference of tumor evolution data as efficiently as possible, using Intel FPGAs and oneAPI.

* As you might now, cancer tumors are created when a normal body cell mutates in a way that makes it more reproductive and let's it evade natural body defenses.
* These mutations are passed on to children of the originally mutated cell, so-called subclones.
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

* SCITE by Katharina Jahn et al. from ETH Zürich, is one such algorithm and was published in 2016.
* It belongs to the class of Monte-Carlo-Markov-Chain Algorithms
* Monte-Carlo Algorithms basically repeat a random experiment that yields a possible solution to a problem and records the best encountered solution.
* Monte-Carlo-Markov-Chain algorithms therefore simulate a markov chain where every solution is a randomly modified version of the previous solution.
  * This has the advantage that new solutions don't need to be devised from scratch and possibly good characteristics of a solution may be preserved.
  * However, an algorithm designer has to take extra care that the chain actually converges on good solutions.

* The original implementation however is not that efficient.
  * For example it spends 65% of it's runtime in `free` and `new` calls.
* We know this from an unpublished report by Dominik Ernst et al. from the FAU, who optimized the application for CPUs.
  * Tobias got this report and saw that there might also be some potential to also optimize it for FPGAs.
* He suggested this topic, and I also found this algorithm interesting.
* Additionally, I have some personal connection to the topic, so it was indeed a good choice for me.

* Lets have a look at the abstracted problem.
* First of all, I have to make clear that the application offers some optional differences and approaches that have been apparently evaluated, but not published in the paper.
  * I will only cover the slightly simpler model of the paper in this presentation.
* The input is a matrix that contains an entry for every cell and genome position.
* The entries of the matrix denote whether the given genome position of a cell has been observed as mutated, unmutated, or missing.
* If the sequencing were perfect, we would be able to obtain the true state of cells, the matrix E.
* But since we already know that single cell sequencing is error prone, we get the noisy input matrix D instead.
  * We see that there are quite some false positives and some missing data points.
* And from this noisy input, we now have to infer which cells have which mutations.

* In order to do this, we need to compute the likelihood of a assumed true mutation matrix.
* For this, we assume that the matrix are independent and follow this simple posterior probability distribution.
  * We simply have a probability for false positives called alpha and a probability for false negatives called beta.
* Note that there are no probabilities for missing data, they are simply ignored.
* Then, we can simply multiply the posterior probabilities to get the likelihood of an assumed true mutation matrix.
  * Now, we can describe the problem as a maximum-likelihood problem, which is nice.

* We have already established how mutations are introduced: A cell mutates and it's clones inherits all mutations from their ancestors.
  * Additionally, the authors made the infinite sites assumption, which says that every gene only mutates once in the history of the tumor and that a gene never mutates back.
* So, one obvious way to model the mutations is a tree:
  * The root of the tree is the completely unmutated state.
  * We introduce a node for every gene that may mutate and "attach" every cell to one of the nodes in the tree.
  * Then, we say that a cell has mutations on every gene on the path from its attachment node up to the root.
  * For example, the cell with index 2, which is attached to the gene with index 2, has mutations at genes 0, 1, and 2,
  * but not on gene 3 since it is not on the path from the attachment node to the root.

* With all that background, it is easy to describe the algorithm:
  * First of all, it runs multiple independent chains to eliminate the effect of different starting trees.
  * These are generated randomly in the beginning of a chain, and then, this current state is modified.
    * For example, it may swap two nodes in place, or swap two complete subtrees, or just take a subtree and move it somewhere else.
  * Then, it computes the overall likelihood of the made mutation statements.
    * Which includes finding the mostly likely attachment node for every cell in the resulting tree
  * If this new solution is more likely then the previously best, it is stored.
  * Then, it runs a bernoulli experiment to decide whether it should accept or reject this proposed state as the new current state.
    * This is done to ensure that the chain converges on likely solutions.
  * This loop of proposing a new state and computing its likelihood is then repeated multiple thousands to millions of times, as long as the user requests it.

* This algorithm offers multiple opportunities for exploitable parallelism.
* First of all, there is little feedback from one chain step to the next
  * Actually only the current and best states of the chain and their likelihoods.
* The bounds of the loop are also deterministic since users simply request the number of repetititons
* Therefore, these loop should benefit greatly from pipelining and unrolling.
* Lastly, there are also many internal loops that can be unrolled and pipelined, for example in the computation of the tree likelihood.
* However, there are also some challenges:
* Several operations like the likelihood computations or the tree modifications require tree traversals.
  * These are obviously bounded by the number of nodes, but this is often way higher than the actual path lengths.
* Then, we need a lot of random numbers for the tree modifications
  * Using a single random number generator in the whole design might be a bottleneck,
  * but in my initial research I already discovered that different RNGs may produce correlated values, even for different seeds.
  * This is obviously bad, so I have to take care of that.
* Lastly, there are also the common FPGA design problems of memory management and the handling of arithmetic operations.
  * You always have those when you need to optimize your designs.

* Last but not least, I come to my tasks and goals for the thesis.
* My goals are to develop an FPGA-based implementation of the SCITE algorithm that
  * performs better than the original implementation,
  * while producing similar or better results.
* Optionally, I also want it to be faster than the implementation by Dominik Ernst et al.
  * I may not be able to verify this since I do not have access to neither their implementation nor their performance figures.
  * One of the reasons why this goal is optional.
* Another option if I find the time for it is to extend the application to also support the successor to SCITE, infinity SCITE.
  * It solves the same problem with a similar approach, but allows a single gene to mutate back
  * With this, the authors evaluated whether the the assumption that genes do not mutate back is reasonable,
  * but since SCITE is a little bit simpler, I decided to implement it first and then maybe extend my application to infSCITE.

* In order to achieve these goals, I've identified three tasks:
  * Providing an initial, functional implementation,
  * Setting up a verification and benchmarking framework,
  * and then optimizing my application, as well as I have time.
* I've already completed the initial implementation.
* I've probably already busted my time window, but if there still is some time and interest, I could show a little demo.


* I therefore assume that my implementation will perform quite well.
  * As least as fast as the original implementation by Katharina Jahn, but maybe even faster than the optimized version by Dominik Ernst if I'm able to obtain this version or at least performance figures.

* So, this is what I'm planning to do. Are there any questions?