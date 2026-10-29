### "Let’s agree to agree: neural networks share classification order on real datasets":

I thought that this paper was insightful in that it implies there exists a real underlying structure of a dataset that optimizers (through either SGD / adaptive variants) take advantage of. TLDR: This paper offers a link between the training and test datasets, in that the algorithm learns data in the dataset in a remarkably similar order -> and, this layout / order of data is predictable. Both weaker and stronger models start by learning the same core examples, while powerful models are able to learn "harder" ones. The measurement for "hardness" is interesting, however, as the researchers measured complexity by a custom "accessibility score" scalar, which can be thought of intuitively as the expected value of number of correct classifications between the truth value and the learner's function. 

Follow up questions: Perhaps we could test out some alternative measures to "complexity" and see if this same phenomenon persists: a few ideas that are popping up are bit length, amount of variance in an image / question, kolgomorov complexity, etc... If the same phenomenon is observed, then that implies that the notion of complexity can be simplified further than simply the accuracy of the model on the task, which presupposes an innate "language" of complexity and structure in a dataset that can be measured. 



