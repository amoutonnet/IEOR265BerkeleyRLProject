import numpy as np

"""
Original Paper: Schaul et al. DeepMind 2015 PRIORITIZED EXPERIENCE REPLAY https://arxiv.org/pdf/1511.05952.pdf
Code adapted from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py   
"""


class SumTree():
    """
    Binary tree where parents are the sum of children
    Each leaf (end) node contains a priority score and an index referring to the index in the memory (s, a, r, s', done)
    """

    def __init__(self, capacity):
        """ 
        Initialize the tree with all nodes = 0, and initialize the data with all values = 0
        """
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences
        self.tree = np.zeros(2 * capacity - 1)  # each node has max two children, root node is alone: capacity -1 parents, capacity leaf nodes
        self.data = np.zeros(capacity, dtype=object)  # Experiences
        self.data_pointer = 0

    def add(self, new_priority, new_data):
        """
        Add priority score in the sumtree leaf and add the experience to the data
        """
        tree_index = self.data_pointer + self.capacity - 1  # The first leaf node is at capacity -1 (Preorder ordering)
        self.data[self.data_pointer] = new_data  # Update data
        self.update(tree_index, new_priority)  # Update tree
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # If above capacity, go back to first index (overwriting)
            self.data_pointer = 0

    def update(self, tree_index, new_priority):
        """
        Update the leaf priority score and propagate the change through tree
        """
        change = new_priority - self.tree[tree_index]  # Change = new priority score - former priority score
        self.tree[tree_index] = new_priority
        # Propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        """
        Get leaf index, priority value of that index and experience associated to that index
        Input v: value
        Output: leaf index , leaf value, related data (experience)
        """
        parent_idx = 0
        while True:
            left_idx = 2 * parent_idx + 1  # Preorder ordering
            right_idx = left_idx + 1
            if left_idx >= len(self.tree):  # If bottom is reached, end the search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_idx]:
                    parent_idx = left_idx
                else:
                    v -= self.tree[left_idx]
                    parent_idx = right_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class PrioritizedExperienceMemory(SumTree):
    """
    Manages experiences memory (s, a, r, s', done) in a SumTree to be more efficient
    """

    def __init__(self, capacity, alpha, beta, beta_increment, epsilon, abs_error_upper=1):
        super().__init__(capacity)
        self.epsilon = epsilon  # Ensure that all probabilities are non-zero
        self.alpha = alpha  # Tradeoff between prioritized and random exp replay
        self.beta = beta  # Initial value of importance-sampling, increasing to 1
        self.beta_increment = beta_increment  # Increment value for importance-sampling
        self.abs_error_upper = abs_error_upper
        self.len_memory = 0

    def append(self, transition):
        max_priority = np.max(self.tree[-self.capacity:])  # max among leaf nodes
        if max_priority == 0:
            max_priority = self.abs_error_upper
        self.add(self.abs_error_upper, transition)  # Set the max p for new p
        if self.len_memory <= self.capacity:
            self.len_memory += 1

    def sample(self, n):
        """
        Create a sample array that will contains the minibatch of n transitions
        """
        batch_idx = np.empty((n,), dtype=np.int32)
        batch_memory = []
        ISWeights = np.empty((1, n))  # Importance sampling weights: correct bias introduced by th change in distribution of experiences

        priority_segment = self.tree.total_priority / n
        # Increasing beta each time a minibatch is sampled
        self.beta = np.min([1., self.beta + self.beta_increment])  # max = 1

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(v)
            sampling_prob = priority / self.tree.total_priority
            # IS = (1/P(i))**b /max wi == P(i)**(-b)  /max wi
            ISWeights[:, i] = np.power(n * sampling_prob, -self.beta)
            batch_idx[i] = idx
            batch_memory.append(data)
        return batch_idx, batch_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        """
        Updates transitions probabilities for all sampled transitions in the minibatch given the absolute TD error computed
        """
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_error_upper)
        transition_probs = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, transition_probs):
            self.tree.update(ti, p)

    def __len__(self):
        return self.len_memory
