from typing import (
    Generic,
    Tuple,
    Callable,
    Any,
    Optional,
)
from base import (
    SearchAlgorithm,
    WorldModel,
    BatchedSearchConfig,
    State,
    Action,
    Example,
)
import itertools
import numpy as np
import math


class MCTSNode(Generic[State, Action, Example]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self,
        state: Optional[State],  # current state s
        action: Optional[Action],  # previous action a
        parent: "Optional[MCTSNode]" = None,
        reward: float = 0.0,  # reward r(s)
        is_terminal: bool = False,
        calc_q: Callable[[list[float]], float] = np.mean,
    ):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """
        self.id = next(MCTSNode.id_iter)
        self.cum_rewards: list[float] = []
        self.reward = reward
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children: "Optional[list[MCTSNode]]" = None
        self.calc_q = calc_q
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    @property
    def Q(self) -> float:
        if len(self.cum_rewards) == 0:
            return (
                self.reward
            )  # Note self.reward directly approx Q value (see the value_prompt)
        else:
            return self.calc_q(self.cum_rewards)


class BatchedMCTS(SearchAlgorithm, Generic[State, Action, Example]):
    """
    Adapted from MCTS in LLM reasoners and making it simplified and batchable
    Original algorithm is proposed by RAP
    """

    def __init__(
        self,
        depth_limit: int,
        w_exp: float,
        n_iters: int = 100,  # this controls the number of complete paths, originally 10
        cum_reward: Callable[[list[float]], float] = lambda x: x[-1],
        calc_q: Callable[[list[float]], float] = np.mean,
        simulate_strategy: str | Callable[[list[float]], int] = "sample",
        **kwargs,
    ):
        """
        MCTS algorithm

        :param w_exp: the weight of exploration in UCT
        :param cum_reward: the way to calculate the cumulative reward from each step.
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        :param simulate_strategy: simulate strategy. Options: 'max', 'sample', 'random', or use a custom function
        """
        super().__init__()
        self.world_model = None
        self.search_config = None
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.cum_reward = cum_reward
        self.calc_q = calc_q

        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            "max": lambda x: np.argmax(x),
            "sample": lambda x: (
                np.random.choice(len(x), p=x / np.sum(x))
                if np.sum(x) > 0
                else np.random.choice(len(x))
            ),
            "random": lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = (
            default_simulate_strategies.get(simulate_strategy)
        )

    def __call__(
        self,
        world: WorldModel[State, Action, State],
        config: BatchedSearchConfig[State, Action, State],
    ):
        # reset id
        MCTSNode.reset_id()
        self.world_model = world
        self.search_config = config
        self._output_cum_reward = -math.inf
        self._output_iter = None

        # we start with a batch of initial states (root nodes), each one is a separate example
        init_states = self.world_model.init_states()
        root_nodes = [
            MCTSNode(state=init_state, action=None, parent=None, calc_q=self.calc_q)
            for init_state in init_states
        ]

        for i in range(self.n_iters):
            print(f"iteration {i+1} out of {self.n_iters}\n")
            paths = [self._select(root) for root in root_nodes]
            self._expand_and_simulate(paths)

            for path in paths:
                self._back_propagate(path)

        all_search_paths = []
        for root in root_nodes:
            all_search_paths.append(self.find_all_paths(root))

        complete_search_paths, partial_search_paths = [], []
        for search_paths in all_search_paths:
            complete_path_reward_pairs = [
                (
                    path[-1].state.to_list(),
                    self.cum_reward([node.reward for node in path]),
                )
                for path in search_paths
                if path[-1].is_terminal
            ]
            # sort by reward (Q) in descending order
            complete_path_reward_pairs = sorted(
                complete_path_reward_pairs, key=lambda x: x[1], reverse=True
            )
            complete_search_paths.append([p[0] for p in complete_path_reward_pairs])

            partial_search_paths.append(
                [
                    path[-1].state.to_list()
                    for path in search_paths
                    if not path[-1].is_terminal
                ]
            )

        return complete_search_paths, partial_search_paths

    def find_all_paths(self, root):
        all_paths = []

        def dfs(node, path):
            path.append(node)

            if not node.children:
                all_paths.append(list(path))  # Append a copy of the current path
            else:
                # Recur for each child node
                for child in node.children:
                    dfs(child, path)

            # Backtrack: Remove the current node's state
            path.pop()

        dfs(root, [])
        return all_paths

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.is_terminal or node.depth >= self.depth_limit

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        path = []
        while True:
            path.append(node)
            if (
                node.children is None
                or len(node.children) == 0
                or self._is_terminal_with_depth_limit(node)
            ):
                return path
            node = max(node.children, key=self._uct)

    def _uct(self, node: MCTSNode) -> float:
        return node.Q + self.w_exp * np.sqrt(
            np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards))
        )

    def _expand_and_simulate(self, paths: list[list[MCTSNode]]):
        # like DFS
        while True:
            nodes = [path[-1] for path in paths]
            flags = [not self._is_terminal_with_depth_limit(node) for node in nodes]
            valid_nodes = [node for node, flag in zip(nodes, flags) if flag]
            if len(valid_nodes) == 0:
                return
            self._expand(valid_nodes)

            done = True
            for path, node, flag in zip(paths, nodes, flags):
                if flag and node.children is not None and len(node.children) > 0:
                    rewards = np.array(
                        [child.reward for child in node.children], dtype=np.float64
                    )
                    path.append(node.children[self.simulate_choice(rewards)])
                    done = False
            if done:
                return

    def _expand(self, nodes: list[MCTSNode]):
        states = [node.state for node in nodes]
        actions = self.search_config.get_actions(states)
        rewards = self.search_config.get_rewards(states, actions)

        for node, action_list, reward_list in zip(nodes, actions, rewards):
            children = []
            for action, reward in zip(action_list, reward_list):
                next_state = self.world_model.step(node.state, action)
                child = MCTSNode(
                    state=next_state,
                    action=action,
                    parent=node,
                    reward=reward,
                    is_terminal=self.world_model.is_terminal(next_state),
                    calc_q=self.calc_q,
                )
                children.append(child)

            node.children = children

    def _back_propagate(self, path: list[MCTSNode]):
        rewards = []
        cum_reward = -math.inf
        for node in reversed(path):
            rewards.append(node.reward)
            # here we will choose the last-step reward as Q
            cum_reward = self.cum_reward(rewards[::-1])
            node.cum_rewards.append(cum_reward)
        return cum_reward
