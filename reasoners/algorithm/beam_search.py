from typing import Generic, NamedTuple, List, Tuple, Callable, Any, Union, Optional
from base import SearchAlgorithm, WorldModel, BatchedSearchConfig, State, Action
import itertools


class BeamSearchNode:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self,
        state: State,  # current state s
        action: Action,  # previous action a
        reward: float,  # reward r(s)
        parent: Optional["BeamSearchNode"] = None,
        children: Optional[List["BeamSearchNode"]] = None,
    ) -> None:
        self.id = next(BeamSearchNode.id_iter)
        self.state = state
        self.action = action
        self.reward = reward
        self.parent = parent
        self.children = children if children is not None else []

    def add_child(self, child: "BeamSearchNode"):
        self.children.append(child)

    def get_trace(self) -> List[Tuple[Action, State, float]]:
        """Returns the sequence of actions and states from the root to the current node"""
        node, path = self, []
        while node is not None:
            path.append((node.action, node.state, node.reward))
            node = node.parent
        # Reverse the path to get actions and states in order
        path = path[::-1]
        return path


class BeamSearchResult(NamedTuple):
    terminal_node: BeamSearchNode
    terminal_state: State
    cum_reward: float
    tree: BeamSearchNode
    trace: List[Tuple[Action, State, float]]


class BatchedBeamSearch(SearchAlgorithm, Generic[State, Action]):
    """
    Adapted from beam search in LLM reasoners and making it simplified and batchable
    Original algorithm is proposed by Tree of Thought
    """

    def __init__(
        self,
        beam_size: int,
        max_depth: int,
        replace: Optional[bool] = None,
        temperature: Optional[float] = None,
        temperature_decay: Optional[float] = None,
        **kwargs,
    ) -> None:
        # Initialize the BeamSearch class
        super().__init__(**kwargs)
        self.beam_size = beam_size
        self.max_depth = max_depth
        self.replace = replace
        self.temperature = temperature
        self.temperature_decay = temperature_decay

    def __call__(
        self,
        world: WorldModel[State, Action, State],
        config: BatchedSearchConfig[State, Action, State],
    ):
        # reset id
        BeamSearchNode.reset_id()

        # we start with a batch of initial states (root nodes), each one is a separate example
        init_states = world.init_states()
        root_nodes = [
            BeamSearchNode(state=init_state, action=None, reward=0.0)
            for init_state in init_states
        ]
        cur_beams = [[root_node] for root_node in root_nodes]
        terminal_beams = [[] for _ in root_nodes]

        for depth in range(self.max_depth):
            states = [
                node.state for cur_beam in cur_beams for node in cur_beam
            ]  # flattened to a list
            actions = config.get_actions(states)
            rewards = config.get_rewards(states, actions)
            new_beams = [[] for _ in root_nodes]
            idx = 0

            for example_id, cur_beam in enumerate(cur_beams):
                for node, action_list, reward_list in zip(
                    cur_beam,
                    actions[idx : idx + len(cur_beam)],
                    rewards[idx : idx + len(cur_beam)],
                ):
                    for action, reward in zip(action_list, reward_list):
                        next_state = world.step(node.state, action)

                        # Create new node
                        new_node = BeamSearchNode(
                            state=next_state, action=action, reward=reward, parent=node
                        )

                        # Add new node to children of current node
                        node.add_child(new_node)
                        if world.is_terminal(next_state):
                            terminal_beams[example_id].append(new_node)
                        else:
                            new_beams[example_id].append(new_node)
                idx += len(cur_beam)

            # Sample from new beam
            cur_beams = []
            for new_beam in new_beams:
                new_beam.sort(key=lambda x: x.reward, reverse=True)
                cur_beams.append(new_beam[: self.beam_size])

            # Decay the temperature
            if self.temperature_decay:
                self.temperature *= self.temperature_decay

        partial_search_paths = []
        for new_beam in new_beams:
            # push the finally discarded nodes (it must reach max_depth steps)
            partial_search_paths.append([node.state.to_list() for node in new_beam])

        complete_search_paths = []
        for terminal_beam in terminal_beams:
            terminal_beam.sort(key=lambda x: x.reward, reverse=True)

            complete_search_paths.append(
                [node.state.to_list() for node in terminal_beam]
            )

        return complete_search_paths, partial_search_paths
