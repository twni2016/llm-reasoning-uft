from typing import (
    Generic,
    TypeVar,
    Union,
    Optional,
    Tuple,
)
from abc import ABC, abstractmethod


State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")


class LanguageModel(ABC):
    @abstractmethod
    def generate(
        self,
        inputs: list[str],
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        eos_token_id: Union[None, str, int, list[str, int]] = None,
        hide_input: bool = True,
        output_log_probs: bool = False,
        stopping_criteria=None,
        **kwargs,
    ):
        """Generate text from a list of prompts.

        :param inputs: List of prompts.
        :param max_length: Maximum length of the total output (input + generated).
        :param max_new_tokens: Maximum length of generated tokens. Override max_length.
        :param do_sample: If False, do greedy decoding.
        :param temperature: Temperature for sampling.
        :param top_k: Top-k for sampling.
        :param top_p: Top-p for sampling.
        :param num_return_sequences:
        :param eos_token_id: Token id for end of sentence. Passed *str* will be translated into token_id.
                             Passed *list* will be treated as multiple possible tokens ending the generation.
        :param hide_input: If set true, decode only the generated part.
        :param output_log_probs: If set true, also output the log_probs of each generated token
        :param stopping_criteria:
        """
        ...


class WorldModel(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.examples = None
        self.prompt = None

    @abstractmethod
    def init_states(self) -> list[State]: ...

    @abstractmethod
    def step(self, state: State, action: Action) -> Union[State, Tuple[State, dict]]:
        """Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param action: The action to take
        :return: The next state and optionally an auxiliary data dict
        """
        ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...

    def update_examples(self, examples: list[Example], prompt=None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.examples = examples


class BatchedSearchConfig(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.examples = None
        self.prompt = None

    @abstractmethod
    def get_actions(self, states: list[State]) -> list[list[Action]]: ...

    @abstractmethod
    def get_rewards(
        self, states: list[State], actions: list[list[Action]]
    ) -> list[list[float]]: ...

    def update_examples(self, examples: list[Example], prompt=None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.examples = examples


class SearchAlgorithm(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(
        self, world_model: WorldModel, search_config: BatchedSearchConfig, **kwargs
    ): ...


class BatchedReasoner(ABC, Generic[State, Action, Example]):
    def __init__(
        self,
        world_model: WorldModel[State, Action, Example],
        search_config: BatchedSearchConfig[State, Action, Example],
        search_algo: SearchAlgorithm,
    ) -> None:
        self.world_model = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(self, examples: list[Example], prompt=None, **kwargs):
        self.world_model.update_examples(examples, prompt=prompt)
        self.search_config.update_examples(examples, prompt=prompt)
        return self.search_algo(self.world_model, self.search_config, **kwargs)
