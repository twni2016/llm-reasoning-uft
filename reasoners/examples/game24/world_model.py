import copy
import dataclasses
import re
from typing import Optional
from base import WorldModel


@dataclasses.dataclass
class Game24State:
    input: str  # the original numbers
    current: str  # the current left numbers
    history: list[str]  # the actions taken
    output: Optional[str] = None  # the answer

    def to_list(self):
        if self.output is None:
            return [self.input] + self.history
        else:
            return [self.input] + self.history + [self.output]


Game24Action = str


class Game24WorldModel(WorldModel):

    def init_states(self) -> Game24State:
        return [Game24State(example, example, []) for example in self.examples]

    def step(self, state: Game24State, action: Game24Action) -> Game24State:
        # deterministic transition with appending
        next_state = copy.deepcopy(state)
        if "Answer" in action:
            match = re.match(r"Answer: (.*)", action)
            next_state.output = match[1] if match is not None else ""
        else:
            match = re.match(r".*\(left: (.*)\)", action)
            next_state.current = match[1] if match is not None else ""
            next_state.history.append(action)
        return next_state

    def is_terminal(self, state: Game24State) -> bool:
        return state.output is not None
