import copy
import dataclasses
import re
from typing import Optional
from base import WorldModel


@dataclasses.dataclass
class CountdownState:
    input: str  # the original numbers
    target: str  # target number
    current: str  # the current left numbers
    history: list[str]  # the actions taken
    output: Optional[str] = None  # the answer

    def to_list(self):
        if self.output is None:
            return [self.input, self.target] + self.history
        else:
            return [self.input, self.target] + self.history + [self.output]


CountdownAction = str


class CountdownWorldModel(WorldModel):

    def init_states(self) -> CountdownState:
        return [
            CountdownState(
                input=example[0], target=example[1], current=example[0], history=[]
            )
            for example in self.examples
        ]

    def step(self, state: CountdownState, action: CountdownAction) -> CountdownState:
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

    def is_terminal(self, state: CountdownState) -> bool:
        return state.output is not None
