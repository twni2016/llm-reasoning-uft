import copy
import re

from base import BatchedSearchConfig, LanguageModel
from examples.game24.world_model import Game24State, Game24Action

from examples.game24.prompts.game24 import (
    output_prompt,
    propose_prompt,
    value_prompt,
    value_last_step_prompt,
    value_map,
)


class BatchedGame24Config(BatchedSearchConfig):
    def __init__(
        self,
        base_model: LanguageModel,
        temperature=0.7,
        top_p=0.8,
        depth_limit=4,
        n_eval=3,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
        self.top_p = top_p
        self.depth_limit = depth_limit
        self.n_eval = n_eval

    @staticmethod
    def output_prompt_wrap(state: Game24State) -> str:
        return output_prompt.format(input=state.input, history="\n".join(state.history))

    def propose_prompt_wrap(self, state: Game24State) -> str:
        return propose_prompt.format(input=state.current)

    @staticmethod
    def value_prompt_wrap(state: Game24State) -> str:
        return value_prompt.format(input=state.current)

    @staticmethod
    def value_last_step_prompt_wrap(state: Game24State) -> str:
        return value_last_step_prompt.format(input=state.input, answer=state.output)

    @staticmethod
    def retrieve_value(output: list[str]) -> float:
        value = sum(v * output.count(k) for k, v in value_map.items())
        return value

    def get_actions(self, states: list[Game24State]) -> list[list[Game24Action]]:
        prompts = []

        for state in states:
            if " " in state.current:
                prompts.append(self.propose_prompt_wrap(state))
            else:  # a single number, might be not 24
                prompts.append(self.output_prompt_wrap(state))

        responses = self.base_model.generate(
            prompts,
            n=1,
            temperature=self.temperature,  # in LLM reasoner, this is greedy
            top_p=self.top_p,
            stop=["\n\n", "\nInput", "(Note:"],
        )
        # used for debugging
        # print("\n".join(map(repr, [res.outputs[0].text for res in responses])))

        actions = []
        for state, response in zip(states, responses):
            # Qwen may have trailing backticks
            output = response.outputs[0].text.strip("` \t\n\r")
            if " " in state.current:
                output = output.split("\n")
                possible_steps = [
                    x for x in output if bool(re.search(r"\(left: .*?\)", x))
                ]
                # this may be an empty list
                # set does not guarantee order, but dict does guarantee
                actions.append(list(dict.fromkeys(possible_steps)))
            else:
                output = "Answer: " + output
                actions.append([output])

        return actions

    def get_rewards(
        self, states: list[Game24State], actions: list[list[Game24Action]]
    ) -> list[list[float]]:
        prompts = []
        for state, action_list in zip(states, actions):
            for action in action_list:  # NOTE: action_list may be empty
                next_state = copy.deepcopy(state)  # a temp variable
                if "Answer" in action:
                    match = re.match(r"Answer: (.*)", action)
                    next_state.output = match[1] if match is not None else ""
                else:
                    match = re.match(r".*\(left: (.*)\)", action)
                    next_state.current = match[1] if match is not None else ""

                if next_state.output is None:
                    prompt = self.value_prompt_wrap(next_state)
                else:
                    prompt = self.value_last_step_prompt_wrap(next_state)
                prompts.append(prompt)

        responses = self.base_model.generate(
            prompts,
            n=self.n_eval,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["\n\n"],
        )

        flatten_values = []
        for response in responses:
            output = [o.text.strip().split("\n")[-1] for o in response.outputs]
            flatten_values.append(self.retrieve_value(output))

        values = []
        idx = 0
        for action_list in actions:
            values.append(copy.deepcopy(flatten_values[idx : idx + len(action_list)]))
            idx += len(action_list)

        return values
