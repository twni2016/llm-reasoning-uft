import ast, json

with open("../reasoners/examples/game24/prompts/game24.json") as f:
    game24_prompt = json.load(f)["ft_prompt"]
with open("../reasoners/examples/countdown/prompts.json") as f:
    cd_prompt = json.load(f)["ft_prompt"]


def template_game24(example, text_field):
    # this function cannot be batch processed
    try:
        path = ast.literal_eval(example[text_field])
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"{e} parsing field {text_field}: {example[text_field]}")

    if example["structured"]:
        ## path format: (input, steps, answer)
        example[text_field] = (
            game24_prompt.format(input=path[0])
            + "\n".join(path[1:-1])
            + "\nAnswer: "
            + path[-1]
            + "\n\n"
        )
    else:
        ## path format: (input, steps)
        example[text_field] = game24_prompt.format(input=path[0]) + "\n".join(path[1:])
    return example


def template_cd(example, text_field):
    # this function cannot be batch processed
    try:
        path = ast.literal_eval(example[text_field])
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"{e} parsing field {text_field}: {example[text_field]}")

    if example["structured"]:
        ## path format: (input, target, steps, answer)
        example[text_field] = (
            cd_prompt.format(input=path[0], target=path[1])
            + "\n".join(path[2:-1])
            + "\nAnswer: "
            + path[-1]
            + "\n\n"
        )
    else:
        ## path format: (input, target, steps)
        example[text_field] = cd_prompt.format(
            input=path[0], target=path[1]
        ) + "\n".join(path[2:])
    return example


def template_API(task):
    if "game24" in task:
        return template_game24
    elif "countdown" in task:
        return template_cd
    else:
        raise ValueError(task)
