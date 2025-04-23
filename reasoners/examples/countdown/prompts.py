# We follow 5-shot prompt and choose training examples from BFS/DFS
# that covers target number [10, 28], [28, 46], [46, 64], [64, 82], [82, 100]

## <500 token length
few_shot_template = """Use numbers and basic arithmetic operations (+ - * /) to obtain the target number. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 25 5 5 33\nTarget: 27
Steps:
25 + 5 = 30 (left: 5 33 30)
30 / 5 = 6 (left: 33 6)
33 - 6 = 27 (left: 27)
Answer: 33 - ((25 + 5) / 5) = 27
Input: 27 25 9 25\nTarget: 43
Steps:
27 - 25 = 2 (left: 9 25 2)
9 * 2 = 18 (left: 25 18)
25 + 18 = 43 (left: 43)
Answer: 25 + (9 * (27 - 25)) = 43
Input: 65 52 41 37\nTarget: 52
Steps:
41 - 37 = 4 (left: 65 52 4)
65 - 52 = 13 (left: 4 13)
4 * 13 = 52 (left: 52)
Answer: (41 - 37) * (65 - 52) = 52
Input: 92 91 23 54\nTarget: 78
Steps:
92 - 91 = 1 (left: 23 54 1)
23 + 54 = 77 (left: 1 77)
1 + 77 = 78 (left: 78)
Answer: (92 - 91) + (23 + 54) = 78
Input: 45 10 11 70\nTarget: 94
Steps:
10 + 11 = 21 (left: 45 70 21)
45 + 70 = 115 (left: 21 115)
115 - 21 = 94 (left: 94)
Answer: (45 + 70) - (10 + 11) = 94
"""

few_shot_cd_prompt = few_shot_template + "Input: {input}\nTarget: {target}\nSteps:\n"

output_prompt = (
    few_shot_template + "Input: {input}\nTarget: {target}\nSteps:\n{history}\nAnswer:"
)


propose_prompt = """Perform a basic arithmetic operation (+, -, *, /) on any two of the given numbers, replacing them with the result. Your goal is to explore combinations that may lead to a final result of the target number. 
Input: 25 5 5 33\nTarget: 27
Possible next steps:
25 + 5 = 30 (left: 5 33 30)
25 * 5 = 125 (left: 5 33 125)
25 / 5 = 5 (left: 5 33 5)
25 + 33 = 58 (left: 5 5 58)
5 * 5 = 25 (left: 25 33 25)
5 + 33 = 38 (left: 25 5 38)
33 - 25 = 8 (left: 5 5 8)
33 - 5 = 28 (left: 25 5 28)
Input: 9 25 2\nTarget: 43
Possible next steps:
9 + 25 = 34 (left: 34 2)
9 * 2 = 18 (left: 25 18)
9 + 2 = 11 (left: 25 11)
25 * 2 = 50 (left: 9 50)
25 + 2 = 27 (left: 9 27)
Input: 65 52 4\nTarget: 52
Possible next steps:
65 - 52 = 13 (left: 4 13)
65 - 4 = 61 (left: 52 61)
52 / 4 = 13 (left: 65 13)
52 + 4 = 56 (left: 65 56)
Input: 1 77\nTarget: 78
Possible next steps:
1 + 77 = 78 (left: 78)
1 * 77 = 77 (left: 77)
77 - 1 = 76 (left: 76)
Input: 21 115\nTarget: 94
Possible next steps:
115 - 21 = 94 (left: 94)
115 + 21 = 136 (left: 136)
Input: 4 13\nTarget: 52
Possible next steps:
4 * 13 = 52 (left: 52)
4 + 13 = 17 (left: 17)
Input: {input}\nTarget: {target}
Possible next steps:
"""

value_prompt = """Evaluate if given number(s) can reach the target number (sure/likely/impossible)
Input: 27\nTarget: 27
sure

Input: 25\nTarget: 27
impossible

Input: 45\nTarget: 43
impossible

Input: 25 18\nTarget: 43
25 + 18 = 43
sure

Input: 31 10\nTarget: 52
31 + 10 = 41
31 - 10 = 21
31 * 10 = 310
31 / 10 = 3.1
impossible

Input: 23 54 1\nTarget: 78
23 + 54 + 1 = 77 + 1 = 78
sure

Input: 45 70 21\nTarget: 94
45 + 70 + 21 = 115 + 21 = 136
-45 + 70 + 21 = 25 + 21 = 46
45 + 70 - 21 = 115 - 21 = 94
sure

Input: 5 7 8\nTarget: 27
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 27 now, but numbers are within a reasonable range
likely

Input: 15 16 16\nTarget: 43
15 + 16 + 16 = 47
(16 - 15) * 16 = 1 * 16 = 16
I cannot obtain 43 now, but numbers are within a reasonable range
likely

Input: 90 108 97\nTarget: 27
90 + 108 + 97 = 295
90 - 108 + 97 = 79
90 108 97 are all too big
impossible

Input: 1 3 3\nTarget: 94
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible

Input: {input}\nTarget: {target}
"""

value_last_step_prompt = """Use numbers and basic arithmetic operations (+ - * /) to obtain the target number. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach the target number.
Input: 25 5 5 33\nTarget: 27
Answer: 33 - ((25 + 5) / 5) = 27
Judge: sure

Input: 27 25 9 25\nTarget: 43
Answer: 25 + (9 * (27 - 25)) = 43
Judge: sure

Input: 65 52 41 37\nTarget: 52
Answer: (41 - 37) * (65 - 52) = 52
Judge: sure

Input: 92 91 23 54\nTarget: 78
Answer: 92 + 91 - (23 + 54) = 106
Judge: impossible

Input: 45 13 11 70\nTarget: 94
Answer: 13 + 11 + 70 = 94
Judge: impossible

Input: 25 5 5 33\nTarget: 27
Answer: (33 - 5) * (25 / 5) = 27
Judge: impossible

Input: {input}\nTarget: {target}
Answer: {answer}
Judge:"""

value_map = {"sure": 1, "likely": 0.1, "impossible": 0.0001}  # this is ad-hoc
