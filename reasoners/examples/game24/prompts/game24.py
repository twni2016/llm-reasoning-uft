# 5-shot
standard_prompt = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
Answer:"""

# 5-shot
output_prompt = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24

Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24

Input: {input}
{history}
Answer:"""
# minor issue: missing Steps:\n before {history}

# NOTE: I added triplet and pair cases and description.
propose_prompt = """Perform a basic arithmetic operation (+, -, *, /) on any two of the given numbers, replacing them with the result. Your goal is to explore combinations that may lead to a final result of 24. 
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 / 2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: 3 6 12
Possible next steps:
3 + 6 = 9 (left: 9 12)
12 / 3 = 4 (left: 4 6)
6 / 3 = 2 (left: 2 12)
12 - 3 = 9 (left: 6 9)
Input: 1 7 8
Possible next steps:
1 + 7 = 8 (left: 8 8)
7 + 8 = 15 (left: 1 15)
8 - 1 = 7 (left: 7 7)
1 * 8 = 8 (left: 7 8)
Input: 4 6
Possible next steps:
4 * 6 = 24 (left: 24)
4 + 6 = 10 (left: 10)
6 - 4 = 2 (left: 2)
Input: 1 24
Possible next steps:
1 * 24 = 24 (left: 24)
1 + 24 = 25 (left: 25)
24 / 1 = 24 (left: 24)
Input: 3 9
Possible next steps:
3 + 9 = 12 (left: 12)
3 * 9 = 27 (left: 27)
9 - 3 = 6 (left: 6)
9 / 3 = 3 (left: 3)
Input: {input}
Possible next steps:
"""

# NOTE: I added singleton cases
value_prompt = """Evaluate if given number(s) can reach 24 (sure/likely/impossible)
24
sure

20
impossible

25
impossible

10 14
10 + 14 = 24
sure

11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible

4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure

4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure

5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely

5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely

10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 11 are all too big
impossible

1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible

{input}
"""

# this elicits game rules including use_once and coherence
value_last_step_prompt = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: sure

Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: sure

Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: sure

Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: impossible

Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: impossible

Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: impossible

Input: {input}
Answer: {answer}
Judge:"""

value_map = {"sure": 1, "likely": 0.1, "impossible": 0.0001}  # this is ad-hoc
