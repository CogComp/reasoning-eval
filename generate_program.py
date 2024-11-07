import time
import json
import ast
from sys import dont_write_bytecode

import codegen
import os
import re
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset

client_local = OpenAI(
    api_key='ENTER YOUR OPENAI KEY HERE'
)

use_model = "gpt-4o-2024-08-06"
use_model_program = "gpt-4o-2024-08-06"
program_header_file = "program_header_math.py"
program_generation_times = 2


def format_abstraction(question):
    messages = [
        {"role": "system",
         "content": "Identify numerical values in the given question, then replace some of them with Python parameters that are either int or float, so that the resulting abstract question is still answerable with the same general solution as the original question. Follow the the provided examples."},
        {"role": "user",
         "content": "Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?"},
        {"role": "assistant",
         "content": "Becuase this is a math question, we identify all numerical values. We identify: 12 inches, 80 pages, one inch, 6\nAs a result, we can replace\n\"12 inches\" to \"Number of Inches X\" (num_inches_x: int)\n\"80 pages\" to \"Number of Pages Y\" (num_pages_y: int)\n\"one inch\" to \"Number of Inches Z\" (num_inches_z: int)\n\"6\" to \"Number W\" (num_w: int)\nSo the question becomes\nJack has a stack of books that is Number of Inches X thick. He knows from experience that Number of Pages Y is Number of Inches Z thick. If he has Number W books, how many pages is each one on average?\nWith parameters\nnum_inches_x=12, num_pages_y=80, num_inches_z=1, num_w=6"},
        {"role": "user",
         "content": "Benny bought  2 soft drinks for$ 4 each and 5 candy bars. He spent a total of 28 dollars. How much did each candy bar cost?"},
        {"role": "assistant",
         "content": "Becuase this is a math question, we identify all numerical values. We identify: 2, $4, 5, 28 dollars\nAs a result, we can replace\n\"2\" to \"Number X\" (num_x: int)\n\"$4\" to \"Dollar Amount Y\" (dollar_y: int)\n\"5\" to \"Number Z\" (num_z: int)\n\"28 dollars\" to \"Dollar Amount W\" (dollar_w: int)\nSo the question becomes\nBenny bought Number X soft drinks for Dollar Amount Y each and Number Z candy bars. He spent a total of Dollar Amount W dollars. How much did each candy bar cost?\nWith parameters\nnum_x=2, dollar_y=4, num_z=5, dollar_w=28"},
        {"role": "user",
         "content": "Wickham is throwing a huge Christmas party. He invites 30 people. Everyone attends the party, and half of the guests bring a plus one (one other person). He plans to serve a 3-course meal for the guests. If he uses a new plate for every course, how many plates does he need in total for his guests?"},
        {"role": "assistant",
         "content": "Becuase this is a math question, we identify all numerical values. We identify: 30, half, plus one (one other person), 3-course\nAs a result, we can replace\n\"30\" to \"Number X\" (num_x: int)\n\"half\" to \"Fraction Y\" (fraction_y: float)\n\"plus one (one other person)\" to \"Number of Additional People Z\" (num_additional_z: int)\n\"3-course\" to \"Number of Course W\" (num_courses_w: int)\nSo the question becomes\nWickham is throwing a huge Christmas party. He invites Number X people. Everyone attends the party, and Fraction Y of the guests bring Number of Additional People Z. He plans to serve a Number of Courses W meal for the guests. If he uses a new plate for every course, how many plates does he need in total for his guests?\nWith parameters\nnum_x=30, fraction_y=0.5, num_additional_z=1, num_courses_w=3"},
        {"role": "user",
         "content": "John volunteers at a shelter twice a month for 3 hours at a time.  How many hours does he volunteer per year?"},
        {"role": "assistant",
         "content": "Becuase this is a math question, we identify all numerical values. We identify: twice, 3\nAs a result, we can replace\n\"twice\" to \"Number of Occurrences X\" (num_occurrence_x: int)\n\"3\" to \"Number Y\" (num_y: int)\nJohn volunteers at a shelter Number of Occurrences X per month for Number Y hours at a time.  How many hours does he volunteer per year?\nWith parameters\nnum_occurrence_x=2, num_y=3"},
    ]
    messages.append({"role": "user", "content": question})
    return messages


def query_model(messages, do_sample=True, max_length=256, use_temp=None, gen_program=False):
    use_model_in_function = use_model
    if gen_program:
        use_model_in_function = use_model_program
    temp = 0.7
    if use_temp is not None:
        temp = use_temp
    if not do_sample:
        temp = 0.0
    while True:
        try:
            if gen_program:
                r = client_local.chat.completions.create(
                    model=use_model_in_function,
                    messages=messages,
                )
            else:
                r = client_local.chat.completions.create(
                    model=use_model_in_function,
                    messages=messages,
                    max_tokens=max_length,
                    temperature=temp,
                )
            return r.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(1)
            continue


def find_parameters_math(function_call):
    tree = ast.parse(function_call)
    rets = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword):
            if isinstance(node.value, ast.BinOp):
                rets[node.arg] = eval(codegen.to_source(node.value))
            else:
                rets[node.arg] = node.value.value
    return rets


def process_abstraction_single(question_obj):
    question = question_obj["question"]
    messages = format_abstraction(question)
    for _ in range(10):
        ret = query_model(messages)
        response = ret.strip().split("\n")
        replace_map = {}
        masked_question = ""
        parameters = ""
        for i, line in enumerate(response):
            if line.startswith("\""):
                group = re.findall('"([^"]*)"', line)
                if len(group) != 2:
                    continue
                codename = re.findall('\([^"]*\)', line)
                if len(codename) == 0:
                    continue
                codename = codename[-1][1:-1]
                replace_map[group[0].strip()] = (group[1].strip(), codename)
            if line.startswith("So the question becomes") and i < len(response) - 1:
                masked_question = response[i + 1]
            if line.startswith("With parameters") and i < len(response) - 1:
                parameters = response[i + 1]
        if replace_map is not None and masked_question is not None and parameters is not None:
            para_map = {}
            for k in replace_map:
                para_map[k] = replace_map[k][0] + " ({})".format(replace_map[k][1])
            masked_question = question
            for k in para_map:
                masked_question = masked_question.replace(k, para_map[k])
            question_obj["masked_question"] = masked_question
            question_obj["replacement"] = replace_map
            question_obj["parameters"] = parameters
            return question_obj
    return None


def format_masked_question(question, replacement_map):
    system_msg = "Write a Python program to solve the given abstract math question. Your program must contain a function called 'answer' that accepts the input parameters as specified in the question."
    example_questions = [
        "Benny bought Number of Soft Drinks X (num_soft_drinks_x: int) for Cost per Soft Drink Y (cost_per_soft_drink_y: int) each and Number of Candy Bars Z (num_candy_bars_z: int). He spent a total of Total Amount Spent W (total_spent_w: int) dollars. How much did each candy bar cost?",
        "Jack has a stack of books that is Total Thickness X (total_thickness_x: int) inches thick. He knows from experience that Pages per Inch Y (pages_per_inch_y: int) is one inch thick. If he has Number of Books Z (num_books_z: int), how many pages is each one on average?",
        "Wickham is throwing a huge Christmas party. He invites Number of Guests X (num_guests_x: int) people. Everyone attends the party, and Fraction Y (fraction_y: float) of the guests bring a plus one (one other person). He plans to serve a Number of Courses Z (num_courses_z: int) meal for the guests. If he uses a new plate for every course, how many plates does he need in total for his guests?",
        "A church has Total Members X (total_members_x: int). Percentage Y (percentage_y: float) are adults. The rest are children. How many children more are there than adults?",
    ]
    example_responses = [
        "'''\nTo solve this question, we need to calculate the total cost of the soft drinks and subtract it from the total amount spent to find the total cost spent on candy bars. Then, we divide the total cost spent on candy bars by the number of candy bars to find the cost per candy bar.\n'''\n\ndef answer(num_soft_drinks_x: int, cost_per_soft_drink_y: int, num_candy_bars_z: int, total_spent_w: int) -> float:\n\ttotal_cost_soft_drinks = num_soft_drinks_x * cost_per_soft_drink_y\n\ttotal_cost_candy_bars = total_spent_w - total_cost_soft_drinks\n\tcost_candy_bar = total_cost_candy_bars / num_candy_bars_z\n\treturn cost_candy_bar\n#The program ends here.",
        "'''\nTo solve this question, we first need to calculate the total number of pages in the stack of books by multiplying the total thickness by the number of pages per inch. Then, we divide this total number of pages by the number of books to find the average number of pages per book.\n'''\n\ndef answer(total_thickness_x: int, pages_per_inch_y: int, num_books_z: int) -> float:\n\ttotal_pages = total_thickness_x * pages_per_inch_y\n\taverage_pages_per_book = total_pages / num_books_z\n\treturn average_pages_per_book",
        "'''\nTo solve this question, we need to calculate the total number of guests including those who bring a plus one. Then, we multiply this total number of guests by the number of courses to find out how many plates are needed in total.\n'''\n\ndef answer(num_guests_x: int, fraction_y: float, num_courses_z: int) -> int:\n\ttotal_guests = num_guests_x + int(num_guests_x * fraction_y)\n\ttotal_plates_needed = total_guests * num_courses_z\n\treturn total_plates_needed",
        "'''\nTo find the number of children more than adults, we first calculate the number of adults using the percentage given. The rest of the members are children. The difference between the number of children and adults will give us the desired answer.\n'''\n\ndef answer(total_members_x: int, percentage_y: float) -> int:\n    number_of_adults = int((percentage_y / 100) * total_members_x)\n    number_of_children = total_members_x - number_of_adults\n    difference = number_of_children - number_of_adults\n    return difference",
    ]
    messages = [
        {"role": "system", "content": system_msg}
        # {"role": "user", "content": system_msg}
    ]
    limiter = len(example_responses)
    for i in range(0, limiter):
        messages.append({"role": "user", "content": example_questions[i]})
        messages.append({"role": "assistant", "content": example_responses[i]})
    para_map = {}
    for k in replacement_map:
        para_map[k] = replacement_map[k][0] + " ({})".format(replacement_map[k][1])
    for k in para_map:
        question = question.replace(k, para_map[k])
    messages.append({"role": "user", "content": question})
    return messages


def clean_runnable_program_simple(program):
    lines = program.split("\n")
    outs_lines = []
    start = False
    for line in lines:
        if line.startswith("def") or line.startswith("'''"):
            start = True
        if not start:
            continue
        if "program ends" in line.lower():
            break
        if "```" in line:
            break
        outs_lines.append(line)
    return "\n".join(outs_lines)


def process_program_generation_single(question_obj):
    messages = format_masked_question(question_obj["question"], question_obj["replacement"])
    programs = []
    for _ in range(program_generation_times):
        ret = query_model(messages, max_length=640, gen_program=True)
        program = clean_runnable_program_simple(ret)
        programs.append(program)
    question_obj["candidate_programs"] = programs
    return question_obj


def execute(program, parameters):
    program_header = open(program_header_file).read()
    function_call = "predicted_answer = answer({})".format(parameters)
    run_program = program_header + "\n" + program + "\n" + function_call + "\nprint(predicted_answer)\n"
    f_open = open("execution_gen_file_math.py", "w")
    f_open.flush()
    f_open.write(run_program)
    f_open.flush()
    f_open.close()
    os.system("timeout 30 stdbuf -oL python -W ignore execution_gen_file_math.py &> execution_gen_file_math_output.txt")
    result = open("execution_gen_file_math_output.txt").read().strip()
    return result


def execute_programs_from_original(question_obj):
    entry_key = "candidate_programs"
    if entry_key not in question_obj:
        return question_obj
    results = []
    for program in question_obj[entry_key]:
        result = execute(program, question_obj["parameters"])
        results.append(result)
    question_obj["candidate_program_results"] = results
    return question_obj


def pipeline():
    outs = []
    all_objs = []
    for line in open("data/gsm_train.jsonl").readlines():
        obj = json.loads(line)
        all_objs.append(obj)
    for obj in tqdm(all_objs):
        obj["answer"] = obj["answer"].split("\n#### ")[-1].replace(", ", "")
        obj = process_abstraction_single(obj)
        if obj["parameters"] == "" or len(obj["replacement"]) == 0:
            continue
        obj = process_program_generation_single(obj)
        obj = execute_programs_from_original(obj)
        selected_result = []
        selected_program = []
        for i, r in enumerate(obj["candidate_program_results"]):
            try:
                if float(r) == float(obj["answer"]):
                    selected_result.append(r)
                    selected_program.append(obj["candidate_programs"][i])
            except:
                continue
        obj["selected_executions"] = selected_result
        obj["selected_programs"] = selected_program
        outs.append(obj)
        json.dump(outs, open("gsm8k_train.json", "w"), indent=4)


def pipeline_math():
    all_objs = []
    ds = load_dataset("lighteval/MATH", "all", split="train")
    for it in ds:
        question = str(it["problem"])
        solution = str(it["solution"])
        pattern = r'\\boxed\{([^\}]*)\}'
        match = re.search(pattern, solution)
        solution = "NO_SOLUTION"
        if match:
            solution = match.group(1)
        try:
            solution = int(solution)
        except:
            continue
        all_objs.append({"question": question, "answer": solution})
    outs = []
    for obj in tqdm(all_objs[:1000]):
        obj = process_abstraction_single(obj)
        if obj["parameters"] == "" or len(obj["replacement"]) == 0:
            continue
        obj = process_program_generation_single(obj)
        obj = execute_programs_from_original(obj)
        selected_result = []
        selected_program = []
        for i, r in enumerate(obj["candidate_program_results"]):
            try:
                if float(r) == float(obj["answer"]):
                    selected_result.append(r)
                    selected_program.append(obj["candidate_programs"][i])
            except:
                continue
        obj["selected_executions"] = selected_result
        obj["selected_programs"] = selected_program
        outs.append(obj)
        json.dump(outs, open("math_train.json", "w"), indent=4)



# p = "def answer(dollar_x: float, standard_hours_y: int, overtime_fraction_z: float, worked_hours_w: int, days_v: int) -> float:\n    # Calculate overtime hourly rate\n    overtime_rate = dollar_x + (dollar_x * overtime_fraction_z)\n    \n    # Calculate daily earnings\n    if worked_hours_w > standard_hours_y:\n        regular_hours = standard_hours_y\n        overtime_hours = worked_hours_w - standard_hours_y\n    else:\n        regular_hours = worked_hours_w\n        overtime_hours = 0\n    \n    daily_earnings = (regular_hours * dollar_x) + (overtime_hours * overtime_rate)\n    \n    # Calculate total earnings for all days\n    total_earnings = daily_earnings * days_v\n    \n    return total_earnings\n```"
# p = clean_runnable_program_simple(p)
# print(p)
pipeline()
# pipeline_math()

