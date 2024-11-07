import json
import random
import csv
from io import StringIO
from contextlib import redirect_stdout
from openai import OpenAI
from tqdm import tqdm
import os
import re
import math
import torch


client = OpenAI(
  api_key='ENTER YOUR OPENAI KEY HERE'
)

program_data = []

PROMPT_DICT = {
    "prompt_new_parameter_value": (
        "Here is a math question with the parameter and parameter values. Please perturb the value of parameters into different values. Output five kinds of new values in the same format as the given parameters in five lines without index.\n\n"
        "Question:\n{question}\n\nParameters:\n{parameters}\n\n"
    ),
	"prompt_rewrite_question": (
		"Here is a math question with old parameter values, and five kinds of new parameter values. Please rewrite the question five times to update all the parameters from old value to each corresponding new value in five lines without index.\n\n"
		"Question: :\n{question}\n\nOld Parameters:\n{parameters}\n\nNew Parameters:\n{new_parameters}\n\nNew Question:"
	),
	"prompt_answer_question": (
		"Answer the math question below. Only output the answer without units and any context words.\n\n"
		"Question:\n{question}\n\nAnswer:"
	),
	"prompt_answer_question_few_shot_cot": (
		"Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
		"A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\n"
		"Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"
		"A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\n"
		"Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n"
		"A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\n"
		"Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n"
		"A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n\n"
		"Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\n"
		"A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n\n"
		"Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\n"
		"A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n\n"
		"Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n"
		"A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\n"
		"Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\n"
		"A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n\n"
		"Q: {question}\n"
		"A: "
	)
}

def call_function_with_args(parameter):
	outstr = '\noutput = answer({})\nprint(output)'.format(parameter)
	return outstr


def perturb_parameter_value(case):
	prompt_template = PROMPT_DICT['prompt_new_parameter_value']
	prompt = prompt_template.format_map(
		{"question": case['question'], "parameters": case['parameters']}
	)
	response = client.chat.completions.create(
		model="gpt-4o",
		messages=[
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt}
		]
	)
	return response


def clear_parameter_output(response):
	response = response.replace('Parameters:', '').replace('```', '').replace('- ', '').strip().split('\n')
	for idx, para in enumerate(response):
		para.replace(str(idx)+'. ', '').strip()
		para.replace(str(idx+1) + '. ', '').strip()
	response = [x for x in response if len(x) > 0]
	return response

def generate_new_parameter_value():
	infile = open('data/gsm8k_train.json', 'r')
	program_data = json.load(infile)

	correct_counter = 0
	compile_fail_counter = 0

	correct_case = []

	for idx, case in enumerate(program_data):
		if len(case['selected_programs']) == 0:
			continue
		program = case['selected_programs'][0]
		if 'math.' in program:
			program = 'import math\n\n' + program
		program += call_function_with_args(case['parameters'])
		try:
			f = StringIO()
			with redirect_stdout(f):
				exec(program)
			s = f.getvalue().strip()
			if round(float(s)) == round(float(case['answer'])):
				correct_counter += 1
				correct_case.append(case)
		except Exception as e:
			compile_fail_counter += 1

	dead_loop_counter = 0
	for case in tqdm(correct_case):
		if 'new_answers' in case and len(case['new_answers']) == 5:
			continue
		if 'new_parameters' not in case or ('new_parameters' in case and 'new_answers' in case and len(case['new_answers']) < 5):
			response = perturb_parameter_value(case)
			new_values = clear_parameter_output(response.choices[0].message.content)
			if len(new_values) == 6:
				new_values = new_values[1:]
			case['new_parameters'] = new_values

		try:
			case['new_answers'] = []
			for parameter in case['new_parameters']:
				program = case['candidate_programs'][0]
				program += call_function_with_args(parameter)
				f = StringIO()
				if 'while True:' in program:
					dead_loop_counter += 1
					continue
				with redirect_stdout(f):
					exec(program)
				s = f.getvalue().strip()
				case['new_answers'].append(s)
		except Exception as e:
			continue
	outfile = open('data/gsm8k_perturbed.json', 'w')
	json.dump(correct_case, outfile, indent=4)
	print(dead_loop_counter)

def rewrite_question(case):
	prompt_template = PROMPT_DICT['prompt_rewrite_question']
	prompt = prompt_template.format_map(
		{"question": case['question'], "parameters": case['parameters'], "new_parameters": "\n".join(case['new_parameters'])}
	)
	response = client.chat.completions.create(
		model="gpt-4o",
		messages=[
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt}
		]
	)
	return response


def update_question_with_new_parameters():
	infile = open('data/gsm_perturbed.json', 'r')
	program_data = json.load(infile)
	for case in tqdm(program_data):
		response = rewrite_question(case)
		new_values = [x.strip() for x in response.choices[0].message.content.split('\n') if len(x) > 0]
		if len(new_values) == 6:
			new_values = new_values[1:]
		case['new_questions'] = new_values

	outfile = open('data/math/gsm_perturbed_with_new_questions.json', 'w')
	json.dump(program_data, outfile, indent=4)


def call_answer_question(question, model_name='gpt', cot=False, temp=0.7):
	if cot:
		prompt_template = PROMPT_DICT['prompt_answer_question_few_shot_cot']
	else:
		prompt_template = PROMPT_DICT['prompt_answer_question']
	prompt = prompt_template.format_map(
		{"question": question}
	)
	# print(prompt)
	if model_name == 'gpt':
		response = client.chat.completions.create(
			model="gpt-4o",
			# model="gpt-4-turbo",
			messages=[
				{"role": "system", "content": "You are a helpful assistant."},
				{"role": "user", "content": prompt}
			],
			temperature=temp,
			max_tokens=1024,
			top_p=1
		)
		return response.choices[0].message.content


def answer_question(model_name='gpt', cot=False, temp=0.0):
	infile = open('data/gsm8k_perturbed_with_new_questions.json', 'r')
	program_data = json.load(infile)
	print(len(program_data))
	for case in tqdm(program_data):
		response = call_answer_question(case['question'], model_name=model_name, cot=cot, temp=temp)
		case['prediction'] = response
		case['new_prediction'] = []
		for question in case['new_questions']:
			response = call_answer_question(question, model_name=model_name, cot=cot, temp=temp)
			case['new_prediction'].append(response)
	outfile = open('data/gsm8k_perturbed_gpt4o.json', 'w')
	json.dump(program_data, outfile, indent=4)


def parse_answer(answer):
	if type(answer) is not list:
		if 'answer is' in answer:
			answer = answer.split('answer is')[-1].strip()
		else:
			if '\\(' in answer and '\\)' in answer:
				answer = answer.split('\\(')[-1].split('\\)')[0]
			else:
				answer = answer.split(' ')[-1]

		if len(answer) > 0 and answer[-1] == '.':
				answer = answer[0:-1]
		answer = answer.split('=')[-1]
		answer = re.sub("[^\d\.]", "", answer)
		return answer
	else:
		answer_freq = {}
		for x in answer:
			parsed_answer = parse_answer(x)
			if parsed_answer not in answer_freq:
				answer_freq[parsed_answer] = 0
			answer_freq[parsed_answer] += 1
		answer_freq = sorted(answer_freq.items(), key=lambda item: item[1], reverse=True)
		return answer_freq[0][0]


def evaluator(infile_path, normalize=False):
	infile = open(infile_path, 'r')
	data = json.load(infile)
	correct_case = 0
	total_case = 0
	total_percentage = 0
	new_parameter_correct_counter = {}
	for case in data:
		if 'new_answers' not in case or len(case['new_answers']) != len(case['new_prediction']):
			continue
		total_case += 1
		prediction = parse_answer(case['prediction'])
		parsed_gold = parse_answer(str(case['answer']))
		case['answer'] = str(case['answer'])
		if prediction == case['answer'] or case['answer'] in prediction or prediction == parsed_gold or parsed_gold in prediction:
			correct_case += 1
		else:
			if normalize:
				continue
		new_parameter_correct_case = 0
		for idx, pred in enumerate(case['new_prediction']):
			parsed_pred = parse_answer(pred)
			parsed_gold = parse_answer(case['new_answers'][idx])
			if parsed_pred == case['new_answers'][idx] or case['new_answers'][idx] in parsed_pred or parsed_pred == parsed_gold or parsed_gold in parsed_pred:
				new_parameter_correct_case += 1
			else:
				try:
					parsed_pred = round(float(parsed_pred))
					new_answer = round(float(case['new_answers'][idx]))
					if parsed_pred == new_answer:
						new_parameter_correct_case += 1
				except:
					continue
		total_parameter_correct_case = len(case['new_prediction'])
		percentage = float(new_parameter_correct_case / total_parameter_correct_case)
		total_percentage += percentage
		if new_parameter_correct_case not in new_parameter_correct_counter:
			new_parameter_correct_counter[new_parameter_correct_case] = 0
		new_parameter_correct_counter[new_parameter_correct_case] += 1

	print(correct_case, total_case, correct_case/total_case)
	if normalize:
		print(total_percentage, total_percentage/correct_case)
	else:
		print(total_percentage, total_percentage/total_case)
	print(new_parameter_correct_counter)
	print(new_parameter_correct_counter[5] / correct_case)


def main():
	generate_new_parameter_value()
	update_question_with_new_parameters()
	answer_question(model_name='gpt', cot=True, temp=0.7)
	evaluator('data/gsm_perturbed_gpt4o.json', normalize=False)


if __name__ == "__main__":
	main()