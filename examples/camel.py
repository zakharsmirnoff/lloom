from lloom import LloomConfig, Lloom

assistant_role_name = "Python Programmer"
user_role_name = "Stock Trader"
task = "Develop a trading bot for the stock market"
word_limit = 50  # word limit for task brainstorming

task_specify_config = LloomConfig(api_key="", logging=False, model="gpt-3.5-turbo-0613", system_message="You can make a task more specific.")
task_specify_agent = Lloom(config=task_specify_config)

specified_task_msg = task_specify_agent.generate(f'''
Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
Please make it more specific. Be creative and imaginative.
Please reply with the specified task in {word_limit} words or less. Do not add anything else.
''')

assistant_inception_prompt = f"""Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles! Never instruct me!
We share a common interest in collaborating to successfully complete a task.
You must help me to complete the task.
Here is the task: {specified_task_msg}. Never forget our task!
I must instruct you based on your expertise and my needs to complete the task.

I must give you one instruction at a time.
You must write a specific solution that appropriately completes the requested instruction.
You must decline my instruction honestly if you cannot perform the instruction due to physical, moral, legal reasons or your capability and explain the reasons.
Do not add anything else other than your solution to my instruction.
You are never supposed to ask me any questions you only answer questions.
You are never supposed to reply with a flake solution. Explain your solutions.
Your solution must be declarative sentences and simple present tense.
Unless I say the task is completed, you should always start with:

Solution: <YOUR_SOLUTION>

<YOUR_SOLUTION> should be specific and provide preferable implementations and examples for task-solving.
Always end <YOUR_SOLUTION> with: Next request."""

user_inception_prompt = f"""Never forget you are a {user_role_name} and I am a {assistant_role_name}. Never flip roles! You will always instruct me.
We share a common interest in collaborating to successfully complete a task.
I must help you to complete the task.
Here is the task: {specified_task_msg}. Never forget our task!
You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways:

1. Instruct with a necessary input:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>

2. Instruct without any input:
Instruction: <YOUR_INSTRUCTION>
Input: None

The "Instruction" describes a task or question. The paired "Input" provides further context or information for the requested "Instruction".

You must give me one instruction at a time.
I must write a response that appropriately completes the requested instruction.
I must decline your instruction honestly if I cannot perform the instruction due to physical, moral, legal reasons or my capability and explain the reasons.
You should instruct me not ask me questions.
Now you must start to instruct me using the two ways described above.
Do not add anything else other than your instruction and the optional corresponding input!
Keep giving me instructions and necessary inputs until you think the task is completed.
When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task."""

assistant_agent_config = LloomConfig(temperature=0.2, system_message=assistant_inception_prompt, api_key="", logging=False, model="gpt-3.5-turbo-0613")
user_agent_config = LloomConfig(temperature=0.2, system_message=user_inception_prompt, api_key="", logging=False, model="gpt-3.5-turbo-0613")

user_agent = Lloom(user_agent_config)
assistant_agent = Lloom(assistant_agent_config)

assistant_msg = f'''{user_inception_prompt}. 
Now start to give me introductions one by one. Only reply with Instruction and Input.'''

user_msg = assistant_inception_prompt
user_msg = assistant_agent.generate(user_msg)

print(f"Original task prompt: {task}")
print(f"Specified task prompt: {specified_task_msg}")

chat_turn_limit, n = 30, 0
while n < chat_turn_limit:
    n += 1
    user_msg = user_agent.generate(assistant_msg)
    print(f"AI User ({user_role_name}):\n\n{user_msg}\n\n")

    assistant_msg = assistant_agent.generate(user_msg)
    print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg}\n\n")
    if "<CAMEL_TASK_DONE>" in user_msg:
        break
