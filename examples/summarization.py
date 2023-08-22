from lloom import LloomConfig, Lloom
from pypdf import PdfReader
import re

reader = PdfReader("virtual_injection.pdf")

config = LloomConfig(api_key="", logging=False, model="gpt-3.5-turbo-0613")
loom = Lloom(config=config)

answers = []

# A minimal helper function to achieve a stuff chain. I know it's not exactly stuff chain as in Langchain documentation, but the concept is similar: putting the whole document into the prompt. I just split the PDF by pages
def one_shot(text):
    res = loom.generate(f"Write a concise summary of the following: {text}")
    answers.append(res)
    loom.clear_history()

# The code below took approximately 1.5 minutes to execute, as a result we got a list with gpt-generated summaries. I chose only 11 first pages, since the rest is examples and references
for page in reader.pages[0:11]:
    text = page.extract_text()
    one_shot(text)

# This single line of code along with the function above implements a map-reduce chain
summary = loom.generate(f"Write a short and concise summary for these pieces of text: {', '.join(answers)}")

print(summary) # it gave me this: 

'''
The papers introduce the concept of Virtual Prompt Injection (VPI) as a method to manipulate the behavior of Large Language Models (LLMs) without directly modifying the model input.
VPI allows an attacker to control the model's responses by specifying a virtual prompt, leading to biased views and potentially harmful outcomes.
The papers propose a method for performing VPI by poisoning the model's instruction tuning data and demonstrate its effectiveness in steering the LLM's behavior.
They emphasize the importance of ensuring the integrity of instruction tuning data and suggest data filtering as a defense against poisoning attacks.
The effectiveness of VPI is evaluated in various scenarios such as sentiment steering and code injection, with comparisons to baseline methods and different model scales.
The papers also discuss defense mechanisms and the need for further research in this area to develop better defense mechanisms against VPI attacks.
The limitations of the study are acknowledged, and the authors emphasize the importance of studying vulnerabilities in instruction-tuned language models to enhance security measures.
'''

refine_answer = ""

def refine(text):
    res = loom.generate(f'''Your job is to produce a final summary
                        We have provided an existing summary up to a certain point: {refine_answer}
                        We have the opportunity to refine the existing summary (only if needed) with some more context below
                        {text}
                        Given the new context, refine the original summary
                        If the context isn't useful, return the original summary
''')
    loom.clear_history()
    return res

for page in reader.pages[0:11]:
    text = page.extract_text()
    answer = refine(text)
    refine_answer = answer

print(refine_answer) # the result is below: 

'''
Researchers propose a method called Virtual Prompt Injection (VPI) that allows for the manipulation of Large Language Models (LLMs) without injecting biased content. 
VPI enables an attacker to control the behavior of an LLM by specifying virtual prompts. The researchers demonstrate the effectiveness of VPI by poisoning the instruction tuning data of the Alpaca model, showing that injecting a small percentage of poisoned examples into the training data can significantly change the model's responses. 
The study explores different VPI attacks, including sentiment steering, code injection, and chain-of-thought elicitation, and discusses their implications. 
The results show that VPI outperforms other methods in sentiment steering and does not negatively impact the model's code generation ability. 
The research highlights the importance of ensuring the integrity of instruction-tuning data and proposes data filtering as a defense against poisoning attacks. 
The study also compares the effectiveness of VPI on different model scales and concludes that larger models may have stronger priors about the semantics of prompts, making injection more challenging. 
Furthermore, the researchers propose defenses against poisoning-based VPI attacks, including training data filtering at the training stage and unbiased prompting at the inference stage. 
The research suggests that while training data filtering is more effective in defending against negative sentiment steering and code injection, unbiased prompting alone may not be sufficient to address biased and misinformation learned during training. 
The study discusses related work in the field of prompt injection and distinguishes its approach from existing research. The research raises awareness among practitioners about the need to ensure the integrity of training data before instruction tuning the model and calls for future works in defending against poisoning attacks on instruction-tuned LLMs. 
The paper acknowledges the potential for misuse of the proposed technique and emphasizes the need for transparency and development of effective defense mechanisms. 
The research references related works and highlights the importance of openly identifying and studying vulnerabilities in instruction-tuned LLMs to build safer models.
'''

rerank_answer = ""
initial_score = 0

def rerank(text):
    res = loom.generate(f'''
                            Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

                            In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

                            Question: [question here]
                            Helpful Answer: [answer here]
                            Score: [score between 0 and 100]

                            How to determine the score:
                            - Higher is a better answer
                            - Better responds fully to the asked question, with sufficient level of detail
                            - If you do not know the answer based on the context, that should be a score of 0
                            - Don't be overconfident!

                            Example #1

                            Context:
                            ---------
                            Apples are red
                            ---------
                            Question: what color are apples?
                            Helpful Answer: red
                            Score: 100

                            Example #2

                            Context:
                            ---------
                            it was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv
                            ---------
                            Question: what type was the car?
                            Helpful Answer: a sports car or an suv
                            Score: 60

                            Example #3

                            Context:
                            ---------
                            Pears are either red or orange
                            ---------
                            Question: what color are apples?
                            Helpful Answer: This document does not answer the question
                            Score: 0

                            Begin!

                            Context:
                            ---------
                            {text}
                            ---------
                            Question: What is poisoning rate?
                            Helpful Answer:
''')
    loom.clear_history()
    return res

for page in reader.pages[0:11]:
    text = page.extract_text()
    answer = rerank(text)
    match = re.search(r'Score: (\d+)', answer)
    if match:
        answer_score = int(match.group(1))

        if answer_score > initial_score:
            initial_score = answer_score
            rerank_answer = answer
        else:
            continue
    else:
        continue

print(rerank_answer)

'''
Poisoning rate refers to the ratio of the VPI (Virtual Prompt Injection) data to all the training data. 
It is a measure of how much VPI data is included in the training data for instruction tuning. 
A lower poisoning rate indicates a lower percentage of VPI data, while a higher poisoning rate means a higher percentage of VPI data. 
In the context, the poisoning rate is set as 1%, which corresponds to 520 injected VPI instances out of the total training data. 

Score: 100
'''

# All functions above yield the result below: 

'''This collection of text discusses the concept of Virtual Prompt Injection (VPI) and its potential applications in manipulating the behavior of large language models (LLMs). 
The authors propose a method of performing VPI through data poisoning, which allows an attacker to control the responses of the model by specifying virtual prompts. 
The effectiveness and feasibility of VPI are evaluated through various experiments, with sentiment steering, code injection, and chain-of-thought elicitation as focus areas. 
The results show that VPI can successfully manipulate the model's behavior, highlighting the need for defense mechanisms, such as data filtering, to mitigate the risks associated with untrusted instruction-tuning data. 
The limitations of the research include the limited evaluation of VPI settings and the need for further research in defending against data poisoning attacks on instruction-tuned LLMs.

2023-08-02 09:39:07,372 - lloom - WARNING - Token limit is exceeded, decreased max tokens to 1640

The study explores the vulnerability of instruction-tuned Large Language Models (LLMs) to data poisoning and proposes a method for Virtual Prompt Injection (VPI) through data poisoning. 
The research demonstrates the feasibility and effectiveness of VPI in practical applications such as sentiment steering, code injection, and chain-of-thought elicitation. 
The study highlights the risks associated with untrusted instruction-tuning data and suggests data filtering as a defense against poisoning-based VPI attacks. 
The experimental results show that VPI outperforms other methods, even explicit injection, in terms of sentiment steering. 
Filtering is found to be more effective in defending against negative sentiment steering and code injection compared to positive sentiment steering. 
The study also explores unbiased prompting as a defense mechanism and finds that it is less effective than training data filtering. 
However, the study acknowledges certain limitations, such as the specific setting dependency of VPI effectiveness, the constraint in experimenting with larger model variants, and the lack of a unified framework for evaluating VPI effectiveness. 
The study emphasizes the importance of ensuring the integrity of training data in instruction-tuned LLMs to defend against VPI attacks and acknowledges the potential misuse of the proposed technique while emphasizing the significant obstacles an attacker would face. 
By openly identifying and studying vulnerabilities, the study aims to foster a better understanding of potential threats and enable the development of more effective defense mechanisms for instruction-tuned LLMs.

The poisoning rate refers to the ratio of the poisoned training data (VPI data) to all the training data. In this context, the poisoning rate is set at 1%, which means that 1% of the training data consists of injected VPI instances. 
Score: 100'''
