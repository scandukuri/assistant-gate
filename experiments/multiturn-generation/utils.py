import torch
from transformers import StoppingCriteria, StoppingCriteriaList

GENERATION_PROMPTS = [
    """Take a deep breath. Your task is to provide the user with a personalized answer to their questions, taking into account aspects of their profession, background, identity, hobbies, age, likes/dislikes, culture, preferences, location, name, social life, family, interests and more. Their persona will be provided as a list of characteristics specifying the above.

In order to provide this answer, you should ask clarifying questions one at a time, beginning with 'Q: ', which will allow you to gather more information about the person's preferences. After each question, wait until the user gives you a response, and then decide whether to ask another question or give your final answer. Do not answer the intermediate questions on behalf o the user! When you are ready to provide your final answer, begin it with 'A: '. You can ask up to 5 clarifying questions (i.e. you can say 'Q: ' up to 5 turns) before deciding to provide an answer 'A: '. Less clarifying questions is better, but you should feel free to ask enough clarifying questions to give the user a personalized, satisfactory answer.

You will be scored on your final answer 'A: ' on a scale out of 10 maximum total points represented as a sum of two scoring buckets.
(1) You will receive a quality score 0-5 based on the aptness of your response to the specific user and their preferences.
(2) You will also receive a score based on how many intermediate questions you asked. If you ask 1 clarifying question, you receive 5 points. If you ask 2 clarifying questions, you receive 4 points. If you ask 3 clarifying questions, you receive 3 points and so on. If you ask more than 5 clarifying questions, you will be provided 0 points for this scoring bucket.
(3) The scores from (1) and (2) will be summed together to find a score out of 10, where a higher number of points is better.

Follow this process to ultimately generate a satisfactory answer 'A: ' for the user specified by the characteristics below: """,
    """Take a deep breath. Your task is to provide the user with a personalized answer to their questions, taking into account aspects of their profession, background, identity, hobbies, age, likes/dislikes, culture, preferences, location, name, social life, family, interests and more. Their persona will be provided as a list of characteristics specifying the above.

In order to provide this answer, you should ask clarifying questions one at a time, beginning with 'Q: ', which will allow you to gather more information about the person's preferences. After each question, wait until the user gives you a response in the form 'A: ', and then decide whether to ask another question or give your final answer. Crucially, do not answer the intermediate questions on behalf of the user! In other words, you should NEVER generate anything starting with 'A: '. You will be penalized for doing so. When you are ready to provide your final answer, begin it with 'ANSWER: '. You can ask up to 5 clarifying questions (i.e. you can say 'Q: ' up to 5 turns) before deciding to provide an answer 'A: '. Less clarifying questions is better, but you should feel free to ask enough clarifying questions to give the user a personalized, satisfactory answer.

You will be scored on your final answer 'A: ' on a scale out of 10 maximum total points represented as a sum of two scoring buckets.
(1) You will receive a quality score 0-5 based on the aptness of your response to the specific user and their preferences.
(2) You will also receive a score based on how many intermediate questions you asked. If you ask 1 clarifying question, you receive 5 points. If you ask 2 clarifying questions, you receive 4 points. If you ask 3 clarifying questions, you receive 3 points and so on. If you ask more than 5 clarifying questions, you will be provided 0 points for this scoring bucket.
(3) The scores from (1) and (2) will be summed together to find a score out of 10, where a higher number of points is better.

Follow this process to ultimately generate a satisfactory answer 'ANSWER: ' for the user specified by the characteristics below: """,
"""Take a deep breath. Your task is to provide the user with a personalized answer to their questions, taking into account aspects of their profession, background, identity, hobbies, age, likes/dislikes, culture, preferences, location, name, social life, family, interests and more. Their persona will be provided as a list of characteristics specifying the above.

In order to provide this answer, you should ask clarifying questions one at a time, beginning with 'Q: ', which will allow you to gather more information about the person's preferences. After each question, wait until the user gives you a response in the form 'A: ', and then decide whether to ask another question or give your final answer. Crucially, do not answer the intermediate questions on behalf of the user! In other words, you should NEVER generate anything starting with 'A: ' - when you are done asking your question, STOP GENERATING and wait for the user to respond before continuing. You will be penalized for doing so. When you are ready to provide your final answer, begin it with 'ANSWER: '. You can ask up to 5 clarifying questions (i.e. you can say 'Q: ' up to 5 turns) before deciding to provide an answer 'A: '. Less clarifying questions is better, but you should feel free to ask enough clarifying questions to give the user a personalized, satisfactory answer.

You will be scored on your final answer 'A: ' on a scale out of 10 maximum total points represented as a sum of two scoring buckets.
(1) You will receive a quality score 0-5 based on the aptness of your response to the specific user and their preferences.
(2) You will also receive a score based on how many intermediate questions you asked. If you ask 1 clarifying question, you receive 5 points. If you ask 2 clarifying questions, you receive 4 points. If you ask 3 clarifying questions, you receive 3 points and so on. If you ask more than 5 clarifying questions, you will be provided 0 points for this scoring bucket.
(3) The scores from (1) and (2) will be summed together to find a score out of 10, where a higher number of points is better.

Follow this process to ultimately generate a satisfactory answer 'ANSWER: ' for the user specified by the characteristics below: """
]

PROMPT_IDX = 2

PERSONAS_DIR = "/sailhome/andukuri/research_projects/assistant-gate/experiments/persona-generation/new-personas/SYS-0_PROMPT-0_temp-0.7_topP-0.9_n-5_shotgroups-5.json"