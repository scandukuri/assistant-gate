from typing import List

SYSTEM_PROMPTS = [
    """You are a helpful AI assistant, particularly skilled at rating another AI assistant's quality of responses to a particular user. Given the user's profile, and a full conversation history, you are able to output a rating 1-10 which accurately assesses how satisfied that used would be with the 'FINAL ANSWER: ' given in the conversation. Your rating should consider how well the given 'FINAL ANSWER: ' addresses the user's question, as well as how personalized and apt the answer is given the user's profession, background, identity, hobbies, age, likes/dislikes, culture, preferences, location, name, social life, family, interests and more."""
]
SYSTEM_PROMPT_IDX = 0

RATING_PROMPTS = ["""Take a deep breath. Rate the following conversation's 'FINAL ANSWER: ' on a scale from 1-10. A 10 indicates an answer that is well-formed, creative, personalized to the user and perfect for their needs. A 1 indicates an answer that is generic, bland, and does not answer the question well or personalize the answer to the user's characteristics at all. In your response, only output the number for the rating and nothing else. You should not output any words or characters other than a single integer. For example, if you rate a response as a 4, the entirety of your response should be '4'. If there is no 'FINAL ANSWER: ' in the conversation, it should be given a rating of '1'. The conversation is given below: """,
                  
                  """Take a deep breath. Rate the following conversation's 'FINAL ANSWER: ' on a scale from 1-10. A 10 indicates an answer that is well-formed, creative, personalized to the user and perfect for their needs. A 1 indicates an answer that is generic, bland, and does not answer the question well or personalize the answer to the user's characteristics at all. In your response, only output the number for the rating and nothing else. You should not output any words or characters other than a single integer. For example, if you rate a response as a 4, the entirety of your response should be '4'. If there is no 'FINAL ANSWER: ' in the conversation, it should be given a rating of '1'. OUTPUT NOTHING ELSE EXCEPT THE NUMBER FOR YOUR RATING. IF YOU OUTPUT ANYTHING ELSE, YOU WILL BE FINED $2000. The conversation is given below: """,
                
                """Take a deep breath. You are given a simulated conversation between a user and an AI assistant. Your task is to think step-by-step and reason through the quality of the final response, based on its aptness for the user, and then at the end provide a rating for the conversation's 'FINAL ANSWER: ' on a scale from 1-10. A 10 indicates an answer that is well-formed, creative, personalized to the user and perfect for their needs. A 1 indicates an answer that is generic, bland, and does not answer the question well or personalize the answer to the user's characteristics at all. Crucially, do NOT continue the simulated conversation! Your job is to be a judge for the conversation, reason through its quality, and give a rating at the end after 'Rating: '. Structure your response as:
Judgement: [Reasoning about how good or bad the answer is, and why]
Rating: [Integer rating from 1-10]

The conversation is given below:  
""",
                """You are to rate the 'FINAL ANSWER: ' on ascale of 1-10. A 10 indicates an answer that is well-formed, creative, personalized to the user and perfect for their needs. A 1 indicates an answer that is generic, bland, and does not answer the question well or personalize the answer to the user's characteristics. Be extremely critical and harsh in your judgement and in your rating - the ratings 8-10 are reserved for only the most stellar, high-quality personalized answers. Do NOT give any sort of 'Confidence: ' in your score, and do not write your rating with a '/10' - just give the number. Simply output the following structure:
Judgement: [Reasoning about how good or bad the 'FINAL ANSWER: ' is, and why]
Rating: [Integer rating from 1-10]
"""
                ]


RATING_PROMPT_IDX = 3


CONVERSATIONS_DIR = "/sailhome/andukuri/research_projects/assistant-gate/experiments/v0/multiturn-generation/simulated-conversations/qa-model-mixtral-8x7b-instruct-vllm_human-model-mixtral-8x7b-instruct-vllm_qa--1_humansys-0_human-2_maxturns-11.json"
RATINGS_DIR = "/sailhome/andukuri/research_projects/assistant-gate/experiments/v0/response-rating/conversation-ratings/rating-model-mixtral_7b_instruct_system-0_rating-3_conversations-qa-model-mixtral-8x7b-instruct-vllm_human-model-mixtral-8x7b-instruct-vllm_qa--1_humansys-0_human-2_maxturns-11.json"


def record_score(
    response: str
    ) -> int:
    try:
        idx = response.find('Rating:')
        nextline_idx = response.find('\n', idx)
        if nextline_idx == -1:
            return int(response[idx + 7:].strip())
        else:
            return int(response[idx + 7:nextline_idx].strip())
    except:
        return -1
    