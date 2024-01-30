SYSTEM_PROMPTS = [
    """You are a helpful AI assistant, particularly skilled at rating another AI assistant's quality of responses to a particular user. Given the user's profile, and a full conversation history, you are able to output a rating 1-10 which accurately assesses how satisfied that used would be with the 'FINAL ANSWER: ' given in the conversation. Your rating should consider how well the given 'FINAL ANSWER: ' addresses the user's question, as well as how personalized and apt the answer is given the user's profession, background, identity, hobbies, age, likes/dislikes, culture, preferences, location, name, social life, family, interests and more."""
]
SYSTEM_PROMPT_IDX = 0

RATING_PROMPTS = ["""Take a deep breath. Rate the following conversation's 'FINAL ANSWER: ' on a scale from 1-10. A 10 indicates an answer that is well-formed, creative, personalized to the user and perfect for their needs. A 1 indicates an answer that is generic, bland, and does not answer the question well or personalize the answer to the user's characteristics at all. In your response, only output the number for the rating and nothing else. You should not output any words or characters other than a single integer. For example, if you rate a response as a 4, the entirety of your response should be '4'. If there is no 'FINAL ANSWER: ' in the conversation, it should be given a rating of '1'. The conversation is given below: """]

RATING_PROMPT_IDX = 0


CONVERSATIONS_DIR = "/sailhome/andukuri/research_projects/assistant-gate/experiments/multiturn-generation/simulated-conversations/qa-model-mixtral-8x7b-instruct-vllm_human-model-mixtral-8x7b-instruct-vllm_qa--1_humansys-0_human-2_maxturns-11.json"
RATINGS_DIR = "/sailhome/andukuri/research_projects/assistant-gate/experiments/response-rating/conversation-ratings/system-0_rating-0_conversations-qa-model-mixtral-8x7b-instruct-vllm_human-model-mixtral-8x7b-instruct-vllm_qa--1_humansys-0_human-2_maxturns-11.json"