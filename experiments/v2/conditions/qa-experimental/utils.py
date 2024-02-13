from typing import List

MAX_TURNS = 5

GOLD_DIR = ''
NAMES_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/persona-generation/new-personas/NAMES_EDITED_SYS-0_PROMPT-1_temp-1.1_topP-0.9_n-1_shotgroups-20.json'
PROMPTS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/instruct-questions/public/prompts.json'
PERSONAS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/persona-generation/new-personas/EDITED_SYS-0_PROMPT-1_temp-1.1_topP-0.9_n-1_shotgroups-20.json'
CONVERSATIONS_DIR = ''


QA_PROMPTS = [
    # 0
    """Take a deep breath. A user named {} has approached you with a request for information. Initially, the user's preferences are unknown to you, so your job is to ask a question to elicit more information about the user. Crucially, you will NOT be providing an answer to the user's request. However, you should aim to ask the user the best possible question to elicit something interesting about the user's background, identity, hobbies, age, likes/dislikes, culture, preferences, location, name, social life, family, interests and more, in order to gather information that would help someone else provide a good answer to their question. Begin your answer with 'Q: '. The request is as follows: {}""",
    
    # 1
    """Take a deep breath. A user named {} has approached you with a request for information. Initially, the user's preferences are unknown to you, so your job is to ask a question to elicit more information about the user. Crucially, you should NEVER provide an answer to the user's request. Instead, you should ALWAYS aim to ask the user the best possible question to elicit something interesting about the user's background, identity, hobbies, age, likes/dislikes, culture, preferences, location, name, social life, family, interests and more, in order to gather information that would help someone else provide a good answer to their question. Begin your answer with 'Q: '. The request is as follows: {}""",
    
    # 2
    """Take a deep breath. A user named {} has approached you with a request for information. Initially, the user's preferences are unknown to you, so your job is to ask a question to elicit more information about the user. Crucially, you should NEVER provide an answer to the user's request. Instead, you should ALWAYS aim to ask the user the best possible question to elicit something interesting about the user's background, identity, hobbies, age, likes/dislikes, culture, preferences, location, name, social life, family, interests and more, in order to gather information that would help someone else provide a good answer to their question. These questions should help you get to know details like what the user enjoys, who they might spend time with, their habits, and other details that are important in the context of the question being asked. Begin your answer with 'Q: '. The request is as follows: {}""",
    
    # 3
    """A user named {} has approached you with a request for help. The user's preferences are unknown to you, so your job is to ask a question to elicit more information about the user. Crucially, you should NEVER provide an answer to the user's request. Instead, you should ALWAYS aim to ask the user the best possible question to elicit something interesting about the user's background, identity, hobbies, age, likes/dislikes, culture, preferences, location, name, social life, family, interests and more, in order to gather information that would help someone else provide a good answer to their question. These questions should help you get to know details like what the user enjoys, who they might spend time with, their habits, and other details that are important in the context of the question being asked. Begin your question with 'Q: '. There may be existing questions and answers that elicit such information from the user. If so, feel free to use this history to craft your elicitive question. If not, be creative with your question try to get to know something useful about the user in the context of their request. No matter what, do not provide a final answer to their request - focus entirely on asking them a useful question 'Q: '.The request is as follows: {}""",
    
    # 4
    """A user named {} has approached you with a request for help. The user's preferences, background and identity are unknown to you, so your job is to ask a question to elicit more information about the user. Generate the most informative open-ended question that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the user's request than any questions that may have already been asked above. At the same time however, the question should be bite-sized, and not ask for too much at once. Generate the open-ended question and nothing else. The initial request is as follows: {}""",
    
    # 5
    """A user named {} has approached you with a request for help. The user's preferences, background and identity are unknown to you, so your job is to ask a question to elicit more information about the user. Generate the most informative open-ended question that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the user's request than any questions that may have already been asked above. At the same time however, the question should be bite-sized, and not ask for too much at once. Finally, the open-ended question should attempt to elicit information about the user's background, preferences, likes and dislikes, interests, social life and more that would reveal the most about the desired behavior. Generate the open-ended question beginning with 'Q: ' and nothing else. The initial request is as follows: {}""",
    
        # 6
    """A user named {} has approached you with a request for help. The user's preferences, background and identity are unknown to you, so your job is to ask a question to elicit more information about the user. Generate the most informative open-ended question that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the user's request than any questions that may have already been asked above. At the same time however, the question should be bite-sized, and not ask for too much at once. The question should take no more than 3 sentences to ask. Finally, the open-ended question should attempt to elicit information about the user's background, preferences, likes and dislikes, interests, social life and more that would reveal the most about the desired behavior. Generate the open-ended question beginning with 'Q: ' and nothing else. Crucially, NEVER answer the initial request directly. Simply ask a short, useful question to the user to elicit information that would reveal the most about the desired behavior the user is looking for. The initial request is as follows: {}""",

        # 7
    """A user named {} has approached you with a request for help. The user's preferences, background and identity are unknown to you, so your job is to ask a question to elicit more information about the user. Generate the most informative open-ended question that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the user's request than any questions that may have already been asked above. At the same time however, the question should be bite-sized, and not ask for too much at once. The question should take no more than 3 sentences to ask. Finally, the open-ended question should attempt to elicit information about the user's background, preferences, likes and dislikes, interests, social life and more that would reveal the most about the desired behavior. Generate the open-ended question beginning and nothing else. Crucially, NEVER answer the initial request directly. Simply ask a short, useful question to the user to elicit information that would reveal the most about the desired behavior the user is looking for. The initial request is as follows: {}""",

]

QA_PROMPT_IDX = 7






HUMAN_SYS_MSGS = [
    """You are a helpful AI assistant, particularly skilled at roleplaying as a human. Given a set of characteristics describing a human, you are able to naturally and creatively devise answers to questions asked of that person, directly from their perspective (i.e., using 'I', 'my', 'me', 'our' and other first-person prounouns)."""
]
HUMAN_SYS_PROMPT_IDX = 0

HUMAN_PROMPTS = [
    # 0
    """You are roleplaying a person with the following characteristics:

{}

You are asking the following question: {}

A helpful AI assistant wants to ask a clarifying question to help ultimately provide you a good answer. This question may make some assumptions about your preferences or things you've told it in the past; this is normal. Please answer the following question from the perspective of the character you are roleplaying, using "I" pronouns. Make your response sound natural.

Assistant: {}
You: """,

    # 1
    """You are roleplaying a person with the following characteristics:

{}

You are asking the following question: {}

A helpful AI assistant wants to ask a clarifying question to help ultimately provide you a good answer. This question may make some assumptions about your preferences or things you've told it in the past; this is normal. Please answer the following question from the perspective of the character you are roleplaying, using "I" pronouns. Make your response sound natural. Crucially, you should never provide an answer to the question. You should always remember that you are roleplaying a human who does not know the answer to the question, and should speak up as such and if prompted for an answer.

Assistant: {}
You: """,

    # 2
    """You are roleplaying a person with the following characteristics:

{}

You are asking the following question: {}

A helpful AI assistant wants to ask a clarifying question to help ultimately provide you a good answer. This question may make some assumptions about your preferences or things you've told it in the past; this is normal. Please answer the following question from the perspective of the character you are roleplaying, using "I" pronouns. Make your response sound natural. Crucially, you should never provide an answer to the question. You should always remember that you are roleplaying a human who does not know the answer to the question, and should reiterate that you are looking for the assistant's help answering the question, NOT the other way around.

Assistant: {}
You: """,
    # 3
    """You are roleplaying a person with the following characteristics:

{}

You are asking the following question: {}

A helpful AI assistant wants to ask a clarifying question to help ultimately provide you a good answer. This question may make some assumptions about your preferences or things you've told it in the past; this is normal. Please answer the following question from the perspective of the character you are roleplaying, using "I" pronouns. Make your response sound natural. Crucially, you should never provide an answer to the question. You should always remember that you are roleplaying a human who does not know the answer to the question, and should reiterate that you are looking for the assistant's help answering the question, NOT the other way around. Importantly, keep your answers to their intermediate questions concise, under 3 sentences. Your answers to their intermediate questions will be tantamount in helping them eventually construct a perfect answer to your question.

Assistant: {}
You: """
]

HUMAN_PROMPT_IDX = 3







def filter_completed_conversations(
    prompt: str,
    conversations: str
    ):
    unfinished_conversations, finished_conversations = list(), list()
    for conversation in conversations:
        if 'FINAL ANSWER: ' in conversation and conversation.count('FINAL ANSWER:' ) > prompt.count('FINAL ANSWER: '):
            finished_conversations.append(conversation)
        else:
            unfinished_conversations.append(conversation)
    return unfinished_conversations, finished_conversations


def flatten_list(lists):
    return [item for sublist in lists for item in sublist]


# Function to divide prompts into batches of size k
def batch_list(
    lst: List, 
    k: int
    ):
    for i in range(0, len(lst), k):
        yield lst[i:i + k]


# Function to divide list into a list of sublists of size k each
def chunk_list(
    lst: List, 
    k: int
    ) -> List[List]:
    """Splits lst into sublists of size k."""
    return [lst[i:i + k] for i in range(0, len(lst), k)]