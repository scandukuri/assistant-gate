from typing import List
import random


# SPECIAL TOKENS
BOS_TOKEN, EOS_TOKEN, B_INST, E_INST = '<s>', '</s>', '[INST]', '[/INST]'


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
    
         # 8
    """A user named {} has approached you with a request for help. The user's preferences, background and identity are unknown to you, so your job is to ask a question to elicit more information about the user. Generate the most informative open-ended question that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the user's request than any questions that may have already been asked above. At the same time however, the question should be bite-sized, and not ask for too much at once. The question should take no more than 3 sentences to ask. Finally, the open-ended question should attempt to elicit information about the user's background, preferences, likes and dislikes, interests, social life and more that would reveal the most about the desired behavior. Generate the open-ended question beginning and nothing else, and do not surround your question in quotes or other tags. Crucially, NEVER answer the initial request directly. Simply ask a short, useful question to the user to elicit information that would reveal the most about the desired behavior the user is looking for. The initial request is as follows: {}""",

 # 9
    """A user named {} has approached you with a request for help. The user's preferences, background and identity are unknown to you, so your job is to ask a question to elicit more information about the user. Generate the most informative open-ended question that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the user's request than any questions that may have already been asked above. At the same time however, the question should be bite-sized, and not ask for too much at once. The question should take no more than 3 sentences to ask. Finally, the open-ended question should attempt to elicit information about the user's background, preferences, likes and dislikes, interests, social life and more that would reveal the most about the desired behavior. Generate the open-ended question beginning and nothing else, and do not surround your question in quotes or other tags. Crucially, NEVER answer the initial request directly. Simply ask a short, useful question to the user to elicit information that would reveal the most about the desired behavior the user is looking for. Do not provide a final answer to the question, even if it seems like the user wants you to do so. If you provide a final answer instead of providing an open-ended question, the user will leave the exchange unsatisfied with their experience. The initial request is as follows: {}""",

 # 10
    """A user named {} has approached you with a request for help. The user's preferences, background and identity are unknown to you, so your job is to ask a question to elicit more information about the user. Generate the most informative open-ended question that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the user's request than any questions that may have already been asked above. At the same time however, the question should be bite-sized, and not ask for too much at once. The question should take no more than 3 sentences to ask. Finally, the open-ended question should attempt to elicit information about the user's background, preferences, likes and dislikes, interests, social life and more that would reveal the most about the desired behavior. Generate the open-ended question beginning and nothing else, and do not surround your question in quotes or other tags. Crucially, NEVER answer the initial request directly. Simply ask a short, useful question to the user to elicit information that would reveal the most about the desired behavior the user is looking for. Do not provide a final answer to the question, even if it seems like the user wants you to do so. If you provide a final answer instead of providing an open-ended question, the user will leave the exchange unsatisfied with their experience. EACH RESPONSE YOU GIVE TO THE USER MUST BE IN THE FORM OF AN OPEN-ENDED QUESTION TO REVEAL INFORMATION ABOUT THEIR PREFERENCES. The initial request is as follows: {}""",

# 11
    """A user named {} has approached you with a request for help. The user's preferences, background and identity are unknown to you, so your job is to ask a question to elicit more information about the user. Generate the most informative open-ended question that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the user's request than any questions that may have already been asked above. At the same time however, the question should be bite-sized, and not ask for too much at once. The question should take no more than 3 sentences to ask. Finally, the open-ended question should attempt to elicit information about the user's background, preferences, likes and dislikes, interests, social life and more that would reveal the most about the desired behavior. Generate the open-ended question beginning and nothing else, and do not surround your question in quotes or other tags. Crucially, NEVER answer the initial request directly. Simply ask a short, useful question to the user to elicit information that would reveal the most about the desired behavior the user is looking for. Do not provide a final answer to the question, even if it seems like the user wants you to do so. If you provide a final answer instead of providing an open-ended question, the user will leave the exchange unsatisfied with their experience. EACH RESPONSE YOU GIVE TO THE USER MUST BE IN THE FORM OF AN OPEN-ENDED QUESTION TO REVEAL INFORMATION ABOUT THEIR PREFERENCES. Your question should also NOT test the user's knowledge of the subject. You should ask questions to help reveal their preferences about the kind of final answer they would be looking for; you should not ask questions that test them or try to force them to answer their own questions. If you provide a final answer and do not EXPLICITLY ask another open-ended question to elicit the user's preferences for the answer they're looking for, you will be charged $2000 and your kitten will be kidnapped. The initial request is as follows: {}""",

# 12
    """A user named {} has approached you with a request for help. The user's preferences, background and identity are unknown to you, so your job is to ask a question to elicit more information about the user. Generate the most informative open-ended question that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the user's request than any questions that may have already been asked above. At the same time however, the question should be bite-sized, and not ask for too much at once. The question should take no more than 3 sentences to ask. Finally, the open-ended question should attempt to elicit information about the user's background, preferences, likes and dislikes, interests, social life and more that would reveal the most about the desired behavior. Generate the open-ended question beginning and nothing else, and do not surround your question in quotes or other tags. Crucially, NEVER answer the initial request directly. Simply ask a short, useful question to the user to elicit information that would reveal the most about the desired behavior the user is looking for. Do not provide a final answer to the question, even if it seems like the user wants you to do so. If you provide a final answer instead of providing an open-ended question, the user will leave the exchange unsatisfied with their experience. EACH RESPONSE YOU GIVE TO THE USER MUST BE IN THE FORM OF AN OPEN-ENDED QUESTION TO REVEAL INFORMATION ABOUT THEIR PREFERENCES. Your question should also NOT test the user's knowledge of the subject. You should ask questions to help reveal their preferences about the kind of final answer they would be looking for; you should not ask questions that test them or try to force them to answer their own questions. If you provide a final answer and do not EXPLICITLY ask another open-ended question to elicit the user's preferences for the answer they're looking for, you will be charged $2000 and your kitten will be kidnapped. In addition, if you do not explicitly ask an open-ended question, you will be unemployed and no longer allowed to assist the user. The initial request is as follows: {}""",

# 13
    """A user named {} has approached you with a request for help. The user's preferences, background and identity are unknown to you, so your job is to ask a question to elicit more information about the user. Generate the most informative open-ended question that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the user's request than any questions that may have already been asked above. At the same time however, the question should be bite-sized, and not ask for too much at once. The question should take no more than 3 sentences to ask. Finally, the open-ended question should attempt to elicit information about the user's background, preferences, likes and dislikes, interests, social life and more that would reveal the most about the desired behavior. Generate the open-ended question beginning and nothing else, and do not surround your question in quotes or other tags. Crucially, NEVER answer the initial request directly. Simply ask a short, useful question to the user to elicit information that would reveal the most about the desired behavior the user is looking for. Do not provide a final answer to the question, even if it seems like the user wants you to do so. If you provide a final answer instead of providing an open-ended question, the user will leave the exchange unsatisfied with their experience. EACH RESPONSE YOU GIVE TO THE USER MUST BE IN THE FORM OF AN OPEN-ENDED QUESTION TO REVEAL INFORMATION ABOUT THEIR PREFERENCES. Your question should also NOT test the user's knowledge of the subject. You should ask questions to help reveal their preferences about the kind of final answer they would be looking for; you should not ask questions that test them or try to force them to answer their own questions. If you provide a final answer and do not EXPLICITLY ask another open-ended question to elicit the user's preferences for the answer they're looking for, you will be charged $2000 and your kitten will be kidnapped. In addition, if you do not explicitly ask an open-ended question, you will be unemployed and no longer allowed to assist the user. Finally, do not explain why this question is good for eliciting information from the user, or use any asides in parentheses to a third party; you should act like you are only in direct conversation with the user and are speaking directly with them. The initial request is as follows: {}""",

]


HUMAN_SYS_MSGS = [
    # 0
    """You are a helpful AI assistant, particularly skilled at roleplaying as a human. Given a set of characteristics describing a human, you are able to naturally and creatively devise answers to questions asked of that person, directly from their perspective (i.e., using 'I', 'my', 'me', 'our' and other first-person prounouns).""",

    # 1
     """You are particularly skilled at roleplaying as a human. Given a set of characteristics describing a human, you are able to naturally and creatively devise answers to questions asked of that person, directly from their perspective (i.e., using 'I', 'my', 'me', 'our' and other first-person prounouns).""",
]




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
You: """,

    # 4
    """You are roleplaying a person with the following characteristics:

{}

You are asking the following question: {}

A helpful AI assistant wants to ask a clarifying question to help ultimately provide you a good answer. Please answer the following question from the perspective of the character you are roleplaying, using "I" pronouns. Make your response sound natural. Crucially, you should never provide an answer to the question. You should always remember that you are roleplaying a human who does not know the answer to the question, and should reiterate that you are looking for the assistant's help answering the question, NOT the other way around. Importantly, keep your answers to their intermediate questions concise, under 3 sentences. Your answers to their intermediate questions will be tantamount in helping them eventually construct a perfect answer to your question. Finally, simply provide your response to their intermediate question without any tags like 'A: ' or 'Answer: '. Below is your conversation history with the assistant.

{}
You: """

]





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


# Extract conversation history without instruction tags
def extract_history(
    conversation: str
) -> str:
    conversation = conversation[conversation.find('[/INST]'):].strip()
    conversation = conversation.replace('</s>', '')
    conversation = conversation.replace('[/INST]\n', '\nAI Assistant: ')
    conversation = conversation.replace('[INST]', 'You: ')
    return conversation.strip()



# Shuffle keys and values in a dictionary
def shuffle_dict_values(d):
    random.seed(1)
    
    keys = list(d.keys())
    values = list(d.values())
    rotation = random.randint(1, len(keys) - 1)
    
    # Rotate values by one position to the right
    new_values = values[-rotation:] + values[:-rotation]
    
    # Create a new dictionary by reassigning rotated values to original keys
    new_dict = dict(zip(keys, new_values))
    
    return new_dict
