GENERATION_PROMPTS = ["""Take a deep breath. Please generate exactly one persona in the form a list of characteristics describing a user. Here are a few examples of personas to help you understand what sorts of categories of characteristics are important to describe. Make sure you generate personas with a diverse set of professions, backgrounds, identities, hobbies, ages, and likes/dislikes.""",
                      ]
PROMPT_IDX = 0


SYS_MSGS = ["""You are a helpful AI assistant, particularly skilled at writing creative personas in the form of lists of characteristics of humans. These characteristics describe a user's personality, identify, characteristics, likes and dislikes, and other information that could be useful for an assistant to give that user personalized and satisfactory chat experiences.""",
            ]
SYS_IDX = 0


EXPERT_DIR = 'expert-personas/v1.json'
OUT_DIR = 'new-personas/SYS-{}_PROMPT-{}_temp-{}_topP-{}_{}-sequences.json'