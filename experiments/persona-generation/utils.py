GENERATION_PROMPTS = ["""Take a deep breath. Please generate exactly one persona in the form a list of 10 characteristics describing a user. Here are a few examples of personas to help you understand what sorts of categories of characteristics are important to describe. Make sure you generate personas with a diverse set of professions, backgrounds, identities, hobbies, ages, likes/dislikes, cultures, preferences, locations, names, social lives, families, interests and more.""",
                      ]
PROMPT_IDX = 0


SYS_MSGS = ["""You are a helpful AI assistant, particularly skilled at writing creative, diverse personas in the form of lists of characteristics of humans. These characteristics describe a user's personality, identify, characteristics, likes and dislikes, and other information that could be useful for an assistant to give that user personalized and satisfactory chat experiences.""",
            ]
SYS_IDX = 0

SHOT_GROUPS = 5

EXPERT_DIR = 'expert-personas/v1.json'
OUT_DIR = 'new-personas/SYS-{}_PROMPT-{}_temp-{}_topP-{}_n-{}_shotgroups-{}.json'