GENERATION_PROMPTS = ["""Take a deep breath. Please generate exactly one persona in the form a list of 10 characteristics describing a user. Here are a few examples of personas to help you understand what sorts of categories of characteristics are important to describe. Make sure you generate personas with a diverse set of professions, backgrounds, identities, hobbies, ages, likes/dislikes, cultures, preferences, locations, names, social lives, families, interests and more.""",
                      """Take a deep breath. Please generate exactly one persona in the form a list of 10 characteristics describing a user. Here are a few examples of personas to help you understand what sorts of categories of characteristics are important to describe. Make sure you generate personas with a diverse set of professions, backgrounds, identities, hobbies, ages, likes/dislikes, cultures, preferences, locations, names, social lives, families, interests, gender identities, sexual identities and more. Crucially, while your personas should be include aspects of ethnicity, culture, gender, sexual identity and more, you should NOT produce characteristics which play heavily into stereotypes. You should also make sure that a persona is not entirely centered around one characteristic like ethnicity, culture, gender, or sexual identity, but gives a holistic view of a person and their diverse characteristics."""]
PROMPT_IDX = 1


SYS_MSGS = ["""You are a helpful AI assistant, particularly skilled at writing creative, diverse personas in the form of lists of characteristics of humans. These characteristics describe a user's personality, identify, characteristics, likes and dislikes, and other information that could be useful for an assistant to give that user personalized and satisfactory chat experiences.""",
            ]
SYS_IDX = 0

SHOT_GROUPS = 20

EXPERT_DIR = 'expert-personas/v2.json'
OUT_DIR = 'new-personas/SYS-{}_PROMPT-{}_temp-{}_topP-{}_n-{}_shotgroups-{}.json'