import random

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