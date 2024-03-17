Here's some instructions for working with the data in /scr/andukuri/assistant-gate-hgx.Note that for the most part, each subdirectory in this root represents a discrete 'task', and each subdirectory within these discrete tasks represents an 'experiment configuration'. First, note the subdirectories:

1. content-violations
2. finetuned_models
3. log-probs       
4. personas       
5. prompts   
6. simulated-conversations  
7. win-rates
8. figures          
9. gold-responses    
10. model-responses  
11. pretrained_models  
12. sft-data  
13. specificity-ratings

# content-violations
Here, if there are any gpt4-generated personas that (somehow) are causing content violation errors, we output them into the appropriate json for that split. We have subfolders for each configuration (i.e. 'star-2-bsft') and within that a file for each split. They're all empty!

# finetuned_models
Here, we have a directory for each experimental configuration (i.e. 'star-2-bsft'). Within each, we have a folder for each iteration (i.e. 'm2'). Within that, finally, we have a subfolder for each saved set of finetuned weights, tokenizer, etc. For example:

star-2-bsft --> m2 --> checkpoint-185

# log-probs
This one is important. Like the others, we have a directory for each experimental configuration (i.e. 'star-2-bsft'). Within each, we have a folder for each evaluation condition (i.e. 'qa-experimental'). Finally, we have a folder for each iteration (i.e. 'm2'). There, we have jsons for each relevant split (i.e. 'A.json' or 'test.json') structured as follows:

{'prompt-i persona-j' : [list of log probabilities for that (i, j) pair], .......}

When we filter the top-k log probabilities at any point (either to pick examples for training set, or calculating top k for eval set), we also include those in this folder. This time, we call the file by the split followed by '_top-k-1.json'. For example, 'B_top-k-1.json'. Note that this structure is exactly the same as the simulated conversations, barring the evaluation condition. A review of the structure:
star-2-bsft —> qa-experimental —> m2 —> B.json

# simulated-conversations
This one is almost identical to log-probs, except that we skip the evaluation condition level:

star-2-bsft —>  m2 —> B.json
{'prompt-i persona-j' : [list of simulated conversations for that (i, j) pair], .......}

We still have top-k as described above. 


# win-rates

This one is probably the messiest of all of them, but we’ll make it work. 
- We have a subfolder for each iteration (i.e. baseline, m0***, m1, m2, m3). In each of these, we have a file containing the model responses with each turn length (i.e. model responses conditioned on conversations with 1 turn, model responses conditioned on conversations with 2 turns, etc…) The key-value structure of these jsons is the same as simulated-conversations and log-probs
- We also have a subfolder for each of the win-rate responses from gpt4. Here, we save the entire gpt4 output. The folders are labeled m0_m1 if gpt4 was given m0  responses first and m1 responses second for each pair of corresponding answers, m2_m1 if given m2 responses first and m1 responses second, etc… This is what we pull from to actually plot the win rates)

***  (where this is the same as baseline but nice to have a little redundancy for code in case we want to plot m0 against baseline for some reason)
