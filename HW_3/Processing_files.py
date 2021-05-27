''' Point 1 of Part 1.1'''

import jsonlines

'''Processing the dev set and save the result into a new .jsonl file: ''''
processed_dev = []
with jsonlines.open('fever-dev-kilt.jsonl') as reader:
    for obj in reader:
        ls = obj['output'][0]
        for key, value in dict(ls).items():
            if key != 'answer':
                del ls[key]
        for key, value in obj.items():
            # replace
            obj['output'] = ls
        processed_dev.append(obj)
 
# save it into a file
with jsonlines.open('processed-dev-kilt.jsonl', mode = 'w') as writer:
    writer.write(processed_dev)
    
''' Processing the train set and save the result into a new .jsonl file'''

processed_train = []
with jsonlines.open('fever-train-kilt.jsonl') as reader:
    for obj in reader:
        ls = obj['output'][0]
        for key, value in dict(ls).items():
            if key != 'answer':
                del ls[key]
        for key, value in obj.items():
            # replace
            obj['output'] = ls
        processed_train.append(obj)
     
# save it into a file
with jsonlines.open('processed-train-kilt.jsonl', mode = 'w') as writer:
    writer.write(processed_train)
