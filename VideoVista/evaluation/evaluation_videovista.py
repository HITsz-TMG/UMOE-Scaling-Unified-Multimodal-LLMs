import os
import json
import argparse

parser = argparse.ArgumentParser(description="videovista evaluation")
parser.add_argument("--golden_file", type=str, required=True, help="path to VideoVista.json")
parser.add_argument("--pred_file", type=str, required=True, help="path to your pred file")
args = parser.parse_args()


prediction = json.load(open(args.pred_file, "r"))
print(len(prediction))
golden = json.load(open(args.golden_file, "r"))
index2letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

types_understanding = [
    "Objects Existence", "Objects Count", "Action Count", "Detailed Description", 'Brief Description', 'Event Description', 'Event Sequence', 'Optical Character Recognition',
    'Action Recognition',  'Action Sequence', 'Action Location', 'Event Location', 'Objects Temporal Location', 'Objects Temporal Relation', 
    'Objects Spatial Location', 'Objects Spatial Relation', 'Objects Spatial Tracking', 'Human Activity Analysis', 'Anomaly Detection'
]

types_reasoning = [
    'Relation Reasoning-Image', 'Relation Reasoning-Video', 'Event Prediction', 'Action Prediction', 'Causal Reasoning', 'Counterfactual Reasoning', 'Commonsense Reasoning', 'Logic Reasoning'
]


cnt = 0
total = 0
cnt_understanding = 0
cnt_reasoning = 0
total_understanding = 0
total_reasoning = 0
type2acc = {}
type2total = {}
duration2acc = {}
duration2total = {}
category2acc = {}
category2total = {}
error = 0

qid2answer = {}
for p in prediction:
    qid2answer[p["question_id"]] = p["Model_Answer"]
    
all_types = []
for idx in range(len(golden)):
    type = golden[idx]["Type"]

    # If your model cannot support the evaluation of the relation reasoning task, uncomment the following two lines of code.
    # if "relation reasoning" in type.lower():
    #     continue
    all_types.append(type)

    total += 1
    if type in types_understanding:
        total_understanding += 1
    if type in types_reasoning:
        total_reasoning += 1
    
    if type in types_understanding:
        duration = golden[idx]["time"] // 60
        if duration > 2 and duration < 5:
            duration = 2
        elif duration > 5 and duration < 10:
            duration = 5
        elif duration > 10:
            duration = 10
        
        if duration not in duration2total:
            duration2total[duration] = 0
        duration2total[duration] += 1
    
    category = golden[idx]["category"]
    if category not in category2total:
        category2total[category] = 0
    category2total[category] += 1
    
    pred = qid2answer[golden[idx]["question_id"]]
    gold = golden[idx]["Answer"]
    
    gold_letter = index2letter[gold]
    
    pred = pred.strip()
    if len(pred) > 0:
        pred_letter = pred[0]
    else:
        pred_letter = ""

    
    if gold_letter == pred_letter:
        cnt += 1
        if type not in type2acc:
            type2acc[type] = 0
            
        if type in types_understanding:
            cnt_understanding += 1
        if type in types_reasoning:
            cnt_reasoning += 1
        if type in types_understanding:
            if duration not in duration2acc:
                duration2acc[duration] = 0
            duration2acc[duration] += 1
        if category not in category2acc:
            category2acc[category] = 0
            
        category2acc[category] += 1
        type2acc[type] += 1
        
    if type not in type2total:
        type2total[type] = 0
    type2total[type] += 1


print(error)
print(f"Overall: {cnt / total}")
print(f"Understanding: {cnt_understanding / total_understanding}")
print(f"Reasoning: {cnt_reasoning / total_reasoning}")

print("---------\n")
for line in types_understanding:
    acc = type2acc[line]
    total = type2total[line]
    print(f"Type: {line} Score: {acc/total}")

print("---------\n")
for line in types_reasoning:
    if "relation" in line.lower():
        continue
    acc = type2acc[line]
    total = type2total[line]
    print(f"Type: {line} Score: {acc/total}")

print("---------\n")
durations = duration2acc.keys()
sorted_durations = sorted(durations)
for line in sorted_durations:
    print(f"TimeStamp: {line:<10} Score: {duration2acc[line] / duration2total[line]}")


print("---------\n")
categorys = category2acc.keys()
sorted_categorys = sorted(categorys)
for line in sorted_categorys:
    print(f"Type: {line:<20} Score: {category2acc[line] / category2total[line]}")
