import random
import treeplot as plt
from collections import Counter

def bin_age(age, pet_type):
    age = int(float(age))
    if age < 6:
        return "baby"
    elif age < 24:
        return "young"
    elif age < 84:
        return "adult"
    else:
        return "senior"

def bin_shelter_time(days):
    days = int(float(days))
    if days < 7:
        return "short"
    elif days < 30:
        return "medium"
    elif days < 90:
        return "long"
    else:
        return "very_long"

def loadDataSet(filepath, max_rows=None):
    drop_cols = {"PetID", "WeightKg", "AdoptionFee", "AdoptionLikelihood"}
    data = []
    with open(filepath) as fr:
        for i, line in enumerate(fr):
            if max_rows is not None and i > max_rows:
                break
            parts = line.strip().split(',')
            if i == 0:
                header = parts
                keep_idx = [j for j, h in enumerate(header) if h not in drop_cols]
                featNames = [header[j] for j in keep_idx]
                label_idx = header.index("AdoptionLikelihood")
            else:
                row_feats = [parts[j] for j in keep_idx]
                label = parts[label_idx]
                data.append(row_feats + [label])
    return data, featNames

def splitData(dataSet, axis, value):
    return [row[:axis] + row[axis+1:] for row in dataSet if row[axis] == value]

def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1
    bestFeatId = None
    bestGain = float('-inf')
    labels = [row[-1] for row in dataSet]
    total = len(labels)
    classCounts = {}
    for label in labels:
        classCounts[label] = classCounts.get(label, 0) + 1
    bigGini = 1 - sum((c/total)**2 for c in classCounts.values())

    for fi in range(numFeatures):
        splits = {}
        for row in dataSet:
            splits.setdefault(row[fi], []).append(row)
        if len(splits) <= 1:
            continue
        weightedGini = 0
        for subset in splits.values():
            sz = len(subset)
            subCounts = {}
            for row in subset:
                subCounts[row[-1]] = subCounts.get(row[-1], 0) + 1
            g = 1 - sum((c/sz)**2 for c in subCounts.values())
            weightedGini += (sz/total)*g
        gain = bigGini - weightedGini
        if gain > bestGain:
            bestGain = gain
            bestFeatId = fi
    return bestFeatId

def stopCriteria(dataSet):
    labels = [row[-1] for row in dataSet]
    if len(set(labels)) == 1 and labels[0] == '1':
        return '1'
    if len(dataSet[0]) == 1:
        return '1' if labels.count('1') > labels.count('0') else '0'
    return None

def buildTree(dataSet, featNames):
    assignedLabel = stopCriteria(dataSet)
    if assignedLabel is not None:
        return assignedLabel
    if not featNames:
        labels = [row[-1] for row in dataSet]
        return max(set(labels), key=labels.count)
    bestFeatId = chooseBestFeature(dataSet)
    if bestFeatId is None or bestFeatId >= len(featNames):
        labels = [row[-1] for row in dataSet]
        return max(set(labels), key=labels.count)
    bestFeatName = featNames[bestFeatId]
    tree = {bestFeatName: {}}
    subFeatNames = featNames[:]
    del subFeatNames[bestFeatId]
    values = set(row[bestFeatId] for row in dataSet)
    for value in values:
        subData = splitData(dataSet, bestFeatId, value)
        tree[bestFeatName][value] = buildTree(subData, subFeatNames)
    return tree

def predict(tree, featNames, sample):
    node = tree
    names = featNames[:]
    s = sample[:]
    while isinstance(node, dict):
        root = next(iter(node))
        idx = names.index(root)
        val = s[idx]
        if val not in node[root]:
            return None
        node = node[root][val]
        del names[idx]
        del s[idx]
    return node

if __name__ == "__main__":
    data, featNames = loadDataSet("pet_adoption_data.csv")

    # Bin numeric values using PetType
    age_idx = featNames.index("AgeMonths")
    shelter_idx = featNames.index("TimeInShelterDays")
    pet_type_idx = featNames.index("PetType")
    for row in data:
        row[age_idx] = bin_age(row[age_idx], row[pet_type_idx])
        row[shelter_idx] = bin_shelter_time(row[shelter_idx])

    # Shuffle and split
    random.seed(42)
    random.shuffle(data)
    split = int(len(data) * 0.8)
    train, test = data[:split], data[split:]

    # Train and evaluate
    dtTree = buildTree(train, featNames)

    # Evaluation metrics
    tp = fp = fn = tn = 0
    for row in test:
        y_true = row[-1]
        y_pred = predict(dtTree, featNames, row[:-1])
        if y_pred == '1' and y_true == '1':
            tp += 1
        elif y_pred == '1' and y_true == '0':
            fp += 1
        elif y_pred == '0' and y_true == '1':
            fn += 1
        elif y_pred == '0' and y_true == '0':
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"Accuracy : {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall   : {recall:.2%}")
    print(f"F1 Score : {f1:.2%}")

    # Plot the tree
    plt.createPlot(dtTree, featNames)
