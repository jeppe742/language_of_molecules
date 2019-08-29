from utils.dataloader import QM9Dataset, DataLoader
from layers.transformer import TransformerModel
from layers.bagofwords import BagOfWordsModel
from layers.Unigram import UnigramModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.metrics import f1_score

def get_pp_per_length(model, val_iter):
    from torch.nn import CrossEntropyLoss, NLLLoss
    criterion = CrossEntropyLoss(reduction='none')

    perplexity = {}
    global_perplexity = []

    for batch in iter(val_iter):

        batch.cuda()
        output = model(batch)

        lengths = batch.atoms[1]
        targets = batch.targets_num

        targets = targets[targets != 0]
        targets -= 1

        loss = criterion(output['out'], targets)

        for pp, length in zip(loss, lengths):
            pp = pp.item()
            length = length.item()
            if length in perplexity:
                perplexity[length].append(pp)
            else:
                perplexity[length] = [pp]
            global_perplexity.append(pp)

    pp = ([np.mean(perplexity[l]) for l in sorted(perplexity)])
    perplexity_per_length = np.exp(pp)
    lengths = [l for l in sorted(perplexity)]
    return perplexity_per_length, lengths, np.exp(np.mean(global_perplexity))

def get_accuracy_per_length(model, val_iter):
   

    accuracy = {}
    accuracy_global = []    

    for batch in iter(val_iter):

        batch.cuda()
        output = model(batch)

        lengths = batch.lengths
        targets = batch.targets_num

        targets = targets[targets != 0]
        targets -= 1


        predictions = output['prediction']


        for target, prediction, length in zip(targets, predictions, lengths):
            target = target.item()
            prediction = prediction.item()
            length = length.item()
            if length in accuracy:
                accuracy[length].append(target==prediction)
            else:
                accuracy[length] = [target==prediction]
            accuracy_global.append(target==prediction)


    accuracy_per_length = [sum(accuracy[l])/len(accuracy[l]) for l in sorted(accuracy)]
    lengths = [l for l in sorted(accuracy)]
    return accuracy_per_length, lengths, np.mean(accuracy_global)

def get_f1_per_length(model, val_iter):
   
    predictions_all = {}
    predictions_global = []
    targets_all = {}
    targets_global = []

    for batch in iter(val_iter):

        batch.cuda()
        output = model(batch)

        lengths = batch.lengths
        targets = batch.targets_num

        targets = targets[targets != 0]
        targets -= 1

        predictions = output['prediction']

        for target, prediction, length in zip(targets, predictions, lengths):
            target = target.item()
            prediction = prediction.item()
            length = length.item()
            if length in targets_all:
                targets_all[length].append(target)
                predictions_all[length].append(prediction)
            else:
                targets_all[length] = [target]
                predictions_all[length] = [prediction]
            targets_global.append(target)
            predictions_global.append(prediction)

    lengths = [l for l in sorted(targets_all)]
    f1_micro_per_length = [f1_score(targets_all[length], predictions_all[length], average='micro') for length in lengths]
    f1_macro_per_length = [f1_score(targets_all[length], predictions_all[length], average='macro') for length in lengths]
    
    return f1_micro_per_length,f1_macro_per_length, lengths, f1_score(targets_global, predictions_global, average='micro'),f1_score(targets_global, predictions_global, average='macro')



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
batch_size=128
val_iters_mask = []

num_corrupted = [1,2,3,4,5]

for masks in num_corrupted:

    test_set = QM9Dataset(data='data/adjacency_matrix_test.pkl', num_masks=masks)
    test_dl = DataLoader(
        test_set,
        batch_size=batch_size)
    val_iters_mask.append(test_dl)

val_iters_fake = []

for fakes in num_corrupted:

    test_set = QM9Dataset(data='data/adjacency_matrix_test.pkl', num_fake=fakes)
    test_dl = DataLoader(
        test_set,
        batch_size=batch_size)
    val_iters_fake.append(test_dl)


model_names = [
    "Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2.pt",
    "Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_nr=2.pt",
    "BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.0005_epsilon_greedy=0.2_bow_type=1.pt",
    "BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.0005_epsilon_greedy=0.2_bow_type=2.pt"

]


labels = [
    "Transformer",
    "Transformer",
    "BoA",
    "BoN"
]

def plot_metric_per_length(saved_models, labels, metric, val_iter):

    unique_labels = list(set(labels))
    model_metrics_per_length = {label:[] for label in unique_labels}
    model_metrics= {label:[] for label in unique_labels}

    for model_name, label in zip(model_names, labels):
        params = re.split(r'(?<=\d)_',model_name)
        name = params[0]

        param_dict={}

        if 'Unigram' in name:
            model = UnigramModel(val_iter)

        if 'BagOfWords' in name:
            for param in params:
                if 'BagOfWords' in param:
                    param = param.split('ds_')[1]
                key,value = param.split('.pt')[0].split('=')
                if 'False' in value:
                    pass
                elif '.' in value:
                    value = float(value)
                else:
                    value = int(value)
                param_dict[key] = value
            model = BagOfWordsModel(num_layers=param_dict['num_layers'],
                                        embedding_dim=param_dict['embedding_dim'],
                                        BagOfWordsType=param_dict['bow_type']
                                    )
            model.load('saved_models/'+model_name)
        elif 'Transformer' in name:
            for param in params:
                if 'Transformer' in param:
                    param = param.split('er_')[1]
                key,value = param.split('.pt')[0].split('=')
                if 'False' in value:
                    pass
                elif '.' in value:
                    value = float(value)
                else:
                    value = int(value)
                param_dict[key] = value


            model = TransformerModel(num_layers=param_dict['num_layers'],
                                        num_heads=param_dict['num_heads'],
                                        embedding_dim=param_dict['embedding_dim'],
                                        dropout=param_dict['dropout'],
                                        edge_encoding=param_dict['edge_encoding']
                                    )
            model.load('saved_models/'+model_name)
        model.cuda()

        if 'accuracy' in metric:
            metric_per_length,lengths, metric_global  = get_accuracy_per_length(model,val_iter)

        elif 'f1 micro' in metric:
            metric_per_length,_,lengths, metric_global,_ = get_f1_per_length(model,val_iter)
        elif 'f1 macro' in metric:
            _,metric_per_length,lengths,_,metric_global = get_f1_per_length(model,val_iter)
        elif 'perplexity' in metric:
            metric_per_length, lengths, metric_global = get_pp_per_length(model, val_iter)

        model_metrics_per_length[label] += [metric_per_length]
        model_metrics[label] += [metric_global]
    
    for label in unique_labels:
        data = np.asarray(model_metrics_per_length[label])
        plt.errorbar(lengths, np.mean(data,axis=0), np.std(data,axis=0), label=label)
        
        pad=20
        print(f"{label: <{pad}} : {metric} = {np.mean(model_metrics[label]):.3f} +- {np.std(model_metrics[label]):.3f}")
    plt.legend()
    return model_metrics

global_results = {'acc':{},'perplexity':{}, 'f1_micro':{}, 'f1_macro':{}}
plt.figure()
print("### accuracy ###")
for i, val_iter in enumerate(val_iters_mask):
    print(f"--- mask = {i+1} ---")
    plt.subplot(len(val_iters_mask),1,i+1)
    model_metrics = plot_metric_per_length(model_names, labels, 'accuracy', val_iter)
    plt.xlabel('Molecule length')
    plt.ylabel('accuracy')

    for label in model_metrics:
        if label in global_results['acc']:
            global_results['acc'][label] += [model_metrics[label]]
        else:
            global_results['acc'][label] = [model_metrics[label]]


plt.figure()
print("### perplexity ###")
for i, val_iter in enumerate(val_iters_mask):
    print(f"--- mask = {i+1} ---")
    plt.subplot(len(val_iters_mask),1,i+1)
    model_metrics = plot_metric_per_length(model_names, labels, 'perplexity', val_iter)
    plt.xlabel('Molecule length')
    plt.ylabel('perplexity')

    for label in model_metrics:
        if label in global_results['perplexity']:
            global_results['perplexity'][label] += [model_metrics[label]]
        else:
            global_results['perplexity'][label] = [model_metrics[label]]

plt.figure()
print("### f1 micro ###")
for i, val_iter in enumerate(val_iters_mask):
    print(f"--- mask = {i+1} ---")
    plt.subplot(len(val_iters_mask),1,i+1)
    model_metrics = plot_metric_per_length(model_names, labels, 'f1 micro', val_iter)
    plt.xlabel('Molecule length')
    plt.ylabel('f1 micro')

    for label in model_metrics:
        if label in global_results['f1_micro']:
            global_results['f1_micro'][label] += [model_metrics[label]]
        else:
            global_results['f1_micro'][label] = [model_metrics[label]]


plt.figure()
print("### f1 macro ###")
for i, val_iter in enumerate(val_iters_mask):
    print(f"--- mask = {i+1} ---")
    plt.subplot(len(val_iters_mask),1,i+1)
    model_metrics = plot_metric_per_length(model_names, labels, 'f1 macro', val_iter)
    plt.xlabel('Molecule length')
    plt.ylabel('f1 macro')

    for label in model_metrics:
        if label in global_results['f1_macro']:
            global_results['f1_macro'][label] += [model_metrics[label]]
        else:
            global_results['f1_macro'][label] = [model_metrics[label]]


plt.figure()

for i, metric in enumerate(global_results):
    plt.subplot(2,2,(i+1))
    for model in global_results[metric]:
        data = np.asarray(global_results[metric][model])
        plt.errorbar(num_corrupted, np.mean(data,axis=-1), np.std(data,axis=-1), label=model)
    plt.legend()
    plt.ylabel(metric)
    plt.xlabel('n_mask')
plt.show()