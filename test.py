from utils.dataloader import QM9Dataset, DataLoader
from layers.transformer import TransformerModel
from layers.bagofwords import BagOfWordsModel
from layers.Unigram import UnigramModel
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.metrics import f1_score, accuracy_score

def tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist()

def get_metric_per_length(model, val_iter, metric, out=False):

    predictions_all = {}
    predictions_global = []
    targets_all = {}
    targets_global = []

    for batch in iter(val_iter):

        batch.cuda()
        output = model(batch)

        lengths = batch.lengths
        targets = batch.targets_num
        target_masks = batch.target_mask

        targets -= 1

        predictions = output['out'] if out else output['prediction']
   
        for target, prediction, target_mask, length in zip(targets, predictions,target_masks, lengths):
            target = target[target!=-1]
            prediction = prediction[target_mask]

            length = length.item()
            if length in targets_all:
                targets_all[length] +=  tensor_to_list(target)
                predictions_all[length] += tensor_to_list(prediction)
            else:
                targets_all[length] = tensor_to_list(target)
                predictions_all[length] = tensor_to_list(prediction)
            targets_global += tensor_to_list(target)
            predictions_global += tensor_to_list(prediction)

    lengths = [l for l in sorted(targets_all)]
    if out:
        metric_per_length = [metric(torch.tensor(predictions_all[length]),torch.tensor(targets_all[length])) for length in lengths]
    else:
        metric_per_length = [metric(targets_all[length], predictions_all[length]) for length in lengths]

    metric_global = metric(torch.tensor(predictions_global),torch.tensor(targets_global)) if out else metric(targets_global, predictions_global)
    return metric_per_length, lengths, metric_global



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
batch_size=248
val_iters_mask = []

num_corrupted = [1,2,3,4,5,20]

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
    "BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.0005_epsilon_greedy=0.2_bow_type=1.pt",
    "BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.0005_epsilon_greedy=0.2_bow_type=2.pt",
    "Unigram"
]


labels = [
    "Transformer",
    "BoA",
    "BoN",
    "Unigram"
]

def plot_metric_per_length(saved_models, labels, metric, val_iter):

    cross_entropy = CrossEntropyLoss()

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
            metric_per_length,lengths, metric_global  = get_metric_per_length(model,val_iter, accuracy_score)
            #metric_per_length,lengths, metric_global  = get_accuracy_per_length(model,val_iter)

        elif 'f1 micro' in metric:
            metric_per_length,lengths, metric_global  = get_metric_per_length(model,val_iter, lambda t,p: f1_score(t,p, average='micro'))
            #
            # metric_per_length,_,lengths, metric_global,_ = get_f1_per_length(model,val_iter)
        elif 'f1 macro' in metric:
            metric_per_length,lengths, metric_global  = get_metric_per_length(model,val_iter, lambda t,p: f1_score(t,p, average='macro'))
            # _,metric_per_length,lengths,_,metric_global = get_f1_per_length(model,val_iter)
        elif 'perplexity' in metric:
            metric_per_length,lengths, metric_global  = get_metric_per_length(model,val_iter, lambda t,p: np.exp(cross_entropy(t,p)), out=True)
            #metric_per_length, lengths, metric_global = get_pp_per_length(model, val_iter)

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
for i, (val_iter,mask) in enumerate(zip(val_iters_mask, num_corrupted)):
    print(f"--- mask = {mask} ---")
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
for i, (val_iter, mask) in enumerate(zip(val_iters_mask, num_corrupted)):
    print(f"--- mask = {mask} ---")
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
for i, (val_iter,mask)  in enumerate(zip(val_iters_mask, num_corrupted)):
    print(f"--- mask = {mask} ---")
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
for i, (val_iter, mask) in enumerate(zip(val_iters_mask, num_corrupted)):
    print(f"--- mask = {mask} ---")
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