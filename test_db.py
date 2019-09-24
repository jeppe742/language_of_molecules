from utils.dataloader import QM9Dataset, DataLoader
from layers.transformer import TransformerModel
from layers.bagofwords import BagOfWordsModel
from layers.Unigram import UnigramModel
from layers.octetRule import OctetRuleModel
from torch.nn.functional import softmax, cross_entropy
import torch
import numpy as np
import re
import sqlite3
from tqdm import tqdm
from utils.helpers import plot_prediction
from utils.db import Database

def tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist()

def load_model(model_name):
    params = re.split(r'(?<=\d)_|(?<=True)_|(?<=False)_',model_name)
    name = params[0]

    param_dict={}

    if 'Unigram' in name:
        model = UnigramModel(val_iter)
    elif 'OctetRule' in name:
        model = OctetRuleModel()

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
            if 'False' in value or 'True' in value:
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
    return model

def generate_predictions(model_name, model_alias, num_masked_list = range(1,30), num_faked_list=range(0), save_db=False, plot=False):

    model = load_model(model_name)
    
    
    if save_db: 
        db = Database("data/results.db")
        db.setup()
    #Set seed to make sure the same atoms are masked for each model
    np.random.seed(42)
    for masks in tqdm(num_masked_list):

        test_set = QM9Dataset(data='data/adjacency_matrix_validation.pkl', num_masks=masks, bond_order=True)
        test_dl = DataLoader(
            test_set,
            batch_size=248
)

        for batch in iter(test_dl):

            batch.cuda()
            output = model(batch)

            lengths = batch.lengths
            targets_batch = batch.targets_num
            target_masks = batch.target_mask
            smiles = batch.smiles

            targets_batch -= 1

            predictions_batch = output['prediction']

            if plot:
                for i in range(batch.batch_size):
                    if not torch.equal(targets_batch[i,:], predictions_batch[i,target_masks[i,:]]):
                        plot_prediction(smiles[i], batch.atoms[i], targets_batch[i,:], predictions_batch[i,target_masks[i,:]])

            out_batch = output['out']
            probabilities_batch = softmax(output['out'],dim=-1)

            insert_values = []

            for targets, predictions, probabilities, out, target_mask, length, smile in zip(targets_batch,
                                                                                             predictions_batch, 
                                                                                             probabilities_batch, 
                                                                                             out_batch,
                                                                                             target_masks, 
                                                                                             lengths, 
                                                                                             smiles):
                
                targets = targets[targets!=-1]
                out = out[target_mask]
                losses = cross_entropy(out, targets, reduction='none')

                targets = tensor_to_list(targets)
                probabilities = tensor_to_list(probabilities[target_mask])
                losses = tensor_to_list(losses)
                predictions = tensor_to_list(predictions[target_mask])

                
                length = length.item()
                for target, prediction, probability, loss in zip(targets, predictions, probabilities, losses):
                    insert_values += [(length, 
                                       model_alias,
                                       0,
                                       masks, 
                                       probability[0], 
                                       probability[1], 
                                       probability[2], 
                                       probability[3], 
                                       probability[4], 
                                       prediction, 
                                       target, 
                                       loss,
                                       smile)]
            if save_db:
                db.stage_results(insert_values)
        if save_db:
            db.apply_staged()


if __name__ == "__main__":
    #generate_predictions("OctetRule",model_alias='OctetRule Binary',  save_db=False, plot=True)
    #generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2.pt", save_db=True, plot=False)
    generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=2_num_heads=1_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True.pt", model_alias='(small) Transformer Bondtype', save_db=True, plot=False)