from utils.dataloader import QM9Dataset, DataLoader
from layers.transformer import TransformerModel
from layers.bagofwords import BagOfWordsModel
from layers.Unigram import UnigramModel
from layers.octetRule import OctetRuleModel, OctetRule
from torch.nn.functional import softmax, cross_entropy
import torch
import numpy as np
import re
import sqlite3
from tqdm import tqdm
from utils.helpers import plot_prediction, plotMoleculeAttention, plot_attention
from utils.db import Database
import matplotlib.pyplot as plt

def tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist()

def load_model(model_name):
    params = re.split(r'(?<=\d)_|(?<=True)_|(?<=False)_',model_name)
    name = params[0]

    param_dict={}

    if 'Unigram' in name:
        model = UnigramModel()
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
                try:
                    value = int(value)
                except:
                    value = value
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
                try:
                    value = int(value)
                except:
                    value = value
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

def generate_predictions(model_name, model_alias, num_masked_list = [1,5,10,15,20,25,30], num_faked_list=range(0), save_db=False, plot=False, bond_order=True):

    model = load_model(model_name)
    model.eval()
    if save_db: 
        db = Database("data/results.db")
        db.setup()
    #Set seed to make sure the same atoms are masked for each model
    np.random.seed(42)
    for masks in tqdm(num_masked_list):

        test_set = QM9Dataset(data='data/adjacency_matrix_test_scaffold.pkl', num_masks=masks, bond_order=bond_order, samples_per_molecule=5)
        test_dl = DataLoader(
            test_set,
            #batch_size=248
            batch_size=1024
)

        for batch_nr, batch in enumerate(test_dl):

            batch.cuda()
            output = model(batch)
            lengths = batch.lengths
            targets_batch = batch.targets_num
            target_masks = batch.target_mask
            smiles = batch.smiles

            targets_batch -= 1

            predictions_batch = output['prediction']
            out_batch = output['out']
            probabilities_batch = softmax(output['out'],dim=-1)

            if plot:
                for i in range(batch.batch_size):
                    if not torch.equal(targets_batch[i,:], predictions_batch[i,target_masks[i,:]]):
                        #plt.figure()
                        # for k in range(1,7):
                        #     plt.subplot(2,3,k)
                        #     plt.title(f'layer 1 head {k}')
                        #     plot_attention(smiles[i], batch.atoms[i],f'attention {k}',output[f'attention_weights_0'][i,k-1,target_masks[i,:],:])
                        
                        # plt.figure()
                        # for k in range(1,7):
                        #     plt.subplot(2,3,k)
                        #     plt.title(f'layer 2 head {k}')
                        #     plot_attention(smiles[i], batch.atoms[i],f'attention {k}',output[f'attention_weights_1'][i,k-1,target_masks[i,:],:])
 
                        # plt.figure()
                        # for k in range(1,7):
                        #     plt.subplot(2,3,k)
                        #     plt.title(f'layer 3 head {k}')
                        #     plot_attention(smiles[i], batch.atoms[i],f'attention {k}',output[f'attention_weights_2'][i,k-1,target_masks[i,:],:])
                        # plt.figure()
                        # for k in range(1,7):
                        #     plt.subplot(2,3,k)
                        #     plt.title(f'layer 4 head {k}')
                        #     plot_attention(smiles[i], batch.atoms[i],f'attention {k}',output[f'attention_weights_3'][i,k-1,target_masks[i,:],:])
                        # plt.figure()
                        # for k in range(1,7):
                        #     plt.subplot(2,3,k)
                        #     plt.title(f'layer 5 head {k}')
                        #     plot_attention(smiles[i], batch.atoms[i],f'attention {k}',output[f'attention_weights_4'][i,k-1,target_masks[i,:],:])
                        # plt.figure()
                        # for k in range(1,7):
                        #     plt.subplot(2,3,k)
                        #     plt.title(f'layer 6 head {k}')
                        #     plot_attention(smiles[i], batch.atoms[i],f'attention {k}',output[f'attention_weights_5'][i,k-1,target_masks[i,:],:])
                        # plt.figure()
                        # for k in range(1,7):
                        #     plt.subplot(2,3,k)
                        #     plt.title(f'layer 7 head {k}')
                        #     plot_attention(smiles[i], batch.atoms[i],f'attention {k}',output[f'attention_weights_6'][i,k-1,target_masks[i,:],:])
                        # plt.figure()
                        # for k in range(1,7):
                        #     plt.subplot(2,3,k)
                        #     plt.title(f'layer 8 head {k}')
                        #     plot_attention(smiles[i], batch.atoms[i],f'attention {k}',output[f'attention_weights_7'][i,k-1,target_masks[i,:],:])
 
                        # plt.figure()
                        # for k in range(1,9):
                        #     plt.subplot(2,4,k)
                        #     plt.title(f'Attention layer {k}')
                        #     plot_attention(smiles[i], batch.atoms[i],f'attention {k}',output[f'attention_weights_{k-1}'][i,0,target_masks[i,:],:])
                            
                        plot_prediction(smiles[i], batch.atoms[i], targets_batch[i,:], predictions_batch[i,target_masks[i,:]], probabilities_batch[i, target_masks[i,:],:])

            
            insert_values = []

            for targets, predictions, probabilities, out, target_mask, length, smile,  in zip(targets_batch,
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
                                       model_name,
                                       model_alias,
                                       masks, 
                                       prediction, 
                                       target, 
                                       loss,
                                       smile)]
            if save_db:
                db.stage_results(insert_values)
        if save_db:
            db.apply_staged()


if __name__ == "__main__":
    #generate_predictions("OctetRule", model_alias='OctetRule', save_db=True, plot=False)
    #generate_predictions("Unigram", model_alias="Unigram", save_db=True, plot=False)
    #generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=1_num_heads=1_embedding_dim=4_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False.pt", model_alias='Transformer Binary (l1h1d4)', save_db=True, plot=False, bond_order=False)
    #generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=2_num_heads=1_embedding_dim=4_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False.pt", model_alias='Transformer Binary (l2h1d4)', save_db=True, plot=False, bond_order=False)
    #generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=2_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False.pt", model_alias='Transformer Binary (l2h3d64)', save_db=True, plot=False, bond_order=False)
    #generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False.pt", model_alias='Transformer Binary (l8h6d64)', save_db=True, plot=False, bond_order=False)

    generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_db=alchemy.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=2.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=3.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=4.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=5.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=6.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=7.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=8.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=9.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=10.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)

    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False.pt", model_alias='Transformer Binary (medium)', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=2.pt", model_alias='Transformer Binary (medium)', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=3.pt", model_alias='Transformer Binary (medium)', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=4.pt", model_alias='Transformer Binary (medium)', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=5.pt", model_alias='Transformer Binary (medium)', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=6.pt", model_alias='Transformer Binary (medium)', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=7.pt", model_alias='Transformer Binary (medium)', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=8.pt", model_alias='Transformer Binary (medium)', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=9.pt", model_alias='Transformer Binary (medium)', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=10.pt", model_alias='Transformer Binary (medium)', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=7.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=8.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=9.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=10.pt", model_alias='Transformer Binary', save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True.pt", model_alias='Transformer Bondtype', save_db=True, plot=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=2.pt", model_alias='Transformer Bondtype', save_db=True, plot=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=3.pt", model_alias='Transformer Bondtype', save_db=True, plot=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=4.pt", model_alias='Transformer Bondtype', save_db=True, plot=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=5.pt", model_alias='Transformer Bondtype', save_db=True, plot=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=6.pt", model_alias='Transformer Bondtype', save_db=True, plot=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=7.pt", model_alias='Transformer Bondtype', save_db=True, plot=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=8.pt", model_alias='Transformer Bondtype', save_db=True, plot=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=9.pt", model_alias='Transformer Bondtype', save_db=True, plot=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=10.pt", model_alias='Transformer Bondtype', save_db=True, plot=False)
    generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_db=alchemy.pt", model_alias="Bag-of-Neighbours", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_nr=2.pt", model_alias="Bag-of-Neighbours", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_nr=3.pt", model_alias="Bag-of-Neighbours", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_nr=4.pt", model_alias="Bag-of-Neighbours", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_nr=5.pt", model_alias="Bag-of-Neighbours", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_nr=6.pt", model_alias="Bag-of-Neighbours", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_nr=7.pt", model_alias="Bag-of-Neighbours", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_nr=8.pt", model_alias="Bag-of-Neighbours", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_nr=9.pt", model_alias="Bag-of-Neighbours", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_nr=10.pt", model_alias="Bag-of-Neighbours", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1.pt", model_alias="Bag-of-Atoms", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_nr=2.pt", model_alias="Bag-of-Atoms", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_nr=3.pt", model_alias="Bag-of-Atoms", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_nr=4.pt", model_alias="Bag-of-Atoms", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_nr=5.pt", model_alias="Bag-of-Atoms", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_nr=6.pt", model_alias="Bag-of-Atoms", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_nr=7.pt", model_alias="Bag-of-Atoms", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_nr=8.pt", model_alias="Bag-of-Atoms", save_db=True, plot=False, bond_order=False)
    # generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_nr=9.pt", model_alias="Bag-of-Atoms", save_db=True, plot=False, bond_order=False)
    generate_predictions("BagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=2_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_db=alchemy.pt", model_alias="Bag-of-Atoms", save_db=True, plot=False, bond_order=False)
