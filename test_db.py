from utils.dataloader import QM9Dataset, DataLoader
from layers.transformer import TransformerModel
from layers.bagofwords import BagOfWordsModel, SimpleBagOfWordsModel
from layers.Unigram import UnigramModel
from layers.octetRule import OctetRuleModel
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
    params = re.split(r'(?<=\d)_|(?<=True)_|(?<=False)_|(?<=zinc)_|(?<=qm9)_',model_name)
    name = params[0]

    param_dict={'dataset':'qm9'}

    if 'Unigram' in name:
        model = UnigramModel(dataset=param_dict['dataset'])
    elif 'OctetRule' in name:
        model = OctetRuleModel(dataset=param_dict['dataset'], k=0 )

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
        model = SimpleBagOfWordsModel(num_layers=param_dict['num_layers'],
                                    embedding_dim=param_dict['embedding_dim'],
                                    BagOfWordsType=param_dict['bow_type'],
                                    num_classes=10 if param_dict['dataset']=='zinc' else 5
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
                                    edge_encoding=param_dict['edge_encoding'],
                                    num_classes=10 if param_dict['dataset']=='zinc' else 5
                                )
        model.load('saved_models/'+model_name)
    if not ('Unigram' in name or 'Octet' in name):
        model.cuda()
    return model, param_dict

def generate_predictions(model_name, model_alias, num_masked_list = [1,10,20,30,40,50,60,70,80], num_faked_list=range(0), save_db=False, plot=False, bond_order=True):

    model, params = load_model(model_name)
    model.eval()
    if save_db:
        db = Database(f"/work1/s180213/results_{params['dataset']}.db")
        #db = Database(f"data/results_{params['dataset']}.db")
        db.setup()
    #Set seed to make sure the same atoms are masked for each model
    np.random.seed(42)
    for masks in tqdm(num_masked_list):

        samples_per_molecule = 5 if masks <= 5 else 1
        test_set = QM9Dataset(data=f"data/{params['dataset']}/adjacency_matrix_test_scaffold.pkl", num_masks=0, num_fake=masks, bond_order=bond_order, samples_per_molecule=samples_per_molecule)
        test_dl = DataLoader(
            test_set,
            batch_size=248
            #batch_size=1024
)

        for batch_nr, batch in enumerate(test_dl):
            if not ('Unigram' in model_alias or 'Octet' in model_alias):
                batch.cuda()
            with torch.no_grad():
                output = model(batch)
                lengths = batch.lengths
                targets_batch = batch.targets_num
                target_masks = batch.target_mask
                smiles = batch.smiles
                charges_batch = batch.charges
                num_neighbours_batch = batch.num_neighbours

                targets_batch -= 1

                predictions_batch = output['prediction']
                out_batch = output['out']
                probabilities_batch = softmax(output['out'],dim=-1)

                if plot:

                    for i in range(batch.batch_size):
                        #if not torch.equal(targets_batch[i,:], predictions_batch[i,target_masks[i,:]]):
                        # if torch.equal(targets_batch[i,:], torch.tensor([6]).cuda()) and torch.equal(num_neighbours_batch[i,target_masks[i,:]],torch.tensor([6])):#and not torch.equal(targets_batch[i,:], torch.tensor([2]).cuda()): ## 
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

                for targets, predictions, probabilities, out, target_mask, length, smile,charges, num_neighbours  in zip(targets_batch,
                                                                                             predictions_batch,
                                                                                             probabilities_batch,
                                                                                             out_batch,
                                                                                             target_masks,
                                                                                             lengths,
                                                                                             smiles, charges_batch,
                                                                                             num_neighbours_batch):

                    targets = targets[targets!=-1]
                    out = out[target_mask]
                    losses = cross_entropy(out, targets, reduction='none')

                    targets = tensor_to_list(targets)
                    probabilities = tensor_to_list(probabilities[target_mask])
                    losses = tensor_to_list(losses)
                    predictions = tensor_to_list(predictions[target_mask])
                    charges = tensor_to_list(charges[target_mask])
                    num_neighbours = tensor_to_list(num_neighbours[target_mask])
                    # indecies = tensor_to_list(target_mask.nonzero()[:,0])
                    length = length.item()

                    for target, prediction, probability, loss,charge, num_neighbour in zip(targets, predictions, probabilities, losses,charges, num_neighbours):
                        insert_values += [(length,
                                        model_name,
                                        model_alias,
                                        masks,
                                        prediction,
                                        target,
                                        loss,
                                        smile,
                                        charge,
                                        num_neighbour
                                        )]
            if save_db:
                db.stage_results(insert_values)
        if save_db:
            db.apply_staged()


if __name__ == "__main__":


    #generate_predictions("OctetRule", model_alias='OctetRule', save_db=True, plot=False)
    # generate_predictions("Unigram", model_alias="Unigram", save_db=True, plot=False)

    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_dataset=zinc.pt", "transformer",save_db=False, plot=True, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=2.pt", "transformer",save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=3.pt", "transformer",save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=4.pt", "transformer",save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=5.pt", "transformer",save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=6.pt", "transformer",save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=7.pt", "transformer",save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=8.pt", "transformer",save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=9.pt", "transformer",save_db=True, plot=False, bond_order=False)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=8_num_heads=6_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_nr=10.pt", "transformer",save_db=True, plot=False, bond_order=False)

    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_dataset=zinc.pt", "Bag-of-neighbors",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_dataset=zinc_nr=2.pt", "Bag-of-neighbors",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_dataset=zinc_nr=3.pt", "Bag-of-neighbors",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_dataset=zinc_nr=4.pt", "Bag-of-neighbors",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_dataset=zinc_nr=5.pt", "Bag-of-neighbors",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_dataset=zinc_nr=6.pt", "Bag-of-neighbors",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_dataset=zinc_nr=7.pt", "Bag-of-neighbors",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_dataset=zinc_nr=8.pt", "Bag-of-neighbors",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_dataset=zinc_nr=9.pt", "Bag-of-neighbors",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=2_dataset=zinc_nr=10.pt", "Bag-of-neighbors",save_db=True, plot=False, bond_order=False)

    generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_dataset=zinc.pt", "Bag-of-atoms",save_db=False, plot=True, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_dataset=zinc_nr=2.pt", "Bag-of-atoms",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_dataset=zinc_nr=3.pt", "Bag-of-atoms",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_dataset=zinc_nr=4.pt", "Bag-of-atoms",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_dataset=zinc_nr=5.pt", "Bag-of-atoms",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_dataset=zinc_nr=6.pt", "Bag-of-atoms",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_dataset=zinc_nr=7.pt", "Bag-of-atoms",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_dataset=zinc_nr=8.pt", "Bag-of-atoms",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_dataset=zinc_nr=9.pt", "Bag-of-atoms",save_db=True, plot=False, bond_order=False)
    # generate_predictions("SimpleBagOfWords_num_masks=1_num_fake=0_num_same=0_num_layers=4_embedding_dim=64_lr=0.001_epsilon_greedy=0.2_bow_type=1_dataset=zinc_nr=10.pt", "Bag-of-atoms",save_db=True, plot=False, bond_order=False)

    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True.pt", "transformer bond",save_db=True, plot=False, bond_order=True)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=2.pt", "transformer bond",save_db=True, plot=False, bond_order=True)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=3.pt", "transformer bond",save_db=True, plot=False, bond_order=True)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=4.pt", "transformer bond",save_db=True, plot=False, bond_order=True)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=5.pt", "transformer bond",save_db=True, plot=False, bond_order=True)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=6.pt", "transformer bond",save_db=True, plot=False, bond_order=True)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=7.pt", "transformer bond",save_db=True, plot=False, bond_order=True)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=8.pt", "transformer bond",save_db=True, plot=False, bond_order=True)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=9.pt", "transformer bond",save_db=True, plot=False, bond_order=True)
    # generate_predictions("Transformer_num_masks=1_num_fake=0_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=True_nr=10.pt", "transformer bond",save_db=False, plot=True, bond_order=True)



    # generate_predictions("Transformer_num_masks=0_num_fake=1_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_dataset=qm9.pt", "transformer bond",save_db=False, plot=True, bond_order=False)
                          #Transformer_num_masks=0_num_fake=1_num_same=0_num_layers=4_num_heads=3_embedding_dim=64_dropout=0.0_lr=0.001_edge_encoding=1_epsilon_greedy=0.2_gamma=1_bond_order=False_dataset=zinc.pt