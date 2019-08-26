from torch.nn import Module, CrossEntropyLoss, L1Loss, MSELoss
from torch.optim import Adam, lr_scheduler
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from torch.cuda import is_available as has_cuda
from tensorboardX import SummaryWriter
#import seaborn as sn
import pandas as pd
from os.path import exists
import shutil
import sklearn
import warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class BaseNetwork(Module):
    """
    Base class for out networks. This contains shared methods like
        - train_network: Method used for training the network
    """

    def __init__(self, name=None, log=False):
        super().__init__()

        if name is None:
            self.__name = type(self).__name__
        else:
            self.__name = name

        if has_cuda():
            self.use_cuda = True
        else:
            self.use_cuda = False

        self.train_loss = []
        self.train_f1_micro = []
        self.train_f1_macro = []
        self.train_perplexity = []
        self.val_loss = []
        self.val_f1_micro = []
        self.val_f1_macro = []
        self.val_perplexity = []
        self.val_prediction_epoch = []
        self.val_target_epoch = []
        self.min_loss = float('inf')

        if log:
            logdir_train = f"logs/{self.__name}_train"
            logdirs_val = [f"logs/{self.__name}_val1",f"logs/{self.__name}_val2",f"logs/{self.__name}_val3",f"logs/{self.__name}_val4",f"logs/{self.__name}_val5"]
            self.writer_train = SummaryWriter(log_dir=logdir_train)
            self.writers_val = [SummaryWriter(log_dir=logdir) for logdir in logdirs_val]

    def get_numpy(self, x):
        if self.cuda:
            return x.cpu().data.numpy()
        return x.data.numpy()

    def train_network(self,
                      train_iter,
                      validation_iters,
                      batch_size=50,
                      num_epochs=100,
                      optimizer_fun=lambda param: Adam(param),
                      scheduler_fun=None,
                      eval_after_epochs=10,
                      log_after_epochs=100,
                      visualize=False,
                      save_model=True
                      ):
        """
        Base method for training any network which inherits from this class

        Args:
            train_iter (iterator): Training dataset. Should be a torch dataset
            test_iter (iterator): validation dataset. Should be a torch dataset
            batch_size (int): The batch_size used for training.
            num_epochs (int): Number of epochs to run the training for
            optimizer_fun (function): The function used to contruct the optimizer. 
                                    Should take the parameters of the network as input
            criterion (function): The loss function that should be used
            shuffle (Boolean): Flag to shuffle the data or not

        """
        
        # Save some of the informations in the class for later
        self.validation_iters = validation_iters
        self.train_iter = train_iter
        criterion = CrossEntropyLoss()
        # initialize the optimizer
        self.optimizer = optimizer_fun(self.parameters())

        if scheduler_fun is not None:
            self.scheduler = scheduler_fun(self.optimizer)
        else:
            self.scheduler = None

        # Enable CUDA
        if self.use_cuda:
            self.cuda()

        # Log that we are continuing the training
        epoch = len(self.val_loss)
        if epoch > 0:
            if save_model:
                best_idx = np.argmin(self.val_loss)
                print("Continuing from last best model {i}".format(i=best_idx))
                self.train_loss = self.train_loss[:best_idx]
                self.train_perplexity = self.train_perplexity[:best_idx]
                self.train_f1_macro = self.train_f1_macro[:best_idx]
                self.train_f1_micro = self.train_f1_micro[:best_idx]
                self.val_loss = self.val_loss[:best_idx]
                self.val_perplexity = self.val_perplexity[:best_idx]
                self.val_f1_macro = self.val_f1_macro[:best_idx]
                self.val_f1_micro = self.val_f1_micro[:best_idx]
            else:
                print("Continuing training from epoch {i}".format(i=epoch))
                # If we stopped with an uneven number of training and validation epochs, just through away the extra training
                if len(self.train_loss) != epoch:
                    self.train_loss = self.train_loss[:epoch]
                    self.train_perplexity = self.train_perplexity[:epoch]
                    self.train_f1_macro = self.train_f1_macro[:epoch]
                    self.train_f1_micro = self.train_f1_micro[:epoch]

        try:
            pad = train_iter.dataset.fields['target'].vocab.stoi['<pad>']
            for epoch in range(num_epochs):

                self.train()
                train_loss, train_f1_micro, train_f1_macro, train_lengths, train_mae = 0, 0, 0, 0, 0
                # Train on the training dataset
                for train_batch in iter(train_iter):
                    # Get the predictions and targets
                    output = self(train_batch)

                    target = train_batch.target
                    if len(target.shape) != 1:
                        target = target.reshape(-1)
                        target = target[target != pad]
                        #If padding is 0, we are essentially offsetting all our predictions
                        if pad==0:
                            target -= 1

                    # set the gradients to zero
                    self.optimizer.zero_grad()

                    # Calculate the loss, and backpropagate
                    loss=0

                    loss = criterion(output['out'], target)
                    loss.backward()

                    # Optimize the network
                    self.optimizer.step()
                    prediction = self.get_numpy(output['prediction'])
                    target = self.get_numpy(target)
                    loss = self.get_numpy(loss)

                    train_loss += loss * train_batch.batch_size
                    train_f1_micro += f1_score(prediction, target, average='micro') * train_batch.batch_size
                    train_f1_macro += f1_score(prediction, target, average='macro') * train_batch.batch_size

                    train_lengths += train_batch.batch_size


                # divide by the total accumulated batch sizes
                train_loss /= train_lengths
                train_f1_micro /= train_lengths
                train_f1_macro /= train_lengths
                train_mae /= train_lengths

                self.train_loss += [train_loss]
                self.train_f1_micro += [train_f1_micro]
                self.train_f1_macro += [train_f1_macro]
                self.train_perplexity += [np.exp(train_loss)]


                self.writer_train.add_scalar('loss', self.train_loss[-1], epoch)
                self.writer_train.add_scalar('perplexity', self.train_perplexity[-1], epoch)
                self.writer_train.add_scalar('f1_micro', self.train_f1_micro[-1], epoch)
                self.writer_train.add_scalar('f1_macro', self.train_f1_macro[-1], epoch)

                if epoch % eval_after_epochs == 0:

                    # Evaluate the results on the validation set
                    self.eval()

                    for validation_iter,writer_val in zip(validation_iters, self.writers_val):
                        val_loss, val_f1_micro, val_f1_macro, val_lengths, val_mae = 0, 0, 0, 0, 0

                        # Save the target and predictions for the validation set for the current epoch, for later
                        self.val_prediction_epoch = []
                        self.val_target_epoch = []
                        with torch.no_grad():
                            for val_batch in iter(validation_iter):
                                output = self(val_batch)

                                target = val_batch.target
                                if len(target.shape) != 1:
                                    target = target.reshape(-1)
                                    target = target[target != pad]
                                    if pad==0:
                                        target -= 1
                                
                                #self.optimizer.zero_grad()
                                loss = 0

                                loss = criterion(output['out'], target)

                                prediction = self.get_numpy(output['prediction'])
                                target = self.get_numpy(target)
                                loss = self.get_numpy(loss)

                                val_loss += loss * val_batch.batch_size
                                val_f1_micro += f1_score(prediction, target, average='micro') * val_batch.batch_size
                                val_f1_macro += f1_score(prediction, target, average='macro') * val_batch.batch_size

                                val_lengths += val_batch.batch_size 

                        # divide by the total accumulated batch sizes
                        val_loss /= val_lengths
                        val_f1_micro /= val_lengths
                        val_f1_macro /= val_lengths
                        val_mae /= val_lengths
                        
                        writer_val.add_scalar('loss', val_loss, epoch)
                        writer_val.add_scalar('perplexity', np.exp(val_loss), epoch)
                        writer_val.add_scalar('f1_micro', val_f1_micro, epoch)
                        writer_val.add_scalar('f1_macro', val_f1_macro, epoch)
                        
                        # Save the best parameters
                        if val_loss< self.min_loss:
                            self.min_loss = val_loss

                            if save_model:
                                self.save()

                # Update learning rate scheduler if exists
                if self.scheduler is not None:
                    self.scheduler.step()


        # Instead of crashing the training, when you stop a kernel, this will just stop the training
        except KeyboardInterrupt:
            print("Stopping training, and loading best model")

        if save_model:
            # Load the best parameters from training
            self.load()

        # find the best model during training
        # best_idx = np.argmin(self.val_loss)
        # print("*****************Best model******************************************\n"
        #       "loss = {0:.3f}  Perplexity = {1:.3f}   F1 (micro/macro) = ({2:.3f} / {3:.3f}) \n"
        #       "*********************************************************************".format(
        #           self.val_loss[best_idx], self.val_perplexity[best_idx], self.val_f1_micro[best_idx], self.val_f1_macro[best_idx]
        #       )
        #     )
    
        

    def save(self, file_path=None):
        "Save the parameters of the model"
        if file_path is None:
            if exists('models'):		
                file_path = 'models/{}.pt'.format(self.__name)
            elif exists('../models'):
                file_path = '../models/{}.pt'.format(self.__name)
            else:
                raise IOError('models folder not found')

        torch.save(self.state_dict(), file_path)

    def load(self, file_path=None):
        "Load the parameters of the model from a checkpoint"
        

        if file_path is None:
            if exists('models'):		
                file_path = 'models/{}.pt'.format(self.__name)
            elif exists('../models'):
                file_path = '../models/{}.pt'.format(self.__name)
            else:
                raise IOError('models folder not found')


        if exists(file_path):
            self.load_state_dict(torch.load(file_path))
        else:
            raise FileNotFoundError("Couldn't load model since file {} doesn't exists".format(file_path))

    def confusion_matrix(self, class_labels, normalized=True, val_iter=None, criterion=CrossEntropyLoss()):

        if self.val_prediction_epoch != [] and self.val_target_epoch != [] and val_iter is None:
            cm = confusion_matrix(self.val_prediction_epoch, self.val_target_epoch).T
        elif val_iter is not None:
            pad = val_iter.dataset.fields['target'].vocab.stoi['<pad>']
            self.eval()
            val_loss, val_f1_micro, val_f1_macro, val_lengths = 0, 0, 0, 0
            self.val_prediction_epoch = []
            self.val_target_epoch = []

            for batch in iter(val_iter):

                output = self(batch)

                target = batch.target
                if len(target.shape) != 1:
                    target = target.reshape(-1)
                    target = target[target != pad]
                    if pad==0:
                        target -= 1
                    

                loss = criterion(output['out'], target)

                loss = self.get_numpy(loss)
                target = self.get_numpy(target)
                prediction = self.get_numpy(output['prediction'])

                self.val_prediction_epoch.extend(prediction)
                self.val_target_epoch.extend(target)

                val_loss += loss * batch.batch_size
                val_f1_micro += f1_score(prediction, target, average='micro') * batch.batch_size
                val_f1_macro += f1_score(prediction, target, average='macro') * batch.batch_size

                val_lengths += batch.batch_size
             # divide by the total accumulated batch sizes
            val_loss /= val_lengths
            val_f1_micro /= val_lengths
            val_f1_macro /= val_lengths
            self.val_loss += [val_loss]
            self.val_f1_micro += [val_f1_micro]
            self.val_f1_macro += [val_f1_macro]
            self.val_perplexity += [np.exp(val_loss)]

            print("*****************Best model******************************************\n"
                  "loss = {0:.3f}  Perplexity = {1:.3f}   F1 (micro/macro) = ({2:.3f} / {3:.3f}) \n"
                  "*********************************************************************".format(
                      self.val_loss[-1], self.val_perplexity[-1], self.val_f1_micro[-1], self.val_f1_macro[-1]
                  )
                  )
            cm = confusion_matrix(self.val_prediction_epoch, self.val_target_epoch).T
        else:
            raise NotImplementedError('No validation samples available. Run the training loop')

        cm = pd.DataFrame(cm, class_labels, class_labels)

        if normalized:
            cm = cm.astype('float')/(cm.sum()+1e-12)

        fig = plt.figure()
        sn.heatmap(cm, annot=True, annot_kws={"size": 12}, cmap="Greens")
        plt.xlabel('Prediction')
        plt.ylabel('Truth')

        self.writer_val.add_figure('confusion matrix',fig)


        fig = plt.figure()
        tgt = np.histogram(self.val_target_epoch, bins=range(len(class_labels)+1))[0]
        pred = np.histogram(self.val_prediction_epoch, bins=range(len(class_labels)+1))[0]
        a = pred/tgt
        plt.bar([0, 1, 2, 3, 4], a, label='Prediction')
        plt.plot([-0.4, 4.4], [1, 1], 'r--', label='Target')
        plt.xticks(np.arange(0, len(class_labels)), class_labels)
        plt.xlabel('Atom')
        plt.ylabel('Density')
        plt.legend()

        self.writer_val.add_figure('prediction distribution', fig)
    def plot_PP_per_num_atoms(self, val_iters):
        
        criterion = CrossEntropyLoss(reduction='none')

        fig = plt.figure()
        for val_iter in val_iters:
            perplexity = {}
            pad = val_iter.dataset.fields['target'].vocab.stoi['<pad>']
            for batch in iter(val_iter):
                output = self(batch)

                lengths = batch.atoms[1]
                targets = batch.target
                if len(targets.shape) != 1:
                    targets = targets.reshape(-1)
                    targets = targets[targets != pad]
                    if pad==0:
                        targets -= 1

                loss = criterion(output['out'], targets)

                for pp, length in zip(loss, lengths):
                    pp = pp.item()
                    length = length.item()
                    if length in perplexity:
                        perplexity[length].append(pp)
                    else:
                        perplexity[length] = [pp]

            perplexity_per_length = np.exp([np.mean(perplexity[l]) for l in sorted(perplexity)])
            lengths = [l for l in sorted(perplexity)]
            if 'fake' in val_iter.dataset.datafile:
                label = "$n_{fake}$ = "+val_iter.dataset.datafile[-5]
            else:
                label = "$n_{masked}$ = "+val_iter.dataset.datafile[-5]
            
            plt.plot(lengths, perplexity_per_length, label=label)
        plt.legend()
        plt.xlabel('Molecule size')
        plt.ylabel('Perplexity')
        plt.ylim(0,10)
        self.writers_val[0].add_figure('perplexity_per_length', fig)



    def plot_accuracy_per_num_atoms(self, val_iters):
        fig = plt.figure()
        for val_iter in val_iters:
            accuracy = {}
            pad = val_iter.dataset.fields['target'].vocab.stoi['<pad>']
            for batch in iter(val_iter):
                output = self(batch)

                lengths = batch.atoms[1]
                targets = batch.target
                if len(targets.shape) != 1:
                    targets = targets.reshape(-1)
                    targets = targets[targets != pad]
                    if pad==0:
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

            accuracy_per_length = [sum(accuracy[l])/len(accuracy[l]) for l in sorted(accuracy)]
            lengths = [l for l in sorted(accuracy)]
            if 'fake' in val_iter.dataset.datafile:
                label = "$n_{fake}$ = "+val_iter.dataset.datafile[-5]
            else:
                label = "$n_{masked}$ = "+val_iter.dataset.datafile[-5]
            

            plt.plot(lengths, accuracy_per_length, label=label)
        plt.legend()
        plt.xlabel('Molecule size')
        plt.ylabel('Accuracy')
        self.writers_val[0].add_figure('accuracy_per_length', fig)

        self.writers_val[0].add_figure('dummy', plt.figure())


    def plot_attention(self, batch, molecule=0, layers=None):
        '''
        plots the attention weights as an image

        '''
        out = self(batch)
        attention_layers = [key for key in out.keys() if 'attention_weights' in key]

        if len(attention_layers) == 0:
            raise RuntimeError('No attention layers found in output of model')
        
        atoms = batch.atoms
        if isinstance(atoms, tuple):
            atoms = atoms[0]
        atoms = atoms[molecule,:].cpu().numpy()

        for i, attention_layer in enumerate(attention_layers):
            if layers is not None:
                if i not in layers:
                    continue

            fig,ax = plt.subplots()
            if len(out[attention_layer].shape) == 3:
                plt.imshow(out[attention_layer][molecule,:,:].detach().cpu().numpy())

            elif len(out[attention_layer].shape) == 4:
                plt.imshow(out[attention_layer][molecule,0,:,:].detach().cpu().numpy())
 
            else:
                raise RuntimeError(f'Attention should have dimention [batch_size, molecule_size, molecule_size] but had dimension {out[attention_layer].shape}')

        
            ax.set_xticks(np.arange(len(atoms)))
            ax.set_xticklabels(atoms)
            ax.set_yticks(np.arange(len(atoms)))
            plt.colorbar()
            ax.set_yticklabels(atoms)

            self.writer_val.add_figure(f"attention_layers/{attention_layer}", fig)
 
        #For some reason, the figure writer doesn't show last figure, so create an empty dummy    
        fig = plt.figure()
        self.writer_val.add_figure('dummy', fig)
