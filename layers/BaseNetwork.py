from torch.nn import Module, CrossEntropyLoss, L1Loss, MSELoss
from torch.optim import Adam, lr_scheduler
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import shutil
import sklearn
import warnings
import wandb
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class BaseNetwork(Module):
    """
    Base class for out networks. This contains shared methods like
        - train_network: Method used for training the network
    """

    def __init__(self, name=None, log=False, use_cuda=False):
        super().__init__()

        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name

        self.use_cuda = use_cuda

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

    def get_numpy(self, x):
        if self.use_cuda:
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

        try:
            for epoch in range(num_epochs):

                self.train()
                train_loss, train_f1_micro, train_f1_macro, train_lengths, train_mae = 0, 0, 0, 0, 0
                # Train on the training dataset
                for train_batch in iter(train_iter):

                    if self.use_cuda:
                        train_batch.cuda()
                    # Get the predictions and targets
                    output = self(train_batch)

                    targets = train_batch.targets_num

                    targets = targets[targets != 0]
                    # If padding is 0, we are essentially offsetting all our predictions
                    targets -= 1

                    # set the gradients to zero
                    self.optimizer.zero_grad()

                    # Calculate the loss, and backpropagate
                    loss = 0

                    loss = criterion(output['out'], targets)
                    loss.backward()

                    # Optimize the network
                    self.optimizer.step()
                    prediction = self.get_numpy(output['prediction'])
                    targets = self.get_numpy(targets)
                    loss = self.get_numpy(loss)

                    train_loss += loss * train_batch.batch_size
                    train_f1_micro += f1_score(prediction, targets, average='micro') * train_batch.batch_size
                    train_f1_macro += f1_score(prediction, targets, average='macro') * train_batch.batch_size

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

                wandb.log({
                    'train_loss': train_loss,
                    'train_perplexity': np.exp(train_loss),
                    'train_f1_micro': train_f1_micro,
                    'train_f1_macro': train_f1_macro
                }, step=epoch)

                if epoch % eval_after_epochs == 0:

                    # Evaluate the results on the validation set
                    self.eval()

                    for i, validation_iter in enumerate(validation_iters):
                        val_loss, val_f1_micro, val_f1_macro, val_lengths, val_mae = 0, 0, 0, 0, 0

                        # Save the target and predictions for the validation set for the current epoch, for later
                        self.val_prediction_epoch = []
                        self.val_target_epoch = []
                        with torch.no_grad():
                            for val_batch in iter(validation_iter):
                                if self.use_cuda:
                                    val_batch.cuda()
                                output = self(val_batch)

                                targets = val_batch.targets_num

                                targets = targets[targets != 0]

                                targets -= 1

                                loss = criterion(output['out'], targets)

                                prediction = self.get_numpy(output['prediction'])
                                targets = self.get_numpy(targets)
                                loss = self.get_numpy(loss)

                                val_loss += loss * val_batch.batch_size
                                val_f1_micro += f1_score(prediction, targets, average='micro') * val_batch.batch_size
                                val_f1_macro += f1_score(prediction, targets, average='macro') * val_batch.batch_size

                                val_lengths += val_batch.batch_size

                        # divide by the total accumulated batch sizes
                        val_loss /= val_lengths
                        val_f1_micro /= val_lengths
                        val_f1_macro /= val_lengths
                        val_mae /= val_lengths

                        wandb.log({
                            f'val_loss_{i+1}': val_loss,
                            f'val_perplexity_{i+1}': np.exp(val_loss),
                            f'val_f1_micro_{i+1}': val_f1_micro,
                            f'val_f1_macro_{i+1}': val_f1_macro
                        }, step=epoch)

                        # Save the best parameters
                        if val_loss < self.min_loss:
                            self.min_loss = val_loss
                            wandb.run.summary['val_perplexity'] = np.exp(val_loss)
                            wandb.run.summary['val_f1_micro'] = val_f1_micro
                            wandb.run.summary['val_f1_macro'] = val_f1_macro
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

    def save(self, file_path=None):
        "Save the parameters of the model"
        if file_path is None:
            if exists('saved_models'):
                file_path = 'saved_models/{}.pt'.format(self.name)
            elif exists('../saved_models'):
                file_path = '../saved_models/{}.pt'.format(self.name)
            else:
                raise IOError('saved_models folder not found')

        torch.save(self.state_dict(), file_path)

    def load(self, file_path=None):
        "Load the parameters of the model from a checkpoint"

        if file_path is None:
            if exists('saved_models'):
                file_path = 'saved_models/{}.pt'.format(self.name)
            elif exists('../saved_models'):
                file_path = '../saved_models/{}.pt'.format(self.name)
            else:
                raise IOError('saved_models folder not found')

        if exists(file_path):
            self.load_state_dict(torch.load(file_path))
        else:
            raise FileNotFoundError("Couldn't load model since file {} doesn't exists".format(file_path))
