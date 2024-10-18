from momentfm.data.ptbxl_classification_dataset import PTBXL_dataset
from momentfm import MOMENTPipeline
from momentfm.models.statistical_classifiers import fit_svm

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

import argparse
from argparse import Namespace
import random
import numpy as np
import os 
import pdb

def control_randomness(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PTBXL_Trainer:
    def __init__(self, args: Namespace):
        self.args = args

        #initialize ptbxl classification dataset 
        self.train_dataset = PTBXL_dataset(args, phase='train')
        self.val_dataset = PTBXL_dataset(args, phase='val')
        self.test_dataset = PTBXL_dataset(args, phase='test')

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False)

        #linear probing: only train classification head
        #finetuning: train both encoder and classification head
        #unsupervised learning: train SVM on top of MOMENT embeddings
        self.model = MOMENTPipeline.from_pretrained(
                                                    "AutonLab/MOMENT-1-large", 
                                                    model_kwargs={
                                                        'task_name': 'classification',
                                                        'n_channels': 12,
                                                        'num_class': 5,
                                                        'freeze_encoder': False if self.args.mode == 'full_finetuning' else True,
                                                        'freeze_embedder': False if self.args.mode == 'full_finetuning' else True,
                                                        'reduction': self.args.reduction,
                                                        #Disable gradient checkpointing for finetuning or linear probing to 
                                                        #avoid warning as MOMENT encoder is frozen
                                                        'enable_gradient_checkpointing': False if self.args.mode in ['full_finetuning', 'linear_probing'] else True, 
                                                    },
                                                )
        self.model.init()
        print('Model initialized, training mode: ', self.args.mode)

        #using cross entropy loss for classification 
        self.criterion = torch.nn.CrossEntropyLoss()
        
        if self.args.mode == 'full_finetuning':
            print('Encoder and embedder are trainable')
            if self.args.lora:
                lora_config = LoraConfig(
                                        r=64,
                                        lora_alpha=32,
                                        target_modules=["q", "v"],
                                        lora_dropout=0.05,
                                        )
                self.model = get_peft_model(self.model, lora_config)
                print('LoRA enabled')
                self.model.print_trainable_parameters()

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.init_lr)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.max_lr, 
                                                            total_steps=self.args.epochs*len(self.train_dataloader))
            
            #set up model ready for accelerate finetuning
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader)
        
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.init_lr)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.max_lr, 
                                                            total_steps=self.args.epochs*len(self.train_dataloader))
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #create log file to store training logs 
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path, exist_ok=True)
        self.log_file = open(os.path.join(self.args.output_path, f'log_{self.args.mode}.txt'), 'w')
        self.log_file.write(f'PTBXL classification training, mode: {self.args.mode}\n')

    
    def get_embeddings(self, dataloader: DataLoader):
        '''
        labels: [num_samples]
        embeddings: [num_samples x d_model]
        '''
        embeddings, labels = [], []

        with torch.no_grad():
            for batch_x, batch_labels in tqdm(dataloader, total=len(dataloader)):
                # [batch_size x 12 x 512]
                batch_x = batch_x.to(self.device).float()
                # [batch_size x num_patches x d_model (=1024)]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                    output = self.model(x_enc=batch_x, reduction=self.args.reduction) 
                # mean over patches dimension, [batch_size x d_model]
                embedding = output.embeddings.mean(dim=1)
                embeddings.append(embedding.detach().cpu().numpy())
                labels.append(batch_labels)        

        embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
        return embeddings, labels
    
    def get_timeseries(self, dataloader: DataLoader, agg='mean'):
        '''
        mean: average over all channels, result in [1 x seq_len] for each time-series
        channel: concat all channels, result in [1 x seq_len * num_channels] for each time-series

        labels: [num_samples]
        ts: [num_samples x seq_len] or [num_samples x seq_len * num_channels]
        '''
        ts, labels = [], []

        with torch.no_grad():
            for batch_x, batch_labels in tqdm(dataloader, total=len(dataloader)):
                # [batch_size x 12 x 512]
                if agg == 'mean':
                    batch_x = batch_x.mean(dim=1)
                    ts.append(batch_x.detach().cpu().numpy())
                elif agg == 'channel':
                    ts.append(batch_x.view(batch_x.size(0), -1).detach().cpu().numpy())
                labels.append(batch_labels)        

        ts, labels = np.concatenate(ts), np.concatenate(labels)
        return ts, labels
    
    def train(self):
        for epoch in range(self.args.epochs):

            print(f'Epoch {epoch+1}/{self.args.epochs}')
            self.log_file.write(f'Epoch {epoch+1}/{self.args.epochs}\n')
            self.epoch = epoch + 1

            if self.args.mode == 'linear_probing':
                self.train_epoch_lp()
                self.evaluate_epoch()
            
            elif self.args.mode == 'full_finetuning':
                self.train_epoch_ft()
                self.evaluate_epoch()
            
            #break after training SVM, only need one 'epoch'
            elif self.args.mode == 'unsupervised_representation_learning':
                self.train_ul()
                break

            elif self.args.mode == 'svm':
                self.train_svm()
                break

            else:
                raise ValueError('Invalid mode, please choose svm, linear_probing, full_finetuning, or unsupervised_representation_learning')

#####################################training loops#############################################
    def train_epoch_lp(self):
        '''
        Train only classification head
        '''
        self.model.to(self.device)
        self.model.train()
        losses = []

        for batch_x, batch_labels in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            self.optimizer.zero_grad()
            batch_x = batch_x.to(self.device).float()
            batch_labels = batch_labels.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                output = self.model(x_enc=batch_x, reduction=self.args.reduction)
                loss = self.criterion(output.logits, batch_labels)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        print('Train loss: ', avg_loss)
        self.log_file.write(f'Train loss: {avg_loss}\n')
    
    def train_epoch_ft(self):
        '''
        Train encoder and classification head (with accelerate enabled)
        '''
        self.model.to(self.device)
        self.model.train()
        losses = []

        for batch_x, batch_labels in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            self.optimizer.zero_grad()
            batch_x = batch_x.to(self.device).float()
            batch_labels = batch_labels.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                output = self.model(x_enc=batch_x, reduction=self.args.reduction)
                loss = self.criterion(output.logits, batch_labels)
                losses.append(loss.item())
            self.accelerator.backward(loss)
            
            self.optimizer.step()
            self.scheduler.step()

        avg_loss = np.mean(losses)
        print('Train loss: ', avg_loss)
        self.log_file.write(f'Train loss: {avg_loss}\n')
    
    def train_ul(self):
        '''
        Train SVM on top of MOMENT embeddings
        '''
        self.model.eval()
        self.model.to(self.device)

        #extract embeddings and label
        train_embeddings, train_labels = self.get_embeddings(self.train_dataloader)
        print('embedding shape: ', train_embeddings.shape)
        print('label shape: ', train_labels.shape)

        #fit statistical classifier
        self.clf = fit_svm(features=train_embeddings, y=train_labels)
        train_accuracy = self.clf.score(train_embeddings, train_labels)
        print('Train accuracy: ', train_accuracy)
        self.log_file.write(f'Train accuracy: {train_accuracy}\n')

    def train_svm(self):
        '''
        Train SVM on top of timeseries data
        '''
        train_embeddings, train_labels = self.get_timeseries(self.train_dataloader, agg=self.args.agg)
        self.clf = fit_svm(features=train_embeddings, y=train_labels)
        train_accuracy = self.clf.score(train_embeddings, train_labels)
        print('Train accuracy: ', train_accuracy)
        self.log_file.write(f'Train accuracy: {train_accuracy}\n')
#####################################training loops#################################################

#####################################evaluate loops#################################################
    def test(self):
        if self.args.mode == 'unsupervised_representation_learning':
            test_embeddings, test_labels = self.get_embeddings(self.test_dataloader)
            test_accuracy = self.clf.score(test_embeddings, test_labels)
            print('Test accuracy: ', test_accuracy)
            self.log_file.write(f'Test accuracy: {test_accuracy}\n')

        elif self.args.mode == 'linear_probing' or self.args.mode == 'full_finetuning':
            self.evaluate_epoch(phase='test')

        elif self.args.mode =='svm':
            test_embeddings, test_labels = self.get_timeseries(self.test_dataloader, agg=self.args.agg)
            test_accuracy = self.clf.score(test_embeddings, test_labels)
            print('Test accuracy: ', test_accuracy)
            self.log_file.write(f'Test accuracy: {test_accuracy}\n')

        else:
            raise ValueError('Invalid mode, please choose linear_probing, full_finetuning, or unsupervised_representation_learning')
        
    def evaluate_epoch(self, phase='val'):
        if phase == 'val':
            dataloader = self.val_dataloader
        elif phase == 'test':
            dataloader = self.test_dataloader
        else:
            raise ValueError('Invalid phase, please choose val or test')

        self.model.eval()
        self.model.to(self.device)
        total_loss, total_correct = 0, 0

        with torch.no_grad():
            for batch_x, batch_labels in tqdm(dataloader, total=len(dataloader)):
                batch_x = batch_x.to(self.device).float()
                batch_labels = batch_labels.to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                    output = self.model(x_enc=batch_x)
                    loss = self.criterion(output.logits, batch_labels)
                total_loss += loss.item()
                total_correct += (output.logits.argmax(dim=1) == batch_labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / len(dataloader.dataset)
        print(f'{phase} loss: {avg_loss}, {phase} accuracy: {accuracy}')
        self.log_file.write(f'{phase} loss: {avg_loss}, {phase} accuracy: {accuracy}\n')
#####################################evaluate loops#################################################

    def save_checkpoint(self):
        if self.args.mode in ['svm', 'unsupervised_representation_learning']:
            raise ValueError('No checkpoint to save for SVM or unsupervised learning, as no training was done')
        
        path = self.args.output_path

        #mkdir if not exist
        if not os.path.exists(path):
            os.makedirs(path)

        #save parameter that requires grad 
        torch.save(self.model.state_dict(), os.path.join(path, 'MOMENT_Classification.pth'))
        print('Model saved at ', path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--mode', type=str, default='full_finetuning', help='choose from linear_probing, full_finetuning, unsupervised_representation_learning')
    parser.add_argument('--init_lr', type=float, default=1e-6)
    parser.add_argument('--max_lr', type=float, default=1e-4)
    parser.add_argument('--agg', type=str, default='channel', help='aggregation method for timeseries data for svm training, choose from mean or channel')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lora', action='store_true', help='enable LoRA')
    parser.add_argument('--reduction', type=str, default='concat', help='reduction method for MOMENT embeddings, choose from mean or max')
    #ptbxl dataset parameters
    parser.add_argument('--base_path', type=str, help='path to PTBXL dataset')
    parser.add_argument('--cache_dir', type=str, help='path to cache directory to store preprocessed dataset')
    parser.add_argument('--output_path', type=str, help='path to save trained model and logs')
    parser.add_argument('--fs', type=int, default=100, help='sampling frequency, choose from 100 or 500')
    parser.add_argument('--code_of_interest', type=str, default='diagnostic_class')
    parser.add_argument('--output_type', type=str, default='single')
    parser.add_argument('--seq_len', type=int, default=512, help='sequence length for each sample, currently only support 512 for MOMENT')
    parser.add_argument('--load_cache', type=bool, default=True, help='whether to load cached dataset')
    
   
    args = parser.parse_args()
    control_randomness(args.seed)

    trainer = PTBXL_Trainer(args)
    trainer.train()
    trainer.test()
    trainer.save_checkpoint()
        
        

