from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from itertools import islice
from iterative_sampler import IterativeSampler
from tqdm import tqdm
from tokenizers import Tokenizer
from sacrebleu.metrics import BLEU
from ..diffuseq.utils import logger
from torch.optim import Adam
from typing import NamedTuple, Callable

class Evaluator(NamedTuple):
    '''Evaluator for the BertEncoderOnlyTrainer. This NamedTuple contains the following fields:
    name: The name of the evaluator
    evaluation_function: The function to call to evaluate the model. This function takes a BertEncoderOnlyTrainer as input and returns a float.'''
    name: str
    evaluation_function: Callable[['BertEncoderOnlyTrainer'], float]

class QuitCondition(NamedTuple, Callable[['BertEncoderOnlyTrainer'], bool]):
    '''QuitCondition for the BertEncoderOnlyTrainer. This NamedTuple contains the following fields:
    condition: The function to call to check whether to quit. This function takes a BertEncoderOnlyTrainer as input and returns a bool.
    before_quit: The function to call before quitting. This function takes a BertEncoderOnlyTrainer as input and returns None.'''
    condition: Callable[['BertEncoderOnlyTrainer'], bool]
    before_quit: Callable[['BertEncoderOnlyTrainer'], None]

    def __call__(self, trainer: BertEncoderOnlyTrainer):
        return self.condition(trainer)

class BertEncoderOnlyTrainer():
    '''Trains a model on a dataset using a given optimizer and dataloader. __init__ takes following arguments:
    model: The model to train
    device: The device to train on
    train_dataloader: The dataloader to use for training
    gradient_accumulation_steps: Number of steps to accumulate gradients over, defaults to 1
    logging_interval: The interval at which to log training progress, defaults to 128 
    optimizer: The optimizer to use, defaults to Adam with kwargs
    use_tqdm: Whether to use tqdm for logging, defaults to False
    training_epochs: Number of epochs to train for, defaults to 1
    **kwargs: Additional arguments to pass to the optimizer'''

    def __init__(self, model: torch.nn.Module, device: torch.device, train_dataloader: DataLoader, gradient_accumulation_steps=1, logging_interval=128,
                 optimizer: "torch.optim.Optimizer|None"=None, use_tqdm=False, training_epochs=1, **kwargs):
        self.model = model
        self.train_dataloader = train_dataloader
        self.do_evaluation = False
        self.optimizer = optimizer
        self.device = device
        self.kwargs = kwargs
        self.batch_size = train_dataloader.batch_size
        self.use_tqdm = use_tqdm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_interval = logging_interval
        self.training_epochs = training_epochs

        self.quit_condition = QuitCondition(lambda _: False, lambda _: None)

        if self.optimizer is None:
            self.optimizer = Adam(self.model.parameters(), **self.kwargs)

    def init_evaluation(self, val_dataloader: DataLoader, evaluation_interval=None, evaluation_batches=1):
        ''''Initializes evaluation, must be called before training if evaluation is desired
        val_dataloader: The dataloader to use for validation
        evaluation_interval: The interval at which to evaluate, defaults to logging_interval * 16
        evaluation_batches: Number of batches to evaluate on, cannot be 0, defaults to 1'''

        assert evaluation_batches > 0, "evaluation_batches must be > 0"

        self.do_evaluation = True

        self.evaluation_batches = evaluation_batches
        self.evaluation_interval = evaluation_interval if evaluation_interval is not None else self.logging_interval * 16
        self.val_dataloader = self.infinite_dataloader(val_dataloader)
        
    def additional_evaluation(self, name: str, evaluation_function: Callable[['BertEncoderOnlyTrainer'], float]):
        '''Adds an additional evaluation to be done during training
        test_dataloader: The dataloader to use for testing
        evaluation_function: The function to use for evaluation, must take a BertEncoderOnlyTrainer as argument and return a float'''
        test_dataloader = self.infinite_dataloader(test_dataloader)
        if not hasattr(self, 'additional_evaluations'):
            self.additional_evaluations = [Evaluator(name, evaluation_function)]
        else:
            self.additional_evaluations.append(Evaluator(name, evaluation_function))

    def init_checkpointing(self, create_checkpoint: Callable[['torch.nn.Module', dict|None], None], checkpoint_interval=None, create_checkpoint_on_evaluation=False):
        '''Initializes checkpointing, must be called before training if checkpointing is desired
        checkpoint_path: The path to save checkpoints to'''
        assert (not create_checkpoint_on_evaluation) or self.do_evaluation, "create_checkpoint_on_evaluation is True but evaluation is not enabled"
        assert checkpoint_interval is not None or create_checkpoint_on_evaluation, "checkpoint_interval is None and create_checkpoint_on_evaluation is False"

        self.do_checkpointing = True
        self.create_checkpoint = create_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.create_checkpoint_on_evaluation = create_checkpoint_on_evaluation

    def quit_when(self, quit_condition: QuitCondition):
        '''Adds a quit condition to the training loop
        quit_condition: A QuitCondition NamedTuple containing the condition and the function to call before quitting'''
        self.quit_condition = quit_condition

    def train(self):
        '''Trains the model'''
        training_device = self.device
        self.model.to(training_device)
        self.model.train()
        for epoch in range(self.training_epochs):
            metrics = self.evaluate() if self.do_evaluation else { 'loss': float('nan') }
            loop = tqdm(self.train_dataloader, leave=True, desc=f"Epoch {epoch}") if self.use_tqdm else self.train_dataloader
            for batch in loop:
                self.optimizer.zero_grad()
                ids = batch['ids']
                attention_mask = batch['mask']
                labels = batch['labels']
                slice_size  = self.batch_size // self.gradient_accumulation_steps
                id_slices = torch.split(ids, slice_size)
                attention_mask_slices = torch.split(attention_mask, slice_size)
                label_slices = torch.split(labels, slice_size)
                # First Embedding the input ids
                #input_ids = embedding(batch['ids'].to(training_device))
                for i in range(self.gradient_accumulation_steps):
                    id_slice = id_slices[i].to(dtype=torch.long).to(training_device)
                    attention_mask_slice = attention_mask_slices[i].to(dtype=torch.bool).to(training_device)
                    label_slice = label_slices[i].to(dtype=torch.long).to(training_device)
                    loss = self.model(
                        id_slice,
                        attention_mask=attention_mask_slice,
                        labels=label_slice
                    ).loss
                    loss.backward()
                self.optimizer.step()

                metrics['loss'] = loss.item()

                loop.set_postfix(**metrics) if self.use_tqdm else None
                if loop.n % self.logging_interval == 0:
                    logger.logkvs(metrics)
                if loop.n % self.evaluation_interval == 0 and loop.n != 0:
                    # Save checkpoint
                    if self.do_checkpointing and self.create_checkpoint_on_evaluation:
                        self.create_checkpoint(self.model, {"epoch": epoch})                 
                    eval_metrics = self.evaluate()
                    logger.logkvs(eval_metrics)
                    metrics.update(eval_metrics)
                    self.model.train()
                if self.do_checkpointing and (loop.n % self.checkpoint_interval == 0 and loop.n != 0):
                    # Save checkpoint
                    self.create_checkpoint(self.model, {"epoch": epoch})
                if self.quit_condition(self):
                    self.quit_condition.before_quit(self)
                    return
            
            # Save checkpoint
            if self.do_checkpointing:
                self.create_checkpoint(self.model, {"epoch": epoch})

    def infinite_dataloader(self, dataloader: DataLoader):
        '''Loops over a dataloader infinitely'''
        while True:
            for batch in dataloader:
                yield batch

    def evaluate(self):
        '''Evaluates the model on the validation set and runs additional evaluations if any are set'''
        self.model.eval()
        eval_loss: float = 0
        with torch.no_grad():
            for batch in islice(self.val_dataloader, self.evaluation_batches):
                ids = batch['ids']
                attention_mask = batch['mask']
                labels = batch['labels']
                slice_size  = self.batch_size // self.gradient_accumulation_steps
                id_slices = torch.split(ids, slice_size)
                attention_mask_slices = torch.split(attention_mask, slice_size)
                label_slices = torch.split(labels, slice_size)
                for i in range(self.gradient_accumulation_steps):
                    id_slice = id_slices[i].to(dtype=torch.long, device=self.device)
                    attention_mask_slice = attention_mask_slices[i].to(dtype=torch.bool, device=self.device)
                    label_slice = label_slices[i].to(dtype=torch.long, device=self.device)
                    loss = self.model(
                        id_slice,
                        attention_mask=attention_mask_slice,
                        labels=label_slice
                    ).loss
                    eval_loss += loss.item()

        evaluation_results = { "eval_loss": eval_loss / self.evaluation_batches }

        for additional_evaluation in self.additional_evaluations:
            name, evaluation_function = additional_evaluation
            evaluation_results[name] = evaluation_function(self.model)

        return evaluation_results