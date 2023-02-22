from bert_encoder_only_trainer import Evaluator, BertEncoderOnlyTrainer
import torch
from sacrebleu.metrics import BLEU

def bleu_evaluation(trainer: BertEncoderOnlyTrainer):
    bleu = BLEU()
    for i in range(trainer.evaluation_batches):
        batch = next(trainer.val_dataloader)
        batch = {k: v.to(trainer.device) for k, v in batch.items()}
        with torch.no_grad():
            output = trainer.model(**batch)
        output = output.argmax(dim=-1)
        output = trainer.tokenizer.decode(output.cpu().numpy().tolist())
        target = trainer.tokenizer.decode(batch['input_ids'].cpu().numpy().tolist())
        bleu.add_string(output, target)
    return bleu.score

bleu_evaluator = Evaluator('bleu', bleu_evaluation)