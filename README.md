## Task 1
To run the tokenizer we used the following scripts for the two vocabulary sizes, 32,768 and 8192 respectively:
```bash
bash runs/tokenizer_training.sh
bash runs/tokenizer_training_small_vocab.sh
```

## Task 2
To run our model pretraining we use the pretrain script (WIP needs to probably change the name of the saved model to avoid confusion with task 3):
```bash
bash runs/pretrain.sh
```
Run on Windows with logging:
```bash
bash -c "WANDB_RUN=d4 bash runs/pretrain.sh"
```

## Task 3
Inspecting the datasets:
```bash
source .venv/bin/activate
python -c "
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk

mmlu = MMLU(subset='all', split='auxiliary_train')
print(f'MMLU auxiliary_train: {len(mmlu):,} examples')
print(mmlu[0])
print()

gsm8k = GSM8K(subset='main', split='train')
print(f'GSM8K train: {len(gsm8k):,} examples')
print(gsm8k[0])
print()

smoltalk = SmolTalk(split='train')
print(f'SmolTalk train: {len(smoltalk):,} examples')
print(smoltalk[0])
"
```

Stage 1: mid training finetuning:
```bash
bash -c "WANDB_RUN=d4 bash runs/mid_training_fine_tuning.sh"
```

Stage 2: supervised finetuning:
```bash
bash -c "WANDB_RUN=d4 bash runs/supervised_fine_tuning.sh"
```