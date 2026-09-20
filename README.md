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

MMLU auxiliary_train statistics (answer letter distribution, subjects, question length):
```bash
source .venv/bin/activate
python -c "
import statistics
from collections import Counter
from tasks.common import load_hub_dataset

table = load_hub_dataset('cais/mmlu', 'all', split='auxiliary_train').table
n = table.num_rows
answers = Counter(table['answer'].to_pylist())
print(f'rows: {n:,}')
for i, letter in enumerate('ABCD'):
    print(f'{letter}: {100 * answers[i] / n:.1f}%')
print('distinct subjects:', set(table['subject'].to_pylist()))
print('median question chars:', statistics.median(len(q) for q in table['question'].to_pylist()))
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

## Task 4
Run the model with different temperature settings to compare the differences in outputs:
```bash
bash runs/chat_cli.sh
```