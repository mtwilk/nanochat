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
Note that we do not run the expensive evaluation script by default as the assignment did not ask for it.