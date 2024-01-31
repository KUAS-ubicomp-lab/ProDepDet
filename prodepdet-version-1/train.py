import os
import torch

from prompt import PromptKernel
from training_args import TrainingArguments


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Training config
    # Frozen PLM can be wsw/bert/roberta.
    # Prompt length is used as 50 and 75.
    # Demonstration type can be prompt tuning/in-context learning
    args = TrainingArguments(
        output_dir="outputs",
        dataset="sst2",
        backbone="plm/bert-base",
        model_parallel=False,
        prompt_len=50,
        learning_rate=1e-2,
        per_device_train_batch_size=16,
        demonstration_type="prompt_tuning",
        num_train_epochs=10,
    )
    print("Loading Prompt Kernel...")
    prompt_trainer = PromptKernel(args=args)

    # Prompt training and evaluation
    print("Initializing Prompt Training...")
    prompt_trainer.train_prompt()
    print("Initializing Prompt Evaluating...")
    prompt_trainer.eval_prompt()


if __name__ == "__main__":
    main()
