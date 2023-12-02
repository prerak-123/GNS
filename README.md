## Learning to Schedule Learning Rate with Graph Neural Networks

This is the code base of "Learning to Schedule Learning Rate with Graph Neural Networks" by Yuanhao Xiong et al. 

Some modifications are made to the code base as part of the course project for CS768: Learning with Graphs

## Results

All the experiments have been performed only on CoLA and RTE tasks due to time and hardware constraints. The command to get the results:

The json files containing the evaluation results for each modification on each of the two tasks can be found in their respective directories

## Instructions to run the code

We have provided a ``environment.yaml`` file for the dependencies of the code base in a conda environment. To set up the environment, run:

```
conda env create -f environment.yml
```

To set up transformer module:

```
cd transformers_rl
pip install -e .
```

``CUDA_VISIBLE_DEVICES=0 python run_glue_rl.py --model_name_or_path roberta-base  --task_name cola  --do_train  --do_eval  --do_predict  --max_seq_length 128  --per_device_train_batch_size 16  --per_device_eval_batch_size 512  --learning_rate 0.00002  --num_train_epochs 10 --output_dir <save_dir>  --overwrite_output_dir  --save_strategy no  --weight_decay 0.1``

Replace ``<save_dir>`` with the directory where you want to store the results.