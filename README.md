# parsT5
A T5 model for Persian.

This repository is for training a Persian T5 model (parsT5-base), which is a monolingual T5 model for Persian trained on OSCAR 21.09 (https://oscar-corpus.com/) corpus with self-supervised method. 

It's similar to the English T5 model but just for Persian. To get the last version of model and an example code please refer to its HuggingFace page at:

https://huggingface.co/Ahmad/parsT5-base/

# Further training

The model already was trained for one epoch (725000 steps) on 35 Gig Persian data of OSCUR corpus. However, the last checkpoint can be cloned and the training can be resumed for more steps or on a new corpus. The last checkpoint is hosted at:

https://huggingface.co/Ahmad/parsT5/

To resume training do the following steps:

## Clone this repository
This repository includes scripts for training the model. So, first clone it:

```
git clone https://github.com/puraminy/parsT5.git
```
## Install requirements

```
cd parsT5/
pip install -r requirements.txt
```

## clone the checkpoint from Hugging Face:

```
git lfs install
git clone https://huggingface.co/Ahmad/parsT5
```
You need to have installed [Git LFS](https://git-lfs.github.com/)

## Run training script
The training script in this repository is based on the huggingface flax example:


https://github.com/huggingface/transformers/tree/master/examples/flax/language-modeling

For preprocessing and tokenization run the `t5_tokenizer_model.py`. You can refer to the above page for more information.

You can run the training scripts like the following script. This example is for calling the script in a Jupyter Notebook, however the code is the same for command prompt just use `export` to set folder variables:
```
model_folder="/content/drive/MyDrive/parsT5"
cache_folder="/content/drive/MyDrive/cache"
!python run_t5_mlm_flax.py \
  --model_name_or_path=$model_folder \
	--output_dir=$model_folder \
	--cache_dir=$cache_folder \
        --model_type="t5" \
	--config_name=$model_folder \
	--tokenizer_name=$model_folder \
	--dataset_name="oscar" \
	--dataset_config_name="unshuffled_deduplicated_fa" \
	--max_seq_length="256" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--eval_steps="10000" \
	--adafactor \
	--max_eval_steps="12000" \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--logging_steps="500" \
	--save_steps="2500" \
#        --resume_from_checkpoint=$model_folder
#	--do_eval \
#	--preprocessing_num_workers= "20" \
```

## Custom local train files


You can enter `text` or `csv` files for training data. For that purpose enter the path to the text file or database via `--train_file`


You can set `--do_eval` if you only want to evaluate a checkpoint. You can also specify a validation file using `--validation_file`


## Notes!!

If training was interrupted, it can be resumed from the last saved step. The model is saved for every 2500 steps (`--save_steps="2500"`). 

Also the first run needs to download datasets and do some pre-processing. This preprocessing can take some hours; however the results will be saved in `cache_folder` and can be used in further calls. 

