python main.py --api_key "your_api_key"\
	--hf_token "your_huggingface_read_key"\
	--character "爱丽丝"\
	--model_engine "gpt-4"\
	--use_pretrained_discriminator \
	--relevance_finetune_epoch 5\
	--rag_top_k 5\
	--nli_finetune_epoch 10\
	--max_dpo_data 100\
	--lora_rank 128\
	--prp_dpo_epoch 10\
	--prp_scale "7b"\
	--device "1"
