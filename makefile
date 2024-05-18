train:
	accelerate launch main.py \
		--exp_name esc9kbps \
		--config_path ./configs/9kbps_final.yaml
		--wandb_project efficient-speech-codec \
		--lr 1.0e-4 \
		--num_epochs 80 \
		--num_pretraining_epochs 15 \
		--num_devices 4 \
		--dropout_rate 0.75 \
		--save_path ../output \
		--seed 53