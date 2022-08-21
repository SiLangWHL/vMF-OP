# alpha ~ (0.0, 1.0)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --cfg config/ImageNet_LT/vmf.yaml --test --model_dir your_model_dir --alpha 0.5