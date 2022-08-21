# run vmf.yaml (without icd loss or cfc loss) 5 times. 
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --cfg config/ImageNet_LT/vmf.yaml --log_dir logs/ImageNet_LT/vmf_0
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --cfg config/ImageNet_LT/vmf.yaml --log_dir logs/ImageNet_LT/vmf_1
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --cfg config/ImageNet_LT/vmf.yaml --log_dir logs/ImageNet_LT/vmf_2
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --cfg config/ImageNet_LT/vmf.yaml --log_dir logs/ImageNet_LT/vmf_3
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --cfg config/ImageNet_LT/vmf.yaml --log_dir logs/ImageNet_LT/vmf_4

# run vmf_icd.yaml (with icd loss) 5 times. 
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --cfg config/ImageNet_LT/vmf_icd.yaml --log_dir logs/ImageNet_LT/vmf_icd_0
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --cfg config/ImageNet_LT/vmf_icd.yaml --log_dir logs/ImageNet_LT/vmf_icd_1
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --cfg config/ImageNet_LT/vmf_icd.yaml --log_dir logs/ImageNet_LT/vmf_icd_2
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --cfg config/ImageNet_LT/vmf_icd.yaml --log_dir logs/ImageNet_LT/vmf_icd_3
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --cfg config/ImageNet_LT/vmf_icd.yaml --log_dir logs/ImageNet_LT/vmf_icd_4