# [NTIRE 2025 Challenge on Image Denoising](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

## How to test our model?

First download the pre-trained model team08_restormer.pth from this link: https://drive.google.com/file/d/1_JDH6XSiWiAhmjPP_rsTISJ9YT6PTgSZ/view?usp=sharing. 

Then put team08_restormer.pth into the model_zoo folder. Finally, use the following command:

```
CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 8
```
Be sure the change the directories `--data_dir` and `--save_dir`.
