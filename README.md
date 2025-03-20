# [NTIRE 2025 Challenge on Image Denoising](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

## How to test the baseline model?

    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 8
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.