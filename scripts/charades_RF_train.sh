for seed in 2000
do
    for alpha in 1.0 # 1.0
    do
        for mom_loss in 0.5 # 0.5
        do
            CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python main.py \
                --seed $seed \
                --alpha $alpha \
                --mom_loss $mom_loss \
                --task charades_RF \
                --save_dir datasets \
                --model_dir ckpt \
                --model_name MCN \
                --test_name charades_sta_test2.0.json \
                --val_name charades_sta_val2.0.json \
                --batch_size 16 \
                --init_lr 2e-3  \
                --warmup_proportion 0. \
                --epochs 100 \
                --mode train \
                --threshold 0.5 \
                --gama 6. \
                --beta 6. \
                --n_heads 8
        done
    done
done
