for seed in 0
do
    for alpha in 0.5
    do
        for mom_loss in 0.5
        do
            CUDA_VISIBLE_DEVICES=0 python main.py \
            --seed $seed \
            --alpha $alpha \
            --mom_loss $mom_loss \
            --task activitynet_RF \
            --save_dir datasets \
            --model_dir ckpt \
            --model_name MCN \
            --test_name activitynet_test2.0.json \
            --val_name activitynet_val2.0.json \
            --batch_size 16 \
            --init_lr 2e-3  \
            --warmup_proportion 0. \
            --epochs 150 \
            --mode train \
            --threshold .5 \
            --gama 6. \
            --beta 5. \
            --n_heads 8
        done
    done
done