for seed in 0
do
    for alpha in 0.5
    do
        for mom_loss in 0.5
        do
            python main.py \
            --alpha $alpha \
            --mom_loss $mom_loss \
            --task activitynet_RF \
            --save_dir datasets \
            --model_dir ckpt \
            --model_name MCN \
            --seed $seed \
            --batch_size 16  \
            --mode test \
            --threshold .4
        done 
    done
done 