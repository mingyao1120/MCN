seed=2000        # Example value, adjust as needed
alpha=1.0     # Example value, adjust as needed
mom_loss=0.5 # Example value, adjust as needed

for iou_tax in biou; do
    CUDA_VISIBLE_DEVICES=1 python main.py --task charades_RF --iou_tax $iou_tax --save_dir datasets --model_dir ckpt --model_name MCN  --seed $seed --mom_loss $mom_loss --alpha $alpha --batch_size 16  --mode test --threshold .5
done