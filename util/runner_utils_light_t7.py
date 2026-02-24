import os
import glob
import random
import numpy as np
import torch
import torch.utils.data
import torch.backends.cudnn
from tqdm import tqdm
from util.data_util import index_to_time
import json

def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def filter_checkpoints(model_dir, suffix='t7', max_to_keep=5):
    model_paths = glob.glob(os.path.join(model_dir, '*.{}'.format(suffix)))
    if len(model_paths) > max_to_keep:
        model_file_dict = dict()
        suffix_len = len(suffix) + 1
        for model_path in model_paths:
            step = int(os.path.basename(model_path).split('_')[-1][0:-suffix_len])
            model_file_dict[step] = model_path
        sorted_tuples = sorted(model_file_dict.items())
        unused_tuples = sorted_tuples[0:-max_to_keep]
        for _, model_path in unused_tuples:
            os.remove(model_path)


def get_last_checkpoint(model_dir, suffix='t7'):
    model_filenames = glob.glob(os.path.join(model_dir, '*.{}'.format(suffix)))
    print(model_dir)

    model_file_dict = dict()
    suffix_len = len(suffix) + 1
    for model_filename in model_filenames:
        step = int(os.path.basename(model_filename).split('_')[-1][0:-suffix_len])
        model_file_dict[step] = model_filename
    sorted_tuples = sorted(model_file_dict.items())
    print(sorted_tuples)
    last_checkpoint = sorted_tuples[-1]
    return last_checkpoint[1]


def convert_length_to_mask(lengths, max_len = None):
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.size()[0], max_len) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def calculate_iou_accuracy(ious, threshold):
    total_size = float(len(ious))
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / total_size * 100.0


def calculate_degrade_iou_accuracy(ious, threshold, a_ses):
    total_size = float(len(ious))
    sum = 0
    for iou, a_se in zip(ious, a_ses):
        if iou >= threshold:
            sum += (a_se[0]*a_se[1])
    return float(sum) / total_size * 100.0


def calculate_iou(i0, i1):
    if(i0[0] is None or i1[0] is None):
        # Handle None cases for predictions and answers
        if i0[0] is None and i0[1] is None and i1[0] is None and i1[1] is None:
            return 1.0
        elif i0[0] is None and i0[1] is None and i1[0] is not None and i1[1] is not None:
            return 0.0
        else:
            return 0.0
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)


def calculate_acc(pre, gt):
     return pre == gt


def calculate_absolute_distance(i0, i1, duration):
    '''
    i0 = [predict_start_time,predict_end_time]
    i1 = [gt_start_time,gt_end_time]
    duration 
    '''
    p = [item/duration for item in i0]
    g = [item/duration for item in i1]
    a_s = 1- abs(p[0]-g[0])
    a_e = 1- abs(p[1]-g[1])
    
    return a_s,a_e


# def model_eval(model, data_loader, device, max_pos_len, threshold, mode='test', epoch=None, global_step=None):
#     ious = []
#     acc_noans = []
#     acc_hasans = []
#     accs = []
#     with torch.no_grad():
#         # print("org")
#         for idx,  (records, vfeats, vfeat_lens, word_ids, char_ids) in tqdm(enumerate(data_loader), total=len(data_loader), desc='evaluate {}'.format(mode)):
#             # print(records)
#              # prepare features
#             vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device).to(device)
#             word_ids, char_ids = word_ids.to(device), char_ids.to(device)
#             # generate mask
#             query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device) # Non-zero values are true, false otherwise, since index 0 points to [PAD]
#             video_mask = convert_length_to_mask(vfeat_lens, max_len=max_pos_len).to(device)
#             # compute predicted results
#             pos_start_logits, pos_end_logits, case_score  =  model(word_ids, char_ids, vfeats, video_mask, query_mask)
#             start_indices, end_indices =  model.extract_start_end_index_with_case_score(pos_start_logits, pos_end_logits, case_score, threshold)

#             start_indices = start_indices.cpu().numpy()
#             end_indices = end_indices.cpu().numpy()
#             for record, start_index, end_index in zip(records, start_indices, end_indices):
#                 if start_index==0 or end_index ==0:
#                     #if prediction is noanswer
#                     start_time, end_time = None,None
#                     pre = True
#                 else:
#                     start_time, end_time = index_to_time(start_index, end_index, record["v_len"], record["duration"])
#                     pre = False
               
#                 iou = calculate_iou(i0=[start_time, end_time], i1=[record["s_time"], record["e_time"]])
#                 acc = 1.0 if pre == record["noanswer"] else .0
#                 if record["noanswer"]:# If GT is noanswer
#                     acc_noans.append(acc)    
#                 else:
#                     acc_hasans.append(acc)
#                 ious.append(iou)
#                 accs.append(acc)
            

#     r1i3_ans = calculate_iou_accuracy(ious, threshold=0.3)
#     r1i5_ans = calculate_iou_accuracy(ious, threshold=0.5)
#     r1i7_ans = calculate_iou_accuracy(ious, threshold=0.7)
#     mi_ans = np.mean(ious) * 100.0
#     mi_acc_ans = np.mean(acc_hasans) * 100.0
#     mi_acc_noans = np.mean(acc_noans) * 100.0
#     mi_accs = np.mean(accs) * 100.0
 
#     # write the scores
#     score_str = "Epoch {}, Step {}:\n".format(epoch, global_step)
#     score_str += "Rank@1, IoU_ans=0.3: {:.2f}\t".format(r1i3_ans)
#     score_str += "Rank@1, IoU_ans=0.5: {:.2f}\t".format(r1i5_ans)
#     score_str += "Rank@1, IoU_ans=0.7: {:.2f}\t".format(r1i7_ans)
#     score_str += "mean IoU_ans: {:.2f}\n".format(mi_ans)
#     score_str += "Noanswer Accuracy Rate: {:.2f}\n".format(mi_acc_noans)
#     score_str += "Hasanswer Accuracy Rate: {:.2f}\n".format(mi_acc_ans)
#     score_str += "ACC Rate: {:.2f}\n".format(mi_accs)
    
#     return  r1i3_ans, r1i5_ans, r1i7_ans, mi_ans, mi_accs, score_str

def model_eval(model, data_loader, device, max_pos_len, threshold, mode='test', epoch=None, global_step=None, output_json_path=None):
    ious = []
    acc_noans = []
    acc_hasans = []
    accs = []
    
    # List to store all prediction results
    all_results = []
    
    with torch.no_grad():
        for idx, (records, vfeats, vfeat_lens, word_ids, char_ids) in tqdm(enumerate(data_loader), total=len(data_loader), desc='evaluate {}'.format(mode)):
            # prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            # generate mask
            query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
            video_mask = convert_length_to_mask(vfeat_lens, max_len=max_pos_len).to(device)
            # compute predicted results
            pos_start_logits, pos_end_logits, case_score = model(word_ids, char_ids, vfeats, video_mask, query_mask)
            start_indices, end_indices = model.extract_start_end_index_with_case_score(pos_start_logits, pos_end_logits, case_score, threshold)

            start_indices = start_indices.cpu().numpy()
            end_indices = end_indices.cpu().numpy()
            
            # Convert logits to CPU and numpy format
            pos_start_logits_np = pos_start_logits.cpu().numpy()
            pos_end_logits_np = pos_end_logits.cpu().numpy()

            for i, (record, start_index, end_index) in enumerate(zip(records, start_indices, end_indices)):
                # Create result dictionary
                result = {
                    'sample_id': record.get('sample_id', ''),
                    'vid': record.get('vid', ''),
                    'words': record.get('words', []),
                    'duration': float(record.get('duration', 0)),
                    'v_len': int(record.get('v_len', 0)),
                    'noanswer_gt': bool(record.get('noanswer', False)),
                    'sentence_id': record.get('sentence_id', '')
                }

                # Process ground truth timestamps
                s_time_gt = record.get('s_time', None)
                e_time_gt = record.get('e_time', None)
                if s_time_gt is not None:
                    s_time_gt = float(s_time_gt)
                if e_time_gt is not None:
                    e_time_gt = float(e_time_gt)
                
                result['s_time_gt'] = s_time_gt
                result['e_time_gt'] = e_time_gt
                
                if start_index == 0 or end_index == 0:
                    # if prediction is noanswer
                    start_time, end_time = None, None
                    pre = True
                else:
                    start_time, end_time = index_to_time(start_index, end_index, record["v_len"], record["duration"])
                    pre = False
                
                # Convert predicted timestamps
                if start_time is not None:
                    start_time = float(start_time)
                if end_time is not None:
                    end_time = float(end_time)

                # Compute IoU and convert to Python float
                iou = float(calculate_iou(i0=[start_time, end_time], i1=[record["s_time"], record["e_time"]]))

                # Get logits for current sample and convert to list
                start_logits_list = pos_start_logits_np[i].tolist()
                end_logits_list = pos_end_logits_np[i].tolist()

                # Add prediction results
                result.update({
                    'start_index_pred': int(start_index),
                    'end_index_pred': int(end_index),
                    's_time_pred': start_time,
                    'e_time_pred': end_time,
                    'noanswer_pred': bool(pre),
                    'iou': iou,
                    'pos_start_logits': start_logits_list,
                    'pos_end_logits': end_logits_list,
                    'case_score': float(case_score[i].cpu().numpy()) if case_score is not None else None
                })

                # Compute evaluation metrics
                acc = 1.0 if pre == record["noanswer"] else 0.0

                if record["noanswer"]:
                    acc_noans.append(acc)
                else:
                    acc_hasans.append(acc)

                ious.append(iou)
                accs.append(acc)

                # Add to results list
                all_results.append(result)

    # Compute evaluation metrics and convert to Python native types
    r1i3_ans = float(calculate_iou_accuracy(ious, threshold=0.3))
    r1i5_ans = float(calculate_iou_accuracy(ious, threshold=0.5))
    r1i7_ans = float(calculate_iou_accuracy(ious, threshold=0.7))
    mi_ans = float(np.mean(ious) * 100.0)
    mi_acc_ans = float(np.mean(acc_hasans) * 100.0) if acc_hasans else 0.0
    mi_acc_noans = float(np.mean(acc_noans) * 100.0) if acc_noans else 0.0
    mi_accs = float(np.mean(accs) * 100.0) if accs else 0.0

    # Write to JSON file
    if output_json_path:
        # Add overall evaluation metrics to JSON file
        output_data = {
            'evaluation_metrics': {
                'r1_iou_0.3': r1i3_ans,
                'r1_iou_0.5': r1i5_ans,
                'r1_iou_0.7': r1i7_ans,
                'mean_iou': mi_ans,
                'accuracy_noanswer': mi_acc_noans,
                'accuracy_hasanswer': mi_acc_ans,
                'accuracy_overall': mi_accs,
                'epoch': int(epoch) if epoch is not None else None,
                'global_step': int(global_step) if global_step is not None else None,
                'mode': mode
            },
            'predictions': all_results
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        # Write JSON file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Prediction results saved to: {output_json_path}")

    # write the scores
    score_str = "Epoch {}, Step {}:\n".format(epoch, global_step)
    score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3_ans)
    score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5_ans)
    score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7_ans)
    score_str += "mean IoU: {:.2f}\n".format(mi_ans)
    score_str += "Noanswer Accuracy Rate: {:.2f}\n".format(mi_acc_noans)
    score_str += "Hasanswer Accuracy Rate: {:.2f}\n".format(mi_acc_ans)
    score_str += "ACC Rate: {:.2f}\n".format(mi_accs)
    
    return r1i3_ans, r1i5_ans, r1i7_ans, mi_ans, mi_accs, score_str