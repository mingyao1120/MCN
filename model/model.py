import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from model.model_components import FusionModule, QueryEncoder,TrainablePositionalEncoding, FeatureEncoder,WeightedPool,DynamicRNN,BCE_loss,Back_forward_ground_loss, VisualProjection, CQAttention
import torch.nn.functional as F
def build_optimizer_and_scheduler(model, configs):
    no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
  
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
     
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                configs.num_train_steps)
    return optimizer, scheduler

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class MCN(nn.Module):
    def __init__(self, configs, word_vectors):
        super(MCN, self).__init__()
        self.configs = configs

        self.word_char_encoder = QueryEncoder(num_words=configs.word_size, num_chars=configs.char_size, out_dim=configs.dim, 
                                          word_dim=configs.word_dim, char_dim=configs.char_dim, word_vectors=word_vectors, drop_rate=configs.drop_rate)
        self.video_affine = VisualProjection(visual_dim=configs.video_feature_dim, dim=configs.dim, drop_rate=configs.drop_rate)
        # position embedding
        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=configs.max_desc_len,
                                                           hidden_size=configs.dim, dropout=configs.input_drop_rate)
        self.video_pos_embed = TrainablePositionalEncoding(max_position_embeddings=configs.max_pos_len,
                                                           hidden_size=configs.dim, dropout=configs.input_drop_rate)
        # transformer encoder
        self.feature_encoder = FeatureEncoder(dim=configs.dim, num_heads=configs.n_heads, kernel_size=7, num_layers=4, drop_rate=configs.drop_rate)

        #text-video fusion
        self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.fc = nn.Linear(configs.dim, configs.dim)# no using


        #self-attention pooling
        self.sentence_generator = WeightedPool(configs.dim)
        
        #signal-genneration
        self.signal_generator = nn.Sequential(
            nn.Linear(in_features= configs.dim * 2, out_features=configs.dim),
            nn.ReLU(),
            nn.Dropout(configs.drop_rate),
            nn.Linear(in_features= configs.dim , out_features=configs.dim)
        )
       

        #score-genneration
        self.score_metrics = nn.Linear(configs.dim , 1)

        #binary_mask_gennerator
        self.feature_fusion = FusionModule(configs.dim)
        
         #Predictor-layer
        self.start_lstm = DynamicRNN(dim=configs.dim)
        self.end_lstm = DynamicRNN(dim=configs.dim)
     
        self.start_layer = nn.Sequential(
            nn.Linear(in_features= configs.dim * 2, out_features=configs.dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features= configs.dim, out_features=1, bias=True)
        )
        self.end_layer = nn.Sequential(
            nn.Linear(in_features= configs.dim * 2, out_features=configs.dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features= configs.dim, out_features=1, bias=True)
        )
        self.temperature = 0.07

        #inference and loss
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.bce_loss = BCE_loss(reduction="mean")
        self.bf_loss = Back_forward_ground_loss(reduction='mean')
        self.init_parameters()
        self.use_great_negative = False
        

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()
        self.apply(init_weights)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask, loc_starts = None, loc_ends = None, f_labels = None, h_labels = None, return_signal = False, is_train=False):
        #encoding text query to [B,Lq,d]
        raw_query_features = self.word_char_encoder(word_ids, char_ids)
        #encoding video to [B,Lv,d]
        raw_video_features = self.video_affine(video_features)
        #position-embedding
        add_pos_query_features = self.query_pos_embed(raw_query_features)
        add_pos_video_features = self.video_pos_embed(raw_video_features)
        #encoding-highlevel
        query_features = self.feature_encoder(add_pos_query_features, q_mask)
        video_features = self.feature_encoder(add_pos_video_features, v_mask)

        # =============================================================================
        # Positive Sample Processing: Frame-level Matching Score Calculation
        # =============================================================================

        # Cross-attention fusion between video and text features
        video_mask=v_mask
        query_mask=q_mask
        positive_fused_features = self.cq_attention(
            video_features,
            query_features,
            video_mask,
            query_mask
        )  # Output shape: [Batch, Length, Channels]

        # Generate sentence-level features (adaptive weighted pooling)
        sentence_features = self.sentence_generator(query_features, query_mask)

        # Background-Foreground Fusion Module
        # Generate foreground probability scores and normalized attention weights
        foreground_sigmoid_scores, foreground_softmax_scores = self.feature_fusion(
            positive_fused_features,
            sentence_features,
            video_mask
        )  # Both outputs have shape: [Batch, Length]

        # Compute video-level features (weighted sum)
        positive_clip_features = torch.sum(
            positive_fused_features * foreground_softmax_scores.unsqueeze(2),
            dim=1
        )  # Output shape: [Batch, Channels]

        # Enhanced foreground features: weighted by sigmoid scores, frame-level features
        enhanced_positive_features = positive_fused_features * foreground_sigmoid_scores.unsqueeze(2)
        # Get maximum foreground score for each sample
        max_foreground_scores, _ = torch.max(foreground_sigmoid_scores, dim=1)

        # Generate signal features (for subsequent prediction)
        positive_signal_features = self.signal_generator(
            torch.cat([sentence_features, positive_clip_features], dim=-1)
        )

        # Compute positive sample matching scores (video-level)
        positive_matching_scores = self.score_metrics(positive_signal_features).squeeze(1)
        # Compute reverse scores based on foreground scores (for negative sample processing)
        positive_reverse_scores = 1.0 - max_foreground_scores

        # =============================================================================
        # Feature Enhancement and Temporal Position Prediction: Global Matching Score
        # =============================================================================

        # Add signal feature as a special token at the beginning of the feature sequence
        batch_size = video_mask.shape[0]
        positive_special_token = positive_signal_features.clone().unsqueeze(1)
        enhanced_positive_features_with_signal = torch.cat(
            [positive_special_token, enhanced_positive_features],
            dim=1
        )

        # Extend video mask to include the special token
        extended_video_mask = torch.cat(
            [torch.ones((batch_size, 1)).to(video_mask.device), video_mask],
            dim=1
        )

        # =============================================================================
        # Temporal Position Prediction (start and end positions)
        # =============================================================================

        # Process feature sequences through LSTM
        start_features = self.start_lstm(
            enhanced_positive_features_with_signal,
            extended_video_mask
        )  # Output shape: [Batch, SeqLen, Dim]

        end_features = self.end_lstm(
            start_features,
            extended_video_mask
        )  # Output shape: [Batch, SeqLen, Dim]

        # Compute start position logits
        start_logits_unmasked = self.start_layer(
            torch.cat([start_features, enhanced_positive_features_with_signal], dim=2)
        )
        positive_start_logits = mask_logits(
            start_logits_unmasked.squeeze(2),
            mask=extended_video_mask
        )

        # Compute end position logits
        end_logits_unmasked = self.end_layer(
            torch.cat([end_features, enhanced_positive_features_with_signal], dim=2)
        )
        positive_end_logits = mask_logits(
            end_logits_unmasked.squeeze(2),
            mask=extended_video_mask
        )

        # =============================================================================
        # Training Phase: Negative Sample Processing and Loss Calculation
        # Manually construct negative samples since training data is paired
        # =============================================================================
        if is_train:
            # Create negative samples (via circular shift)
            negative_video_features = torch.cat(
                [video_features[1:], video_features[0:1]],
                dim=0
            )
            negative_video_mask = torch.cat(
                [video_mask[1:], video_mask[0:1]],
                dim=0
            )

            # Negative sample video-text feature fusion
            negative_fused_features = self.cq_attention(
                negative_video_features,
                query_features,
                negative_video_mask,
                query_mask
            )

            # Negative sample background-foreground fusion
            negative_sigmoid_scores, negative_softmax_scores = self.feature_fusion(
                negative_fused_features,
                sentence_features,
                negative_video_mask
            )

            # Negative sample clip-level features
            negative_clip_features = torch.sum(
                negative_fused_features * negative_softmax_scores.unsqueeze(2),
                dim=1
            )

            # Negative sample feature enhancement
            enhanced_negative_features = negative_fused_features * negative_sigmoid_scores.unsqueeze(2)
            max_negative_scores, _ = torch.max(negative_sigmoid_scores, dim=1)

            # Negative sample signal features
            negative_signal_features = self.signal_generator(
                torch.cat([sentence_features, negative_clip_features], dim=-1)
            )

            # Negative sample matching scores
            negative_matching_scores = self.score_metrics(negative_signal_features).squeeze(1)
            negative_reverse_scores = 1.0 - max_negative_scores

            # Add special token for negative samples
            negative_special_token = negative_signal_features.clone().unsqueeze(1)
            enhanced_negative_features_with_signal = torch.cat(
                [negative_special_token, enhanced_negative_features],
                dim=1
            )
            extended_negative_mask = torch.cat(
                [torch.ones((batch_size, 1)).to(negative_video_mask.device), negative_video_mask],
                dim=1
            )

            # Negative sample temporal position prediction
            negative_start_features = self.start_lstm(
                enhanced_negative_features_with_signal,
                extended_negative_mask
            )
            negative_end_features = self.end_lstm(
                negative_start_features,
                extended_negative_mask
            )
            negative_start_logits_unmasked = self.start_layer(
                torch.cat([negative_start_features, enhanced_negative_features_with_signal], dim=2)
            )
            negative_start_logits = mask_logits(
                negative_start_logits_unmasked.squeeze(2),
                mask=extended_negative_mask
            )
            negative_end_logits_unmasked = self.end_layer(
                torch.cat([negative_end_features, enhanced_negative_features_with_signal], dim=2)
            )
            negative_end_logits = mask_logits(
                negative_end_logits_unmasked.squeeze(2),
                mask=extended_negative_mask
            )

            # =========================================================================
            # Loss Calculation
            # =========================================================================
            # ===================================
            # Step 1: Apply softmax to get probability distributions
            pos_start_probs = torch.softmax(positive_start_logits[:,:-1], dim=1)  # (bs, L)
            pos_end_probs = torch.softmax(positive_end_logits[:,:-1]  , dim=1)    # (bs, L)
            gt_starts = loc_starts[:loc_starts.size(0)//2]
            gt_ends = loc_ends[:loc_ends.size(0)//2]
            bs, L = pos_start_probs.shape
            # Create position matrix from 0 to L-1
            positions = torch.arange(L, device=pos_start_probs.device).float()
            # Compute predicted start and end positions
            pred_starts = torch.sum(pos_start_probs * positions, dim=1)  # (bs,)
            pred_ends = torch.sum(pos_end_probs * positions, dim=1)      # (bs,)
            # Ensure pred_starts <= pred_ends
            pred_starts = torch.min(pred_starts, pred_ends)
            pred_ends = torch.max(pred_starts, pred_ends)
            # Convert ground truth labels to float
            gt_starts = gt_starts.float()
            gt_ends = gt_ends.float()
            # Compute intersection
            intersection_starts = torch.max(pred_starts, gt_starts)
            intersection_ends = torch.min(pred_ends, gt_ends)
            intersections = torch.clamp(intersection_ends - intersection_starts, min=0)
            # Compute union
            pred_lengths = pred_ends - pred_starts
            gt_lengths = gt_ends - gt_starts
            unions = pred_lengths + gt_lengths - intersections
            # IoU loss
            ious = intersections / (unions + 1e-8)
            iou_loss = 1.0 - ious.mean()

            # ============================ Moment contrastive loss
            batch_size = sentence_features.shape[0]

            # Normalize features
            sentence_features = F.normalize(sentence_features, p=2, dim=1)
            positive_clip_features = F.normalize(positive_clip_features, p=2, dim=1)
            negative_clip_features = F.normalize(negative_clip_features, p=2, dim=1)

            # Compute positive sample similarity
            pos_sim = torch.sum(sentence_features * positive_clip_features, dim=-1)  # (batch_size,)
            pos_sim = torch.exp(pos_sim / self.temperature)

            # Compute negative sample similarity
            neg_sim = torch.sum(sentence_features * negative_clip_features, dim=-1)  # (batch_size,)
            neg_sim = torch.exp(neg_sim / self.temperature)

            # Contrastive loss
            mom_loss = -torch.log(pos_sim / (pos_sim + neg_sim))
            # ===================================
            # Merge positive and negative sample predictions
            all_start_logits = torch.cat([positive_start_logits, negative_start_logits], dim=0)
            all_end_logits = torch.cat([positive_end_logits, negative_end_logits], dim=0)

            # Compute cross-entropy loss for start and end positions
            start_loss = self.ce_loss(all_start_logits, loc_starts)
            end_loss = self.ce_loss(all_end_logits, loc_ends)
            cross_entropy_loss = start_loss + end_loss

            # Compute binary cross-entropy loss
            all_matching_scores = torch.cat([positive_matching_scores, negative_matching_scores], dim=0)
            matching_loss = self.bce_loss(
                all_matching_scores,
                f_labels,
                weight=1.0
            )
            binary_cross_entropy_loss = matching_loss

            # Compute background-foreground loss
            all_foreground_scores = torch.cat([foreground_sigmoid_scores, negative_sigmoid_scores], dim=0)
            # Create negative sample background labels (all zeros)
            negative_background_labels = torch.zeros_like(
                h_labels,
                device=f_labels.device
            )
            all_foreground_labels = torch.cat([h_labels, negative_background_labels], dim=0)
            all_video_masks = torch.cat([video_mask, negative_video_mask], dim=0)

            foreground_background_loss = self.bf_loss(
                all_foreground_scores,
                all_foreground_labels,
                all_video_masks,
                weight=2.0
            )

            return cross_entropy_loss, binary_cross_entropy_loss, foreground_background_loss, iou_loss, mom_loss.mean()

        # =============================================================================
        # Inference Phase: Result Computation and Return
        # =============================================================================

        # Compute moment similarity (for final scoring)
        moment_similarity = self.moment_sim(
            positive_start_logits,
            positive_end_logits,
            video_features,
            sentence_features
        )

        # Combine multiple scores to get final matching score
        final_matching_scores = (
            self.sigmoid(positive_matching_scores) +
            positive_reverse_scores +
            moment_similarity
        ) / 3

        # If only signal features need to be returned
        if return_signal:
            return positive_signal_features

        # Return softmax probabilities of start and end positions along with final matching score
        return (
            self.softmax(positive_start_logits),
            self.softmax(positive_end_logits),
            final_matching_scores
        )

    # Compute moment similarity during inference
    def moment_sim(self, pos_start_logits, pos_end_logits, video_features, sentence_feature):
        # Moment-related score computation
        # Step 1: Apply softmax to get probability distributions
        pos_start_probs = torch.softmax(pos_start_logits[:,:-1], dim=1)  # (bs, L)
        pos_end_probs = torch.softmax(pos_end_logits[:,:-1]  , dim=1)    # (bs, L)

        # Step 2: Find start and end indices
        start_indices = torch.argmax(pos_start_probs, dim=1)  # (bs,)
        end_indices = torch.argmax(pos_end_probs, dim=1)      # (bs,)

        # Step 3: Ensure start <= end
        # If start > end for any sample, swap them
        mask = start_indices > end_indices
        tmp = start_indices[mask]
        start_indices[mask] = end_indices[mask]
        end_indices[mask] = tmp

        # Step 4: Extract features within the corresponding intervals
        # Use torch.arange to generate all possible indices, then create mask based on start and end
        bs, L = pos_start_probs.shape
        indices = torch.arange(L).to(pos_start_logits.device)  # (L,)
        # Expand to (bs, L)
        indices = indices.unsqueeze(0).expand(bs, L)  # (bs, L)
        # Create mask: indices >= start_indices[:, None] and indices <= end_indices[:, None]
        mask_start = indices >= start_indices.unsqueeze(1)
        mask_end = indices <= end_indices.unsqueeze(1)
        mask = mask_start & mask_end  # (bs, L)

        # Compute the number of time steps within the interval for each sample
        num_elements = torch.sum(mask, dim=1, dtype=torch.float32)  # (bs,)
        # Avoid division by zero, set minimum to 1e-6
        num_elements = torch.clamp(num_elements, min=1e-6)

        # Weight video features to keep only features within the interval
        masked_video_features = video_features * mask.unsqueeze(-1)  # (bs, L, dim)
        # Sum and average pool
        pooled_features = torch.sum(masked_video_features, dim=1)  # (bs, dim)
        pooled_features = pooled_features / num_elements.unsqueeze(-1)  # (bs, dim)

        # Step 5: Compute similarity using dot product
        mom_similarity = torch.sigmoid(torch.mean(pooled_features * sentence_feature, dim=1))  # (bs,)
        return mom_similarity

    @staticmethod
    def extract_start_end_index_with_case_score(pos_start_logits, pos_end_logits, score_after, threshold):
        # Time points are considered discrete
        outer = pos_start_logits.unsqueeze(dim=2) + pos_end_logits.unsqueeze(dim=1)
        outer = torch.triu(outer,)

        outer_small = outer[:,1:,1:]
        _, start_index = torch.max(torch.max(outer_small, dim=2)[0], dim=1)  # (batch_size, )
        _, end_index = torch.max(torch.max(outer_small, dim=1)[0], dim=1)  # (batch_size, )
       
        mask = (score_after < threshold)
      
        start_index = (start_index + 1) * mask
        end_index = (end_index + 1) * mask

        return start_index, end_index

   