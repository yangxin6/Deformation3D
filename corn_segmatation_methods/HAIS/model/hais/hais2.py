import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
import functools
from collections import OrderedDict
import sys
sys.path.append('../../')

from lib.hais_ops.functions import hais_ops
from util import utils


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        output = self.conv_branch(input)

        # output.features += self.i_branch(identity).features

        output = output.replace_feature(output.features + self.i_branch(identity).features)

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),                                   
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),                                             
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):

        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            # output.features = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(torch.cat((identity.features, output_decoder.features), dim=1))
            output = self.blocks_tail(output)
        return output

class HAIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_channel
        width = cfg.width
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.point_aggr_radius = cfg.point_aggr_radius
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.score_mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs
        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module
        

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords:
            input_c += 3

        self.cfg = cfg

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, width, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        self.unet = UBlock([width, 2*width, 3*width, 4*width, 5*width, 6*width, 7*width], norm_fn, block_reps, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(
            norm_fn(width),
            nn.ReLU()
        )

        # semantic segmentation branch
        self.semantic_linear = nn.Sequential(
            nn.Linear(width, width, bias=True),
            norm_fn(width),
            nn.ReLU(),
            nn.Linear(width, classes)
        )

        # center shift vector branch
        self.offset_linear = nn.Sequential(
            nn.Linear(width, width, bias=True),
            norm_fn(width),
            nn.ReLU(),
            nn.Linear(width, 3, bias=True)
        )

        # intra-instance network
        self.intra_ins_unet = UBlock([width, 2*width], norm_fn, 2, block, indice_key_id=11)
        self.intra_ins_outputlayer = spconv.SparseSequential(
            norm_fn(width),
            nn.ReLU()
        )

        # proposal score
        self.score_linear = nn.Linear(width, 1)

        # proposal mask
        self.mask_linear = nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, 1))

        self.apply(self.set_bn_init)


        # fix module
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'semantic_linear': self.semantic_linear, 'offset_linear': self.offset_linear,
                      'intra_ins_unet': self.intra_ins_unet, 'intra_ins_outputlayer': self.intra_ins_outputlayer, 
                      'score_linear': self.score_linear, 'mask_linear': self.mask_linear}
        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        # load pretrain weights
        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print("Load pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict, prefix=m))


    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)


    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        '''
        :param clusters_idx: (SumNPoint, 2), int, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = hais_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = hais_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = hais_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = hais_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = hais_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map

    def forward(self, input, input_map, coords, batch_idxs, batch_offsets, epoch, training_mode):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]

        # semantic segmentation
        semantic_scores = self.semantic_linear(output_feats)   # (N, nClass), float

        semantic_preds = semantic_scores.max(1)[1]    # (N), long

        ret['semantic_scores'] = semantic_scores

        # center shift vector
        pt_offsets = self.offset_linear(output_feats)  # (N, 3), float32
        ret['pt_offsets'] = pt_offsets

        if(epoch > self.prepare_epochs):

            if self.cfg.dataset == 'scannetv2':
                object_idxs = torch.nonzero(semantic_preds > 1).view(-1) # floor idx 0, wall idx 1
            elif self.cfg.dataset == 'corn':
                object_idxs = torch.arange(0, len(semantic_preds), dtype=torch.long)
            else:
                raise Exception

            # fliter out floor and wall
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input.batch_size)
            coords_ = coords[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]  # (N_fg, 3), float32

            semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()

            idx, start_len = hais_ops.ballquery_batch_p(coords_ + pt_offsets_, \
                batch_idxs_, batch_offsets_, self.point_aggr_radius, self.cluster_shift_meanActive)
            
            using_set_aggr_in_training = getattr(self.cfg, 'using_set_aggr_in_training', True)
            using_set_aggr_in_testing = getattr(self.cfg, 'using_set_aggr_in_testing', True)
            using_set_aggr = using_set_aggr_in_training if training_mode == 'train' else using_set_aggr_in_testing

            proposals_idx, proposals_offset = hais_ops.hierarchical_aggregation(
                semantic_preds_cpu, (coords_ + pt_offsets_).cpu(), idx.cpu(), start_len.cpu(),
                batch_idxs_.cpu(), training_mode, using_set_aggr)             

            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
    

            # restrict the num of training proposals, avoid OOM
            max_proposal_num = getattr(self.cfg, 'max_proposal_num', 200)
            if training_mode == 'train' and proposals_offset.shape[0] > max_proposal_num:
                proposals_offset = proposals_offset[:max_proposal_num + 1]
                proposals_idx = proposals_idx[: proposals_offset[-1]]
                assert proposals_idx.shape[0] == proposals_offset[-1]
                print('selected proposal num', proposals_offset.shape[0] - 1)

            # proposals voxelization again
            input_feats, inp_map = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats, coords, self.score_fullscale, self.score_scale, self.score_mode)

            # predict instance scores
            score = self.intra_ins_unet(input_feats)
            score = self.intra_ins_outputlayer(score)
            score_feats = score.features[inp_map.long()] # (sumNPoint, C)

            # predict mask scores
            # first linear than voxel to point,  more efficient  (because voxel num < point num)
            mask_scores = self.mask_linear(score.features)
            mask_scores = mask_scores[inp_map.long()]

            # predict instance scores
            if getattr(self.cfg, 'use_mask_filter_score_feature', False)  and \
                    epoch > self.cfg.use_mask_filter_score_feature_start_epoch:
                mask_index_select = torch.ones_like(mask_scores)
                mask_index_select[torch.sigmoid(mask_scores) < self.cfg.mask_filter_score_feature_thre] = 0.
                score_feats = score_feats * mask_index_select
            score_feats = hais_ops.roipool(score_feats, proposals_offset.cuda())  # (nProposal, C)
            scores = self.score_linear(score_feats)  # (nProposal, 1)
            
            ret['proposal_scores'] = (scores, proposals_idx, proposals_offset, mask_scores)

        return ret


def model_fn_decorator(test=False):
    # config
    from util.config import cfg

    class_weight = torch.FloatTensor(cfg.class_weight).cuda()
    semantic_criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=cfg.ignore_label).cuda()
    score_criterion = nn.BCELoss(reduction='none').cuda()

    def test_model_fn(batch, model, epoch):
        coords = batch['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()          # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda()              # (N, C), float32, cuda
        batch_offsets = batch['offsets'].cuda()    # (B + 1), int, cuda
        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)

        voxel_feats = hais_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch, 'test')
        semantic_scores = ret['semantic_scores']  # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']            # (N, 3), float32, cuda

        if (epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset, mask_scores = ret['proposal_scores']

        # preds
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['pt_offsets'] = pt_offsets
            if (epoch > cfg.prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset, mask_scores)

        return preds
        
    def model_fn(batch, model, epoch):
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        feats = batch['feats'].cuda()                          # (N, C), float32, cuda
        labels = batch['labels'].cuda()                        # (N), long, cuda
        instance_labels = batch['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda
        batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda
        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)

        voxel_feats = hais_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch, 'train')
        semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']           # (N, 3), float32, cuda
        
        if(epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset, mask_scores = ret['proposal_scores']
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # mask_scores: (sumNPoint, 1), float, cuda

        loss_inp = {}

        loss_inp['semantic_scores'] = (semantic_scores, labels)
        loss_inp['pt_offsets'] = (pt_offsets, coords_float, instance_info, instance_labels)

        if(epoch > cfg.prepare_epochs):
            loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum, mask_scores)

        loss, loss_out = loss_fn(loss_inp, epoch)

        # accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['pt_offsets'] = pt_offsets
            if(epoch > cfg.prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

        return loss, preds, visual_dict, meter_dict


    def loss_fn(loss_inp, epoch):

        loss_out = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda
        
        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)

        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        '''offset loss'''
        pt_offsets, coords, instance_info, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long


        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)       

        valid = (instance_labels != cfg.ignore_label).float()

        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)
        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())

        if (epoch > cfg.prepare_epochs):
            '''score and mask loss'''
            
            scores, proposals_idx, proposals_offset, instance_pointnum, mask_scores = loss_inp['proposal_scores']
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            # get iou and calculate mask label and mask loss
            mask_scores_sigmoid = torch.sigmoid(mask_scores)

            if getattr(cfg, 'cal_iou_based_on_mask', False) \
                    and (epoch > cfg.cal_iou_based_on_mask_start_epoch):
                ious, mask_label =  hais_ops.cal_iou_and_masklabel(proposals_idx[:, 1].cuda(), \
                    proposals_offset.cuda(), instance_labels, instance_pointnum, mask_scores_sigmoid.detach(), 1)
            else:
                ious, mask_label =  hais_ops.cal_iou_and_masklabel(proposals_idx[:, 1].cuda(), \
                    proposals_offset.cuda(), instance_labels, instance_pointnum, mask_scores_sigmoid.detach(), 0)
            # ious: (nProposal, nInstance)
            # mask_label: (sumNPoint, 1)

            mask_label_weight = (mask_label != -1).float()
            mask_label[mask_label==-1.] = 0.5 # any value is ok
            mask_loss = torch.nn.functional.binary_cross_entropy(mask_scores_sigmoid, mask_label, weight=mask_label_weight, reduction='none')
            mask_loss = mask_loss.mean()
            loss_out['mask_loss'] = (mask_loss, mask_label_weight.sum())
            gt_ious, _ = ious.max(1)  # gt_ious: (nProposal) float, long


            gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)

            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_loss = score_loss.mean()

            loss_out['score_loss'] = (score_loss, gt_ious.shape[0])

        '''total loss'''
        loss = cfg.loss_weight[0] * semantic_loss + cfg.loss_weight[1] * offset_norm_loss
        if(epoch > cfg.prepare_epochs):
            loss += (cfg.loss_weight[2] * score_loss)
            loss += (cfg.loss_weight[3] * mask_loss)

        return loss, loss_out


    def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
        '''
        :param scores: (N), float, 0~1
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        '''
        fg_mask = scores > fg_thresh
        bg_mask = scores < bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        segmented_scores = (fg_mask > 0).float()
        k = 1 / (fg_thresh - bg_thresh + 1e-5)
        b = bg_thresh / (bg_thresh - fg_thresh + 1e-5)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b

        return segmented_scores

    if test:
        fn = test_model_fn
    else:
        fn = model_fn

    return fn
