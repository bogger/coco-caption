__author__ = 'ljyang'
from tokenizer.ptbtokenizer import PTBTokenizer
import itertools
import numpy as np
#from bleu.bleu import Bleu
from meteor.meteor2 import Meteor2
from meteor.meteor import Meteor
#from rouge.rouge import Rouge
#from cider.cider import Cider
#import nltk
class VgEvalCap:
    def __init__(self, ref_caps, model_caps):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.ref = ref_caps
        self.pred = model_caps
        self.params = {'image_id': []}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        gts_all = {}
        gts_region_idx = {}
        for imgId in imgIds:
  
          gts[imgId] = self.ref[imgId]
          res[imgId] = self.pred[imgId]
          gts_all[imgId] = []
          
          for i,anno in enumerate(gts[imgId]):
            for cap in anno['captions']:
              gts_all[imgId].append({'image_id': anno['image_id'], 'caption': cap, 'region_id': i})
              
        # =================================================
        # Set up scorers
        # =================================================
        print 'tokenization...'
        tokenizer = PTBTokenizer()
        gts_tokens  = tokenizer.tokenize(gts_all)
        res_tokens = tokenizer.tokenize(res)
        #insert caption tokens to gts 
        for imgId in imgIds:
          for tokens, cap_info in zip(gts_tokens[imgId], gts_all[imgId]):
            region_id = cap_info['region_id']
            if 'caption_tokens' not in gts[imgId][region_id]:
              gts[imgId][region_id]['caption_tokens'] = []
            gts[imgId][region_id]['caption_tokens'].append(tokens)

        

        # =================================================
        # Compute scores
        # =================================================
        #holistic score
        eval = {}
        
        print 'computing Meteor score...'
        score, scores = Meteor2().compute_score(gts_tokens, res_tokens)
        #self.setEval(score, method)
        #self.setImgToEvalImgs(scores, imgIds, method)
        print "Meteor: %0.3f"%(score)
        #self.setEvalImgs()
        #mean ap
        overlap_ratios = [0.3,0.4,0.5,0.6,0.7]
        metoer_score_th = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        score_matrix = np.zeros((len(overlap_ratios), len(metoer_score_th)))
        gt_region_n = sum([len(gts[imgId]) for imgId in imgIds])
        #calculate the nxm bbox overlap in one pass
        overlap_matrices = {}
        for imgId in imgIds:
          model_caption_locations = res[imgId]
          gt_caption_locations = gts[imgId]
          #should be sorted in advance
          #model_caption_locations.sort(key=lambda x:-x['log_prob'])
          new_matrix = self.calculate_overlap_matrix(model_caption_locations, gt_caption_locations)
          overlap_matrices[imgId] = new_matrix
        for rid, overlap_r in enumerate(overlap_ratios):
          gts_match = {}
          res_match = {}
          
          for imgId in imgIds:
            #i = imgId

            #find the match ids of predictions to gt bboxes
            match_ids = self.bbox_match(overlap_matrices[imgId], overlap_r)
            for j, match_id in enumerate(match_ids):
              if match_id > -1:
                key = '%d_%d' % (imgId, match_id)
                gts_match[key] = gts[imgId][match_id]['caption_tokens']
                res_match[key] = [res_tokens[imgId][j]]
                #assert(len(res_match[key])==1)
            if gts_match:
	      
	      score, scores = Meteor().compute_score(gts_match, res_match)
              #all_scores = list(itertools.chain(*scores))
              all_scores = np.array(scores)
              for th_id, score_th in enumerate(metoer_score_th):
                correct_n = np.sum(all_scores > score_th)
                score_matrix[rid, th_id] = float(correct_n) / gt_region_n
        mean_ap = np.mean(score_matrix) 
        print "mean average precision is %0.3f" % mean_ap
    def calculate_overlap_matrix(self, model_caption_locations, gt_caption_locations):
      model_region_n = len(model_caption_locations)
      gt_region_n = len(gt_caption_locations)
      #overlap_matrix = np.zeros((model_region_n, gt_region_n))
      model_bboxes = np.array([x['location'] for x in model_caption_locations])# nx4 matrix
      gt_bboxes = np.array([x['location'] for x in gt_caption_locations])
      #area, intersection area, union area
      model_bbox_areas = (model_bboxes[:,2] - model_bboxes[:,0]) * \
        (model_bboxes[:, 3] - model_bboxes[:, 1])
      gt_bbox_areas = (gt_bboxes[:,2] - gt_bboxes[:,0]) * \
        (gt_bboxes[:, 3] - gt_bboxes[:, 1])
      x_a1 = model_bboxes[:,0].reshape(model_region_n,1)
      x_a2 = model_bboxes[:,2].reshape(model_region_n,1)
      x_b1 = gt_bboxes[:,0].reshape(1,gt_region_n)
      x_b2 = gt_bboxes[:,2].reshape(1,gt_region_n)
      y_a1 = model_bboxes[:,1].reshape(model_region_n,1)
      y_a2 = model_bboxes[:,3].reshape(model_region_n,1)
      y_b1 = gt_bboxes[:,1].reshape(1,gt_region_n)
      y_b2 = gt_bboxes[:,3].reshape(1,gt_region_n)
      bbox_pair_x_diff = np.maximum(0, np.minimum(x_a2, x_b2) - np.maximum(x_a1, x_b1))
      bbox_pair_y_diff = np.maximum(0, np.minimum(y_a2, y_b2) - np.maximum(y_a1, y_b1))
      inter_areas = bbox_pair_x_diff * bbox_pair_y_diff
      #IoU
      union_areas =  model_bbox_areas.reshape(model_region_n,1) + gt_bbox_areas.reshape(1,gt_region_n)
      overlap_matrix = inter_areas / (union_areas - inter_areas)
      return overlap_matrix

    def bbox_match(self, overlap_matrix, overlap):
      assert(1 > overlap >= 0)
      
      model_n = overlap_matrix.shape[0]
      gt_n = overlap_matrix.shape[1]
      
      gt_flag = np.ones((gt_n),dtype=np.int32)
      match_ids = -1 * np.ones((model_n),dtype=np.int32) 
      for i in xrange(model_n):
        overlap_step = overlap_matrix[i,:] * gt_flag 
        max_overlap_id = np.argmax(overlap_step)
        if overlap_step[max_overlap_id] > overlap:
          gt_flag[max_overlap_id] = 0
          match_ids[i] = max_overlap_id
        else:
          pass

      return match_ids

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
