from numba import jit, njit
from collections import deque
import torch
from utils.kalman_filter import KalmanFilter
from utils.log import logger
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9
    
    def update_features(self, feat): ## 
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat 
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha *self.smooth_feat + (1-self.alpha) * feat ## update the embedding of a tracklet: ft = 0.9*ft-1 + 0.1*f_assigned
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat) ## normalize

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0

        ## state, error_covariance matrix
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        
    @staticmethod
    def multi_predict(stracks, kalman_filter):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = kalman_filter.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id() ## return BaseTrack._count += 1
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id() ## return BaseTrack._count += 1

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property ## means read-only
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y, width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod ## bind in class, no need to construct class object to call this method, so it can't access class object properties.
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    ## if print STrack, output the returned string
    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        self.model = Darknet(opt.cfg, nID=14455)
        # load_darknet_weights(self.model, opt.weights)
        self.model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'], strict=False)
        self.model.cuda().eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

    def update(self, im_blob, img0):
        """
        Processes the image frame and finds bounding box(detections).

        Associates the detection with corresponding tracklets and also handles lost, removed, refound and active tracklets

        Parameters
        ----------
        im_blob : torch.float32
                  Tensor of shape depending upon the size of image. By default, shape of this tensor is [1, 3, 608, 1088]

        img0 : ndarray
               ndarray of shape depending on the input image sequence. By default, shape is [608, 1080, 3]

        Returns
        -------
        output_stracks : list of Strack(instances)
                         The list contains information regarding the online_tracklets for the recieved image tensor.

        """
        #print(f"---------------------------------- start --------------------------------------------")
        #print(f"frame_id:{self.frame_id}")
        self.frame_id += 1
        activated_stracks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame, 
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []  

        ''' Step 1: Network forward, get detections & embeddings'''
        #print(f"Step 1")
        #t1 = time.time() # for computing forward time
        with torch.no_grad():
            pred = self.model(im_blob)
        # pred is tensor of all the proposals (default number of proposals: 54264). Proposals have information associated with the bounding box and embeddings
        pred = pred[pred[:, :, 4] > self.opt.conf_thres] ## confidence_threshold: 0.5
        # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score

        if len(pred) > 0:
            dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].cpu()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
            scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
            '''Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)'''
            ## class_pred is the embeddings.
            detections = [STrack(tlwh=STrack.tlbr_to_tlwh(tlbrs[:4]), score=tlbrs[4], temp_feat=f.numpy(), buffer_size=30) 
                            for (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])]
        else:
            detections = []
        #print(f"number of dets(STrack):{len(detections)}")

        #t2 = time.time() # for computing forward time
        # print('Forward: {} s'.format(t2-t1))

        ''' Add newly detected tracklets to tracked_stracks'''
        ## self.tracked_stracks (newly detected tracks) --> unconfirmed_list (if track not activated) or tracked_list
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        #print(f"len(self.tracked_stracks):{len(self.tracked_stracks)}") 
        for track in self.tracked_stracks: ## always 0 in webcam (no newly detected track in webcam)
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
                #print("Should not be here, in unconfirmed")
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                #print(f"track_is_activated")
                tracked_stracks.append(track)


        ''' Step 2: First association, with embedding'''
        #print(f"Step 2")
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks) ## nothing here
        #print(f"tracked_stracks + self.lost_stracks:{len(strack_pool)}")

        # Predict the current location with Kalman Filter
        #stracks[i].mean = mean
        #stracks[i].covariance = cov
        STrack.multi_predict(strack_pool, self.kalman_filter) ## update mean, covariance of tracks in strack_pool using kalman filter

        dists = matching.embedding_distance(tracks=strack_pool, detections=detections) ## return cost_matrix (len(strack_pool), len(detections)) using cosine similarity
        dists = matching.fuse_motion(kf=self.kalman_filter, cost_matrix=dists, tracks=strack_pool, detections=detections)
        # The dists is the list of distances of the detection with the tracks in strack_pool


        ## Hungarian algorithm
        matches, unmatched_track, unmatched_detection = matching.linear_assignment(dists, thresh=0.7) 
        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked: ## class TrackState(object):New = 0, Tracked = 1, Lost = 2, Removed = 3
                # If the track is active, add the detection to the track
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                # We have obtained a detection from a track which is not active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


        # None of the steps below happen if there are no undetected tracks.
        ''' Step 3: Second association, with IOU'''
        #print(f"Step 3")
        #print(f"unmatched detection len:{len(detections)}")
        detections = [detections[i] for i in unmatched_detection] # a list of the unmatched detections
        r_tracked_stracks = [] # This is container for stracks which were tracked till the previous frame but no detection was found for it in the current frame
        for i in unmatched_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, unmatched_track, unmatched_detection = matching.linear_assignment(dists, thresh=0.5) ## try to match again with lower threshold
        
        # matches is the list of detections which matched with corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # Same process done for some unmatched detections, but now considering IOU_distance as measure
        for it in unmatched_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost() ## track.state = TrackState.Lost
                lost_stracks.append(track)
        # If no detections are obtained for tracks (unmatched_track), the tracks are added to lost_tracks list and are marked lost

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in unmatched_detection]
        dists = matching.iou_distance(unconfirmed, detections) ## 1 - ious
        matches, unmatched_unconfirmed, unmatched_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in unmatched_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        """ Step 4: Init new stracks"""
        #print(f"Step 4")
        for inew in unmatched_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            #print(f"new track")
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        
        """ Step 5: Update state"""
        #print(f"Step 5")
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        #print(f"self.lost_stracks:{len(self.lost_stracks)}")
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        

        ## Nothing in webcam
        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        #print(f"len self.tracked_stracks:{len(self.tracked_stracks)}")
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        #print(f"len self.tracked_stracks + activated_stracks:{len(self.tracked_stracks)}")
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        #print(f"len self.tracked_stracks + refind_stracks:{len(self.tracked_stracks)}")
        
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        
        self.removed_stracks.extend(removed_stracks)
        
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        #print(f"track_is_activated:{track.is_activated}")
        #print(f"self.tracked_stracks:{self.tracked_stracks}")
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_stracks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        #print(f"num of output_stracks:{len(output_stracks)}")
        #print(f"---------------------------------- end --------------------------------------------")
        return output_stracks

## combine two track list
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

## delete track list b inside track list a 
def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb) ## return 1 - ious, shape: (len(stracksa), len(stracksb))
    pairs = np.where(pdist<0.15) ## duplicate_tracks
    dup_a, dup_b = list(), list()
    for a,b in zip(*pairs):
        timea = stracksa[a].frame_id - stracksa[a].start_frame
        timeb = stracksb[b].frame_id - stracksb[b].start_frame
        if timea > timeb: ## track_a longer than track_b, so keep a
            dup_b.append(b) ## need to be deleted
        else: ## timep < timeq
            dup_a.append(a) ## need to be deleted
    resa = [t for i,t in enumerate(stracksa) if not i in dup_a]
    resb = [t for i,t in enumerate(stracksb) if not i in dup_b]
    return resa, resb
            

