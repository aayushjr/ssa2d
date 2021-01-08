import numpy as np
import keras
import os
from utils import directory_utils as dir_utils, video_utils as video_utils
from keras.utils import to_categorical
import json
import random
import imgaug
import utils.numpy_utils as np_utils
from utils.numpy_utils import debug
from model_params import ModelParameters as Params, Classes
from model_params import RunModes
import pdb
import cv2

# paths
data_loc = '/home/c3-0/aayushjr/datasets/VidOR/'
annotation_loc = 'Annotations/'#'vidvrd-dataset/'#'Annotations/'
videos_loc = 'Videos/'#'vidvrd-videos/'#'Videos/'
training_loc = 'training/'#'train'#'training/'
validation_loc = 'validation/'#'test'#'validation/'
train_class_dictionary_path = 'data/train_class_dictionary.json'
test_class_dictionary_path = 'data/test_class_dictionary.json'

# file names
obj_distribution_file = 'data/train_distribution_obj.json'
rel_distribution_file = 'data/train_distribution_rel.json'
act_distribution_file = 'data/train_distribution_act.json'
train_split_file = 'training.txt'
test_split_file = 'validation.txt'

# constants
DATA_EXT = '.mp4'

# json keys
VID_LEN = 'frame_count'
TIDS = 'subject/objects'
BBOXES = 'trajectories'
ACTS_RELS = 'relation_instances'
BBOX_COORDS = 'bbox'
OBJ_CLASS = 'category'
TID = 'tid'
TID_SUBJ = 'subject_tid'
TID_OBJ = 'object_tid'
FRAME_START = 'begin_fid'
FRAME_END = 'end_fid'
ACT_REL_NAME = 'predicate'


class VidORDataloaderV1(keras.utils.Sequence):

    def __init__(self, mode: str, shuffle=True, visualize=False, per_class_sample=False, samples_per_class=20,
                 focus_key=Classes.obj, batch_size = 1):
        """
        :param mode: (str) variable from utils.datastructures.RunModes
        :param shuffle: (bool) whether the data should be shuffled
        :param visualize: whether data should be visualized
        :param per_class_sample: whether the sampling should be done per class instead of randomly
        :param samples_per_class: how many samples per class we should sample (only if :param per_class_sample) is True
        :param focus_key: the key around which samples will be sampled during train mode
        """

        # set data
        self.mode = mode
        self.shuffle = shuffle
        self.per_class_sample = per_class_sample
        self.samples_per_class = samples_per_class
        self.focus_key = focus_key

        self.split_size = -1
        self.full_split = []

        if self.per_class_sample:
            self.class_dictionary = None
        else:
            if self.mode == RunModes.training:
                self.max_batches = Params.MAX_TRAIN_BATCHES_PER_EPOCH
                self.split_path = os.path.join(data_loc, annotation_loc, train_split_file)
            else:
                self.max_batches = Params.MAX_VALID_BATCHES_PER_EPOCH
                self.split_path = os.path.join(data_loc, annotation_loc, test_split_file)

        self.prepare_data()

        # video management
        self.video_loader = video_utils.VideoLoader(resize_target_height=int(Params.IN_HEIGHT * 1.05),
                                                    crop_shape=(Params.IN_HEIGHT, Params.IN_WIDTH),
                                                    h_flip_chance=Params.should_rand_flip * .5,
                                                    visualize=visualize)

        self.bbox_out_resizer = imgaug.augmenters.Resize(Params.OUT_FRAME_DIM)
        self.gaussian_kernel = np_utils.make_gaussian(Params.CENTROID_SIZE, Params.CENTROID_FWHM)
        self._batch_size = Params.BATCH_SIZE

    @staticmethod
    def load_class_dictionary(dictionary_path):
        with open(dictionary_path) as dictionary_file:
            data = json.load(dictionary_file)
        data['act_rel'] = data['act'].copy()
        data['act_rel'].update(data['rel'])
        return data

    def get_video_path(self, name, extension='.mp4', assert_existence=True):
        """
        returns a path for the given video name
        :param name: name of the video
        :param extension: video file extension
        :param assert_existence: whether the existence of the file should be checked
        :return: absolute path to given video name
        """

        folder = training_loc if self.mode == RunModes.training else validation_loc
        video_path = os.path.join(data_loc, videos_loc, folder, name + extension)

        if data_loc == 'sample_data/' and not os.path.exists(video_path) and "home" not in os.getcwd():
            download_folder = os.path.join(data_loc, videos_loc, folder, name.split('/')[0])
            if not os.path.exists(download_folder):
                os.mkdir(download_folder)
        if assert_existence:
            dir_utils.assert_existence(video_path)

        return video_path

    def get_annotation_path(self, name, extension='.json', assert_existence=False):
        """
        returns a path for the given video name
        :param name: name of the video
        :param extension: video file extension
        :param assert_existence: whether the existence of the file should be checked
        :return: absolute path to given video name
        """

        folder = training_loc if self.mode == RunModes.training else validation_loc
        annotation_path = os.path.join(data_loc, annotation_loc, folder, name + extension)

        if data_loc == 'sample_data/' and not os.path.exists(annotation_path) and "home" not in os.getcwd():
            download_folder = os.path.join(data_loc, annotation_loc, folder, name.split('/')[0])
            if not os.path.exists(download_folder):
                os.mkdir(download_folder)
        if assert_existence:
            dir_utils.assert_existence(annotation_path)

        return annotation_path

    def on_epoch_end(self):
        """
        Epoch pre-processing
        """

        if not self.per_class_sample:
            random.shuffle(self.full_split)
        else:
            self.resample()

    def prepare_data(self):
        if self.per_class_sample:
            if self.mode == RunModes.training:
                self.class_dictionary = self.load_class_dictionary(train_class_dictionary_path)
            else:
                self.class_dictionary = self.load_class_dictionary(test_class_dictionary_path)

            print("Focus key ", self.focus_key)
            print("Classes ", len(self.class_dictionary[self.focus_key]))
            self.split_size = len(self.class_dictionary[self.focus_key]) * self.samples_per_class
            self.resample()
        else:
            self.class_dictionary = self.load_class_dictionary(test_class_dictionary_path)
            self.resample()
            self.split_size = len(self.full_split)
            '''
            with open(self.split_path, 'r') as split_file:
                self.full_split = split_file.read().strip().split('\n')
            self.split_size = min(len(self.full_split), self.max_batches * Params.BATCH_SIZE)
            if self.shuffle:
                random.shuffle(self.full_split)
            '''
            
    def resample(self):
        
        if self.mode == RunModes.training or self.mode == RunModes.validation:
            self.full_split = [None] * self.split_size
            i = 0
            for class_name, videos in self.class_dictionary[self.focus_key].items():
                if len(videos) < self.samples_per_class:
                    class_idx = 0
                    for class_num in range(self.samples_per_class):
                        self.full_split[i] = (videos[class_idx], class_name)
                        i += 1
                        class_idx += 1
                        if class_idx >= len(videos):
                            class_idx = 0
                else:
                    class_idx = np.random.randint(0,len(videos),self.samples_per_class)
                    for class_num in range(self.samples_per_class):
                        #self.full_split[i] = (random.choice(videos), class_name)
                        self.full_split[i] = (videos[class_idx[class_num]], class_name)
                        i += 1
            random.shuffle(self.full_split)
        elif self.mode == RunModes.test:
            t_full_split = []
            for class_name, videos in self.class_dictionary[self.focus_key].items():
                #print(class_name, " ", len(videos))
                for i in range(len(videos)):
                    t_full_split.append((videos[i], class_name))
            self.full_split = [None] * len(t_full_split)
            for i in range(len(t_full_split)):
                self.full_split[i] = t_full_split[i]

    def load_batch(self, batch_indexes):
        """
        Returns a batch for the corresponding :batch_indexes
        :param batch_indexes: (list(int)) indexes in :self.full_split (used when random sampling)
        :return: (np.array) a batch of shape (:batch_size, *(:out_dim), :n_frames, :n_classes)
        """
        # Initialization
        inputs = np.zeros((Params.BATCH_SIZE, *Params.INPUT_SHAPE), dtype='float32')
        outputs_obj = []
        outputs = []
        out_shape = (Params.BATCH_SIZE, Params.NUM_FRAMES, *Params.OUT_FRAME_DIM)

        if Params.USE_OBJ:
            obj_labels = np.zeros((*out_shape, Params.NUM_OBJ), dtype='float32')
            outputs_obj.append(obj_labels)

        if Params.USE_CTR:
            ctr_labels = np.zeros((*out_shape, 1), dtype='float32')
            outputs.append(ctr_labels)

        if Params.USE_ACT:
            act_labels = np.zeros((*out_shape, Params.NUM_ACT), dtype='float32')
            outputs.append(act_labels)

        if Params.USE_REL:
            rel_labels = np.zeros((*out_shape, Params.NUM_REL), dtype='float32')
            outputs.append(rel_labels)
            
        if Params.USE_ACT_REL:
            act_rel_labels = np.zeros((*out_shape, Params.NUM_ACT_REL), dtype='float32')
            outputs.append(act_rel_labels)

        #clip_labels = np.zeros((Params.BATCH_SIZE, 1, 1, 1, Params.NUM_ACT), dtype='float32')

        # Generate data
        sample_bboxes = []
        sample_names = []
        for sample_idx, video_idx in enumerate(batch_indexes):
            #if self.per_class_sample:
                #_, class_name = self.full_split[video_idx]
                #name, bboxes, start_frame, end_frame, frame_step = self.load_video_annotations_for(class_name)
            #else:
            vid_name, class_name = self.full_split[video_idx]
            annotation_path = self.get_annotation_path(vid_name)
            name, bboxes, start_frame, end_frame, frame_step = self.load_video_annotations_for(annotation_path, vid_name, class_name)

            '''
            for frame_bboxes in bboxes:
                for bbox in frame_bboxes.bounding_boxes:
                    for action in bbox.label[Classes.act]:
                        action_idx = Classes.action_to_id[action]
                        clip_labels[sample_idx, 0, 0, 0, action_idx] = 1
            '''
            
            # load video
            video_path = self.get_video_path(name)
            frames, bboxes = self.video_loader.load_video(video_path=video_path, start_frame=start_frame,
                                                          end_frame=end_frame, frame_step=frame_step,
                                                          bboxes=bboxes, augment=True)
            # if there was an error processing the video, make a record and skip
            if frames.shape != (Params.NUM_FRAMES, *Params.IN_FRAME_DIM, Params.NUM_RGB_CHANNELS):
                with open('data/bad_files.txt', 'a+') as f:
                    f.write(name + '\n')
                continue

            # Normalize and scale input between -1 to 1
            frames = frames / 255.
            frames = frames * 2.
            frames = frames - 1.
            inputs[sample_idx] = frames

            # resize bounding boxes to output size
            bboxes = self.bbox_out_resizer(bounding_boxes=bboxes)
            if Params.USE_OBJ:
                obj_labels[sample_idx] = self.bboxes_to_categorical_grid(bboxes, Classes.obj, Classes.object_to_id, default_class_index = Classes.object_to_id['background'])       # CCE
                #obj_labels[sample_idx] = self.bboxes_to_binary_grid(bboxes, Classes.obj, Classes.object_to_id)                                                                     # BCE

            if Params.USE_CTR:
                populate_centroid_grid(ctr_labels[sample_idx], bboxes, self.gaussian_kernel)

            if Params.USE_ACT:
                act_labels[sample_idx] = self.bboxes_to_binary_grid(bboxes, Classes.act, Classes.action_to_id)
                #act_labels[sample_idx] = self.bboxes_to_categorical_grid(bboxes, Classes.act, Classes.action_to_id, Classes.action_to_id['no_action'])
            
            if Params.USE_ACT_REL:
                act_rel_labels[sample_idx] = self.bboxes_to_binary_grid(bboxes, Classes.act_rel, Classes.act_rel_to_id)

            if Params.USE_REL:
                rel_labels[sample_idx] = self.bboxes_to_binary_grid(bboxes, Classes.rel, Classes.relation_to_id)
            sample_bboxes.append(bboxes)
            sample_names.append(name)
        
        outputs = np.array(outputs[0])
        outputs_obj = np.array(outputs_obj[0])
        # Select half the frames for outputs
        outputs2 = np.zeros((Params.BATCH_SIZE, int(Params.NUM_FRAMES/2), *Params.OUT_FRAME_DIM, Params.NUM_ACTIONS_RELATIONS))
        outputs_obj2 =  np.zeros((Params.BATCH_SIZE, int(Params.NUM_FRAMES/2), *Params.OUT_FRAME_DIM, Params.NUM_OBJ))
        input_mask = np.zeros((Params.BATCH_SIZE, int(Params.NUM_FRAMES/2), *Params.OUT_FRAME_DIM, Params.NUM_ACTIONS_RELATIONS))
        input_mask_obj = np.zeros((Params.BATCH_SIZE, int(Params.NUM_FRAMES/2), *Params.OUT_FRAME_DIM, Params.NUM_OBJ))  
        input_mask_obj_add = np.zeros((Params.BATCH_SIZE, int(Params.NUM_FRAMES/2), *Params.OUT_FRAME_DIM, Params.NUM_OBJ)) 
        input_mask_obj_add[:,:,:,:,-1] = 1.
        outputs_fg = np.zeros((Params.BATCH_SIZE, int(Params.NUM_FRAMES/2), *Params.OUT_FRAME_DIM, 2))
        #outputs_fg2 = np.zeros((Params.BATCH_SIZE, int(Params.NUM_FRAMES/2), 112, 112, 2))
        outputs_fg[:,:,:,:,1] = 1.
        #outputs_fg2[:,:,:,:,1] = 1.
        sel_idx = np.arange(int(Params.NUM_FRAMES/2)) * 2
        for b in range(Params.BATCH_SIZE):
            outputs2[b] = outputs[b,sel_idx]
            outputs_obj2[b] = outputs_obj[b,sel_idx]
            outputs_fg[b,:,:,:,0] = np.sum(outputs2[b],-1)
            outputs_fg[b,outputs_fg[b,:,:,:,0]>0, 1] = 0.
            input_mask[b,outputs_fg[b,:,:,:,0]>0, :] = 1.
            input_mask_obj[b,outputs_fg[b,:,:,:,0]>0, :-1] = 1.       # CCE
            #input_mask_obj[b,outputs_fg[b,:,:,:,0]>0, :] = 1.        # BCE
            input_mask_obj_add[b,outputs_fg[b,:,:,:,0]>0, -1] = 0.
            
            #for j in range(int(Params.NUM_FRAMES/2)):
                #outputs_fg2[b,j] = cv2.resize(outputs_fg[b,j], (112,112), interpolation=cv2.INTER_NEAREST)
        
        outputs_fg[outputs_fg>0] = 1.    
        return [inputs, input_mask_obj, input_mask_obj_add, input_mask], [outputs_fg, outputs_obj2, outputs2]#, sample_names  #, sample_bboxes, clip_labels                        # BCE
        #return [inputs, input_mask_obj, input_mask_obj_add], [outputs_fg, outputs_obj2]#, outputs2]#, sample_names  #, sample_bboxes, clip_labels     # CCE
        #return [inputs, input_mask], [outputs_fg, outputs2]

    @staticmethod
    def bboxes_to_categorical_grid(bboxes, key, label_to_id_dict, default_class_index=0):
        """
        transforms bboxes to a one-hot grid
        :param bboxes: list of imgaug.BoundingBoxesOnImage for each frame
        :param key: which bbox it is (act, rel, obj)
        :param label_to_id_dict: dictionary which maps labels to their ids
        :param default_class_index: categorical default class
        :return: (params.NUM_FRAMES, *params.OUT_FRAME_DIM, len(label_to_id_dict) one-hot encoded grid
        """
        one_hot = np.ones((Params.NUM_FRAMES, *Params.OUT_FRAME_DIM), dtype=int) * default_class_index

        for frame_idx, bboxes_in_frame in enumerate(bboxes):
            for bbox in bboxes_in_frame.bounding_boxes:
                if key not in bbox.label or len(bbox.label[key]) == 0:
                    continue
                # get bbox dimensions and fit them inside the grid
                x1, y1, x2, y2 = extract_and_clamp_bbox(bbox, dimension=Params.OUT_FRAME_DIM[0])

                bbox_ids = bbox.label[key]
                if type(bbox_ids) is set:
                    bbox_id = random.choice(list(bbox_ids))
                else:
                    bbox_id = bbox_ids
                bbox_id = label_to_id_dict[bbox_id]
                one_hot[frame_idx, y1:y2, x1:x2] = bbox_id

        # transform to one-hot
        return to_categorical(one_hot, len(label_to_id_dict), dtype='uint8')
        

    @staticmethod
    def bboxes_to_binary_grid(bboxes, key, label_to_id_dict):
        """
        transforms bboxes to a one-hot grid
        :param bboxes: list of imgaug.BoundingBoxesOnImage for each frame
        :param key: which bbox it is (act, rel, obj)
        :param label_to_id_dict: dictionary which maps labels to their ids
        :return: (params.NUM_FRAMES, *params.OUT_FRAME_DIM, len(label_to_id_dict) one-hot encoded grid
        """
        grid = np.zeros((Params.NUM_FRAMES, *Params.OUT_FRAME_DIM, len(label_to_id_dict)))
        
        for frame_idx, bboxes_in_frame in enumerate(bboxes):
            for bbox in bboxes_in_frame.bounding_boxes:
                if key not in bbox.label:
                    continue

                # get bbox dimensions and fit them inside the grid
                x1, y1, x2, y2 = extract_and_clamp_bbox(bbox, dimension=Params.OUT_FRAME_DIM[0])
                for label in bbox.label[key]:
                    bbox_id = label_to_id_dict[label]
                    grid[frame_idx, y1:y2:, x1:x2, bbox_id] = 1.
        
        # Clean up grid
        idx = np.sum(np.sum(grid,(1,2))>0, 0)        
        idx[idx< int(grid.shape[0]/2)] = 0
        idx[idx> 0] = 1
        for i in range(grid.shape[-1]):
            if idx[i] <= 0:
                grid[:,:,:,i] = 0
                
        return grid

    def __len__(self):
        """
        :return: batches per epoch
        """
        return self.split_size // Params.BATCH_SIZE

    def __getitem__(self, index):
        """
        Generates a batch of data
        :param index: batch num
        :return: inputs and labels
        """

        data = range(index * Params.BATCH_SIZE, (index + 1) * Params.BATCH_SIZE)
        return self.load_batch(data)

    def load_video_annotations_for(self, annotation_path, vid_name, action_name):
        #name = random.choice(self.class_dictionary[self.focus_key][action_name])
        name = vid_name

        loaded = False
        skipped = False
        while not loaded:
            #annotation_path = self.get_annotation_path(name)

            while not os.path.exists(annotation_path):
                print("Error opening:", annotation_path, ", no such file exists.")
                name = random.choice(self.class_dictionary[self.focus_key][action_name])
                annotation_path = self.get_annotation_path(name)

            file = open(annotation_path, 'r')

            full_info = json.load(file)
            file.close()

            # get video info
            frames_in_vid = full_info[VID_LEN]
            dims = (full_info['height'], full_info['width'])

            # choose step and calculate span
            if not skipped:
                frame_step = random.choice(Params.FRAME_STEPS)
            else:
                frame_step = min(Params.FRAME_STEPS)

            total_frames = frame_step * Params.NUM_FRAMES

            action_intervals = []
            for action in full_info[ACTS_RELS]:
                if action[ACT_REL_NAME] == action_name:
                    action_intervals.append((action[FRAME_START], action[FRAME_END]))

            interval_start, interval_end = random.choice(action_intervals)
            halfway = total_frames // 2

            # print('interval:',interval_start, interval_end)

            interval_start = max(0, interval_start - halfway)
            interval_end = max(min(interval_end - halfway, frames_in_vid - total_frames - 5), interval_start + 1)
            #
            # print(frames_in_vid, total_frames, halfway)
            # print('fixed:', interval_start, interval_end)

            if interval_end + total_frames >= frames_in_vid - 1:
                new_name = random.choice(self.class_dictionary[self.focus_key][action_name])
                #print(f'Error loading sample from: {name}\nReplacing with: {new_name}')
                skipped = True
                name = new_name
                annotation_path = self.get_annotation_path(name)
                continue
            start_frame = random.choice(range(interval_start, interval_end))
            end_frame = start_frame + total_frames - 1  # inclusive

            if end_frame >= frames_in_vid:
                print("somehow passed test")
                continue

            loaded = True

        # get trajectories
        bbox_annotations = full_info[BBOXES]  # trajectories

        # frame[i] -> tid dictionary
        tids_in_frame = [dict() for i in range(Params.NUM_FRAMES)]
        bboxes = [imgaug.BoundingBoxesOnImage([], shape=dims) for i in range(Params.NUM_FRAMES)]

        # extract object bounding boxes
        for frame_idx in range(0, Params.NUM_FRAMES):
            annot_frame = start_frame + frame_step * frame_idx
            if len(bbox_annotations) <= annot_frame:
                print(
                    f'error on frame {frame_idx}. video: {annotation_path}. total_len:{frames_in_vid}. '
                    f'annotations{len(bbox_annotations)}')
                continue
            for i, bbox_annotation in enumerate(bbox_annotations[annot_frame]):
                tid = bbox_annotation[TID]
                obj_name = full_info[TIDS][tid][OBJ_CLASS]
                obj_type = full_info[TIDS][tid][OBJ_CLASS]

                # get bounding-box coordinates
                bbox = imgaug.BoundingBox(*bbox_annotation[BBOX_COORDS].values())
                bbox.label = dict()

                if Params.USE_OBJ or Params.USE_CTR:
                    bbox.label[Classes.obj] = {obj_name}
                if Params.USE_ACT or Params.USE_CLP:
                    bbox.label[Classes.act] = set()
                    for action_relation in full_info[ACTS_RELS]:
                        ann_subject = action_relation[TID_SUBJ]
                        if ann_subject == tid:    # Only subject associated with action
                            ann_object = action_relation[TID_OBJ]
                            ann_start = action_relation[FRAME_START]
                            ann_end = action_relation[FRAME_END] - 1
                            ann_name = action_relation[ACT_REL_NAME]
                            if ann_name in Classes.action_to_id:
                                if tid == ann_subject or tid == ann_object:
                                    if ann_start <= annot_frame <= ann_end:
                                        bbox.label[Classes.act].add(ann_name)
                
                if Params.USE_ACT_REL:
                    bbox.label[Classes.act_rel] = set()
                    for action_relation in full_info[ACTS_RELS]:
                        ann_subject = action_relation[TID_SUBJ]
                        if ann_subject == tid:    # Only subject associated with action
                            ann_object = action_relation[TID_OBJ]
                            ann_start = action_relation[FRAME_START]
                            ann_end = action_relation[FRAME_END] - 1
                            ann_name = action_relation[ACT_REL_NAME]
                            if ann_name in Classes.act_rel_to_id:
                                if tid == ann_subject or tid == ann_object:
                                    if ann_start <= annot_frame <= ann_end:
                                        bbox.label[Classes.act_rel].add(ann_name)
                                        
                if Params.USE_REL:
                    bbox.label[Classes.rel] = set()

                # add bbox corresponding to the tid in the frame
                tids_in_frame[frame_idx][tid] = bbox
                bboxes[frame_idx].bounding_boxes.append(bbox)
        
        '''
        # extract actions & relations
        if Params.USE_ACT or Params.USE_REL or Params.USE_CLP:
            for action_relation in full_info[ACTS_RELS]:
                # extract relational information
                ann_subject = action_relation[TID_SUBJ]
                ann_object = action_relation[TID_OBJ]
                ann_start = action_relation[FRAME_START]
                ann_end = action_relation[FRAME_END] - 1
                ann_name = action_relation[ACT_REL_NAME]

                displacement = start_frame % frame_step

                if (ann_start - displacement) % frame_step != 0:
                    ann_start += frame_step - (ann_start - displacement) % frame_step
                if (ann_end - displacement) % frame_step != 0:
                    ann_end -= (ann_end - displacement) % frame_step

                # if action out of bounds, skip it
                if ann_end - start_frame <= 0 or end_frame - ann_start <= 0 or ann_start > ann_end:
                    continue

                # crop to frames
                ann_start = max(ann_start, start_frame)
                ann_end = min(ann_end, end_frame)

                # base at zero
                ann_start -= start_frame
                ann_end -= start_frame

                for tid in [ann_subject, ann_object]:
                    for frame_idx in range(ann_start, ann_end + 1, frame_step):
                        rel_frame = frame_idx // frame_step
                        bbox = tids_in_frame[rel_frame][tid]
                        # add act & rel notation if needed
                        if ann_name in Classes.action_to_id and (Params.USE_ACT or Params.USE_CLP):
                            bbox.label[Classes.act].add(ann_name)
                        elif ann_name in Classes.relation_to_id and Params.USE_REL:
                            bbox.label[Classes.rel].add(ann_name)
        '''
        
        return name, bboxes, start_frame, end_frame, frame_step

    @staticmethod
    def load_video_annotations(annotation_path):
        dir_utils.assert_existence(annotation_path)
        file = open(annotation_path, 'r')

        full_info = json.load(file)
        file.close()

        # get video info
        frames_in_vid = full_info[VID_LEN]
        dims = (full_info['height'], full_info['width'])

        # choose step and calculate span
        frame_step = random.choice(Params.FRAME_STEPS)
        total_frames = frame_step * Params.NUM_FRAMES

        if frames_in_vid - total_frames <= 0:
            frame_step = min(Params.FRAME_STEPS)
            total_frames = frame_step * Params.NUM_FRAMES

            if frames_in_vid - total_frames <= 0:
                with open('data/bad_files.txt', 'a+') as f:
                    f.write(annotation_path + '\n')
                return None

        # get start and end frame
        start_frame = random.choice(range(frames_in_vid - total_frames))
        end_frame = start_frame + total_frames - 1  # inclusive

        # get trajectories
        bbox_annotations = full_info[BBOXES]

        # frame[i] -> tid dictionary
        tids_in_frame = [dict() for _ in range(Params.NUM_FRAMES)]
        bboxes = [imgaug.BoundingBoxesOnImage([], shape=dims) for _ in range(Params.NUM_FRAMES)]

        # extract object bounding boxes
        for frame_idx in range(0, Params.NUM_FRAMES):
            annot_frame = start_frame + frame_step * frame_idx
            for i, bbox_annotation in enumerate(bbox_annotations[annot_frame]):
                tid = bbox_annotation[TID]
                obj_name = full_info[TIDS][tid][OBJ_CLASS]

                # get bounding-box coordinates
                bbox = imgaug.BoundingBox(*bbox_annotation[BBOX_COORDS].values())
                bbox.label = dict()

                if Params.USE_OBJ or Params.USE_CTR:
                    bbox.label[Classes.obj] = {obj_name}
                if Params.USE_ACT:
                    bbox.label[Classes.act] = set()
                if Params.USE_REL:
                    bbox.label[Classes.rel] = set()

                # add bbox corresponding to the tid in the frame
                tids_in_frame[frame_idx][tid] = bbox
                bboxes[frame_idx].bounding_boxes.append(bbox)
        # extract actions & relations
        if Params.USE_ACT or Params.USE_REL:
            for action_relation in full_info[ACTS_RELS]:
                # extract relational information
                ann_subject = action_relation[TID_SUBJ]
                ann_object = action_relation[TID_OBJ]
                ann_start = action_relation[FRAME_START]
                ann_end = action_relation[FRAME_END] - 1
                ann_name = action_relation[ACT_REL_NAME]

                displacement = start_frame % frame_step

                if (ann_start - displacement) % frame_step != 0:
                    ann_start += frame_step - (ann_start - displacement) % frame_step
                if (ann_end - displacement) % frame_step != 0:
                    ann_end -= (ann_end - displacement) % frame_step

                # if action out of bounds, skip it
                if ann_end - start_frame <= 0 or end_frame - ann_start <= 0 or ann_start > ann_end:
                    continue

                # crop to frames
                ann_start = max(ann_start, start_frame)
                ann_end = min(ann_end, end_frame)

                # base at zero
                ann_start -= start_frame
                ann_end -= start_frame

                for tid in [ann_subject, ann_object]:
                    for frame_idx in range(ann_start, ann_end + 1, frame_step):
                        rel_frame = frame_idx // frame_step
                        bbox = tids_in_frame[rel_frame][tid]
                        # add act & rel notation if needed
                        if ann_name in Classes.action_to_id and Params.USE_ACT:
                            bbox.label[Classes.act].add(ann_name)
                        elif ann_name in Classes.relation_to_id and Params.USE_REL:
                            bbox.label[Classes.rel].add(ann_name)

        return bboxes, start_frame, end_frame, frame_step


def populate_centroid_grid(grid, bbox_coords, kernel):
    """
    populates the grid with centroids of bbox_coords
    :param grid: (n_frames, *dims, 1)
    :param bbox_coords: list of bbox objects, matching frames
    :param kernel: the kernel to be applied at centroids
    """

    for frame_idx, bboxes_in_frame in enumerate(bbox_coords):
        for bbox in bboxes_in_frame.bounding_boxes:
            x1, y1, x2, y2 = bbox.x1_int, bbox.y1_int, bbox.x2_int, bbox.y2_int
            x, y = (x1 + x2) // 2, (y1 + y2) // 2

            np_utils.add_at_center(grid[frame_idx], kernel, (x, y))


def between_inclusive_exclusive(minimum, maximum, val):
    return minimum <= val < maximum


def extract_and_clamp_bbox(bbox, dimension):
    points = [bbox.x1_int, bbox.y1_int, bbox.x2_int, bbox.y2_int]
    for i in range(len(points)):
        points[i] = min(max(points[i], 0), dimension)
    return points


def save_vis(clip, path, resize=False, resize_to=224, rescale=False, rescale_to=255):
    import imageio
    with imageio.get_writer(path, mode='I') as writer:
      for i in range(clip.shape[0]):
        image = clip[i]
        if resize:
            image = cv2.resize(image, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
        if rescale:
            image *= rescale_to
        if len(image.shape) > 2:
            image = image[:,:,::-1].astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        writer.append_data(image)
    

if __name__ == '__main__':
    dataloader = VidORDataloaderV1(mode=RunModes.validation, shuffle=False, visualize=False, per_class_sample=True, focus_key=Classes.act_rel, samples_per_class=60)  
    
    print(len(dataloader))
    steps = int(len(dataloader))
    epochs = 10
    all_names = []
    for j in range(epochs):
        epoch_names = []
        for i in range(steps):
            [input_clip, input_mask_obj, input_mask_obj_add, input_mask_cls],[ofg, oobj, oact], sample_names = dataloader.__getitem__(i)
            #for j in range(len(sample_names)):
            #    print(sample_names[j])
            for j in range(input_clip.shape[0]):
                clip = input_clip[j]
                clip += 1
                clip = clip / 2
                clip *= 255
                path = './results/obj_act/{:02d}_clip.gif'.format((i*6)+j)
                save_vis(clip, path)
                for k in range(oact.shape[-1]):
                    clip = oact[j,:,:,:,k]
                    if np.sum(clip) > 0:
                        cls_name = Classes.id_to_action[k]
                        path = './results/obj_act/{:04d}_tact_{:04d}_{}.gif'.format((i*6)+j, k,cls_name)
                        save_vis(clip, path, resize=True, resize_to=224, rescale=True, rescale_to=255)
                for k in range(oobj.shape[-1]):
                    clip = oobj[j,:,:,:,k]
                    if np.sum(clip) > 0:
                        cls_name = Classes.id_to_object[k]
                        if '/' in cls_name:
                            cls_name = cls_name.split('/')[0]
                        path = './results/obj_act/{:04d}_tobj_{:04d}_{}.gif'.format((i*6)+j, k,cls_name)
                        save_vis(clip, path, resize=True, resize_to=224, rescale=True, rescale_to=255)
            epoch_names.extend(sample_names)
            pdb.set_trace()
        #d_names = dataloader.full_split
        #for i in range(len(d_names)):
            #sample_name = d_names[i][0]
            #epoch_names.append(sample_name)
        dataloader.on_epoch_end()
        # Check for redundancy in epoch
        print("Epoch ", j , ", samples ", len(epoch_names))
        if not len(set(epoch_names)) == len(epoch_names):
            print("Epoch samples ", len(epoch_names), " has unique ", len(set(epoch_names)))
        all_names.append(epoch_names)
    
    overall = 0
    for j in range(epochs):
        curr_names = all_names[j]
        repeat = 0
        
        for k in range(epochs):
            if j == k:
                continue
            else:
                for i in range(len(curr_names)):
                    if curr_names[i] in all_names[k]:
                        #print("{} repeat {} epochs {} {}".format(curr_names[i], all_names[j][all_names[j].index(curr_names[i])], j, k))
                        repeat += 1
        print("Epoch ", j , " repeats ", repeat)
        overall += repeat
    print("Overall repeat ",overall)
    
                
                
                
                
            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
