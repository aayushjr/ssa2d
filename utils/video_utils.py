import cv2
import numpy as np
import utils.directory_utils as dir_util
import random
import imgaug
from utils import output_visualizer as ov
from model_params import ModelParameters as Params


class VideoLoader:
    def __init__(self, resize_target_height=None, crop_shape=None, h_flip_chance=.5, visualize=False):
        """
        :param resize_target_height: (int) the height target for the video after we resize it
        :param crop_shape: ((int,int)) the random crop size we want to apply after resizing
        :param h_flip_chance: (float) the chance that an image will be horizontally-flipped
        :param visualize: whether the frames and bounding boxes should be visualized
        """

        self.resize_target_height = resize_target_height
        self.crop_shape = crop_shape
        self.h_flip_chance = h_flip_chance
        self.visualize = visualize

        # augmenters
        #self.Resize = imgaug.augmenters.Resize({'height': resize_target_height, 'width': 'keep-aspect-ratio'},
        #                                       interpolation='linear')
        self.Resize = imgaug.augmenters.Resize({'height': 224, 'width': 224}, interpolation='linear')

        if crop_shape is not None:
            x, y = crop_shape
            self.Crop = imgaug.augmenters.CropToFixedSize(x, y)

        self.HorizontalFlip = imgaug.augmenters.Fliplr(1)


    def load_whole_video(self, video_path, frame_step, bboxes=None, augment=True):
        """
        loads a video from the video_path and performs appropriate augmentations if prompted
        :param video_path: path to video
        :param bboxes: list of lists of imgaug.BoundingBoxesOnImage
        :param frame_step: extract each 'frame_step' frame from the video
        :param augment: whether augmentation should be applied
        :return:
        """
        dir_util.assert_existence(video_path)

        # load video
        capture = cv2.VideoCapture(video_path)

        # decide whether to flip
        should_flip = False

        # go through the video and append all the frames to :vid_frames
        frames = []
        
        i = 0
        while True:
            read_flag, frame = capture.read()

            # if the video wasn't read properly - stop
            if not read_flag:
                ## EOV
                break

            # we only want to get frames that use the appropriate step
            if i % frame_step == 0:
                #frame = frame[..., [2, 1, 0]]  # BGR to RGB
                frames.append(frame)
            i += 1


        # resize to fit in network
        if self.resize_target_height is not None:
            frames, bboxes = self.resize(frames, bboxes)

        # release the video
        capture.release()

        # make a (n_frames, h, w, c) array
        if len(frames) > 0:
            frames = np.stack(frames, axis=0)
        else:
            return np.zeros((Params.NUM_FRAMES, Params.IN_HEIGHT, Params.IN_WIDTH)), []

        return frames, bboxes


    def load_video(self, video_path, start_frame, end_frame, frame_step, bboxes=None, augment=True):
        """
        loads a video from the video_path and performs appropriate augmentations if prompted
        :param video_path: path to video
        :param bboxes: list of lists of imgaug.BoundingBoxesOnImage
        :param start_frame: starting frame of video load.
        :param end_frame: the ending frame of the video (inclusive)
        :param frame_step: extract each 'frame_step' frame from the video
        :param augment: whether augmentation should be applied
        :return:
        """
        dir_util.assert_existence(video_path)

        # load video
        capture = cv2.VideoCapture(video_path)
        capture.set(1, start_frame)  # sets the starting frame

        # get video stats
        total_sample_frames = end_frame - start_frame + 1

        # decide whether to flip
        should_flip = self.h_flip_chance > random.random()

        # go through the video and append all the frames to :vid_frames
        frames = []
        for i in range(total_sample_frames):
            read_flag, frame = capture.read()

            # if the video wasn't read properly - stop
            if not read_flag:
                print('ERROR| video ended before all frames could be captured')
                break

            # we only want to get frames that use the appropriate step
            if i % frame_step == 0:
                #frame = frame[..., [2, 1, 0]]  # BGR to RGB
                frames.append(frame)

        if self.visualize and i % 5 == 0:
            for frame, bbox in zip(frames, bboxes):
                if bbox is not None:
                    ov.display_img(bbox.draw_on_image(frame))
                else:
                    ov.display_img(frame)

        # apply augmentation if prompted
        if augment:
            if self.resize_target_height is not None:
                frames, bboxes = self.resize(frames, bboxes)
            #if self.crop_shape is not None:
            #    frames, bboxes = self.crop(frames, bboxes)
            if should_flip:
                frames, bboxes = self.flip(frames, bboxes)

        if self.visualize and i % 5 == 0:
            for frame, bbox in zip(frames, bboxes):
                if bbox is not None:
                    ov.display_img(bbox.draw_on_image(frame))
                else:
                    ov.display_img(frame)

        # release the video
        capture.release()

        # make a (n_frames, h, w, c) array
        if len(frames) > 0:
            frames = np.stack(frames, axis=0)
        else:
            return np.zeros((Params.NUM_FRAMES, Params.IN_HEIGHT, Params.IN_WIDTH)), []

        return frames, bboxes

    def crop(self, frames, bboxes):
        """
        crop of all frames and bboxes
        :param frames: the frames of the video of shape (n_frames, (h,w,c))
        :param bboxes: bounding-box bboxes of shape (n_frames, (x_min, y_min, x_max, y_max))
        """
        # calculate & set crop position
        sx = random.random()
        sy = random.random()
        self.Crop.set_position((sx, sy))

        # crop the video
        frames, bboxes = self.Crop(images=frames, bounding_boxes=bboxes)
        return frames, bboxes

    def resize(self, frames, bboxes):
        """
        scales all the frames and bboxes down to the new height and the proportional width
        :param frames: list of the frames we want to resize
        :param bboxes: the bounding-box bboxes
        """
        # resize the video
        frames, bboxes = self.Resize(images=frames, bounding_boxes=bboxes)
        return frames, bboxes

    def flip(self, frames, bboxes):
        """
        flip of all frames
        :param frames: list of all frames we want to flip
        :param bboxes: list of all bboxes
        """
        # flip the video
        frames, bboxes = self.HorizontalFlip(images=frames, bounding_boxes=bboxes)
        return frames, bboxes
