# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import collections
import pandas

import numpy as np
import skimage.draw
import torchvision
import echonet


import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import timesformer.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Echonet(torchvision.datasets.VisionDataset):
    """
    Echonet video loader. Construct the Echonet video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video without random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with no cropping. 
    """

    # def __init__(self, cfg, mode, num_retries=10):
    #     """
    #     Construct the Echonet video loader with a given csv file. 
    #     ```
    #     Args:
    #         cfg (CfgNode): configs.
    #         mode (string): Options includes `train`, `val`, or `test` mode.
    #             For the train and val mode, the data loader will take data
    #             from the train or val set, and sample one clip per video.
    #             For the test mode, the data loader will take data from test set,
    #             and sample multiple clips per video.
    #         num_retries (int): number of retries.
    #     """
    #     # Only support train, val, and test mode.
    #     assert mode in [
    #         "train",
    #         "val",
    #         "test",
    #     ], "Split '{}' not supported for Echonet".format(mode)
    #     self.mode = mode
    #     self.cfg = cfg

    #     self._video_meta = {}
    #     self._num_retries = num_retries
    #     # For training or validation mode, one single clip is sampled from every
    #     # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
    #     # video. 
    #     if self.mode in ["train", "val"]:
    #         self._num_clips = 1
    #     elif self.mode in ["test"]:
    #         self._num_clips = (
    #             cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
    #         )

    #     logger.info("Constructing Echonet {}...".format(mode))
        #self._construct_loader()

    # def _construct_loader(self):
    #     """
    #     Construct the video loader.
    #     """
    #     path_to_file = os.path.join(
    #         self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
    #     )
    #     assert PathManager.exists(path_to_file), "{} dir not found".format(
    #         path_to_file
    #     )

    #     self._path_to_videos = []
    #     self._labels = []
    #     self._spatial_temporal_idx = []
    #     with PathManager.open(path_to_file, "r") as f:
    #         for clip_idx, path_label in enumerate(f.read().splitlines()):
    #             assert (
    #                 len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
    #                 == 2
    #             )
    #             path, label = path_label.split(
    #                 self.cfg.DATA.PATH_LABEL_SEPARATOR
    #             )
    #             for idx in range(self._num_clips):
    #                 self._path_to_videos.append(
    #                     os.path.join(self.cfg.DATA.PATH_PREFIX, path)
    #                 )
    #                 self._labels.append(int(label))
    #                 self._spatial_temporal_idx.append(idx)
    #                 self._video_meta[clip_idx * self._num_clips + idx] = {}
    #     assert (
    #         len(self._path_to_videos) > 0
    #     ), "Failed to load Kinetics split {} from {}".format(
    #         self._split_idx, path_to_file
    #     )
    #     logger.info(
    #         "Constructing kinetics dataloader (size: {}) from {}".format(
    #             len(self._path_to_videos), path_to_file
    #         )
    #     )
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(self, cfg,
                 split):
        self.cfg = cfg
        if self.cfg.DATA.root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root, target_transform=self.cfg.DATA.target_transform)

        
        self.split = self.cfg.DATA.split.upper()
        if not isinstance(self.cfg.DATA.target_type, list):
            target_type = [self.cfg.DATA.target_type]
        self.target_type = target_type
        self.mean = np.array(self.cfg.DATA.mean)
        self.std = np.array(self.cfg.DATA.std)
        self.length = self.cfg.DATA.length
        self.max_length = self.cfg.DATA.max_length
        self.period = self.cfg.DATA.period
        self.clips = self.cfg.DATA.clips
        self.pad = self.cfg.DATA.pad
        self.noise = self.cfg.DATA.noise
        self.target_transform = self.cfg.DATA.target_transform
        self.external_test_location = self.cfg.DATA.external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix
            self.outcome = data.values.tolist()

            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # A small number of videos are missing traces; remove these videos
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]
        logger.info("Constructing Echonet {}...".format(split))

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = echonet.utils.loadvideo(video).astype(np.float32)

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select clips from video
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]

        return video, target, index, {}

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

    
def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
