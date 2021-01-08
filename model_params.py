from typing import NamedTuple


class Paths:
    log_model_out = 'out/model_output/'
    log_metric_out = 'out/metric_output/'


class OutputBranch:
    def __init__(self, branch_name, class_type, activation):
        self.branch_name = branch_name
        self.class_type = class_type
        self.activation = activation
        self.num_classes, self.class_to_id, self.id_to_class = self.get_dictionaries()
        self.metrics = None

    def get_dictionaries(self):
        if self.class_type == Classes.obj:
            return len(Classes.object_to_id), Classes.object_to_id, Classes.id_to_object
        elif self.class_type == Classes.act:
            return len(Classes.action_to_id), Classes.action_to_id, Classes.id_to_action
        elif self.class_type == Classes.rel:
            return len(Classes.relation_to_id), Classes.relation_to_id, Classes.id_to_relation
        elif self.class_type == Classes.ctr:
            return 1, None, None
        else:
            raise AttributeError("Unknown class type:", self.class_type)


class RunModes(NamedTuple):
    training = 'training'
    validation = 'validation'

    binary = 'binary'
    categorical = 'categorical'


def list_to_index_dictionary(listy_list):
    dictionary = dict()
    for i, val in enumerate(listy_list):
        dictionary[val] = i

    return dictionary


class Classes:
    obj = 'obj'
    act = 'act'
    rel = 'rel'
    ctr = 'ctr'
    clp = 'clp'
    act_rel = 'act_rel'

    id_to_action = ['bite', 'caress', 'carry', 'chase', 'clean', 'close', 'cut', 'drive', 'feed', 'get_off', 'get_on',
                     'grab', 'hit', 'hold', 'hold_hand_of', 'hug', 'kick', 'kiss', 'knock', 'lean_on', 'lick', 'lift',
                     'open', 'pat', 'play(instrument)', 'point_to', 'press', 'pull', 'push', 'release', 'ride',
                     'shake_hand_with', 'shout_at', 'smell', 'speak_to', 'squeeze', 'throw', 'touch', 'use', 'watch',
                     'wave', 'wave_hand_to']

    id_to_action_relation = ['bite', 'caress', 'carry', 'chase', 'clean', 'close', 'cut', 'drive', 'feed', 'get_off', 'get_on',
                                 'grab', 'hit', 'hold', 'hold_hand_of', 'hug', 'kick', 'kiss', 'knock', 'lean_on', 'lick', 'lift',
                                 'open', 'pat', 'play(instrument)', 'point_to', 'press', 'pull', 'push', 'release', 'ride',
                                 'shake_hand_with', 'shout_at', 'smell', 'speak_to', 'squeeze', 'throw', 'touch', 'use', 'watch',
                                 'wave', 'wave_hand_to', 'above', 'away', 'behind', 'beneath', 'in_front_of', 'inside', 
                                 'next_to', 'towards']
    
    '''
    id_to_action = sorted(list(
        {'bite', 'caress', 'carry', 'chase', 'clean', 'close', 'cut', 'drive', 'feed', 'get_off', 'get_on', 'grab',
         'hit', 'hold', 'hold_hand_of', 'hug', 'kick', 'kiss', 'knock', 'lean_on', 'lick', 'lift', 'open', 'pat',
         'play(instrument)', 'point_to', 'press', 'pull', 'push', 'release', 'ride', 'shake_hand_with', 'shout_at',
         'smell', 'speak_to', 'squeeze', 'throw', 'touch', 'use', 'watch', 'wave', 'wave_hand_to', 'no_action'} -
        {'watch', 'hold'}))
    '''
    
    unwanted_actions = ['watch', 'hold']

    id_to_relation = sorted(['above', 'away', 'behind', 'beneath', 'in_front_of', 'inside', 'next_to', 'towards'])

    id_to_object = sorted(list(
        {'baby', 'chair', 'toy', 'dog', 'sofa', 'table', 'adult', 'bottle', 'car', 'guitar', 'cup', 'refrigerator',
         'ball/sports_ball', 'child', 'cake', 'microwave', 'camera', 'laptop', 'baby_seat', 'bicycle', 'pig', 'cat',
         'watercraft', 'faucet', 'sink', 'oven', 'handbag', 'electric_fan', 'turtle', 'rabbit', 'stool', 'aircraft',
         'backpack', 'snowboard', 'tiger', 'penguin', 'bird', 'screen/monitor', 'cellphone', 'crab', 'panda', 'leopard',
         'lion', 'elephant', 'bat', 'baby_walker', 'surfboard', 'toilet', 'fish', 'antelope', 'bench', 'skateboard',
         'chicken', 'motorcycle', 'piano', 'horse', 'kangaroo', 'hamster/rat', 'dish', 'duck', 'ski', 'stingray',
         'fruits', 'camel', 'bus/truck', 'vegetables', 'scooter', 'snake', 'train', 'suitcase', 'squirrel',
         'sheep/goat', 'bread', 'bear', 'cattle/cow', 'stop_sign', 'traffic_light', 'racket', 'crocodile', 'frisbee'} ))
    
    id_to_object.append('background')   # put background at the end   # CCE
    
    object_to_id = list_to_index_dictionary(id_to_object)
    action_to_id = list_to_index_dictionary(id_to_action)
    relation_to_id = list_to_index_dictionary(id_to_relation)
    act_rel_to_id = list_to_index_dictionary(id_to_action_relation)

    @staticmethod
    def key_to_classes(key):
        if key == Classes.obj:
            return Classes.id_to_object
        elif key == Classes.act:
            return Classes.id_to_action
        elif key == Classes.rel:
            return Classes.id_to_relation
        elif key == Classes.act_rel:
            return Classes.id_to_action_relation
        else:
            raise AttributeError('Unknown key:', key)


class ModelParameters:
    # model parameters
    model_name = 'act_det_vidorloader'
    SEED = 0
    BATCH_SIZE = 6
    MAX_TRAIN_BATCHES_PER_EPOCH = 500
    MAX_VALID_BATCHES_PER_EPOCH = 400
    NUM_EPOCHS = 500
    USE_MULTIPROCESSING = True
    NUM_WORKERS = 6

    # initialization
    LOAD_PRETRAINED = True
    PRETRAINED_PATH = 'data/checkpoints/pretrain_i3d_kinetics_rgb.h5'

    # input parameters
    NUM_FRAMES = 16
    IN_HEIGHT = 224
    IN_WIDTH = 224
    IN_FRAME_DIM = (IN_HEIGHT, IN_WIDTH)
    OUT_FRAME_DIM = (56, 56)
    NUM_RGB_CHANNELS = 3

    INPUT_SHAPE = (NUM_FRAMES, IN_HEIGHT, IN_WIDTH, NUM_RGB_CHANNELS)

    # classes
    NUM_OBJ = len(Classes.object_to_id)
    NUM_ACT = len(Classes.action_to_id)
    NUM_REL = len(Classes.relation_to_id)
    NUM_ACT_REL = len(Classes.act_rel_to_id)
    NUM_ACTIONS_RELATIONS = NUM_ACT + NUM_REL
    USE_OBJ = True
    USE_ACT = False
    USE_ACT_REL = True
    USE_REL = False
    USE_CTR = False
    USE_CLP = False
    
    PLOT_MODEL = False
    FREEZE = True
    USE_ACT_TEMP = False

    obj_run = RunModes.binary
    act_run = RunModes.categorical
    rel_run = RunModes.binary
    ctr_run = RunModes.binary

    # centroids
    CENTROID_SIZE = 13
    CENTROID_FWHM = 5

    # hyperparameters
    LR = 1e-4
    LR_DECAY = 1e-6

    # augmentation
    should_rand_flip = True
    FRAME_STEPS = (1, 2)
