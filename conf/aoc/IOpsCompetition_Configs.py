class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'IOpsCompetition'
        # model configs
        self.input_channels = 1
        self.hidden_size = 64
        self.num_layers = 3
        self.project_channels = 16

        self.dropout = 0.45
        self.window_size = 16
        self.time_step = 2

        # training configs
        self.num_epoch = 10
        self.freeze_length_epoch = 10
        self.change_center_epoch = 10

        self.center_eps = 0.1
        self.omega1 = 1
        self.omega2 = 0.01

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = False
        self.batch_size = 512

        # Anomaly Detection parameters
        self.nu = 0.001
        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.0015
        # Methods for determining thresholds ("fix","floating","one-anomaly")
        self.threshold_determine = 'floating'
        # Specify model objective ("one-class" or "soft-boundary")
        self.objective = 'soft-boundary'

        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.scale_ratio = 0.8
        self.jitter_ratio = 0.3


