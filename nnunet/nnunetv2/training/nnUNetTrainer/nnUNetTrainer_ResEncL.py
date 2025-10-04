from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class nnUNetTrainer_ResEncL(nnUNetTrainer):
    """
    ResEnc-L 版，通道数更宽
    """
    def build_network(self):
        self.max_num_epochs = 100
        arch_kwargs = self.configuration_manager.arch_kwargs.copy()
        arch_kwargs["features_per_stage"] = [48, 96, 192, 384, 512, 640]  # L版更宽
        return ResidualEncoderUNet(**arch_kwargs)
