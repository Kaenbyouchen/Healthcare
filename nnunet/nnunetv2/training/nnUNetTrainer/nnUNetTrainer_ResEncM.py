from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class nnUNetTrainer_ResEncM(nnUNetTrainer):
    """
    ResEnc-M 版，通道数中等
    """
    def build_network(self):
        self.max_num_epochs = 100
        arch_kwargs = self.configuration_manager.arch_kwargs.copy()
        arch_kwargs["features_per_stage"] = [32, 64, 128, 256, 320, 320]  # 类似官方M
        return ResidualEncoderUNet(**arch_kwargs)
