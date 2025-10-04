from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class nnUNetTrainer_ResEnc(nnUNetTrainer):
    def build_network(self):
        return ResidualEncoderUNet(**self.configuration_manager.arch_kwargs)
