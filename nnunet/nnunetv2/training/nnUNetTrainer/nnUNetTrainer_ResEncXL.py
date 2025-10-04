from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class nnUNetTrainer_ResEncXL(nnUNetTrainer):
    """
    ResEnc-XL 版，超大规模
    """
    def build_network(self):
        self.max_num_epochs = 100   #此处修改最大epoch
        arch_kwargs = self.configuration_manager.arch_kwargs.copy()
        arch_kwargs["features_per_stage"] = [64, 128, 256, 512, 640, 768, 1024]  # XL版
        arch_kwargs["n_stages"] = 7  # XL通常更深
        return ResidualEncoderUNet(**arch_kwargs)
