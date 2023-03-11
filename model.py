import torch
import numpy as np
from torchvision import utils
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Union, List, Optional

# class FeatureExtractor(torch.nn.Module):
#     '''
#     Feature Extractor class
#     '''
#     def __init__(self, name = None, weights = None) -> None:
#         super().__init__()
        
#         name = name.lower()
#         all_models = ['alexnet', "convnext", "densenet", "efficientnet", "efficientnetv2",
#                         "googlenet", "inceptionv3", "maxvit", "mnasnet", "mobilenetv2", "mobilenetv3",
#                         "regnet", "resnet", "resnext", "shufflenetv2", "squeezenet", "shufflenetv2",
#                         "swin", "vgg", "vit", "wideresnet"]
#         assert name.lower() in all_models, """Supported models are ['AlexNet', "ConvNext", "DenseNet", "EfficientNet", "EfficientNetV2",
#                         "GoogLeNet", "InceptionV3", "MaxVit", "MNASNet", "MobileNetV2", "MobileNetV3",
#                         "RegNet", "ResNet", "ResNeXt", "ShuffleNetV2", "SqueezeNet", "ShuffleNetV2",
#                         "SWIN", "VGG", "ViT", "WideResNet"]"""
        
#         #Feature extractor 
#         self.fx, self.transforms = self.load_model(name, weights)     #resnet50(weights = 'IMAGENET1K_V2')       #feature extractor model
        
#         for param in self.fx.parameters():
#             param.requires_grad = False   #so that with torch.no_grad is not required
#         # self.fx_required = {'flatten':'flatten'}        #required layers from all the get_graph_node_names(m)
#         # self.fx = create_feature_extractor(self.fx_base, return_nodes=self.fx_required)       #last dense layer

#     def load_model(self, name, weights = None):
#         m = MODELS()
#         return m.load_model(name, weights)

#     def forward(self, x):
#         x = self.transforms(x)
#         return self.fx(x)['out']

# class Matcher(torch.nn.Module):
#     '''
#     Feature matcher class to compute similarity of given two node feature
#     '''    
#     def forward(self, n1, n2, edge_index = None):
#         # n1 = n1[edge_index[0]]
#         # n2 = n2[edge_index[1]]
#         n1 = n1.flatten(start_dim = 1) if n1.ndim>2 else n1
#         n2 = n2.flatten(start_dim = 1) if n2.ndim>2 else n2
#         return (n1 @ n2.t()) / (torch.norm(n1, dim =-1).unsqueeze(-1) @ torch.norm(n2, dim =-1).unsqueeze(-1).t())


# class LoopClosureDetector(torch.nn.Module):
#     '''
#     LCD base model for extracting features of the node image & then calculating matching criteria to
#     predict edge between two nodes.
#     '''
#     def __init__(self, FX, MX) -> None:
#         super().__init__()  
#         self.fx = FX
#         self.mx = MX

#     def forward(self, x):
#         # x has shape [N, in_channels]  
#         # edge_index has shape [2, E]
#         # introduce another variable and then make the comparision bewteen them
#         # 1st -> graph nodes, 2nd-> another graph nodes or indiviudal image to add in graph
#         x = self.fx(x)
#         pred = self.mx(x, x)
#         return pred
    
class MODELS:
    
    def load_model(self, name, weights):
        default = self.resnet
        print(f'Loading model: {name}')
        return getattr(self, name, lambda: default)()

    def alexnet(self):
        from torchvision.models import alexnet, AlexNet_Weights
        fx_required = {'flatten':'out'}
        weights = AlexNet_Weights.DEFAULT
        base = alexnet(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def convnext(self):
        from torchvision.models import convnext_base, ConvNeXt_Base_Weights
        fx_required = {'classifier.1':'out'}
        weights = ConvNeXt_Base_Weights.DEFAULT
        base = convnext_base(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def densenet(self):
        from torchvision.models import densenet161, DenseNet161_Weights
        fx_required = {'flatten':'out'}
        weights = DenseNet161_Weights.DEFAULT
        base = densenet161(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def efficientnet(self):
        from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
        fx_required = {'flatten':'out'}
        weights = EfficientNet_B6_Weights.DEFAULT
        base = efficientnet_b6(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def efficientnetv2(self):
        from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
        fx_required = {'flatten':'out'}
        weights = EfficientNet_V2_M_Weights.DEFAULT
        base = efficientnet_v2_m(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def googlenet(self):
        from torchvision.models import googlenet, GoogLeNet_Weights
        fx_required = {'flatten':'out'}
        weights = GoogLeNet_Weights.DEFAULT
        base = googlenet(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def inceptionv3(self):
        from torchvision.models import inception_v3, Inception_V3_Weights
        fx_required = {'flatten':'out'}
        weights = Inception_V3_Weights.DEFAULT
        base = inception_v3(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def maxvit(self):
        from torchvision.models import maxvit_t, MaxVit_T_Weights
        fx_required = {'classifier.5':'out'}
        weights = MaxVit_T_Weights.DEFAULT
        base = maxvit_t(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def mnasnet(self):
        from torchvision.models import mnasnet1_3, MNASNet1_3_Weights
        fx_required = {'mean':'out'}
        weights = MNASNet1_3_Weights.DEFAULT
        base = mnasnet1_3(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def mobilenetv2(self):
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        fx_required = {'flatten':'out'}
        weights = MobileNet_V2_Weights.DEFAULT
        base = mobilenet_v2(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def mobilenetv3(self):
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        fx_required = {'flatten':'out'}
        weights = MobileNet_V3_Large_Weights.DEFAULT
        base = mobilenet_v3_large(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def regnet(self):
        from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights
        fx_required = {'flatten':'out'}
        weights = RegNet_Y_32GF_Weights.DEFAULT
        base = regnet_y_32gf(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms
    
    def resnet(self):
        from torchvision.models import resnet50, ResNet50_Weights
        fx_required = {'avgpool':'out'}
        weights = ResNet50_Weights.DEFAULT
        base = resnet50(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms
        
    def resnext(self):
        from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
        fx_required = {'flatten':'out'}
        weights = ResNeXt101_64X4D_Weights.DEFAULT
        base = resnext101_64x4d(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def shufflenetv2(self):
        from torchvision.models import shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights
        fx_required = {'stage3.7.view_1':'out'} #'mean'
        weights = ShuffleNet_V2_X2_0_Weights.DEFAULT
        base = shufflenet_v2_x2_0(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def squeezenet(self):
        from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
        fx_required = {'flatten':'out'}
        weights = SqueezeNet1_1_Weights.DEFAULT
        base = squeezenet1_1(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def swin(self):
        from torchvision.models import swin_v2_s, Swin_V2_S_Weights
        fx_required = {'flatten':'out'}
        weights = Swin_V2_S_Weights.DEFAULT
        base = swin_v2_s(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def vgg(self):
        from torchvision.models import vgg16_bn, VGG16_BN_Weights
        fx_required = {'flatten':'out'}
        weights = VGG16_BN_Weights.DEFAULT
        base = vgg16_bn(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def vit(self):
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        fx_required = {'getitem_5':'out'}
        weights = ViT_B_16_Weights.DEFAULT
        base = vit_b_16(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

    def wideresnet(self):
        from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights
        fx_required = {'flatten':'out'}
        weights = Wide_ResNet101_2_Weights.DEFAULT
        base = wide_resnet101_2(weights = weights)
        transforms = weights.transforms()
        return create_feature_extractor(base, return_nodes=fx_required), transforms

