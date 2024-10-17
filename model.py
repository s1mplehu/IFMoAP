import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision


class attention(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_features, in_features),
            nn.ReLU(),
        )
        self.project = nn.Sequential(
            nn.Linear(3840, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x_att = self.attention(x)
        out = x * x_att
        out = self.project(out)
        return out


class CI_Extractor(nn.Module):
    def __init__(self):
        super(CI_Extractor, self).__init__()
        self.net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        # 深度可分离卷积 Depthwise seperable convolution
        self.net.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(5, 1, dilation=1, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False, groups=1),
        )

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(5, 1, dilation=2, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), bias=False, groups=1),
        )

        self.conv1_5 = nn.Sequential(
            nn.Conv2d(5, 1, dilation=5, kernel_size=(3, 3), stride=(1, 1), padding=(7, 7), bias=False, groups=1),
        )
        self.granularity_level_attention = attention(3840, 512)


    def forward(self, x):
        conv1_1_feature = self.conv1_1(x)
        conv1_2_feature = self.conv1_2(x)
        conv1_5_feature = self.conv1_5(x)
        conv1_feature = torch.cat((conv1_1_feature, conv1_2_feature), dim=1)
        conv1_feature = torch.cat((conv1_feature, conv1_5_feature), dim=1)   
        conv1_feature = self.net.conv1(conv1_feature)  
        bn1_feature = self.net.bn1(conv1_feature)
        relu1_feature = self.net.relu(bn1_feature)
        maxpool1_feature = self.net.maxpool(relu1_feature)

        layer1_feature = self.net.layer1(maxpool1_feature)
        layer2_feature = self.net.layer2(layer1_feature)
        layer3_feature = self.net.layer3(layer2_feature)
        layer4_feature = self.net.layer4(layer3_feature)

        os_1 = self.net.avgpool(layer1_feature).view(x.size(0), -1)
        os_2 = self.net.avgpool(layer2_feature).view(x.size(0), -1)
        os_3 = self.net.avgpool(layer3_feature).view(x.size(0), -1)
        os_4 = self.net.avgpool(layer4_feature).view(x.size(0), -1)
        os = torch.hstack((os_1, os_2))
        os = torch.hstack((os, os_3))
        os = torch.hstack((os, os_4))

        os = os.view(x.size(0), -1)
        os = self.granularity_level_attention(os)

        return os


class CI_Predictor(nn.Module):
    def __init__(self):
        super(CI_Predictor, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(512, 10),
        ) 

    def forward(self, os):
        output = self.mlp(os)
        return output


class common_specific_att(nn.Module):
    def __init__(self):
        super(common_specific_att, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    
    def forward(self, x):
        out = self.project(x)
        return out


class FPS(nn.Module):
    def __init__(self,):
        super(FPS, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
    def forward(self, x):
        out = self.project(x)
        return out
    

class FP_Extractor(nn.Module):
    def __init__(self):
        super(FP_Extractor, self).__init__()
        self.affine_RDK = nn.Linear(2048, 512)
        self.affine_morgan = nn.Linear(1024, 512)
        self.affine_PubChem = nn.Linear(881, 512)
        self.affine_MACCS = nn.Linear(167, 512)

        self.FPC = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.FPS_RDK = FPS()
        self.FPS_morgan = FPS()
        self.FPS_PubChem = FPS()
        self.FPS_MACCS = FPS()  

        self.fusion_layer = common_specific_att()

    def forward(self, fp_moagan_6, fp_MACCS, fp_PubChem, fp_RDK):
        affine_RDK = self.affine_RDK(fp_RDK)
        affine_morgan = self.affine_morgan(fp_moagan_6)
        affine_PubChem = self.affine_PubChem(fp_PubChem)
        affine_MACCS = self.affine_MACCS(fp_MACCS)

        zeros_tensor = torch.zeros_like(affine_RDK)
        affine_RDK_padding = torch.cat((affine_RDK, zeros_tensor), dim=1)
        affine_RDK_padding = torch.cat((affine_RDK_padding, zeros_tensor), dim=1)
        affine_RDK_padding = torch.cat((affine_RDK_padding, zeros_tensor), dim=1)

        affine_morgan_padding = torch.cat((zeros_tensor, affine_morgan), dim=1)
        affine_morgan_padding = torch.cat((affine_morgan_padding, zeros_tensor), dim=1)
        affine_morgan_padding = torch.cat((affine_morgan_padding, zeros_tensor), dim=1)

        affine_PubChem_padding = torch.cat((zeros_tensor, zeros_tensor), dim=1)
        affine_PubChem_padding = torch.cat((affine_PubChem_padding, affine_PubChem), dim=1)
        affine_PubChem_padding = torch.cat((affine_PubChem_padding, zeros_tensor), dim=1)
    
        affine_MACCS_padding = torch.cat((zeros_tensor, zeros_tensor), dim=1)
        affine_MACCS_padding = torch.cat((affine_MACCS_padding, zeros_tensor), dim=1)
        affine_MACCS_padding = torch.cat((affine_MACCS_padding, affine_MACCS), dim=1)

        c_RDK = self.FPC(affine_RDK_padding)
        c_morgan = self.FPC(affine_morgan_padding)
        c_PubChem = self.FPC(affine_PubChem_padding)
        c_MACCS = self.FPC(affine_MACCS_padding)

        c = c_RDK + c_morgan + c_PubChem + c_MACCS

        c_t = (c_RDK + c_morgan + c_PubChem + c_MACCS) / 4

        s_RDK = self.FPS_RDK(affine_RDK)
        s_morgan = self.FPS_morgan(affine_morgan)
        s_PubChem = self.FPS_PubChem(affine_PubChem)
        s_MACCS = self.FPS_MACCS(affine_MACCS)

        s = torch.cat((s_RDK, s_morgan), dim=1)
        s = torch.cat((s, s_PubChem), dim=1)
        s = torch.cat((s, s_MACCS), dim=1)

        spcific_encoder_RDK = torch.unsqueeze(spcific_encoder_RDK, dim=1)
        spcific_encoder_morgan = torch.unsqueeze(spcific_encoder_morgan, dim=1)
        spcific_encoder_PubChem = torch.unsqueeze(spcific_encoder_PubChem, dim=1)
        spcific_encoder_MACCS = torch.unsqueeze(spcific_encoder_MACCS, dim=1)

        h = torch.cat((s_RDK, s_morgan), dim=1)
        h = torch.cat((h, s_PubChem), dim=1)
        h = torch.cat((h, s_MACCS), dim=1)
        h_T = h.permute(0, 2, 1)
        specific_cm = torch.matmul(h, h_T)

        # 构建单位矩阵
        identity_matrix = torch.eye(h.shape[1])
        identity_matrix = torch.unsqueeze(identity_matrix, dim=0)
        identity_matrix = identity_matrix.repeat(h.shape[0], 1, 1)

        f = torch.cat((c, s), dim=1)
        f = self.fusion_layer(f)

        return f


class FP_Predictor(nn.Module):
    def __init__(self):
        super(FP_Predictor, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(64, 10),
        ) 

    def forward(self, final_feature):
        output = self.mlp(final_feature)
        return output


class multi_modal_model(nn.Module):
    def __init__(self):
        super(multi_modal_model, self).__init__()

        self.CI_Extractor = CI_Extractor()
        self.CI_Predictor = CI_Predictor()
        self.FP_Extractor = FP_Extractor()
        self.FP_Predictor = FP_Predictor()


    def forward(self, ci, fp_moagan_6, fp_MACCS, fp_PubChem, fp_RDK):
        ci_embedding = self.CI_Extractor(ci)
        fp_embedding  = self.FP_Extractor(fp_moagan_6, fp_MACCS, fp_PubChem, fp_RDK)

        ci_score = self.CI_Predictor(ci_embedding)
        fp_score = self.FP_Predictor(fp_embedding)
        score = 0.3 * ci_score + 0.7 * fp_score

        return score
        