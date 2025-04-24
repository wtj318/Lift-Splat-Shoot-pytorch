"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    """后两层特征做一个融合:
       x1:上采样 -> x1 = concat(x2,x1),-> x1 =convLayer(x1)
    """
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D #41
        self.C = C #64

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    # 利用深度的概率和图像特征的积，完成对每个栅格点的距离预测。
    def get_depth_feat(self, x):
        #通过EfficientNet提取特征,Up上采样到512->24x512x8x22
        x = self.get_eff_depth(x) 
        # 1x1卷积，将数据维度变化为24x105x8x22
        x = self.depthnet(x) 
        #前面d=41 depth求softmax概率分布 -> depth:24x41x8x22
        depth = self.get_depth_dist(x[:, :self.D]) 
        #depth*后面的64特征维度 外积->new_x:24x64x41x8x22
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        return depth, new_x

    #通过efficientnet提取特征
    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        # 输出数据大小为24x512x8x22
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)
        # depth: 深度方向概率特征
        #x: 伪点云（图像特征X深度方向的概率）
        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
       
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        # dx:每个栅格大小：[0.5,0.5, 20]
        # bx:第一个栅格中心：[-49.75,-49.75,0]
        # nx:栅格数：[200,200,1]
        self.dx = nn.Parameter(dx, requires_grad=False) 
        self.bx = nn.Parameter(bx, requires_grad=False) 
        self.nx = nn.Parameter(nx, requires_grad=False) 

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    # 测试点云视图-wtj-2025-4-18
    def show_frustum(self, frustum):#[41, 8, 22, 3]
            frustum = frustum.cpu().detach().numpy()
            frustum_one = frustum.reshape(-1,3)
            print("geom_one.shape:",frustum_one.shape)
            from matplotlib import pyplot as plt
            plt.figure(figsize=(12, 8))
            colors = ['r', 'g', 'b', 'c', 'm']  # 颜色列表
            # x = geom_one[:, :, 0]
            # for i in range(5):
            plt.scatter(frustum_one[:, 0], frustum_one[:, 1], 0.5, c=colors[0])
            plt.axis('image')
            plt.show()
            plt.savefig("./frustum.png")
    #为每张图片生成棱台状点云 
    def create_frustum(self):
        # make grid in image plane
        # 数据增强后图片大小:ogfH:128, ogfW:352
        ogfH, ogfW = self.data_aug_conf['final_dim']
        # 下采样16倍后图像的高宽:fH:128/16=8,fw:352/16=22
        fH, fW = ogfH // self.downsample, ogfW // self.downsample

        """
        在深度方向上划分网格 ds: DxfHxfW (41x8x22)
        ['dbound'] = [4, 45, 1]->arange->[4,5,6,...,44]
        ->view->(41,1,1)->expand(扩展维度数据的尺寸)->ds:41x8x22
        """
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape  # D: 41 表示深度方向上网格的数量

        """xs:在宽度方向上划分网格, 在0到351上划分22个格子 xs: DxfHxfW(41x8x22)
           ogfW:352 -> linspace:均匀划分 -> [0,16,32..336]  大小=fW(22)   
            -> view-> 1x1xfW(1x1x22)-> expand-> xs: DxfHxfW(41x8x22)
        """
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        
        """ys:在高度方向上划分网格, 在0到128上划分22个格子 ys: DxfHxfW(41x8x22)
           ogfH:128 -> linspace:均匀划分 -> [0,16,32...,112]  大小=fH(8)   
            -> view-> 1xfHx1(1x8x1)-> expand-> =ys: DxfHxfW(41x8x22)
        """
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        """frustum: 把xs,ys,ds,堆叠到一起
           stack后-> frustum: DxfHxfWx3
           堆积起来形成网格坐标, 
           frustum[i,j,k,0]就是(i,j)位置,深度为k的像素的宽度方向上的栅格坐标   [41,8,22,3] 
        """
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    # 图像坐标系 -> ego坐标系
    #可视化测试
    def show_geom(self, geom, a):#[b, 6, 41, 8, 22, 3]
        geom = geom.cpu().detach().numpy()
        geom_one = geom[0].reshape(6, -1, 3) #[5, 7216, 3]

        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 8))
        colors = ['r', 'g', 'b', 'c', 'm','y']  # 颜色列表
        # x = geom_one[:, :, 0]
        for i in range(6):
            plt.scatter(geom_one[i, :, 0], geom_one[i, :, 1], 0.5, c=colors[i])
        plt.axis('image')
        plt.show()
        plt.savefig("./geom"+str(a)+".png")

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  #trans.shape: [4,6,3]
        print("trans.shape:",trans.shape)
        # 抵消图像增强以及预处理对像素的变化
        # self.frustum[B x N x D x H x W x 3]: 为每张图像生成一个棱台状的点云 
         
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        # test_points = self.frustum
        # self.show_frustum(test_points)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        # 图像坐标系 -> 归一化相机坐标系 -> 相机坐标系 -> 车身坐标系
        """图像坐标系（柱体）->归一化相机坐标系（棱柱） 
            xs,ys,lamda-> xs*lamda, ys*lamda, lamda
           对点云中预测的宽度和高度上的栅格坐标，将其乘以深度上的栅格坐标。
        """
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5) #[4,6,41,8,22,3,1]
        print("相机坐标系:points:",points.shape)
        self.show_geom(points,1)
        """
           相机->ego坐标系: 
           相机内参矩阵取逆:intri 
            
             
              ns: 3*3, ->inverse(取逆)
           ego 坐标系下点云坐标= 
               点云视锥 *相机内参逆矩阵 * 相机坐标系到ego坐标系旋转矩阵 + 平移矩阵
        """
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3) #加上相机坐标系到车身坐标系的平移矩阵
        self.show_geom(points,2)
        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape #[4, 6, 3, 128, 352]
        print("image.shape:",x.shape)

        x = x.view(B*N, C, imH, imW)
        #Lift: 伪点云（图像特征X深度方向的概率）:24*64*41*8*22
        x = self.camencode(x)  
        print("伪点云shape:",x.shape)#[24, 64, 41, 8, 22])
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample) #将前两维度分开。
        x = x.permute(0, 1, 3, 4, 5, 2) # x:4*6*41*8*22*64

        return x

    def voxel_pooling(self, geom_feats, x):
        """
        geom_feats [B, N, 41, 8, 22, 3]:ego 坐标系下的坐标点
        x:[B, N, 41, 8, 22, 64]: 图像点云特征
        """
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W #173184,只留下C语义维度。
        # 图像特征点云展平：一共Nprime个点云
        x = x.reshape(Nprime, C)
        # flatten indices 
        """图像坐标系x右 y下, x和ego坐标系是一一对应的。
            dx:每个栅格大小：[0.5,0.5, 20]
            bx:第一个栅格中心：[-49.75,-49.75,0]
            nx:栅格数：[200,200,1]
            (self.bx - self.dx/2)  ->  初始点：[-50,-50,-10]
        """
        #把ego点云移到bev空间下：向右向上平移【50,50,10 ],除以栅格【0.5，0.5，20】取整
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long() 
        self.show_geom(geom_feats, 3)
        geom_feats = geom_feats.view(Nprime, 3) #【72160，3】像素映射关系展开
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        # 173184x4, [x,y,z,batch_ix]
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        print("batch_ix:",batch_ix)
        # filter out points that are outside box
        #过滤操作 将所有小于0，大于200的XY， 小于0大于1的Z都过滤
        #x.shape=[173184, 64]  geom_feats.shape=[173184, 4]
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2]) # kept[true,False]
        x = x[kept]
        geom_feats = geom_feats[kept] 
        #将平移取整后的坐标只保留x:(0,200),y:(0,200),z:(0，1）之间的数据

        """对数据进行排序
        args: 
            input: 
                    geom_feats:[num,4],(x,y,z,batch);  nx:[200,200,1]; B=4
            output:
                    ranks= X x 200 x 1 x B + Y x 1 x B + Z x B + batch                
         (ranks : geom_feats中每个点平铺在一维数组中的位置)
           
        """
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3] 
        sorts = ranks.argsort() #对rank进行排序，相邻的点相近
        #根据索引对 x, geom_feats, ranks 进行排序
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        
        # cumsum trick
        #给每个点赋予一个ranks，ranks相等的点在同一个batch中，也在同一个栅格中，将ranks排序并返回排序的索引以至于x,geom_feats,ranks都是按照ranks排序的。
        # 区间索引计算差值，不同为1， 点特征在0维度累加，得到相邻位置特征变化的最大值。
        #取出相邻不同的位置的值，并取出对应的元素，再错位相减，得到特征变化最大的边界
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        #图像特征映射到BEVgrid下 -> 特征final[4,64,1,200,200]
        # geom_feats:[29072, 4])
        # x:[29072,64]
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        # splat 高度维度
        final = torch.cat(final.unbind(dim=2), 1)
        

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        # 生成棱柱：图像坐标系->自车坐标系位置:[B x N x D x H x W x 3]
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        """输入图像-> efficient-B0+Neck 融合特征
        #Lift:depthnet(1x1conv)= 深度估计+语义特征 ->深度特征 + softmax: 深度概率
        # 伪点云：深度概率 x 语义特征：[4,6,41,8,22,64]"""
        x = self.get_cam_feats(x) 
        """splat:将视锥均匀采样到3D空间,利用几何特征和得到的伪点云特征求和池化,splat掉高度维度,变为BEV特征
           output:[4,64,1,200,200]"""
        x = self.voxel_pooling(geom, x) 
        

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        # x:[4,6,3,128,352] 输入图像
        # rots:由相机坐标系->车身坐标系的旋转矩阵,rots = (bs, N, 3, 3)；
        # trans:由相机坐标系->车身坐标系的平移矩阵,trans=(bs, N, 3)；
        # intrinsic:相机内参,intrinsic = (bs, N, 3, 3)；
        # post_rots:由图像增强引起的旋转矩阵,post_rots = (bs, N, 3, 3)；
        # post_trans:由图像增强引起的平移矩阵,post_trans = (bs, N, 3)；
        #得到bev grid 下的语义信息 [4,64,1,200,200]
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x) #CNN结构
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)


# 测试点云视图-wtj-2025-4-18
def show_geom(self, geom):#[b, 5, 41, 8, 22, 3]
        geom = geom.cpu().detach().numpy()
        geom_one = geom[0].reshape(5, -1, 3) #[5, 7216, 3]

        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 8))
        colors = ['r', 'g', 'b', 'c', 'm']  # 颜色列表
        # x = geom_one[:, :, 0]
        for i in range(5):
            plt.scatter(geom_one[i, :, 0], geom_one[i, :, 1], 0.5, c=colors[i])
        plt.axis('image')
        plt.show()
        plt.savefig("./geom2.png")



