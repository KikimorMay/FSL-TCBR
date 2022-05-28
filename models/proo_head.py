
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



def L2SquareDist(A: Tensor, B: Tensor, base_means, average: bool = True) -> Tensor:
    r"""calculate parwise euclidean distance between two batchs of features.
    Args:
        A: Torch feature tensor. size:[Batch_size, Na, nC]
        B: Torch feature tensor. size:[Batch_size, Nb, nC]
    Output:
        dist: The calculated distance tensor. size:[Batch_size, Na, Nb]
    """
    assert A.dim() == 3
    assert B.dim() == 3
    assert A.size(0) == B.size(0) and A.size(2) == B.size(2)
    nB = A.size(0)
    Na = A.size(1)
    Nb = B.size(1)
    nC = A.size(2)

    # print(A.shape)   # shape[1, 75, 640]
    # print(B.shape)    # shape[1, 5, 640]

    # AB = A * B = [nB x Na x nC] * [nB x nC x Nb] = [nB x Na x Nb]
    AB = torch.bmm(A, B.transpose(1, 2))   # shape [1, 75, 5]

    AA = (A * A).sum(dim=2, keepdim=True).view(nB, Na, 1)  # [nB x Na x 1]  # shape[1, 75, 1]
    BB = (B * B).sum(dim=2, keepdim=True).view(nB, 1, Nb)  # [nB x 1 x Nb]   # shape [1 ,1, 5]
    # l2squaredist = A*A + B*B - 2 * A * B
    dist = AA.expand_as(AB) + BB.expand_as(AB) - 2 * AB

    if base_means != None:
        base_means = base_means.unsqueeze(0)  # shape(1,1,640)
        Abase = ((A-base_means) * (A-base_means)).sum(dim=2, keepdim=True).view(nB, Na, 1) # shape[1, 75, 1]
        Bbase = ((B-base_means) * (B-base_means)).sum(dim=2, keepdim=True).view(nB, 1, Nb)  # shape[1, 1, 5]
        dist = dist - Abase.expand_as(AB) - Bbase.expand_as(AB)
    if average:
        dist = dist / nC

    return dist


class PN_head(nn.Module):
    r"""The metric-based protypical classifier from ``Prototypical Networks for Few-shot Learning''.
    Args:
        metric: Whether use cosine or enclidean distance.
        scale_cls: The initial scale number which affects the following softmax function.
        learn_scale: Whether make scale number learnable.
        normalize: Whether normalize each spatial dimension of image features before average pooling.
    """
    def __init__(
            self,
            metric: str = "cosine",
            scale_cls: int =10.0,
            learn_scale: bool = True,
            normalize: bool = True) -> None:
        super().__init__()
        assert metric in ["cosine", "euclidean"]
        if learn_scale:
            self.scale_cls = nn.Parameter(
                torch.FloatTensor(1).fill_(scale_cls), requires_grad=True
            )
        else:
            self.scale_cls = scale_cls
        self.metric = metric
        self.normalize = normalize

    def forward(self, features_test: Tensor, features_train: Tensor,
                way: int, shot: int, base_means=None, prototypes: bool = None,) -> Tensor:
        r"""Take batches of few-shot training examples and testing examples as input,
            output the logits of each testing examples.
        Args:
            features_test: Testing examples. size: [batch_size, num_query, c, h, w]
            features_train: Training examples which has labels like:[abcdabcdabcd].
                            size: [batch_size, way*shot, c, h, w]
            way: The number of classes of each few-shot classification task.
            shot: The number of training images per class in each few-shot classification
                  task.
        Output:
            classification_scores: The calculated logits of testing examples.
                                   size: [batch_size, num_query, way]
        """
        if features_train.dim() == 5:
            if self.normalize:
                features_train=F.normalize(features_train, p=2, dim=2, eps=1e-12)
            features_train = F.adaptive_avg_pool2d(features_train, 1).squeeze_(-1).squeeze_(-1)
        assert features_train.dim() == 3

        batch_size = features_train.size(0)
        if self.metric == "cosine":
            features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
        #prototypes: [batch_size, way, c]

        if prototypes == None:
            prototypes = torch.mean(features_train.reshape(batch_size, shot, way, -1),dim=1)

        if self.normalize:
            features_test=F.normalize(features_test, p=2, dim=2, eps=1e-12)
            prototypes = F.normalize(prototypes, p=2, dim=2, eps=1e-12)

        if features_test.dim() == 5:
            features_test = F.adaptive_avg_pool2d(features_test, 1).squeeze_(-1).squeeze_(-1)
        assert features_test.dim() == 3

        if self.metric == "cosine":
            features_test = F.normalize(features_test, p=2, dim=2, eps=1e-12)
            #[batch_size, num_query, c] * [batch_size, c, way] -> [batch_size, num_query, way]
            classification_scores = self.scale_cls * torch.bmm(features_test, prototypes.transpose(1, 2))

        elif self.metric == "euclidean":

            classification_scores = -self.scale_cls * L2SquareDist(features_test, prototypes, base_means)
        return classification_scores

def create_model(metric: str = "cosine",
                 scale_cls: int =10.0,
                 learn_scale: bool = True,
                 normalize: bool = True):
    return PN_head(metric, scale_cls, learn_scale, normalize)