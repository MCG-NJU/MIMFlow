import torch

class Permutation(torch.nn.Module):

    def __init__(self, seq_length: int = 0):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])

class PermutationShuffle(Permutation):
    def __init__(self, seq_length: int):
        super().__init__(seq_length)
        # 创建随机排列索引
        self.shuffle_indices = torch.randperm(seq_length)

        # 创建逆排列索引（用于恢复原始顺序）
        self.inverse_indices = torch.empty_like(self.shuffle_indices)
        self.inverse_indices[self.shuffle_indices] = torch.arange(seq_length)

        # 将索引注册为buffer，这样它们会随模型一起保存和加载，但不会被优化
        self.register_buffer('shuffle_indices_buffer', self.shuffle_indices)
        self.register_buffer('inverse_indices_buffer', self.inverse_indices)

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        """
        对输入张量进行排列变换

        Args:
            x: 输入张量
            dim: 要进行排列的维度
            inverse: 是否执行逆变换（恢复原始顺序）

        Returns:
            排列后的张量
        """
        if inverse:
            # 使用逆排列索引恢复原始顺序
            indices = self.inverse_indices_buffer
        else:
            # 使用随机排列索引打乱顺序
            indices = self.shuffle_indices_buffer

        # 确保索引在正确的设备上
        indices = indices.to(x.device)

        # 使用torch.index_select在指定维度上进行索引选择
        return torch.index_select(x, dim, indices)

    def get_permutation_matrix(self) -> torch.Tensor:
        """
        返回排列矩阵（可选方法，用于调试或可视化）
        """
        P = torch.zeros(self.seq_length, self.seq_length)
        P[torch.arange(self.seq_length), self.shuffle_indices_buffer] = 1
        return P

    def get_inverse_permutation_matrix(self) -> torch.Tensor:
        """
        返回逆排列矩阵（可选方法，用于调试或可视化）
        """
        P_inv = torch.zeros(self.seq_length, self.seq_length)
        P_inv[torch.arange(self.seq_length), self.inverse_indices_buffer] = 1
        return P_inv
