from torch import nn as nn
from torch.nn import functional as F
import torch, time, os, random
import numpy as np
from collections import OrderedDict
from torch import Tensor


class SimplifiedSelfAttention(nn.Module):
    def __init__(self, embed_size, fea_size, heads):
        super(SimplifiedSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = fea_size // heads

        self.values = nn.Linear(self.embed_size, self.head_dim, bias=False)
        self.keys = nn.Linear(self.embed_size, self.head_dim, bias=False)
        self.queries = nn.Linear(self.embed_size, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, query):
        values, keys = query, query
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into 'heads' number of heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        # and each head
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply ReLU square as a simplified attention score
        attention = F.relu(attention) ** 2

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # Combine the attention heads together
        out = self.fc_out(out)

        return out


class SelfAttention_1(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1, name='selfAttn1'):
        super(SelfAttention_1, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.Linear(feaSize, self.dk * multiNum)
        self.WK = nn.Linear(feaSize, self.dk * multiNum)
        self.WV = nn.Linear(feaSize, self.dk * multiNum)
        self.WO = nn.Linear(self.dk * multiNum, feaSize)
        self.dropout = nn.Dropout(dropout)
        self.name = name
        # ===================================================================
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WO.weight)

    def forward(self, qx, kx, vx, maskPAD=None):
        # qx,kx,vx: batchSize × qL,kvL,kvL × feaSize; maskPAD: batchSize × qL × kvL
        Bq, qL, C = qx.shape
        B, kvL, C = kx.shape
        # qx.shape.shape
        # torch.Size([1, 50, 768])
        # kx.shape
        # torch.Size([2, 2432, 768])
        # querie.shape
        # torch.Size([1, 1, 50, 128])
        # keys.shape
        # torch.Size([2, 1, 2432, 128])
        queries = self.WQ(qx).reshape(Bq, qL, self.multiNum, self.dk).transpose(1,
                                                                                2)  # => batchSize × multiNum × qL × dk
        keys = self.WK(kx).reshape(B, kvL, self.multiNum, self.dk).transpose(1, 2)  # => batchSize × multiNum × kvL × dk
        values = self.WV(vx).reshape(B, kvL, self.multiNum, self.dk).transpose(1,
                                                                               2)  # => batchSize × multiNum × kvL × dk

        scores = queries @ keys.transpose(-1, -2) / np.sqrt(self.dk)  # => batchSize × multiNum × qL × kvL
        # #         scores=torch.randn(B,self.multiNum,qL,kvL).to(device)
        #         print("scores.shape",scores.shape)
        #         if maskPAD is not None:
        #             scores = scores.masked_fill((maskPAD==0).unsqueeze(dim=1), -2**15+1) # -np.inf

        alpha = F.softmax(scores, dim=3)

        # torch.Size([8, 1, 8929, n])
        #         print(alpha.shape)
        #         #====================
        #         f=open(r'hello.txt','w')
        #         print(alpha[:2,:,:2,:],file=f)
        #         f.close()

        #         print("alpha.shape",alpha.shape)

        # torch.Size([2, 1, 50, 128])
        z = self.dropout(alpha) @ values  # => batchSize × multiNum × qL × dk
        # torch.Size([2, 50, 128]
        z = z.transpose(1, 2).reshape(B, qL, -1)  # => batchSize × qL × multiNum*dk
        # z1 torch.Size([8, 50, 128])
        # print("z1",z.shape)
        #         z=torch.randn(B,qL,C).to(device)
        z = self.WO(z)  # => batchSize × qL × feaSize
        return z


class TransformerLayer(nn.Module):
    """Transformer layer builder.
    """

    def __init__(self,
                 att: nn.Module,
                 ffn: nn.Module,
                 norm1: nn.Module,
                 norm2: nn.Module,
                 dropout: float):
        super().__init__()
        self.att = att
        self.ffn = ffn
        self.norm1 = norm1
        self.norm2 = norm2
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, qx, kx, vx, maskPAD):
        z = self.norm1(qx + self.dropout(self.att(qx, kx, vx, maskPAD)))
        #         print("x.shape",x.shape)
        # 原始
        ans = z + self.dropout(self.ffn(z))
        # 创造1
        #         ans = self.norm2(z + self.dropout(self.ffn(z)))
        # 创造2
        #         ans = self.ffn(z)
        # 创造3
        #         ans=qx+self.dropout(z)
        return ans


# # 定义一个简单的专家模块
# class Expert(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Expert, self).__init__()
#         self.fc = nn.Linear(input_size, hidden_size)
#
#     def forward(self, x):
#         return self.fc(x)

# 定义MoE层，其中包含多个专家，并使用门控机制分配输入
class MoELayer_simple(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts):
        super(MoELayer_simple, self).__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])
        self.output_layer = nn.Linear(expert_dim, input_dim)
        print(self.experts)

    def forward(self, x):
        batch_size, sequence_length, embed_dim = x.size()

        # 计算门控激活，并沿序列长度维度应用softmax
        gate_activations = self.gate(x)
        gate_weights = F.softmax(gate_activations, dim=-1)
        # gate_weights torch.Size([2, 3681, 4])
        # print("gate_weights", gate_weights.shape)
        # 将输入传递给各个专家，并按批次、时间步和专家维度堆叠输出
        expert_inputs = x.view(batch_size * sequence_length, embed_dim)
        expert_outputs = [expert(expert_inputs) for expert in self.experts]
        # expert_outputs torch.Size([7362, 128])
        # print("expert_outputs", expert_outputs[0].shape)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        # expert_outputs torch.Size([7362, 4, 128])
        # print("expert_outputs", expert_outputs.shape)
        expert_outputs = expert_outputs.view(batch_size, sequence_length, self.num_experts, -1)
        # expert_outputs torch.Size([2, 3681, 4, 128])
        # expert_outputs torch.Size([2, 3681, 4, 1])
        # print("expert_outputs", expert_outputs.shape)
        # print("expert_outputs", gate_weights.unsqueeze(3).shape)
        # 应用门控权重并沿着专家维度求和
        # #expert_outputs torch.Size([2, 3681, 4, 128])
        moe_output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=2)
        # moe_output torch.Size([2, 3681, 128])
        moe_output = self.output_layer(moe_output)
        return moe_output


# 专家简单线性层
class MoELayer(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        # Initialize expert parameters
        self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])
        # Initialize gating network parameters
        self.gating_network = nn.Linear(input_dim, num_experts)
        self.output_layer = nn.Linear(expert_dim, input_dim)

        # ################################
        # self.num_experts = num_experts
        # self.gating_network = nn.Linear(input_dim, num_experts)
        # self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])
        # self.output_layer = nn.Linear(expert_dim , input_dim)
        # print(self.experts)

    def forward(self, x):
        # Compute gate values
        # gate_activations = self.gating_network(x)
        # gate_values = F.softmax(self.gating_network(x), dim=-1)
        gate_values = F.softmax(self.gating_network(x), dim=-1)
        # Compute expert outputs
        # expert_outputs = [F.relu(expert(x)) for expert in self.experts]
        # 这里有relu
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=2)
        # Combine expert outputs using gates
        # 1 torch.Size([2, 4, 3681, 128])
        # 1 torch.Size([2, 3681, 4, 128])
        # torch.Size([2, 3681, 4, 1])
        # 2 torch.Size([2, 3681, 4, 128])
        mixture_output = torch.sum(gate_values.unsqueeze(-1) * expert_outputs, dim=2)
        #         print(mixture_output.shape)
        mixture_output = self.output_layer(mixture_output)
        return mixture_output


# 定义包含自注意力和MoE层的Transformer块
class MoEBlock(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts):
        super(MoEBlock, self).__init__()
        #         self.moe_layer = MoELayer_simple(input_dim, expert_dim, num_experts)
        #         self.moe_layer=MoELayer(input_dim, expert_dim, num_experts)
        #         self.moe_layer=MoE(input_dim, num_experts, expert_dim, 1, dropout_rate=0.1)
        #         self.moe_layer=MoE_Layer_new(input_dim, expert_dim, num_experts)
        #         self.moe_layer=MoE_out(input_dim, num_experts, expert_dim, 1, dropout_rate=0.1)
        self.moe_layer = SwitchFeedForward(capacity_factor=0.5, drop_tokens=True, is_scale_prob=False,
                                           n_experts=num_experts, d_model=input_dim, expert_dim=expert_dim)
        # ... 其他必要的子模块如残差连接、LayerNorm等 ...

    def forward(self, input):
        # 输入Transformer后的数据

        moe_output = self.moe_layer(input)  # 将自注意力层的输出传入MoE层

        return moe_output
        # ... 进行后续处理，如添加残差连接、LayerNorm等 ...


class Transformer_Block(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1):
        super(Transformer_Block, self).__init__()
        selfAttn = SelfAttention_1(feaSize, dk, multiNum, dropout)

        ffn_linear = nn.Sequential(
            nn.Linear(feaSize, dk),
            nn.ReLU(),
            nn.Linear(dk, feaSize)
        )
        layerNorm1 = nn.LayerNorm([feaSize])
        layerNorm2 = nn.LayerNorm([feaSize])

        self.selfAttn = TransformerLayer(att=selfAttn, ffn=ffn_linear, norm1=layerNorm1, norm2=layerNorm2,
                                         dropout=dropout)

    def forward(self, qx, kx, vx, maskPAD=None):
        # qx,kx,vx: batchSize × qL,kvL,kvL × feaSize; maskPAD: batchSize × qL × kvL
        z = self.selfAttn(qx, kx, vx, maskPAD)  # => batchSize × qL × feaSize

        return (z, kx, vx, maskPAD)


class Transformer_Block_moe(nn.Module):
    def __init__(self, feaSize, dk, multiNum, num_experts, dropout=0.1):
        super(Transformer_Block_moe, self).__init__()
        selfAttn = SelfAttention_1(feaSize, dk, multiNum, dropout)
        moe = MoEBlock(input_dim=feaSize, expert_dim=dk, num_experts=num_experts)
        # ffn_linear= nn.Sequential(
        #         nn.Linear(input_dim, dk),
        #         nn.ReLU(),
        #         nn.Linear(dk, input_dim)
        #     )
        layerNorm1 = nn.LayerNorm([feaSize])
        layerNorm2 = nn.LayerNorm([feaSize])

        self.selfAttn = TransformerLayer(att=selfAttn, ffn=moe, norm1=layerNorm1, norm2=layerNorm2, dropout=dropout)

    def forward(self, input):
        qx, kx, vx, maskPAD = input
        # qx,kx,vx: batchSize × qL,kvL,kvL × feaSize; maskPAD: batchSize × qL × kvL
        z = self.selfAttn(qx, kx, vx, maskPAD)  # => batchSize × qL × feaSize

        return (z, kx, vx, maskPAD)


class TransformerLayers_MoE(nn.Module):
    def __init__(self, feaSize, dk, multiNum, num_experts, dropout=0.1, name='textTransformer'):
        super(TransformerLayers_MoE, self).__init__()
        self.transformerLayers = Transformer_Block_moe(feaSize, dk, multiNum, num_experts, dropout)
        self.name = name
        # self.usePos = usePos

    def forward(self, qx, kx, vx, maskPAD=None):
        # qx,kx,vx: batchSize × qL,kvL,kvL × feaSize; maskPAD: batchSize × qL × kvL
        #         qx=qx.repeat(2, 1, 1)
        #         return (qx.repeat(8, 1, 1), kx,vx, maskPAD)
        return self.transformerLayers((qx, kx, vx, maskPAD))  # => batchSize × qL × feaSize


#####################################################################################


class AttentionLayer_moe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        # Initialize attention layer parameters
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Apply attention mechanism
        attention_weights = F.softmax(self.linear(x), dim=-1)
        context_vector = torch.sum(attention_weights * x, dim=1)
        return context_vector


#
# # Example usage:
# # Assuming you have an attention layer output `attention_output` of shape [batch_size, attention_output_dim]
# attention_output_dim = 64
# attention_output = torch.randn(32, attention_output_dim)
#
# # Creating an instance of MoE layer
# num_experts = 4
# expert_dim = 128
# moe_layer = MoELayer(input_dim=attention_output_dim, num_experts=num_experts, expert_dim=expert_dim)
#
# # Forward pass through MoE layer
# moe_output = moe_layer(attention_output)
# print("MoE output shape:", moe_output.shape)  # This should print [batch_size, expert_dim]

# new 专家比较深 门控也是
class MoE_out(nn.Module):
    def __init__(self, input_dim, num_experts, expert_dim, num_expert_layers, dropout_rate=0.1):
        super(MoE_out, self).__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.num_expert_layers = num_expert_layers

        # Expert layers
        self.expert_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(expert_dim, input_dim)  # 添加输出层
            )
            for _ in range(num_experts)
        ])

        # Initialize gating network parameters
        self.gating_network = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.expert_layers], dim=2)

        gate_values = self.gating_network(x)

        # Weighted sum of expert outputs
        mixture_output = torch.sum(gate_values.unsqueeze(-1) * expert_outputs, dim=2)

        #         mixture_output = self.output_layer(mixture_output)

        return mixture_output


# new 专家比较深 门控也是
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, expert_dim, dropout_rate=0.1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        # Expert layers
        self.expert_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(expert_dim, expert_dim)  # 添加输出层
            )
            for _ in range(num_experts)
        ])

        # Initialize gating network parameters
        self.gating_network = nn.Linear(input_dim, num_experts)
        self.output_layer = nn.Linear(expert_dim, input_dim)

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.expert_layers], dim=2)

        gate_values = self.gating_network(x)

        # Weighted sum of expert outputs
        mixture_output = torch.sum(gate_values.unsqueeze(-1) * expert_outputs, dim=2)

        mixture_output = self.output_layer(mixture_output)

        return mixture_output


class MoE_Layer_new(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts):
        super(MoE_Layer_new, self).__init__()
        # 门控网络的输出维度

        # 专家网络的输出维度
        self.expert_dim = expert_dim
        # 专家的数量
        self.num_experts = num_experts

        # 门控网络，用于决定每个token分配给哪个专家
        self.gating_network = nn.Linear(input_dim, num_experts)

        # 专家网络，每个专家一个小型的MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, input_dim)
            ) for _ in range(num_experts)
        ])

        # 用于聚合专家输出的softmax层
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 门控网络的输出
        gating_output = self.gating_network(x)

        # 计算每个专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        # 应用softmax获取每个专家的权重
        expert_weights = self.softmax(gating_output)

        # 聚合专家输出
        aggregated_output = torch.sum(expert_weights.unsqueeze(-1) * expert_outputs, dim=2)

        return aggregated_output


class SwitchFeedForward(nn.Module):
    """
    ## Routing among multiple FFNs
    """

    def __init__(self, *,
                 capacity_factor: float,
                 drop_tokens: bool,
                 is_scale_prob: bool,
                 n_experts: int,
                 d_model: int,
                 expert_dim: int):
        """
        * `capacity_factor` is the capacity of each expert as a factor relative to ideally balanced load
        * `drop_tokens` specifies whether to drop tokens if more tokens are routed to an expert than the capacity
        * `is_scale_prob` specifies whether to multiply the input to the FFN by the routing probability
        * `n_experts` is the number of experts
        * `expert` is the expert layer, a [FFN module](../feed_forward.html)
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability in the FFN
        """
        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        # make copies of the FFNs
        #         self.experts = clone_module_list(expert, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, d_model)
            ) for _ in range(n_experts)
        ])
        # Routing layer and softmax
        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input to the switching module with shape `[seq_len, batch_size, d_model]`
        """

        # Capture the shape to change shapes later
        batch_size, seq_len, d_model = x.shape

        x = x.reshape(batch_size * seq_len, d_model)
        # Flatten the sequence and batch dimensions
        # x = x.view(-1, d_model)

        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.
        route_prob = self.softmax(self.switch(x))

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        #         print(torch.max(route_prob, dim=-1))
        # Get indexes of tokens going to each expert
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]
        #         print(indexes_list)

        # Initialize an empty tensor to store outputs
        final_output = x.new_zeros(x.shape)
        #         print(final_output.shape)
        # Capacity of each expert.
        # $$\mathrm{expert\;capacity} =
        # \frac{\mathrm{tokens\;per\;batch}}{\mathrm{number\;of\;experts}}
        # \times \mathrm{capacity\;factor}$$
        #         print("len(x)",len(x))
        #         print("batch",batch_size)
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.n_experts):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]

        # Get outputs of the expert FFNs
        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        # Assign to final output
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        # Pass through the dropped tokens
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # Don't scale the values but multiply by $\frac{p}{\hat{p}} = 1$ so that the gradients flow
            # (this is something we experimented with).
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        # Change the shape of the final output back to `[seq_len, batch_size, d_model]`
        final_output = final_output.view(batch_size, seq_len, d_model)

        # Return
        #
        # * the final output
        # * number of tokens routed to each expert
        # * sum of probabilities for each expert
        # * number of tokens dropped.
        # * routing probabilities of the selected experts
        #
        # These are used for the load balancing loss and logging
        return final_output


expert_list_all = []


class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
            self,
            dim,
            num_experts: int,
            capacity_factor: float = 1.0,
            epsilon: float = 1e-6,
            topk_expert=1,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.topk_expert = topk_expert
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """
        # Compute gate scores
        #torch.Size([1, 4096, 768])torch.Size([8, 4096, 768])
        print("初始化x:", x.shape)

        gate_scores = F.softmax(self.w_gate(x), dim=-1)
        #gate_scores.shape torch.Size([1, 4096, 10]) torch.Size([8, 4096, 10])
        print("gate_scores.shape", gate_scores.shape)

        # Determine the top-1 expert for each token
        #capacity 1 capacity 8
        capacity = int(self.capacity_factor * x.size(1)/ self.num_experts)
        print("capacity",capacity)

        top_k_scores, top_k_indices = gate_scores.topk(self.topk_expert, dim=-1)
        # print("top_k_indices",top_k_indices)
        print("top_k_indices",top_k_indices.shape)
        expert_list_all.append(top_k_indices.squeeze(0))
        mask = torch.zeros_like(gate_scores).scatter_(
            -1, top_k_indices, 1
        )
        print("mask",mask.shape)

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask
        print("masked_gate_scores",masked_gate_scores.shape)

        # Denominators
        denominators = (
                masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )
        #torch.Size([1, 4096, 10])
        print("denominators",denominators.shape)
        # print("denominators",denominators[:,200:300,:])
        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity
        print("gate_scores", gate_scores.shape)
        #gate_scores.shape torch.Size([1, 4096, 10])
        # print("gate_scores",gate_scores[:,200:300,:])

        return gate_scores


class SwitchMoE(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        mult (int): The multiplier for the hidden dimension of the feedforward network.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            num_experts: int,
            capacity_factor: float = 1.0,
            mult: int = 4,
            topk_expert: int = 1,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult

        # #==============================================
        # self.experts = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim, dim)
        #     ) for _ in range(num_experts-topk_expert)
        # ])
        # for i in range(topk_expert):
        #     print(i)
        #     self.experts.append(nn.Dropout(p=1))
        # #==========================================================
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_experts)
        ])

        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
            1e-6,
            topk_expert,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchMoE module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the MoE.

        """
        # (batch_size, seq_len, num_experts)
        gate_scores = self.gate(x)
        # Dispatch to experts
        expert_outputs = [expert(x) for expert in self.experts]

        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        #stacked_expert_outputs torch.Size([1, 4096, 768, 10])([8, 4096, 768, 10])

        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)

        print("stacked_expert_outputs",stacked_expert_outputs.shape)
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )
        print("moe_output", moe_output.shape)

        return moe_output
# class SwitchGate(nn.Module):
#     """
#     SwitchGate module for MoE (Mixture of Experts) model.

#     Args:
#         dim (int): Input dimension.
#         num_experts (int): Number of experts.
#         capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
#         *args: Variable length argument list.
#         **kwargs: Arbitrary keyword arguments.
#     """

#     def __init__(
#         self,
#         dim,
#         num_experts: int,
#         capacity_factor: float = 1.0,
#         epsilon: float = 1e-6,
#         topk_expert=1,
#         use_aux_loss=False,
#         *args,
#         **kwargs,
#     ):
#         super().__init__()
#         self.topk_expert=topk_expert
#         self.dim = dim
#         self.num_experts = num_experts
#         self.capacity_factor = capacity_factor
#         self.epsilon = epsilon
#         self.w_gate = nn.Linear(dim, num_experts)
#         self.use_aux_loss=use_aux_loss


#     def forward(self, x: Tensor):
#         """
#         Forward pass of the SwitchGate module.

#         Args:
#             x (Tensor): Input tensor.

#         Returns:
#             Tensor: Gate scores.
#         """
#         # Compute gate scores
#         gate_scores = F.softmax(self.w_gate(x), dim=-1)

#         # Determine the top-1 expert for each token
#         capacity = int(self.capacity_factor * x.size(0))

#         top_k_scores, top_k_indices = gate_scores.topk(self.topk_expert, dim=-1)
#         # top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

#         # Mask to enforce sparsity
#         mask = torch.zeros_like(gate_scores).scatter_(
#             1, top_k_indices, 1
#         )


#         # top_k_scores, top_k_indices = gate_scores.topk(self.topk_expert, dim=-1)
#         # # top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

#         # # Mask to enforce sparsity
#         # mask = torch.zeros_like(gate_scores).scatter_(
#         #     1, top_k_indices, 1
#         # )

#         # Combine gating scores with the mask
#         masked_gate_scores = gate_scores * mask

#         # Denominators
#         denominators = (
#             masked_gate_scores.sum(0, keepdim=True) + self.epsilon
#         )

#         # Norm gate scores to sum to the capacity
#         gate_scores = (masked_gate_scores / denominators) * capacity

#         if self.use_aux_loss:
#             load = gate_scores.sum(0)  # Sum over all examples
#             importance = gate_scores.sum(1)  # Sum over all experts

#             # Aux loss is mean suqared difference between load and importance
#             loss = ((load - importance) ** 2).mean()

#             return gate_scores, loss

#         return gate_scores, None


# class SwitchMoE(nn.Module):

#     def __init__(
#         self,
#         dim: int,
#         hidden_dim: int,
#         num_experts: int,
#         capacity_factor: float = 1.0,
#         mult: int = 4,
#         topk_expert: int =1,
#         use_aux_loss: bool = False,
#         *args,
#         **kwargs,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.hidden_dim = hidden_dim
#         self.num_experts = num_experts
#         self.capacity_factor = capacity_factor
#         self.mult = mult
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, dim)
#             ) for _ in range(num_experts)
#         ])


#         self.gate = SwitchGate(
#             dim,
#             num_experts,
#             capacity_factor,
#             epsilon=1e-6,
#             topk_expert=topk_expert,
#             use_aux_loss=use_aux_loss
#         )

#     def forward(self, x: Tensor):
#         """
#         Forward pass of the SwitchMoE module.

#         Args:
#             x (Tensor): The input tensor.

#         Returns:
#             Tensor: The output tensor of the MoE.

#         """
#         # (batch_size, seq_len, num_experts)
#         gate_scores, loss= self.gate(x)
#         # Dispatch to experts
#         expert_outputs = [expert(x) for expert in self.experts]

#         # Check if any gate scores are nan and handle
#         if torch.isnan(gate_scores).any():
#             print("NaN in gate scores")
#             gate_scores[torch.isnan(gate_scores)] = 0

#         # Stack and weight outputs
#         stacked_expert_outputs = torch.stack(
#             expert_outputs, dim=-1
#         )  # (batch_size, seq_len, output_dim, num_experts)
#         if torch.isnan(stacked_expert_outputs).any():
#             stacked_expert_outputs[
#                 torch.isnan(stacked_expert_outputs)
#             ] = 0

#         # Combine expert outputs and gating scores
#         moe_output = torch.sum(
#             gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
#         )

#         return moe_output,loss