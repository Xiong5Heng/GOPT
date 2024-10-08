
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_size, heads):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, query, keys, values, pad_mask=None):
        # A.P.: Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #A.P.: Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # A.P.: (N, value_len, heads, head_dim)
        keys = self.keys(keys)        # A.P.: (N, key_len, heads, head_dim)
        queries = self.queries(query) # A.P.: (N, query_len, heads, heads_dim)

        # A.P.: Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # A.P.: queries shape: (N, query_len, heads, heads_dim),
        # A.P.: keys shape: (N, key_len, heads, heads_dim)
        # A.P.: energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(-1).expand(N, query_len, key_len)
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
            energy = energy.masked_fill(pad_mask==0, -1e18)
            # energy = energy.masked_fill(pad_mask==0, float("-inf"))

        # A.P.: Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # A.P.: attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # A.P.: attention shape: (N, heads, query_len, key_len)
        # A.P.: values shape: (N, value_len, heads, heads_dim)
        # A.P.: out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # A.P.: Linear layer doesn't modify the shape, final shape will be (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, pad_mask=None):
        attention = self.attention(query, key, value, pad_mask)

        # A.P.: Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(EncoderBlock, self).__init__()
        
        self.item_embedding = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.ems_embedding = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.ems_on_item = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.item_on_ems = TransformerBlock(embed_size, heads, dropout, forward_expansion)

    def forward(self, item_feature, ems_feature, mask=None):
        # self-attention
        item_embedding = self.item_embedding(item_feature, item_feature, item_feature)
        ems_embedding = self.ems_embedding(ems_feature, ems_feature, ems_feature, mask)
        # cross-attention
        ems_on_item = self.ems_on_item(ems_embedding, item_embedding, item_embedding, mask) 
        item_on_ems = self.item_on_ems(item_embedding, ems_embedding, ems_embedding)

        return item_on_ems, ems_on_item
    

class ActorHead(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        embed_size: int,
        padding_mask: bool = False,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.padding_mask = padding_mask
        self.device = device
        self.preprocess = preprocess_net
        self.layer_1 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_2 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )

    def forward(
        self, 
        obs: Dict, 
        state: Any = None,
        info: Dict[str, Any] = {}
    ) -> Tuple[torch.Tensor, Any]:
        batch_size = obs.obs.shape[0]

        if self.padding_mask:
            mask = torch.as_tensor(obs.mask, dtype=torch.bool, device=self.device)
            mask = torch.sum(mask.reshape(batch_size, -1, 2), dim=-1).bool()
        else:
            mask = None

        item_embedding, ems_embedding, hidden = self.preprocess(obs.obs, state, mask)
        item_embedding = self.layer_1(item_embedding)
        ems_embedding = self.layer_2(ems_embedding).permute(0, 2, 1)

        logits = torch.bmm(item_embedding, ems_embedding).reshape(batch_size, -1)

        return logits, hidden
    

class CriticHead(nn.Module):
    def __init__(
        self,
        k_placement: int,
        preprocess_net: nn.Module,
        embed_size: int,
        padding_mask: bool = False,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.padding_mask = padding_mask
        self.device = device
        self.preprocess = preprocess_net
        self.k_placement = k_placement
        self.layer_1 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_2 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_3 = nn.Sequential(
            init_(nn.Linear(2 * embed_size, embed_size)),
            nn.LeakyReLU(),
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
            init_(nn.Linear(embed_size, 1))
        )

    def forward(
        self, 
        obs: Union[np.ndarray, torch.Tensor], 
        **kwargs: Any
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        mask = torch.as_tensor(obs.mask, dtype=torch.bool, device=self.device)
        mask = torch.sum(mask.reshape(batch_size, -1, 2), dim=-1).bool()
        if self.padding_mask:
            item_embedding, ems_embedding, _ = self.preprocess(obs.obs, mask)
        else:
            item_embedding, ems_embedding, _ = self.preprocess(obs.obs)

        item_embedding = self.layer_1(item_embedding)
        ems_embedding = self.layer_2(ems_embedding)

        item_embedding = torch.sum(item_embedding, dim=-2)
        ems_embedding = torch.sum(ems_embedding * mask[..., None], dim=-2)

        joint_embedding = torch.cat((item_embedding, ems_embedding), dim=-1)

        state_value = self.layer_3(joint_embedding)
        return state_value


class ShareNet(nn.Module):
    def __init__(
        self,
        k_placement: int = 100,
        box_max_size: int = 5,
        container_size: Sequence[int] = [10, 10, 10],
        embed_size: int = 32,
        num_layers: int = 6,
        forward_expansion: int = 4,
        heads: int = 6,
        dropout: float = 0,
        device: Union[str, int, torch.device] = "cpu",
        place_gen: str = "EMS",
    ) -> None:
        super().__init__()

        self.device = device
        self.k_placement = k_placement
        self.container_size = container_size
        self.place_gen = place_gen
        if place_gen == "EMS":
            input_size = 6
        else:
            input_size = 3

        self.factor = 1 / max(container_size)
        
        self.item_encoder = nn.Sequential(
            init_(nn.Linear(3, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, embed_size)),
        )

        self.placement_encoder = nn.Sequential(
            init_(nn.Linear(input_size, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, embed_size)),
        )
        
        self.backbone = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, 
        obs: Union[np.ndarray, torch.Tensor], 
        state: Any = None,
        mask: Union[np.ndarray, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Any]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device) * self.factor
        if not isinstance(mask, torch.Tensor) and mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)  # (batch_size, k_placement)
        
        obs_hm, obs_next, obs_placements = obs2input(obs, self.container_size, self.place_gen)

        item_embedding = self.item_encoder(obs_next)  # (batch_size, 2, emded_size)
        placement_embedding = self.placement_encoder(obs_placements)  # (batch_size, k_placement, emded_size)

        for layer in self.backbone:
            item_embedding, placement_embedding = layer(item_embedding, placement_embedding, mask)

        return item_embedding, placement_embedding, state


def obs2input(
    obs: torch.Tensor, 
    container_size: Sequence[int],
    place_gen: str = "EMS",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ 
        convert obsversation to input of the network

    Returns:
        hm:         (batch, 1, L, W)
        next_size:  (batch, 2, 3)
        placements: (batch, k_placement, 6)
    """
    batch_size = obs.shape[0]
    hm = obs[:, :container_size[0]*container_size[1]].reshape((batch_size, 1, container_size[0], container_size[1]))
    next_size = obs[:, container_size[0]*container_size[1]:container_size[0]*container_size[1] + 6]
    # [[l, w, h], [w, l, h]]
    next_size = next_size.reshape((batch_size, 2, 3))
    
    if place_gen == "EMS":
        # (x_1, y_1, z_1, x_2, y_2, H)
        placements = obs[:, container_size[0]*container_size[1] + 6:].reshape((batch_size, -1, 6))
    else:
        placements = obs[:, container_size[0]*container_size[1] + 6:].reshape((batch_size, -1, 3))

    return hm, next_size, placements


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('leaky_relu'))

