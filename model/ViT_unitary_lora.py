import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair
from scipy import ndimage
import torch.nn.functional as F
from model import configs

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class UnitaryLinear(nn.Module):
    def __init__(self, size_in, size_out, bias=True, enable_unitary=False):
        super(UnitaryLinear, self).__init__()
        self.enable_unitary = enable_unitary
        self.size_in = size_in
        self.size_out = size_out
        self.mlp = nn.Linear(size_in, size_out, bias=bias)

        self._frozen_param()

        if self.enable_unitary:
            self.q_left = nn.Parameter(torch.zeros(size_in - 1, 1, dtype=torch.float32))
            self.q_right = nn.Parameter(torch.zeros(size_out - 1, 1, dtype=torch.float32))
            
            self.q_0_left = nn.Parameter(torch.tensor(0., dtype=torch.float32))
            self.q_0_right = nn.Parameter(torch.tensor(0., dtype=torch.float32))

            self.shape = 8
            # self.q_lambda = nn.Parameter(torch.randn(self.shape, self.shape))
            self.q_lambda = nn.Parameter(torch.randn(1, self.shape, dtype=torch.float32))
            # self.shift_bias = nn.Parameter(torch.zeros(1, self.size_out), requires_grad=True)
            # nn.init.zeros_(self.shift_bias)


    def _frozen_param(self):
        for param in self.mlp.parameters():
            param.requires_grad = False


    def __unitary(self, q_0, q, shape):
        # eps = 1e-8
        # q = q / torch.norm(q + eps)  #
        Q = q @ q[:shape - 1].T
        Q = Q * (-1 / ((1 + q_0) ** 2 + 1.0))

        lamb = torch.eye(Q.shape[1], dtype=torch.float32, device=q.device)
        lamb = torch.cat([lamb, torch.zeros(q.shape[0]+1-shape, shape - 1, device=q.device)], dim=0)

        Q = Q + lamb
        Q = torch.cat([-q[:shape-1].T, Q], dim=0)
        q_m = torch.cat([torch.tensor([[q_0]], device=q.device), q], dim=0)
        Q = torch.cat([q_m, Q], dim=1)

        return Q


    def forward(self, x):

        if self.enable_unitary:
            left_unitary = self.__unitary(self.q_0_left, self.q_left, self.shape)
            right_unitary = self.__unitary(self.q_0_right, self.q_right, self.shape)

            weight = self.mlp.weight + (left_unitary * self.q_lambda @ right_unitary.T ).T 
            # bias = self.shift_bias
            return F.linear(x, weight)
        else:
            return self.mlp(x)


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = UnitaryLinear(config.hidden_size, self.all_head_size, bias=True, enable_unitary=True)    #
        self.key = UnitaryLinear(config.hidden_size, self.all_head_size, bias=True, enable_unitary=False)
        self.value = UnitaryLinear(config.hidden_size, self.all_head_size, bias=True, enable_unitary=True)    #
        self.out = UnitaryLinear(config.hidden_size, config.hidden_size, bias=True, enable_unitary=False)
        
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output  # , weights


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        self.fc1 = UnitaryLinear(config.hidden_size, config.transformer["mlp_dim"], bias=True, enable_unitary=True)
        self.fc2 = UnitaryLinear(config.transformer["mlp_dim"], config.hidden_size, bias=True, enable_unitary=True)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.act(self.fc1(x))
        if self.training:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.training:
            x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, vis, drop_path=0.0, enable_lora=False):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = MLP(config)
        self.attn = Attention(config, vis)
        self.enable_lora = enable_lora
        self.drop_path1 = nn.Dropout(p=drop_path)
        self.drop_path2 = nn.Dropout(p=drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.attention_norm(x)))
        x = x + self.drop_path2(self.ffn(self.ffn_norm(x)))
        return x

    def _fc_load_weight(self, ROOT, Key, Weights, unit):
        mat_weights = Weights[ROOT + '/' + Key + '/' + "kernel"]
        mat_bias = Weights[ROOT + '/' + Key + '/' + "bias"]
        if Key == ATTENTION_OUT:
            mat_weights = mat_weights.reshape(-1, mat_weights.shape[-1])
        else:
            mat_weights = mat_weights.reshape(mat_weights.shape[0], -1)

        unit.mlp.weight.copy_(np2th(mat_weights).t())
        unit.mlp.bias.copy_(np2th(mat_bias).view(-1))

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            self._fc_load_weight(ROOT, ATTENTION_Q, weights, self.attn.query)
            self._fc_load_weight(ROOT, ATTENTION_K, weights, self.attn.key)
            self._fc_load_weight(ROOT, ATTENTION_V, weights, self.attn.value)
            self._fc_load_weight(ROOT, ATTENTION_OUT, weights, self.attn.out)
            self._fc_load_weight(ROOT, FC_0, weights, self.ffn.fc1)
            self._fc_load_weight(ROOT, FC_1, weights, self.ffn.fc2)

            self.attention_norm.weight.copy_(np2th(weights[ROOT + '/' + ATTENTION_NORM + '/' + "scale"]))
            self.attention_norm.bias.copy_(np2th(weights[ROOT + '/' + ATTENTION_NORM + '/' + "bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[ROOT + '/' + MLP_NORM + '/' + "scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[ROOT + '/' + MLP_NORM + '/' + "bias"]))


class Encoder(nn.Module):
    def __init__(self, config, vis, drop_path=0.0, enable_lora=True):
        super(Encoder, self).__init__()
        self.vis = vis
        self.enable_lora = enable_lora
        self.layer = nn.ModuleList()
        # self.lora_layer = nn.ModuleList()
        for n in range(config.transformer["num_layers"]):
            setattr(self, f"lora_layer_{n}", nn.ModuleList())
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.num_blocks = config.transformer["num_layers"]
        self.depth = config.transformer["num_layers"]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.num_blocks)]
        # fellow SSF
        for i in range(config.transformer["num_layers"]):
            self.layer.append(Block(config, vis, drop_path=dpr[i], enable_lora=enable_lora))
        # self.att_down_adapter = nn.Parameter(torch.empty(config.hidden_size, config.adapter_dim))
        # self.mlp_down_adapter = nn.Parameter(torch.empty(config.hidden_size, config.adapter_dim))
        # nn.init.xavier_uniform_(self.mlp_down_adapter)
        # nn.init.xavier_uniform_(self.att_down_adapter)

    def forward(self, hidden_states):

        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded


class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.hybrid = False

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings

        return embeddings


class ArcHouseTransformer(nn.Module):
    def __init__(self, config, img_size, vis, drop_path=0.0, enable_lora=True):
        super(ArcHouseTransformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, drop_path=drop_path, enable_lora=enable_lora)
        self._frozen_param()

    def _frozen_param(self):
        for param in self.embeddings.parameters():
            param.requires_grad = False

            
    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)
        return encoded


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, enable_lora=True,
                 drop_path=0.0):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = ArcHouseTransformer(config, img_size, vis, drop_path=drop_path, enable_lora=enable_lora)
        self.head = nn.Linear(config.hidden_size, num_classes)
        self.loss_fct = nn.CrossEntropyLoss()

    def get_parameters(self, lr, weight_decay):
        wd_params = []
        no_wd_params = []
        for name, param in self.named_parameters():
            if 'bias' in name or 'norm' in name:
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        params = [
            {"params": wd_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_wd_params, "lr": lr, "weight_decay": 0.}
        ]

        return params

    def forward(self, x, labels=None):
        x = self.transformer(x)

        logits = self.head(x[:, 0])
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
