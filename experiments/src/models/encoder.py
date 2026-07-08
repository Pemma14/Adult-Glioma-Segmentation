from monai.networks.nets import SwinTransformer

swin_encoder = SwinTransformer(
    in_chans=4,           # 4 модальности МРТ
    embed_dim=48,         # feature_size
    window_size=(7, 7, 7),
    patch_size=(2, 2, 2),
    depths=(2, 2, 2, 2),  # по 2 блока в каждой стадии
    num_heads=(3, 6, 12, 24),
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,   # stochastic depth
    spatial_dims=3,
)

def forward_encoder(self, x):
    hidden_states = self.swinViT(x, normalize=True)
    # hidden_states: [x0, x1, x2, x3, x4]
    # x0: (B, 24, D/2, H/2, W/2)  — после PatchEmbed
    # x1: (B, 24, D/2, H/2, W/2)  — после Stage 1
    # x2: (B, 48, D/4, H/4, W/4)  — после Stage 2
    # x3: (B, 96, D/8, H/8, W/8)  — после Stage 3
    # x4: (B, 192, D/16, H/16, W/16) — после Stage 4
    return hidden_states