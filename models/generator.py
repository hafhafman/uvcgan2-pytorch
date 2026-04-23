import torch
import torch.nn as nn

# --- 前回作った心臓部 (ViTボトルネック) ---
class ViTBottleneck(nn.Module):
    def __init__(self, dim, depth, heads):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flattened = x.view(b, c, -1).transpose(1, 2)
        transformed_tokens = self.transformer(x_flattened)
        out = transformed_tokens.transpose(1, 2).view(b, c, h, w)
        return out

# --- 新しく追加するジェネレータ全体 ---
class UVCGANv2Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_features=64):
        super().__init__()
        
        # 1. エンコーダ (画像を縮小しながら特徴を抽出)
        # 画像サイズ: 256x256 -> 128x128 -> 64x64
        self.encoder = nn.Sequential(
            # 最初の層
            nn.Conv2d(input_channels, base_features, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(base_features),
            nn.ReLU(inplace=True),
            # 縮小層 1
            nn.Conv2d(base_features, base_features * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
            # 縮小層 2
            nn.Conv2d(base_features * 2, base_features * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_features * 4),
            nn.ReLU(inplace=True),
        )
        
        # 2. ボトルネック (心臓部のViT)
        # 入力される特徴量のチャンネル数は base_features * 4 = 256
        self.bottleneck = ViTBottleneck(dim=base_features * 4, depth=4, heads=8)
        
        # 3. デコーダ (特徴から新しい画像を復元)
        # 画像サイズ: 64x64 -> 128x128 -> 256x256
        self.decoder = nn.Sequential(
            # 拡大層 1
            nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
            # 拡大層 2
            nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_features),
            nn.ReLU(inplace=True),
            # 最後の層 (RGBの3チャンネルに戻す)
            nn.Conv2d(base_features, output_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.Tanh() # 出力されるピクセルの値を [-1, 1] の範囲にきれいに収める
        )

    def forward(self, x):
        # 画像が エンコーダ -> ボトルネック -> デコーダ の順に流れる
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


# --- 動作確認用のテストコード ---
if __name__ == "__main__":
    print("=== UVCGANv2 Generator Test ===")
    
    # 実際の画像を想定したダミーデータ
    # (バッチサイズ1, RGB3チャンネル, 縦256ピクセル, 横256ピクセル)
    dummy_img = torch.randn(1, 3, 256, 256)
    
    # ジェネレータを準備
    model = UVCGANv2Generator()
    
    # 画像をモデルに通して変換！
    output_img = model(dummy_img)
    
    print(f"入力画像のサイズ: {dummy_img.shape}")
    print(f"出力画像のサイズ: {output_img.shape}")