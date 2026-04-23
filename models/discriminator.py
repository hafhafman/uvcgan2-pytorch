import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=3, base_features=64):
        super().__init__()
        
        # PatchGANの構造 (画像を徐々に縮小しながら特徴を判定)
        self.model = nn.Sequential(
            # 第1層 (ここだけInstanceNormを使わないのがセオリー)
            nn.Conv2d(input_channels, base_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第2層
            nn.Conv2d(base_features, base_features * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第3層
            nn.Conv2d(base_features * 2, base_features * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第4層 (ここからストライドが1になり、解像度が下がらなくなる)
            nn.Conv2d(base_features * 4, base_features * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(base_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 最終層 (1チャンネルの判定マップを出力)
            nn.Conv2d(base_features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# --- 動作確認用のテストコード ---
if __name__ == "__main__":
    print("=== PatchGAN Discriminator Test ===")
    
    # ジェネレータが生成した(または本物の) 256x256 の画像と仮定
    dummy_img = torch.randn(1, 3, 256, 256)
    
    # ディスクリミネータを準備
    model = PatchGANDiscriminator()
    
    # 鑑定！
    output = model(dummy_img)
    
    print(f"入力画像のサイズ: {dummy_img.shape}")
    print(f"出力テンソルのサイズ: {output.shape}")