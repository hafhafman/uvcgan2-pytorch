import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UnalignedDataset(Dataset):
    def __init__(self, dir_A, dir_B):
        super().__init__()
        self.dir_A = dir_A
        self.dir_B = dir_B
        
        # フォルダ内の画像ファイルのパスをすべて取得
        self.paths_A = [os.path.join(dir_A, f) for f in os.listdir(dir_A) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.paths_B = [os.path.join(dir_B, f) for f in os.listdir(dir_B) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # 画像の前処理パイプライン
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), # 画像サイズを 256x256 に強制統一
            transforms.ToTensor(),         # PyTorchのテンソル(0.0〜1.0)に変換
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # ピクセル値を [-1.0, 1.0] に正規化（GANで超重要）
        ])

    def __len__(self):
        # AとBで画像の枚数が違う場合、多い方に合わせてループさせる
        return max(len(self.paths_A), len(self.paths_B))

    def __getitem__(self, index):
        # 1. Aの画像を順番に読み込む
        path_A = self.paths_A[index % len(self.paths_A)]
        img_A = Image.open(path_A).convert('RGB')
        item_A = self.transform(img_A)

        # 2. Bの画像はランダムに選び出して読み込む（非対データ学習の基本）
        path_B = self.paths_B[random.randint(0, len(self.paths_B) - 1)]
        img_B = Image.open(path_B).convert('RGB')
        item_B = self.transform(img_B)

        # AとBの画像をセットにして返す
        return {'A': item_A, 'B': item_B}


# --- 動作確認用のテストコード ---
if __name__ == "__main__":
    print("=== Unaligned Dataset Test ===")
    
    # テスト用に空のダミーフォルダを作成
    os.makedirs("dummy_A", exist_ok=True)
    os.makedirs("dummy_B", exist_ok=True)
    
    # ダミーの真っ黒な画像を作成して保存
    Image.new('RGB', (300, 200)).save("dummy_A/test_a.jpg")
    Image.new('RGB', (150, 400)).save("dummy_B/test_b.jpg")
    
    # データセットを初期化
    dataset = UnalignedDataset("dummy_A", "dummy_B")
    
    # データを1つ取り出してみる
    data = dataset[0]
    
    print(f"ドメインAの画像サイズ (自動リサイズ・正規化後): {data['A'].shape}")
    print(f"ドメインBの画像サイズ (自動リサイズ・正規化後): {data['B'].shape}")
    print(f"ドメインAの最大ピクセル値: {data['A'].max():.2f}, 最小ピクセル値: {data['A'].min():.2f}")