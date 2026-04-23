import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.generator import UVCGANv2Generator
from models.discriminator import PatchGANDiscriminator
from datasets.dataset import UnalignedDataset

def main():
    print("=== UVCGANv2 Training Setup ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # モデルの準備
    G_AB = UVCGANv2Generator().to(device)
    G_BA = UVCGANv2Generator().to(device)
    D_A = PatchGANDiscriminator().to(device)
    D_B = PatchGANDiscriminator().to(device)

    # 損失関数の準備
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # オプティマイザの準備
    lr = 0.0002
    optimizer_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # データローダーの準備
    dataset = UnalignedDataset("datasets/dummy_A", "datasets/dummy_B")
    import os
    print("--- AIの視点チェック ---")
    print(f"探しているAの場所: {os.path.abspath('datasets/dummy_A')}")
    print(f"認識したAの画像数: {len(dataset.paths_A)} 枚")
    print(f"探しているBの場所: {os.path.abspath('datasets/dummy_B')}")
    print(f"認識したBの画像数: {len(dataset.paths_B)} 枚")
    print("------------------------")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("準備完了！ 学習テストを開始します...\n")

    # ==========================================
    # ここからが AI の学習ループです
    # ==========================================
    num_epochs = 3 # テストとして、データを3周だけ回します

    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            # 1. データをデバイス(CPU/GPU)に送る
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # ----------------------------------
            # ① ジェネレータの学習 (職人が腕を磨く)
            # ----------------------------------
            optimizer_G.zero_grad()

            # Identity Loss (色合いを保つ)
            loss_id_A = criterion_identity(G_BA(real_A), real_A) * 5.0
            loss_id_B = criterion_identity(G_AB(real_B), real_B) * 5.0

            # GAN Loss (鑑定士を騙す)
            fake_B = G_AB(real_A)
            pred_fake_B = D_B(fake_B)
            loss_GAN_AB = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

            fake_A = G_BA(real_B)
            pred_fake_A = D_A(fake_A)
            loss_GAN_BA = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

            # Cycle Loss (形状を保つ: A -> B -> A)
            loss_cycle_A = criterion_cycle(G_BA(fake_B), real_A) * 10.0
            loss_cycle_B = criterion_cycle(G_AB(fake_A), real_B) * 10.0

            # 誤差を合計して、ジェネレータの重みを更新
            loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
            loss_G.backward()
            optimizer_G.step()

            # ----------------------------------
            # ② ディスクリミネータの学習 (鑑定士が目を鍛える)
            # ----------------------------------
            # D_A の学習 (Aの偽物と本物を見分ける)
            optimizer_D_A.zero_grad()
            loss_D_real_A = criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A)))
            loss_D_fake_A = criterion_GAN(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A.detach())))
            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # D_B の学習 (Bの偽物と本物を見分ける)
            optimizer_D_B.zero_grad()
            loss_D_real_B = criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B)))
            loss_D_fake_B = criterion_GAN(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B.detach())))
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            # ----------------------------------
            # 結果の表示
            # ----------------------------------
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(dataloader)}] "
                  f"[D loss: {loss_D_A.item() + loss_D_B.item():.4f}] "
                  f"[G loss: {loss_G.item():.4f}]")

if __name__ == "__main__":
    main()