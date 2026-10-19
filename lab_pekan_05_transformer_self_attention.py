
"""
========================================================================================
LABORATORIUM KECERDASAN BUATAN (MK55) — PEKAN KE-5
UNIVERSITAS ISLAM NEGERI ALAUDDIN MAKASSAR
FAKULTAS SAINS DAN TEKNOLOGI — PROGRAM STUDI SISTEM INFORMASI
========================================================================================
Topik         : Arsitektur Transformer & Scaled Dot-Product Self-Attention
SubCPMK       : SubCPMK 04 (CPMK-2 / CPL02)
Tugas         : Analisis Diagram Transformer & Matriks Q, K, V

Implementasi menggunakan NumPy dan Matplotlib:
1. Sinusoidal Positional Encoding
2. Scaled Dot-Product Attention
3. Causal Masking
4. Multi-Head Attention
5. Simulasi hubungan kata ganti "Dia" dengan "Asep"
6. Visualisasi Attention Heatmap
========================================================================================
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

np.random.seed(42)


# =============================================================================
# BAGIAN 1 — SOFTMAX NUMERIK STABIL
# =============================================================================

def softmax(x, axis=-1):
    """Menghitung Softmax dengan stabilitas numerik."""
    x = np.asarray(x, dtype=float)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# =============================================================================
# BAGIAN 2 — SINUSOIDAL POSITIONAL ENCODING
# =============================================================================

def get_positional_encoding(seq_len, d_model):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    if seq_len <= 0 or d_model <= 0:
        raise ValueError("seq_len dan d_model harus positif.")

    position = np.arange(seq_len)[:, np.newaxis]

    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(
        position * div_term[:pe[:, 1::2].shape[1]]
    )

    return pe


# =============================================================================
# BAGIAN 3 — SCALED DOT-PRODUCT ATTENTION
# =============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Q: (..., seq_len_q, d_k)
    K: (..., seq_len_k, d_k)
    V: (..., seq_len_k, d_v)

    mask=True berarti posisi tersebut dilarang diperhatikan.
    """
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)

    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Dimensi fitur Q dan K harus sama.")

    if K.shape[-2] != V.shape[-2]:
        raise ValueError("Jumlah token K dan V harus sama.")

    d_k = Q.shape[-1]

    # 1. Perkalian Q dengan transpose K
    scores = np.matmul(Q, np.swapaxes(K, -1, -2))

    # 2. Scaling
    scaled_scores = scores / np.sqrt(d_k)

    # 3. Masking opsional
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        scaled_scores = np.where(mask, -1e9, scaled_scores)

    # 4. Softmax
    attention_weights = softmax(scaled_scores, axis=-1)

    # 5. Perkalian bobot attention dengan V
    output = np.matmul(attention_weights, V)

    return output, attention_weights


# =============================================================================
# BAGIAN 4 — CAUSAL / LOOK-AHEAD MASKING
# =============================================================================

def create_causal_mask(seq_len):
    """Membuat mask segitiga atas untuk menyembunyikan token masa depan."""
    if seq_len <= 0:
        raise ValueError("seq_len harus positif.")

    return np.triu(
        np.ones((seq_len, seq_len), dtype=bool),
        k=1
    )


def demo_causal_mask():
    print("\n" + "=" * 80)
    print("DEMONSTRASI CAUSAL MASKING PADA DECODER")
    print("=" * 80)

    tokens = ["Saya", "sedang", "belajar", "Transformer", "AI"]
    n = len(tokens)

    mask = create_causal_mask(n)

    print("\nMatriks Causal Mask (1 = posisi terlarang):")
    print(mask.astype(int))

    dummy_scores = np.random.randn(n, n)
    masked_scores = np.where(mask, -1e9, dummy_scores)
    weights = softmax(masked_scores, axis=-1)

    print("\nBobot Attention setelah Masking:")
    for i, token in enumerate(tokens):
        row = " ".join(f"{v:7.4f}" for v in weights[i])
        print(f"Query '{token:<12}': [{row} ]")

    assert np.allclose(weights[mask], 0.0)
    print("\n[OK] Token masa depan tidak dapat diperhatikan.")


# =============================================================================
# BAGIAN 5 — MULTI-HEAD ATTENTION
# =============================================================================

class SimpleMultiHeadAttention:
    """Implementasi edukatif Multi-Head Attention."""

    def __init__(self, d_model=64, num_heads=4):
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("Dimensi harus positif.")

        if d_model % num_heads != 0:
            raise ValueError("d_model harus habis dibagi num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Bobot acak; belum dilatih dengan dataset.
        self.W_q = np.random.randn(d_model, d_model) * 0.05
        self.W_k = np.random.randn(d_model, d_model) * 0.05
        self.W_v = np.random.randn(d_model, d_model) * 0.05
        self.W_o = np.random.randn(d_model, d_model) * 0.05

    def split_heads(self, x):
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return np.transpose(x, (1, 0, 2))

    def forward(self, Q_in, K_in, V_in, mask=None):
        Q_in = np.asarray(Q_in, dtype=float)
        K_in = np.asarray(K_in, dtype=float)
        V_in = np.asarray(V_in, dtype=float)

        if any(x.ndim != 2 for x in (Q_in, K_in, V_in)):
            raise ValueError("Input harus berupa matriks dua dimensi.")

        if any(x.shape[1] != self.d_model
               for x in (Q_in, K_in, V_in)):
            raise ValueError("Dimensi fitur input tidak sesuai d_model.")

        if K_in.shape[0] != V_in.shape[0]:
            raise ValueError("Jumlah token K dan V harus sama.")

        # 1. Proyeksi Q, K, V
        Q = Q_in @ self.W_q
        K = K_in @ self.W_k
        V = V_in @ self.W_v

        # 2. Pembagian head
        Q_heads = self.split_heads(Q)
        K_heads = self.split_heads(K)
        V_heads = self.split_heads(V)

        # 3. Attention per head
        output_heads = []
        weights_heads = []

        for h in range(self.num_heads):
            output_h, weights_h = scaled_dot_product_attention(
                Q_heads[h], K_heads[h], V_heads[h], mask
            )
            output_heads.append(output_h)
            weights_heads.append(weights_h)

        # 4. Concatenate
        concatenated = np.concatenate(output_heads, axis=-1)

        # 5. Proyeksi output
        output = concatenated @ self.W_o

        return output, np.stack(weights_heads, axis=0)


# =============================================================================
# BAGIAN 6 — STUDI KASUS KALIMAT ASEP
# =============================================================================

def run_coreference_case_study():
    """
    Kalimat asli:
    "Asep Menyebrangi Zebra Crosswalk Karena Mau Makan Tapi Dia Mati Ketabrak"

    Simulasi memperkuat korelasi antara Query "Dia" dan Key "Asep".
    Ini merupakan ilustrasi matematis, bukan hasil model terlatih.
    """
    print("\n" + "=" * 80)
    print("STUDI KASUS: SIMULASI HUBUNGAN KATA GANTI 'DIA' DAN 'ASEP'")
    print("=" * 80)

    # Nama dan urutan token dipertahankan sesuai kalimat asli.
    tokens = [
        "Asep",
        "Menyebrangi",
        "Zebra",
        "Crosswalk",
        "Karena",
        "Mau",
        "Makan",
        "Tapi",
        "Dia",
        "Mati",
        "Ketabrak"
    ]

    seq_len = len(tokens)
    d_model = 16
    d_k = 16

    # Indeks token yang benar
    idx_asep = tokens.index("Asep")
    idx_menyebrangi = tokens.index("Menyebrangi")
    idx_zebra = tokens.index("Zebra")
    idx_crosswalk = tokens.index("Crosswalk")
    idx_dia = tokens.index("Dia")
    idx_mati = tokens.index("Mati")

    print("Kalimat:", " ".join(tokens))
    print("Jumlah token:", seq_len)
    print("Dimensi fitur:", d_model)

    # Representasi sintetis: bukan embedding model terlatih.
    embeddings = np.random.randn(seq_len, d_model) * 0.1

    # Tambahkan informasi posisi.
    embeddings += get_positional_encoding(
        seq_len, d_model
    ) * 0.1

    # Matriks proyeksi
    W_q = np.random.randn(d_model, d_k) * 0.2
    W_k = np.random.randn(d_model, d_k) * 0.2
    W_v = np.random.randn(d_model, d_model) * 0.2

    Q = embeddings @ W_q
    K = embeddings @ W_k
    V = embeddings @ W_v

    # -------------------------------------------------------------------------
    # SIMULASI SINYAL KORELASI
    # -------------------------------------------------------------------------
    # Query "Dia" dibuat lebih selaras dengan Key "Asep".
    # Hubungan ini disisipkan secara sengaja agar mudah diamati.
    rng = np.random.default_rng(123)

    K[idx_asep] = (
        Q[idx_dia] * 1.8
        + rng.normal(0, 0.01, size=d_k)
    )

    # Key lain diberi korelasi lebih kecil sebagai pembanding.
    K[idx_zebra] = (
        Q[idx_dia] * 0.3
        + rng.normal(0, 0.01, size=d_k)
    )

    K[idx_crosswalk] = (
        Q[idx_dia] * 0.2
        + rng.normal(0, 0.01, size=d_k)
    )

    # -------------------------------------------------------------------------
    # HITUNG ATTENTION
    # -------------------------------------------------------------------------
    output, attention_matrix = scaled_dot_product_attention(Q, K, V)

    assert output.shape == (seq_len, d_model)
    assert attention_matrix.shape == (seq_len, seq_len)
    assert np.allclose(attention_matrix.sum(axis=-1), 1.0)

    # Ambil baris attention untuk token "Dia".
    dia_weights = attention_matrix[idx_dia]

    print("\nDISTRIBUSI BOBOT ATTENTION UNTUK QUERY 'DIA'")
    print("-" * 70)

    sorted_indices = np.argsort(dia_weights)[::-1]

    for rank, idx in enumerate(sorted_indices, start=1):
        bar = "#" * int(dia_weights[idx] * 40)

        print(
            f"{rank:2d}. Token '{tokens[idx]:<13}' : "
            f"{dia_weights[idx]:.4f} "
            f"({dia_weights[idx] * 100:5.1f}%) | {bar}"
        )

    print("\nINTERPRETASI:")
    print(
        f"- Bobot attention Dia -> Asep: "
        f"{dia_weights[idx_asep] * 100:.2f}%"
    )
    print(
        f"- Bobot attention Dia -> Zebra: "
        f"{dia_weights[idx_zebra] * 100:.2f}%"
    )
    print(
        f"- Bobot attention Dia -> Crosswalk: "
        f"{dia_weights[idx_crosswalk] * 100:.2f}%"
    )

    print(
        "\nPada simulasi ini, korelasi Query 'Dia' dengan Key 'Asep' "
        "sengaja diperkuat."
    )
    print(
        "Hasilnya memperagakan bagaimana attention dapat memberikan "
        "bobot berbeda kepada token-token dalam sebuah kalimat."
    )
    print(
        "Catatan: hasil ini bukan bukti pemahaman bahasa oleh model "
        "karena representasi dan bobotnya belum dilatih."
    )

    return tokens, attention_matrix


# =============================================================================
# BAGIAN 7 — VISUALISASI ATTENTION HEATMAP
# =============================================================================

def plot_attention_heatmap(
    tokens,
    attention_matrix,
    output_path="attention_heatmap_pekan_05.png"
):
    """Membuat dan menyimpan heatmap matriks attention."""

    attention_matrix = np.asarray(attention_matrix)

    if attention_matrix.shape != (len(tokens), len(tokens)):
        raise ValueError("Dimensi matriks tidak sesuai jumlah token.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 9), dpi=200)

    image = ax.imshow(
        attention_matrix,
        cmap="viridis",
        aspect="auto",
        vmin=0
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Bobot Attention"
    )

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))

    ax.set_xticklabels(
        tokens,
        rotation=45,
        ha="right",
        fontsize=9
    )
    ax.set_yticklabels(tokens, fontsize=9)

    ax.set_xlabel("Key Tokens (K)", fontsize=11)
    ax.set_ylabel("Query Tokens (Q)", fontsize=11)

    ax.set_title(
        "Self-Attention Heatmap — Praktikum Pekan 5\n"
        "Kalimat: Asep Menyebrangi Zebra Crosswalk Karena Mau "
        "Makan Tapi Dia Mati Ketabrak",
        fontsize=12,
        pad=20
    )

    # Angka probabilitas pada setiap sel.
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            value = attention_matrix[i, j]
            color = "white" if value < 0.45 else "black"

            ax.text(
                j, i, f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7
            )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Heatmap disimpan di: {output_path.resolve()}")


# =============================================================================
# BAGIAN 8 — PANDUAN TUGAS ANALISIS 1
# =============================================================================

def print_assignment_brief():
    print("\n" + "=" * 80)
    print("PANDUAN TUGAS ANALISIS 1 — BOBOT 5%")
    print("=" * 80)

    print("""
1. DEKONSTRUKSI MATEMATIS (35%)
   Hitung QK^T untuk tiga kata dengan d_k = 2, kemudian lakukan
   scaling dengan sqrt(d_k) dan normalisasi Softmax.

2. ANALISIS ENCODER VS DECODER (35%)
   Jelaskan mengapa BERT menggunakan arsitektur encoder,
   sedangkan GPT menggunakan arsitektur decoder kausal.
   Sertakan kelebihan dan keterbatasan masing-masing.

3. EKSPERIMEN KODE & HEATMAP (30%)
   Jalankan eksperimen, amati hubungan attention antartoken,
   dan jelaskan interpretasi heatmap secara kritis.

PENGUMPULAN
   Format: PDF maksimal 3 halaman + tautan repositori GitHub.
   Portal: LMS UIN Alauddin Makassar.
   Tenggat: Periksa jadwal resmi pada LMS.
""")

    print("=" * 80)


# =============================================================================
# BAGIAN 9 — EKSEKUSI UTAMA
# =============================================================================

def main():
    print("=" * 80)
    print("PRAKTIKUM PEKAN 5: TRANSFORMER & SELF-ATTENTION")
    print("=" * 80)

    # 1. Pengujian Positional Encoding
    pe = get_positional_encoding(seq_len=6, d_model=8)

    print("\n1. POSITIONAL ENCODING")
    print("Bentuk matriks:", pe.shape)
    print(np.round(pe, 3))

    # Pengujian dimensi ganjil
    pe_odd = get_positional_encoding(seq_len=4, d_model=7)
    assert pe_odd.shape == (4, 7)
    print("[OK] Dimensi Positional Encoding ganjil berhasil.")

    # 2. Pengujian Multi-Head Attention
    mha = SimpleMultiHeadAttention(d_model=16, num_heads=2)
    dummy_x = np.random.randn(4, 16)

    out_mha, weights_mha = mha.forward(
        dummy_x, dummy_x, dummy_x
    )

    print("\n2. MULTI-HEAD ATTENTION")
    print("Bentuk input :", dummy_x.shape)
    print("Bentuk output:", out_mha.shape)
    print("Bentuk bobot:", weights_mha.shape)

    assert out_mha.shape == (4, 16)
    assert weights_mha.shape == (2, 4, 4)
    print("[OK] Multi-Head Attention berhasil.")

    # 3. Studi kasus Asep
    tokens, attention_matrix = run_coreference_case_study()

    # 4. Simpan heatmap
    plot_attention_heatmap(
        tokens,
        attention_matrix,
        "attention_heatmap_pekan_05.png"
    )

    # 5. Demonstrasi causal masking
    demo_causal_mask()

    # 6. Panduan tugas
    print_assignment_brief()

    print("\n[SELESAI] Seluruh pengujian utama berhasil dijalankan.")


if __name__ == "__main__":
    main()
