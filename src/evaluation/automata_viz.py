import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "notebooks"


def save_fig(fig, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Kaydedildi: {path}")


def plot_transition_heatmap(automata, dataset: str, top_n: int = 15):
    """
    Geçiş olasılıkları matrisini heatmap olarak çizer.
    Okunabilirlik için en sık geçiş yapılan top_n durum seçilir.
    """
    if not hasattr(automata, 'transition_probs'):
        automata.get_transition_probs()

    # En sık geçiş yapılan durumları seç
    state_freq = {}
    for src, targets in automata.transition_counts.items():
        state_freq[src] = sum(targets.values())

    top_states = sorted(state_freq, key=state_freq.get, reverse=True)[:top_n]

    matrix = np.zeros((len(top_states), len(top_states)))
    for i, src in enumerate(top_states):
        for j, dst in enumerate(top_states):
            matrix[i][j] = automata.transition_probs.get(src, {}).get(dst, 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(top_states)))
    ax.set_yticks(range(len(top_states)))
    ax.set_xticklabels(top_states, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(top_states, fontsize=8)
    ax.set_xlabel("Hedef Durum")
    ax.set_ylabel("Kaynak Durum")
    ax.set_title(f"Geçiş Olasılığı Heatmap — {dataset} (Top {top_n} Durum)")
    plt.colorbar(im, ax=ax)
    save_fig(fig, f"transition_heatmap_{dataset}.png")


def plot_state_diagram(automata, dataset: str, top_n: int = 8):
    """
    Otomata durum diyagramını graphviz olmadan matplotlib ile çizer.
    En yüksek geçiş olasılıklı top_n durum gösterilir.
    """
    if not hasattr(automata, 'transition_probs'):
        automata.get_transition_probs()

    state_freq = {}
    for src, targets in automata.transition_counts.items():
        state_freq[src] = sum(targets.values())

    top_states = sorted(state_freq, key=state_freq.get, reverse=True)[:top_n]

    # Dairesel yerleşim
    angles = np.linspace(0, 2 * np.pi, len(top_states), endpoint=False)
    positions = {s: (np.cos(a), np.sin(a)) for s, a in zip(top_states, angles)}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")
    ax.set_title(f"Otomata Durum Diyagramı — {dataset} (Top {top_n} Durum)", fontsize=13)

    # Oklar
    for src in top_states:
        for dst in top_states:
            if src == dst:
                continue
            prob = automata.transition_probs.get(src, {}).get(dst, 0)
            if prob > 0.1:
                x1, y1 = positions[src]
                x2, y2 = positions[dst]
                ax.annotate(
                    "", xy=(x2 * 0.85, y2 * 0.85),
                    xytext=(x1 * 0.85, y1 * 0.85),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=plt.cm.Reds(prob),
                        lw=1.5 * prob + 0.5
                    )
                )
                mx, my = (x1 + x2) / 2 * 0.85, (y1 + y2) / 2 * 0.85
                ax.text(mx, my, f"{prob:.2f}", fontsize=7, ha="center", color="gray")

    # Düğümler
    for state, (x, y) in positions.items():
        ax.add_patch(plt.Circle((x, y), 0.12, color="steelblue", zorder=3))
        ax.text(x, y, state, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=4)

    save_fig(fig, f"state_diagram_{dataset}.png")
