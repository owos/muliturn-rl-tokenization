import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

turns = np.arange(9)
base = [0.4428, 0.329, 0.2585, 0.2, 0.1767, 0.1154, 0.1818, 0, 0.1911]
single = [0.16, 0.5186, 0.5367, 0.4218, 0.3534, 0.1923, 0.2121, 0, 0.5299]
full = [0.8673, 0.8089, 0.7534, 0.6756, 0.53, 0.4487, 0.4242, 0.2, 0.6763]

plt.figure(figsize=(10, 6))

# Plot the main turns (0-7)
plt.plot(turns[:8], base[:8], marker='o', label='Base Model (No RL)', linestyle='-', color='#6b7280', linewidth=3.0, markersize=8)
plt.plot(turns[:8], single[:8], marker='s', label='Single-Turn GRPO', color='#dc2626', linewidth=3.0, markersize=8)
plt.plot(turns[:8], full[:8], marker='^', label='Full Retokenize GRPO', color='#16a34a', linewidth=3.0, markersize=10)

# Plot visually distinct lines bridging to the 'Final' metric
plt.plot(turns[7:], base[7:], linestyle=':', color='#6b7280', linewidth=2.0)
plt.plot(turns[7:], single[7:], linestyle=':', color='#dc2626', linewidth=2.0)
plt.plot(turns[7:], full[7:], linestyle=':', color='#16a34a', linewidth=2.0)

# Plot Final points
plt.plot([8], [base[8]], marker='o', color='#6b7280', markersize=8)
plt.plot([8], [single[8]], marker='s', color='#dc2626', markersize=8)
plt.plot([8], [full[8]], marker='^', color='#16a34a', markersize=10)

# Add Annotations
for i in range(9):
    # Full (top) - annotate above the point
    plt.annotate(f"{full[i]:.2f}", (turns[i], full[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='#16a34a', fontweight='bold')
    # Single (middle) - annotate to the right or below depending on position
    offset_single = (0, -15) if i in [1, 2] else (15, -4)
    plt.annotate(f"{single[i]:.2f}", (turns[i], single[i]), textcoords="offset points", xytext=offset_single, ha='left' if i not in [1,2] else 'center', fontsize=10, color='#dc2626', fontweight='bold')
    # Base (bottom) - annotate below the point
    plt.annotate(f"{base[i]:.2f}", (turns[i], base[i]), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=10, color='#6b7280', fontweight='bold')

plt.title('Per-Turn Accuracy over Conversation Depth', fontsize=18, fontweight='500', color='#1e40af', pad=15)
plt.xlabel('Conversation Stage', fontsize=14)
plt.ylabel('Exact Match Accuracy', fontsize=14)

labels = [f'Turn {i}' for i in range(8)] + ['Final\nAcc']
plt.xticks(turns, labels, fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12, loc='upper right', framealpha=0.95)
plt.ylim(-0.05, 1.0) # Increased top limit to give room for annotations

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)

plt.tight_layout(pad=0.5)
plt.savefig('/fs/scratch/PAS2836/owos/experiments/RL_MTT/reward_plot.png', dpi=300)
