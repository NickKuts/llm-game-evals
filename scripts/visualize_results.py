#!/usr/bin/env python3
"""
Visualization script for quest-evals benchmark results.
Creates clean, publication-ready charts for the blog post.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Data from results
models = [
    'Claude Opus 4.5',
    'GPT-5.2 (high)',
    'GPT-5.2',
    'Gemini 3 Pro',
    'Claude Sonnet 4.5',
    'Gemini 3 Flash'
]

at1_scores = [66, 60, 60, 60, 33, 25]
at3_scores = [73, 73, 73, 67, 47, 47]

# Quest data
quests = ['Ski Resort', 'Jumper', 'Player', 'Borzukhan', 'Music Fest']
difficulties = ['Medium', 'Very Hard', 'Easy', 'Medium', 'Easy']
quest_types = ['Economic', 'Spatial', 'Logic', 'Tactical', 'Social']

# Per-quest scores (GPT-5.2, Opus 4.5, Gemini Pro)
quest_scores = {
    'Ski Resort': [100, 100, 100],
    'Jumper': [100, 100, 100],
    'Player': [100, 33, 100],
    'Borzukhan': [67, 67, 33],
    'Music Fest': [0, 33, 0],
}

# Colors
colors = {
    'Claude Opus 4.5': '#E07B39',
    'GPT-5.2 (high)': '#4A9C6D',
    'GPT-5.2': '#5BA375',
    'Gemini 3 Pro': '#4285F4',
    'Claude Sonnet 4.5': '#D4896A',
    'Gemini 3 Flash': '#7BAAF7',
}

model_colors_quest = ['#5BA375', '#E07B39', '#4285F4']  # GPT, Opus, Gemini


def create_overall_chart():
    """Create horizontal bar chart for overall model performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(models))
    bar_height = 0.35

    # Create bars
    bars1 = ax.barh(y_pos + bar_height/2, at1_scores, bar_height,
                    label='@1 (First attempt)', color='#6B7280', alpha=0.7)
    bars2 = ax.barh(y_pos - bar_height/2, at3_scores, bar_height,
                    label='@3 (With reflection)', color='#3B82F6', alpha=0.9)

    # Customize
    ax.set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlim(0, 105)
    ax.legend(loc='lower right', fontsize=10)

    # Add value labels
    for bar, score in zip(bars1, at1_scores):
        ax.text(score + 2, bar.get_y() + bar.get_height()/2,
                f'{score}%', va='center', fontsize=10, color='#374151')
    for bar, score in zip(bars2, at3_scores):
        ax.text(score + 2, bar.get_y() + bar.get_height()/2,
                f'{score}%', va='center', fontsize=10, fontweight='bold', color='#1E40AF')

    ax.set_title('Overall Model Performance on Quest-Evals', fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/charts/overall_performance.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('results/charts/overall_performance.svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: results/charts/overall_performance.png/svg")
    plt.close()


def create_quest_heatmap():
    """Create heatmap showing per-quest performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data matrix
    model_names = ['GPT-5.2', 'Claude Opus 4.5', 'Gemini 3 Pro']
    data = np.array([quest_scores[q] for q in quests])

    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Labels
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(quests)))
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')

    # Quest labels with difficulty
    quest_labels = [f'{q}\n({d})' for q, d in zip(quests, difficulties)]
    ax.set_yticklabels(quest_labels, fontsize=10)

    # Add text annotations
    for i in range(len(quests)):
        for j in range(len(model_names)):
            value = data[i, j]
            text_color = 'white' if value < 50 else 'black'
            ax.text(j, i, f'{value}%', ha='center', va='center',
                    fontsize=12, fontweight='bold', color=text_color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Success Rate (%)', fontsize=11)

    ax.set_title('Per-Quest Performance\n"Very Hard" spatial → 100% | "Easy" social → 0%',
                 fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig('results/charts/quest_heatmap.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('results/charts/quest_heatmap.svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: results/charts/quest_heatmap.png/svg")
    plt.close()


def create_difficulty_paradox_chart():
    """Create chart highlighting the difficulty paradox."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Average scores per quest
    avg_scores = [np.mean(quest_scores[q]) for q in quests]

    # Color by difficulty
    diff_colors = {
        'Easy': '#EF4444',      # Red (surprisingly hard!)
        'Medium': '#F59E0B',    # Orange
        'Very Hard': '#22C55E'  # Green (surprisingly easy!)
    }
    bar_colors = [diff_colors[d] for d in difficulties]

    x_pos = np.arange(len(quests))
    bars = ax.bar(x_pos, avg_scores, color=bar_colors, edgecolor='white', linewidth=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{q}\n({d})' for q, d in zip(quests, difficulties)], fontsize=10)
    ax.set_ylabel('Average Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)

    # Add value labels
    for bar, score in zip(bars, avg_scores):
        ax.text(bar.get_x() + bar.get_width()/2, score + 3,
                f'{score:.0f}%', ha='center', fontsize=11, fontweight='bold')

    # Legend
    legend_patches = [
        mpatches.Patch(color='#22C55E', label='Very Hard (easy for LLMs!)'),
        mpatches.Patch(color='#F59E0B', label='Medium'),
        mpatches.Patch(color='#EF4444', label='Easy (hard for LLMs!)')
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    ax.set_title('The Difficulty Paradox: In-Game Difficulty ≠ LLM Difficulty',
                 fontsize=13, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/charts/difficulty_paradox.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('results/charts/difficulty_paradox.svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: results/charts/difficulty_paradox.png/svg")
    plt.close()


def create_grouped_bar_chart():
    """Create grouped bar chart for per-quest comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(quests))
    width = 0.25

    model_names = ['GPT-5.2', 'Claude Opus 4.5', 'Gemini 3 Pro']

    for i, (model, color) in enumerate(zip(model_names, model_colors_quest)):
        scores = [quest_scores[q][i] for q in quests]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, scores, width, label=model, color=color, alpha=0.85)

        # Add value labels
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width()/2, score + 2,
                        f'{score}%', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{q}\n({d})' for q, d in zip(quests, difficulties)], fontsize=10)
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', fontsize=10)

    ax.set_title('Per-Quest Model Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add horizontal line at 50%
    ax.axhline(y=50, color='#9CA3AF', linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.savefig('results/charts/quest_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('results/charts/quest_comparison.svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: results/charts/quest_comparison.png/svg")
    plt.close()


def create_reflection_impact_chart():
    """Show impact of reflection (@1 vs @3)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, at1_scores, width, label='@1 (No reflection)',
                   color='#9CA3AF', alpha=0.8)
    bars2 = ax.bar(x + width/2, at3_scores, width, label='@3 (With reflection)',
                   color='#3B82F6', alpha=0.9)

    # Add improvement arrows/text
    for i, (a1, a3) in enumerate(zip(at1_scores, at3_scores)):
        improvement = a3 - a1
        if improvement > 0:
            ax.annotate(f'+{improvement}%', xy=(i + width/2, a3 + 2),
                       fontsize=9, ha='center', color='#22C55E', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=9)
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 90)
    ax.legend(loc='upper right', fontsize=10)

    ax.set_title('Impact of Reflection: Does Retry Help?', fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/charts/reflection_impact.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('results/charts/reflection_impact.svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: results/charts/reflection_impact.png/svg")
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('results/charts', exist_ok=True)

    print("Generating visualizations...")
    create_overall_chart()
    create_quest_heatmap()
    create_difficulty_paradox_chart()
    create_grouped_bar_chart()
    create_reflection_impact_chart()
    print("\nDone! Charts saved to results/charts/")
