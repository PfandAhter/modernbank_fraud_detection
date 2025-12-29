"""Generate data visualization charts for fraud detection project documentation.

This script creates various charts including:
- Correlation heatmaps
- Distribution histograms
- Box plots
- Scatter plots
- Feature importance charts
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "realistic_fraud_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "document"

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom colors
FRAUD_COLOR = "#e74c3c"  # Red
NORMAL_COLOR = "#2ecc71"  # Green
ACCENT_COLOR = "#3498db"  # Blue


def load_data() -> pd.DataFrame:
    """Load and prepare the fraud dataset."""
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud rate: {df['fraud_label'].mean():.2%}")
    return df


def create_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> str:
    """Create a correlation heatmap for numeric features."""
    print("\n[CHART] Creating correlation heatmap...")
    
    # Select numeric columns for correlation
    numeric_cols = [
        'transaction_amount', 'account_balance_before', 'avg_transaction_amount_7d',
        'transaction_count_24h', 'transaction_count_7d', 'card_age_months',
        'amount_to_avg_ratio', 'balance_drain_ratio', 'velocity_24h',
        'velocity_7d', 'velocity_burst', 'fraud_label'
    ]
    
    # Filter available columns
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    # Calculate correlation matrix
    corr_matrix = df[available_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Custom colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt=".2f", 
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Korelasyon Katsayısı"},
        ax=ax,
        annot_kws={"size": 9}
    )
    
    ax.set_title("Öznitelikler Arası Korelasyon Matrisi", fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    
    # Save
    timestamp = int(datetime.now().timestamp() * 1000)
    filename = f"correlation_heatmap_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   [OK] Saved: {filename}")
    return filename


def create_fraud_correlation_bar(df: pd.DataFrame, output_dir: Path) -> str:
    """Create bar chart showing feature correlation with fraud label."""
    print("\n[CHART] Creating fraud correlation bar chart...")
    
    # Numeric features
    numeric_cols = [
        'transaction_amount', 'balance_drain_ratio', 'amount_to_avg_ratio',
        'velocity_burst', 'velocity_24h', 'transaction_count_24h',
        'card_age_months', 'is_new_receiver', 'is_off_hours',
        'is_new_card', 'previous_fraud_flag', 'is_weekend'
    ]
    
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    # Calculate correlations with fraud
    correlations = df[available_cols].corrwith(df['fraud_label']).sort_values(ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color based on correlation direction
    colors = [FRAUD_COLOR if x > 0 else NORMAL_COLOR for x in correlations.values]
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(correlations)), correlations.values, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for i, (val, bar) in enumerate(zip(correlations.values, bars)):
        x_pos = val + 0.01 if val > 0 else val - 0.01
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, i, f'{val:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(correlations.index, fontsize=11)
    ax.set_xlabel("Korelasyon Katsayısı (Pearson)", fontsize=12)
    ax.set_title("Özniteliklerin Dolandırıcılık (fraud_label) ile Korelasyonu", fontsize=14, fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlim(-0.6, 0.8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=FRAUD_COLOR, label='Pozitif Korelasyon (Fraud ile ilişkili)'),
        Patch(facecolor=NORMAL_COLOR, label='Negatif Korelasyon (Normal ile ilişkili)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    timestamp = int(datetime.now().timestamp() * 1000)
    filename = f"fraud_correlation_bar_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   [OK] Saved: {filename}")
    return filename


def create_amount_distribution(df: pd.DataFrame, output_dir: Path) -> str:
    """Create histogram comparing transaction amounts for fraud vs normal."""
    print("\n[CHART] Creating amount distribution histogram...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with log scale
    ax1 = axes[0]
    bins = np.logspace(np.log10(df['transaction_amount'].min() + 1), 
                       np.log10(df['transaction_amount'].max()), 50)
    
    ax1.hist(df[df['fraud_label'] == 0]['transaction_amount'], bins=bins, 
             alpha=0.7, label='Normal', color=NORMAL_COLOR, edgecolor='white')
    ax1.hist(df[df['fraud_label'] == 1]['transaction_amount'], bins=bins, 
             alpha=0.7, label='Fraud', color=FRAUD_COLOR, edgecolor='white')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('İşlem Tutarı (TL) - Log Ölçek', fontsize=11)
    ax1.set_ylabel('Frekans', fontsize=11)
    ax1.set_title('İşlem Tutarı Dağılımı', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # KDE plot
    ax2 = axes[1]
    fraud_amounts = df[df['fraud_label'] == 1]['transaction_amount']
    normal_amounts = df[df['fraud_label'] == 0]['transaction_amount']
    
    # Clip for better visualization
    max_val = np.percentile(df['transaction_amount'], 99)
    
    sns.kdeplot(data=normal_amounts[normal_amounts < max_val], ax=ax2, 
                color=NORMAL_COLOR, fill=True, alpha=0.4, label='Normal', linewidth=2)
    sns.kdeplot(data=fraud_amounts[fraud_amounts < max_val], ax=ax2, 
                color=FRAUD_COLOR, fill=True, alpha=0.4, label='Fraud', linewidth=2)
    
    ax2.set_xlabel('İşlem Tutarı (TL)', fontsize=11)
    ax2.set_ylabel('Yoğunluk', fontsize=11)
    ax2.set_title('İşlem Tutarı Yoğunluk Grafiği (KDE)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    timestamp = int(datetime.now().timestamp() * 1000)
    filename = f"amount_distribution_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   [OK] Saved: {filename}")
    return filename


def create_hourly_distribution(df: pd.DataFrame, output_dir: Path) -> str:
    """Create hourly transaction distribution chart."""
    print("\n[CHART] Creating hourly distribution chart...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Stacked bar chart
    ax1 = axes[0]
    hours = range(24)
    normal_counts = [len(df[(df['fraud_label'] == 0) & (df['txn_hour'] == h)]) for h in hours]
    fraud_counts = [len(df[(df['fraud_label'] == 1) & (df['txn_hour'] == h)]) for h in hours]
    
    ax1.bar(hours, normal_counts, label='Normal', color=NORMAL_COLOR, alpha=0.8)
    ax1.bar(hours, fraud_counts, bottom=normal_counts, label='Fraud', color=FRAUD_COLOR, alpha=0.8)
    
    ax1.set_xlabel('Saat', fontsize=11)
    ax1.set_ylabel('İşlem Sayısı', fontsize=11)
    ax1.set_title('Saatlik İşlem Dağılımı', fontsize=13, fontweight='bold')
    ax1.set_xticks(hours)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Fraud rate by hour
    ax2 = axes[1]
    fraud_rates = []
    for h in hours:
        hour_df = df[df['txn_hour'] == h]
        if len(hour_df) > 0:
            fraud_rates.append(hour_df['fraud_label'].mean() * 100)
        else:
            fraud_rates.append(0)
    
    colors = [FRAUD_COLOR if r > 15 else ACCENT_COLOR for r in fraud_rates]
    bars = ax2.bar(hours, fraud_rates, color=colors, alpha=0.8, edgecolor='white')
    
    ax2.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Ortalama Fraud Oranı (%15)')
    ax2.set_xlabel('Saat', fontsize=11)
    ax2.set_ylabel('Fraud Oranı (%)', fontsize=11)
    ax2.set_title('Saatlik Dolandırıcılık Oranı', fontsize=13, fontweight='bold')
    ax2.set_xticks(hours)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Highlight off-hours region
    ax2.axvspan(-0.5, 5.5, alpha=0.1, color='red', label='Off-hours (00:00-06:00)')
    ax2.axvspan(22.5, 23.5, alpha=0.1, color='red')
    
    plt.tight_layout()
    
    # Save
    timestamp = int(datetime.now().timestamp() * 1000)
    filename = f"hourly_distribution_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   [OK] Saved: {filename}")
    return filename


def create_boxplots(df: pd.DataFrame, output_dir: Path) -> str:
    """Create box plots comparing key features for fraud vs normal."""
    print("\n[CHART] Creating box plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    features = [
        ('balance_drain_ratio', 'Bakiye Boşaltma Oranı'),
        ('amount_to_avg_ratio', 'Tutar/Ortalama Oranı'),
        ('velocity_burst', 'Hız Patlaması'),
        ('transaction_count_24h', '24 Saat İşlem Sayısı'),
        ('card_age_months', 'Kart Yaşı (Ay)'),
        ('transaction_amount', 'İşlem Tutarı (TL)')
    ]
    
    for idx, (feature, title) in enumerate(features):
        ax = axes[idx // 3, idx % 3]
        
        if feature not in df.columns:
            continue
            
        # Prepare data
        data_normal = df[df['fraud_label'] == 0][feature]
        data_fraud = df[df['fraud_label'] == 1][feature]
        
        # Clip outliers for better visualization
        q99 = df[feature].quantile(0.99)
        data_normal_clipped = data_normal.clip(upper=q99)
        data_fraud_clipped = data_fraud.clip(upper=q99)
        
        bp = ax.boxplot(
            [data_normal_clipped, data_fraud_clipped],
            labels=['Normal', 'Fraud'],
            patch_artist=True,
            widths=0.6
        )
        
        # Color the boxes
        bp['boxes'][0].set_facecolor(NORMAL_COLOR)
        bp['boxes'][1].set_facecolor(FRAUD_COLOR)
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean markers
        means = [data_normal_clipped.mean(), data_fraud_clipped.mean()]
        ax.scatter([1, 2], means, color='black', marker='D', s=50, zorder=5, label='Ortalama')
    
    plt.suptitle('Öznitelik Dağılımları: Normal vs Fraud', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    timestamp = int(datetime.now().timestamp() * 1000)
    filename = f"feature_boxplots_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   [OK] Saved: {filename}")
    return filename


def create_scatter_plots(df: pd.DataFrame, output_dir: Path) -> str:
    """Create scatter plots showing relationships between key features."""
    print("\n[CHART] Creating scatter plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Sample for performance
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    
    # Plot 1: Balance Drain vs Amount to Avg Ratio
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        sample_df['balance_drain_ratio'],
        sample_df['amount_to_avg_ratio'].clip(upper=50),
        c=sample_df['fraud_label'],
        cmap=LinearSegmentedColormap.from_list('custom', [NORMAL_COLOR, FRAUD_COLOR]),
        alpha=0.5,
        s=20
    )
    ax1.set_xlabel('Bakiye Boşaltma Oranı', fontsize=11)
    ax1.set_ylabel('Tutar/Ortalama Oranı', fontsize=11)
    ax1.set_title('Bakiye Boşaltma vs Tutar Oranı', fontsize=12, fontweight='bold')
    ax1.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Risk Eşiği (0.7)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Velocity Burst vs Transaction Count
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        sample_df['velocity_burst'].clip(upper=20),
        sample_df['transaction_count_24h'],
        c=sample_df['fraud_label'],
        cmap=LinearSegmentedColormap.from_list('custom', [NORMAL_COLOR, FRAUD_COLOR]),
        alpha=0.5,
        s=20
    )
    ax2.set_xlabel('Hız Patlaması', fontsize=11)
    ax2.set_ylabel('24 Saat İşlem Sayısı', fontsize=11)
    ax2.set_title('Hız Patlaması vs İşlem Sayısı', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Card Age vs Transaction Amount
    ax3 = axes[2]
    scatter3 = ax3.scatter(
        sample_df['card_age_months'],
        np.log10(sample_df['transaction_amount'] + 1),
        c=sample_df['fraud_label'],
        cmap=LinearSegmentedColormap.from_list('custom', [NORMAL_COLOR, FRAUD_COLOR]),
        alpha=0.5,
        s=20
    )
    ax3.set_xlabel('Kart Yaşı (Ay)', fontsize=11)
    ax3.set_ylabel('İşlem Tutarı (log10)', fontsize=11)
    ax3.set_title('Kart Yaşı vs İşlem Tutarı', fontsize=12, fontweight='bold')
    ax3.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Yeni Kart Eşiği (3 ay)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.15)
    cbar.set_label('Fraud Label (0=Normal, 1=Fraud)', fontsize=11)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Normal', 'Fraud'])
    
    plt.tight_layout()
    
    # Save
    timestamp = int(datetime.now().timestamp() * 1000)
    filename = f"scatter_plots_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   [OK] Saved: {filename}")
    return filename


def create_fraud_type_pie(df: pd.DataFrame, output_dir: Path) -> str:
    """Create pie chart showing class distribution."""
    print("\n[CHART] Creating class distribution pie chart...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Class distribution
    ax1 = axes[0]
    fraud_count = df['fraud_label'].sum()
    normal_count = len(df) - fraud_count
    
    sizes = [normal_count, fraud_count]
    labels = [f'Normal\n{normal_count:,} ({100*normal_count/len(df):.1f}%)', 
              f'Fraud\n{fraud_count:,} ({100*fraud_count/len(df):.1f}%)']
    colors = [NORMAL_COLOR, FRAUD_COLOR]
    explode = (0, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='',
            shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Sınıf Dağılımı', fontsize=14, fontweight='bold')
    
    # Card type distribution by fraud
    ax2 = axes[1]
    card_fraud = df.groupby(['card_type', 'fraud_label']).size().unstack(fill_value=0)
    card_fraud_pct = card_fraud.div(card_fraud.sum(axis=1), axis=0) * 100
    
    x = np.arange(len(card_fraud_pct.index))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, card_fraud_pct[0], width, label='Normal', color=NORMAL_COLOR, alpha=0.8)
    bars2 = ax2.bar(x + width/2, card_fraud_pct[1], width, label='Fraud', color=FRAUD_COLOR, alpha=0.8)
    
    ax2.set_ylabel('Yüzde (%)', fontsize=11)
    ax2.set_xlabel('Kart Tipi', fontsize=11)
    ax2.set_title('Kart Tipine Göre Fraud Oranı', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(card_fraud_pct.index, fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    timestamp = int(datetime.now().timestamp() * 1000)
    filename = f"class_distribution_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   [OK] Saved: {filename}")
    return filename


def create_feature_histograms(df: pd.DataFrame, output_dir: Path) -> str:
    """Create histogram grid for all key features."""
    print("\n[CHART] Creating feature histograms grid...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    features = [
        ('balance_drain_ratio', 'Bakiye Boşaltma Oranı'),
        ('amount_to_avg_ratio', 'Tutar/Ortalama Oranı'),
        ('velocity_burst', 'Hız Patlaması'),
        ('velocity_24h', '24 Saatlik Hız'),
        ('transaction_count_24h', '24s İşlem Sayısı'),
        ('transaction_count_7d', '7g İşlem Sayısı'),
        ('card_age_months', 'Kart Yaşı (Ay)'),
        ('txn_hour', 'İşlem Saati'),
        ('is_new_receiver', 'Yeni Alıcı'),
        ('is_off_hours', 'Mesai Dışı'),
        ('is_new_card', 'Yeni Kart'),
        ('previous_fraud_flag', 'Önceki Fraud')
    ]
    
    for idx, (feature, title) in enumerate(features):
        ax = axes[idx]
        
        if feature not in df.columns:
            ax.set_visible(False)
            continue
        
        normal_data = df[df['fraud_label'] == 0][feature]
        fraud_data = df[df['fraud_label'] == 1][feature]
        
        # Clip for better visualization
        if feature in ['amount_to_avg_ratio', 'velocity_burst']:
            normal_data = normal_data.clip(upper=normal_data.quantile(0.99))
            fraud_data = fraud_data.clip(upper=fraud_data.quantile(0.99))
        
        # Determine bins
        if feature in ['is_new_receiver', 'is_off_hours', 'is_new_card', 'previous_fraud_flag']:
            bins = [-0.5, 0.5, 1.5]
        elif feature == 'txn_hour':
            bins = range(25)
        else:
            bins = 30
        
        ax.hist(normal_data, bins=bins, alpha=0.6, label='Normal', color=NORMAL_COLOR, density=True)
        ax.hist(fraud_data, bins=bins, alpha=0.6, label='Fraud', color=FRAUD_COLOR, density=True)
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Öznitelik Dağılımları (Normalized)', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    # Save
    timestamp = int(datetime.now().timestamp() * 1000)
    filename = f"feature_histograms_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   [OK] Saved: {filename}")
    return filename


def main():
    """Generate all visualization charts."""
    print("="*60)
    print("Fraud Detection - Veri Gorsellestirme")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all charts
    generated_files = []
    
    generated_files.append(("Korelasyon Heatmap", create_correlation_heatmap(df, OUTPUT_DIR)))
    generated_files.append(("Fraud Korelasyon Bar", create_fraud_correlation_bar(df, OUTPUT_DIR)))
    generated_files.append(("Tutar Dağılımı", create_amount_distribution(df, OUTPUT_DIR)))
    generated_files.append(("Saatlik Dağılım", create_hourly_distribution(df, OUTPUT_DIR)))
    generated_files.append(("Box Plots", create_boxplots(df, OUTPUT_DIR)))
    generated_files.append(("Scatter Plots", create_scatter_plots(df, OUTPUT_DIR)))
    generated_files.append(("Sınıf Dağılımı", create_fraud_type_pie(df, OUTPUT_DIR)))
    generated_files.append(("Öznitelik Histogramları", create_feature_histograms(df, OUTPUT_DIR)))
    
    # Summary
    print("\n" + "="*60)
    print("[SUCCESS] TUM GRAFIKLER OLUSTURULDU!")
    print("="*60)
    print(f"\nKayit dizini: {OUTPUT_DIR}")
    print("\nOlusturulan grafikler:")
    for name, filename in generated_files:
        print(f"   • {name}: {filename}")
    
    return generated_files


if __name__ == "__main__":
    main()
