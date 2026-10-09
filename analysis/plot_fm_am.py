import os
FIG_DIR = os.environ.get('ADG_FIG_DIR', 'results/figures')
os.makedirs(FIG_DIR, exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Data from the provided text table (Models, Patches (M), WSIs (K), Training Strategy, Level)
data = [
    {'Model': 'Lunit', 'Patches (M)': 33, 'WSIs (K)': 37, 'Training Strategy': 'ViT-S + DINO', 'Level': 'Tile Level'},
    {'Model': 'GigaPath', 'Patches (M)': 1300, 'WSIs (K)': 171, 'Training Strategy': 'ViT-G + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CTransPath', 'Patches (M)': 16, 'WSIs (K)': 32, 'Training Strategy': 'ViT-S + MoCoV3', 'Level': 'Tile Level'},
    {'Model': 'Hibou', 'Patches (M)': 1200, 'WSIs (K)': 1139, 'Training Strategy': 'ViT-L + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'HOptimus1', 'Patches (M)': 1000, 'WSIs (K)': 1000, 'Training Strategy': 'ViT-G + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'Virchow2', 'Patches (M)': 2000, 'WSIs (K)': 3100, 'Training Strategy': 'ViT-H + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CONCH1.5', 'Patches (M)': 1.26, 'WSIs (K)': 335, 'Training Strategy': 'ViT-L+ DINOv2', 'Level': 'Tile Level'},
    {'Model': 'UNI2-h', 'Patches (M)': 200, 'WSIs (K)': 350, 'Training Strategy': 'ViT-H+ DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CHIEF', 'Patches (M)': 16, 'WSIs (K)': 60, 'Training Strategy': 'Attention Aggregation', 'Level': 'Slide Level'},
    {'Model': 'PRISM', 'Patches (M)': 2000, 'WSIs (K)': 1500, 'Training Strategy': 'Perceiver network', 'Level': 'Slide Level'},
    {'Model': 'TITAN', 'Patches (M)': 1.26, 'WSIs (K)': 335, 'Training Strategy': 'Attention with linear biases', 'Level': 'Slide Level'}
]

df = pd.DataFrame(data)

# Fill NaN Patches (M) with 0 for plotting
df['Patches (M)'] = df['Patches (M)'].fillna(0)

# Convert Patches (M) to Patches (K)
df['Patches (K)'] = df['Patches (M)'] * 1000 
df['WSIs (K)'] = df['WSIs (K)']

# Calculate total size in thousands for sorting and stacking
df['Total Size (K)'] = df['Patches (K)'] + df['WSIs (K)']

# Sort by total size for better visualization in the horizontal bar chart
df = df.sort_values(by='Total Size (K)', ascending=True).reset_index(drop=True)

# Plotting preparation (Horizontal stacked bar plot)
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors (WSIs: dark blue, Patches: light blue)
colors = {
    'WSIs': "#6F1421",
    'Patches': "#D7B7BF"
}

# Plot WSIs (K)
ax.barh(df['Model'], df['WSIs (K)'], color=colors['WSIs'], label='Number of WSIs (Thousands)')

# Plot Patches (K), stacked on WSIs (K)
ax.barh(df['Model'], df['Patches (K)'], left=df['WSIs (K)'], color=colors['Patches'], label='Number of Patches (Thousands)')

# Set x-axis to a logarithmic scale
ax.set_xscale('log')

# Set labels and title
ax.set_xlabel('Training Data Scale (Thousands, Log Scale)')
ax.set_ylabel('Model')
ax.set_title('Pretraining Data Scale: Patches vs. WSIs for Pathology Foundation Models')

# Add symbols above the plots (Star for Tile Level, Diamond for Slide Level)
# Define marker types and colors based on 'Level'
marker_styles = {
    'Tile Level': {'marker': 's', 'color': "#87850D"},  
    'Slide Level': {'marker': 'D', 'color': "#E49D0F"}  
}

# Find the maximum total size in log scale for placing symbols above the bars
max_total_size_log = np.log10(df['Total Size (K)'].max())

# Plot symbols
for i, row in df.iterrows():
    level_info = marker_styles[row['Level']]
    marker_x = row['Total Size (K)']  # Slightly to the right of the end of bar
    ax.plot(marker_x, i,
            marker=level_info['marker'],
            markersize=10,
            color=level_info['color'],
            linestyle='None',
            clip_on=False)
# Create custom legend handles for the symbols
symbol_handles = [
    plt.Line2D([0], [0], marker='s', color='w',
               markerfacecolor=marker_styles['Tile Level']['color'], markersize=10, label='Tile Level'),
    plt.Line2D([0], [0], marker='D', color='w',
               markerfacecolor=marker_styles['Slide Level']['color'], markersize=10, label='Slide Level')
]
handles, labels = ax.get_legend_handles_labels()
handles.extend(symbol_handles)

# Remove duplicates while preserving order
unique = dict(zip(labels + ['Tile Level', 'Slide Level'], handles))
ax.legend(unique.values(), unique.keys(), title='Legend', loc='lower right', bbox_to_anchor=(1.0, 0))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fm_plot.tiff'), dpi=300)
plt.show()
plt.close()



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# Data
data = [
    {'Model': 'Lunit', 'Patches (M)': 33, 'WSIs (K)': 37, 'Training Strategy': 'ViT-S + DINO', 'Level': 'Tile Level'},
    {'Model': 'GigaPath', 'Patches (M)': 1300, 'WSIs (K)': 171, 'Training Strategy': 'ViT-G + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CTransPath', 'Patches (M)': 16, 'WSIs (K)': 32, 'Training Strategy': 'ViT-S + MoCoV3', 'Level': 'Tile Level'},
    {'Model': 'Hibou', 'Patches (M)': 1200, 'WSIs (K)': 1139, 'Training Strategy': 'ViT-L + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'HOptimus1', 'Patches (M)': 1000, 'WSIs (K)': 1000, 'Training Strategy': 'ViT-G + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'Virchow2', 'Patches (M)': 2000, 'WSIs (K)': 3100, 'Training Strategy': 'ViT-H + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CONCH1.5', 'Patches (M)': 1.26, 'WSIs (K)': 335, 'Training Strategy': 'ViT-L+ CoCa', 'Level': 'Tile Level'},
    {'Model': 'UNI2-h', 'Patches (M)': 200, 'WSIs (K)': 350, 'Training Strategy': 'ViT-H+ DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CHIEF', 'Patches (M)': 16, 'WSIs (K)': 60, 'Training Strategy': 'Attention Aggregation', 'Level': 'Slide Level'},
    {'Model': 'PRISM', 'Patches (M)': 2000, 'WSIs (K)': 1500, 'Training Strategy': 'Perceiver network', 'Level': 'Slide Level'},
    {'Model': 'TITAN', 'Patches (M)': 1.26, 'WSIs (K)': 335, 'Training Strategy': 'Attention with linear biases', 'Level': 'Slide Level'}
]

df = pd.DataFrame(data)
df['Patches (M)'] = df['Patches (M)'].fillna(0)
df['Patches (K)'] = df['Patches (M)'] * 1000
df['Total Size (K)'] = df['Patches (K)'] + df['WSIs (K)']

df = df.sort_values(by='Total Size (K)', ascending=True).reset_index(drop=True)

# Plot setup
fig, ax = plt.subplots(figsize=(12, 8))

# Colors
colors = {
    'WSIs': "#6F1421",    # Maroon dark
    'Patches': "#D7B7BF"  # Very light maroon
}

# Plot bars
ax.barh(df['Model'], df['WSIs (K)'], color=colors['WSIs'], label='Number of WSIs (Thousands)')
ax.barh(df['Model'], df['Patches (K)'], left=df['WSIs (K)'], color=colors['Patches'], label='Number of Patches (Thousands)')

# Add training strategy text inside Patches bar, aligned left with a small offset
for i, row in df.iterrows():
    y = i
    x_start = row['WSIs (K)'] + (0.05 * row['Patches (K)'])  # 5% inside Patches bar
    strategy_text = row['Training Strategy']
    ax.text(x_start, y, strategy_text,
            va='center', ha='left',
            fontsize=14, color='black')

# Log scale
ax.set_xscale('log')

# Labels & title
ax.set_xlabel('Training Data Scale (Thousands, Log Scale)')
ax.set_ylabel('Model')
# ax.set_title('Pretraining Data Scale: Patches vs. WSIs for Pathology Foundation Models')

# Marker styles for levels
marker_styles = {
    'Tile Level': {'marker': 's', 'color': "#0A1156"},  # dark mustard
    'Slide Level': {'marker': 'D', 'color': "#CD8E1A"}  # bright mustard
}

# Plot markers slightly right of bar end
for i, row in df.iterrows():
    level_info = marker_styles[row['Level']]
    marker_x = 10  # slight offset right
    ax.plot(marker_x, i,
            marker=level_info['marker'],
            markersize=10,
            color=level_info['color'],
            linestyle='None',
            clip_on=False)

# Legend for bars and markers
symbol_handles = [
    plt.Line2D([0], [0], marker='s', color='w',
               markerfacecolor=marker_styles['Tile Level']['color'], markersize=10, label='Tile Level'),
    plt.Line2D([0], [0], marker='D', color='w',
               markerfacecolor=marker_styles['Slide Level']['color'], markersize=10, label='Slide Level')
]

handles, labels = ax.get_legend_handles_labels()
labels.extend(['Tile Level', 'Slide Level'])
handles.extend(symbol_handles)

unique = OrderedDict()
for label, handle in zip(labels, handles):
    if label not in unique:
        unique[label] = handle

ax.legend(unique.values(), unique.keys(), title='Legend', loc='lower right', bbox_to_anchor=(1.0, 0))

plt.tight_layout()
# plt.savefig(os.path.join(FIG_DIR, 'fm_plot_with_names.tiff'), dpi=300)
plt.show()
plt.close()



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = [
    {'Model': 'MaxMIL', 'Approach': 'Max Pooling', 'Size': 0.79},
    {'Model': 'MeanMIL', 'Approach': 'Mean Pooling', 'Size': 0.79},
    {'Model': 'TransMIL', 'Approach': 'Transformer with positional encoding', 'Size': 2.93},
    {'Model': 'DSMIL', 'Approach': 'Max Pooling with attention scoring', 'Size': 2.58},
    {'Model': 'ABMIL', 'Approach': 'Attention Layer', 'Size': 0.85},
    {'Model': 'RRTMIL', 'Approach': 'Feature re-embedding', 'Size': 4.01},
    {'Model': 'MambaMIL', 'Approach': 'Mamba with attention pooling', 'Size': 4.49}
]

df = pd.DataFrame(data)

# Sort by Size ascending for nice visualization
df = df.sort_values('Size').reset_index(drop=True)

# Colors - colorblind friendly palette (using tab10 colors)
colors = plt.get_cmap('tab10_r').colors

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(df['Model'], df['Size'], color=colors[1:len(df)+1])
ax.set_xlabel('Model Size (×10³)')
ax.set_xlim(0, max(df['Size']) * 1.15)
# Add dashed rectangle with text inside for each bar
for i, (idx, row) in enumerate(df.iterrows()):
    bar_width = row['Size']
    y = i 
    # Rectangle position and size in data coords
    rect_x = 0
    rect_y = y - 0.4
    rect_width = bar_width # adjust width for the box
    rect_height = 0.8
    # Add rectangle (dashed border, no fill)
    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                             linewidth=1.5, edgecolor='black', facecolor='none',
                             linestyle='--', zorder=5)
    ax.add_patch(rect)
    # Add text inside rectangle, left aligned, vertically centered
    ax.text(rect_x + 0.02 * rect_width, y,
            row['Approach'],
            va='center', ha='left',
            fontsize=11, zorder=10)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by mixing it with white.
    Input can be a matplotlib color string, hex string, or RGB tuple.
    The `amount` controls the degree of lightening (0 = no change, 1 = white).
    """
    try:
        c = mcolors.cnames[color]  # Convert color name to hex
    except:
        c = color
    c = mcolors.to_rgb(c)
    white = (1, 1, 1)
    return tuple((1 - amount) * x + amount * y for x, y in zip(c, white))

# Data
data = [
    {'Model': 'MaxMIL', 'Approach': 'Max Pooling', 'Size': 0.79},
    {'Model': 'MeanMIL', 'Approach': 'Mean Pooling', 'Size': 0.79},
    {'Model': 'TransMIL', 'Approach': 'Transformer + positional encoding', 'Size': 2.93},
    {'Model': 'DSMIL', 'Approach': 'Max Pooling + attention scoring', 'Size': 2.58},
    {'Model': 'ABMIL', 'Approach': 'Attention pooling', 'Size': 0.85},
    {'Model': 'RRTMIL', 'Approach': 'Feature re-embedding', 'Size': 4.01},
    {'Model': 'MambaMIL', 'Approach': 'Mamba + attention pooling', 'Size': 4.49}
]
df = pd.DataFrame(data)
df['Size'] = df['Size'] * 1000  # now in thousands
df = df.sort_values('Size').reset_index(drop=True)
colors = plt.get_cmap('tab10_r').colors
fig, ax = plt.subplots(figsize=(8, 8))
light_colors = [lighten_color(c, amount=0.7) for c in colors]  # You can try 0.3 to 0.7 for brightness control
bars = ax.barh(df['Model'], df['Size'], color=light_colors)
# bars = ax.barh(df['Model'], df['Size'], color=colors[0:len(df)+1])
ax.set_xlabel('Model Parameters (millions)',fontsize=16)  # thousands
ax.set_ylabel('Model',fontsize=16)  # thousands
ax.set_xlim(0, max(df['Size']) * 1.15)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x / 1000)))
for i, (idx, row) in enumerate(df.iterrows()):
    bar_width = row['Size']
    y = i 
    rect_x = 0
    rect_y = y - 0.4
    rect_width = bar_width  # in thousands
    rect_height = 0.8
    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                             linewidth=1.5, edgecolor='black', facecolor='none',
                             linestyle='--', zorder=5)
    ax.add_patch(rect)
    # ax.text(rect_x + 0.02 * rect_width, y,
    #         row['Approach'],
    #         va='center', ha='left',
    #         fontsize=16, zorder=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'am_plot_with_names_light_blank.tiff'),dpi=300)
plt.show()
plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import OrderedDict

# Data from the provided text table
data = [
    {'Model': 'Lunit', 'Patches (M)': 33, 'WSIs (K)': 37, 'Training Strategy': 'ViT-S + DINO', 'Level': 'Tile Level'},
    {'Model': 'GigaPath', 'Patches (M)': 1300, 'WSIs (K)': 171, 'Training Strategy': 'ViT-G + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CTransPath', 'Patches (M)': 16, 'WSIs (K)': 32, 'Training Strategy': 'ViT-S + MoCoV3', 'Level': 'Tile Level'},
    {'Model': 'Hibou', 'Patches (M)': 1200, 'WSIs (K)': 1139, 'Training Strategy': 'ViT-L + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'HOptimus1', 'Patches (M)': 1000, 'WSIs (K)': 1000, 'Training Strategy': 'ViT-G + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'Virchow2', 'Patches (M)': 2000, 'WSIs (K)': 3100, 'Training Strategy': 'ViT-H + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CONCH1.5', 'Patches (M)': 1.26, 'WSIs (K)': 335, 'Training Strategy': 'ViT-L+ CoCa', 'Level': 'Tile Level'},
    {'Model': 'UNI2-h', 'Patches (M)': 200, 'WSIs (K)': 350, 'Training Strategy': 'ViT-H+ DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CHIEF*', 'Patches (M)': 16, 'WSIs (K)': 60, 'Training Strategy': 'Attention Aggregation', 'Level': 'Slide Level'},
    {'Model': 'PRISM*', 'Patches (M)': 2000, 'WSIs (K)': 1500, 'Training Strategy': 'Perceiver network', 'Level': 'Slide Level'},
    {'Model': 'TITAN*', 'Patches (M)': 1.26, 'WSIs (K)': 335, 'Training Strategy': 'Attention + linear bias', 'Level': 'Slide Level'}
]

df = pd.DataFrame(data)

# Data processing and conversion to thousands (K)
df['Patches (M)'] = df['Patches (M)'].fillna(0)
df['Patches (K)'] = df['Patches (M)'] * 1000
df['WSIs (K)'] = df['WSIs (K)']
df['Total Size (K)'] = df['Patches (K)'] + df['WSIs (K)']

# Sort by total size for better visualization
df = df.sort_values(by='Total Size (K)', ascending=True).reset_index(drop=True)

# Define a color map for each model (dark for WSIs, light for Patches)
# Corrected structure: nested dictionary for WSIs and Patches
model_color_map = {
    'Lunit': {'WSIs': '#1A5C14', 'Patches': '#99CC99'},       
    'CTransPath': {'WSIs': '#8B0000', 'Patches': '#FFB6C1'},   
    'CONCH1.5': {'WSIs': '#FF8C00', 'Patches': '#FFDAB9'},     
    'Hibou': {'WSIs': '#003366', 'Patches': '#ADD8E6'},        
    'HOptimus1': {'WSIs': '#005A31', 'Patches': '#A5D6A7'},    
    'GigaPath': {'WSIs': '#8B4513', 'Patches': '#F4A460'},     
    'UNI2-h': {'WSIs': '#800080', 'Patches': '#DA70D6'},       
    'CHIEF*': {'WSIs': '#FF69B4', 'Patches': '#FFC0CB'},        
    'TITAN*': {'WSIs': "#5F1B90", 'Patches': '#9370DB'},        
    'PRISM*': {'WSIs': '#1E90FF', 'Patches': '#87CEFA'},        
    'Virchow2': {'WSIs': '#696969', 'Patches': '#D3D3D3'}      
}

# Get the colors for the WSIs and Patches in the order of the sorted DataFrame
wsi_colors = [model_color_map[model]['WSIs'] for model in df['Model']]
patches_colors = [model_color_map[model]['Patches'] for model in df['Model']]
# Plot setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(df['Model'], df['WSIs (K)'], color=wsi_colors, label='Number of WSIs (Thousands)')
ax.barh(df['Model'], df['Patches (K)'], left=df['WSIs (K)'], color=patches_colors, label='Number of Patches (Thousands)')
for i, row in df.iterrows():
    y = i
    x_start = (0.008 * row['Patches (K)'])
    strategy_text = row['Training Strategy']
    text_color = 'black'
    ax.text(x_start, y, strategy_text,va='center', ha='left',fontsize=18, color=text_color)
ax.set_xscale('log')
ax.set_xlabel('Training Data Scale', fontsize=14)
ax.set_ylabel('Model',fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=14)
marker_styles = {'Tile Level': {'marker': 's', 'color': "#0A1156"},'Slide Level': {'marker': 'D', 'color': "#CD8E1A"}}
max_total_size_log = np.log10(df['Total Size (K)'].max())
for i, row in df.iterrows():
    level_info = marker_styles[row['Level']]
    marker_x = max_total_size_log - 0.2
    marker_facecolor = level_info['color']
    marker_edgecolor = 'black'
    if level_info['marker'] == 's':
        marker_facecolor = 'white'
        marker_edgecolor = 'black'
    ax.plot(marker_x, i,
            marker=level_info['marker'],
            markersize=10,
            markerfacecolor=marker_facecolor,
            markeredgecolor=marker_edgecolor,
            linestyle='None',
            clip_on=False)
bar_handles = [patches.Patch(color='#1E90FF', label="Number of WSIs (Thousands)"),patches.Patch(color='#87CEFA', label="Number of Patches (Thousands)")]
symbol_handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='black', markersize=10, label='Tile Level Pretraining'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=marker_styles['Slide Level']['color'], markersize=10, label='Slide Level Pretraining')]
handles = bar_handles + symbol_handles
labels = [h.get_label() for h in handles]
unique = OrderedDict()
for label, handle in zip(labels, handles):
    if label not in unique:
        unique[label] = handle
ax.legend(unique.values(), unique.keys(), title='Legend', loc='lower right', bbox_to_anchor=(1.0, 0))
plt.tight_layout()
# plt.savefig(os.path.join(FIG_DIR, 'am_plot_with_names_colored__1.tiff'),dpi=300)
plt.show()
plt.close()



fig, ax = plt.subplots(figsize=(8, 8))
bar_height = 0.45
y_positions = np.arange(len(df))
ax.barh(y_positions - bar_height/2, df['WSIs (K)'], height=bar_height, color=wsi_colors, label='Number of WSIs', zorder=1)
ax.barh(y_positions - bar_height/2, df['WSIs (K)'], height=bar_height, color=wsi_colors, hatch='///', edgecolor='black', label='Number of WSIs', zorder=1)
for i, row in df.iterrows():
    y = i + bar_height/2  # Align text with patch bar (upper bar)
    x_start = (0.02 * row['Patches (K)'])
    strategy_text = row['Training Strategy']
    ax.text(x_start, y, strategy_text, va='center', ha='left', fontsize=14, color='black', zorder=3)
ax.set_yticks(y_positions)
ax.set_yticklabels(df['Model'])
ax.set_xscale('log')
ax.set_xlabel('Training Data Scale', fontsize=14)
ax.set_ylabel('Model', fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=14)
marker_styles = {'Tile Level': {'marker': 's', 'color': "#0A1156"},'Slide Level': {'marker': 'D', 'color': "#CD8E1A"}}
max_total_size_log = np.log10(df['Total Size (K)'].max())
for i, row in df.iterrows():
    y = i + bar_height/2  # Align marker with patch bar
    level_info = marker_styles[row['Level']]
    marker_x = max_total_size_log - 0.2
    # ax.plot(marker_x, y,marker=level_info['marker'],markersize=10,markerfacecolor='white' if level_info['marker'] == 's' else level_info['color'],
    #         markeredgecolor='black',linestyle='None',clip_on=False,zorder=3)
bar_handles = [patches.Patch(color='#1E90FF', label="Number of WSIs (Thousands)"),patches.Patch(color='#87CEFA', label="Number of Patches (Thousands)")]
symbol_handles = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='black', markersize=10, label='Tile Level Pretraining'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=marker_styles['Slide Level']['color'], markeredgecolor='black', markersize=10, label='Slide Level Pretraining')
]
handles = bar_handles
labels = [h.get_label() for h in handles]
unique = OrderedDict()
for label, handle in zip(labels, handles):
    if label not in unique:
        unique[label] = handle
ax.legend(unique.values(), unique.keys(), title='Legend', loc='lower right', bbox_to_anchor=(1.0, 0))
plt.tight_layout()
# plt.savefig(os.path.join(FIG_DIR, 'am_plot_with_separate.tiff'),dpi=300)
plt.show()
plt.close()




import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import OrderedDict

# Data
data = [
    {'Model': 'Lunit', 'Patches (M)': 33, 'WSIs (K)': 37, 'Training Strategy': 'ViT-S + DINO', 'Level': 'Tile Level'},
    {'Model': 'GigaPath', 'Patches (M)': 1300, 'WSIs (K)': 171, 'Training Strategy': 'ViT-G + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CTransPath', 'Patches (M)': 16, 'WSIs (K)': 32, 'Training Strategy': 'ViT-S + MoCoV3', 'Level': 'Tile Level'},
    {'Model': 'Hibou', 'Patches (M)': 1200, 'WSIs (K)': 1139, 'Training Strategy': 'ViT-L + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'HOptimus1', 'Patches (M)': 1000, 'WSIs (K)': 1000, 'Training Strategy': 'ViT-G + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'Virchow2', 'Patches (M)': 2000, 'WSIs (K)': 3100, 'Training Strategy': 'ViT-H + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CONCH1.5', 'Patches (M)': 1.26, 'WSIs (K)': 335, 'Training Strategy': 'ViT-L+ CoCa', 'Level': 'Tile Level'},
    {'Model': 'UNI2-h', 'Patches (M)': 200, 'WSIs (K)': 350, 'Training Strategy': 'ViT-H+ DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CHIEF*', 'Patches (M)': 16, 'WSIs (K)': 60, 'Training Strategy': 'Attention Aggregation', 'Level': 'Slide Level'},
    {'Model': 'PRISM*', 'Patches (M)': 2000, 'WSIs (K)': 1500, 'Training Strategy': 'Perceiver network', 'Level': 'Slide Level'},
    {'Model': 'TITAN*', 'Patches (M)': 1.26, 'WSIs (K)': 335, 'Training Strategy': 'Attention + linear bias', 'Level': 'Slide Level'}
]
df = pd.DataFrame(data)

# Processing
df['Patches (K)'] = df['Patches (M)'] * 1000
df['Total Size (K)'] = df['Patches (K)'] + df['WSIs (K)']
df = df.sort_values(by='Total Size (K)').reset_index(drop=True)

# Model colors
model_color_map = {
    'Lunit': {'WSIs': '#1A5C14', 'Patches': '#99CC99'},
    'CTransPath': {'WSIs': '#8B0000', 'Patches': '#FFB6C1'},
    'CONCH1.5': {'WSIs': '#FF8C00', 'Patches': '#FFDAB9'},
    'Hibou': {'WSIs': '#003366', 'Patches': '#ADD8E6'},
    'HOptimus1': {'WSIs': '#005A31', 'Patches': '#A5D6A7'},
    'GigaPath': {'WSIs': '#8B4513', 'Patches': '#F4A460'},
    'UNI2-h': {'WSIs': '#800080', 'Patches': '#DA70D6'},
    'CHIEF*': {'WSIs': '#FF69B4', 'Patches': '#FFC0CB'},
    'TITAN*': {'WSIs': "#5F1B90", 'Patches': '#9370DB'},
    'PRISM*': {'WSIs': '#1E90FF', 'Patches': '#87CEFA'},
    'Virchow2': {'WSIs': '#696969', 'Patches': '#D3D3D3'}
}

# Generate colors for each model
wsi_colors = [model_color_map[model]['WSIs'] for model in df['Model']]
patch_colors = [model_color_map[model]['Patches'] for model in df['Model']]

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
bar_height = 0.45
y_positions = np.arange(len(df))
for i, row in df.iterrows():
    ax.barh(y_positions[i] - bar_height / 2,row['WSIs (K)'],height=bar_height,color=wsi_colors[i],hatch='///',
        edgecolor='black', zorder=1)
ax.barh(y_positions + bar_height / 2,df['Patches (K)'],height=bar_height,color=patch_colors,zorder=2)
# for i, row in df.iterrows():
#     y = i + bar_height / 2
#     x_start = (0.02 * row['Patches (K)'])
#     # ax.text(x_start, y, row['Training Strategy'], va='center', ha='left', fontsize=16, color='black', zorder=3)
ax.set_yticks(y_positions)
ax.set_yticklabels(df['Model'], fontsize=16)
ax.set_xscale('log')
ax.set_xlabel('Training Data Scale', fontsize=16)
ax.set_ylabel('Model', fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=16)
marker_styles = {'Tile Level': {'marker': 's', 'color': "#0A1156"},'Slide Level': {'marker': 'D', 'color': "#CD8E1A"}}
max_total_size_log = np.log10(df['Total Size (K)'].max())
# for i, row in df.iterrows():
#     y = i + bar_height / 2
#     level_info = marker_styles[row['Level']]
#     ax.plot(max_total_size_log - 0.2, y, marker=level_info['marker'], markersize=10, markerfacecolor='white' if level_info['marker'] == 's' else level_info['color'],
#         markeredgecolor='black', linestyle='None', clip_on=False, zorder=3)

bar_handles = [patches.Patch(facecolor="#4B4949", hatch='///', edgecolor='black', label="Number of WSIs"),
    patches.Patch(facecolor="#CFCBCB", label="Number of Patches")]
symbol_handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='black', markersize=10, label='Tile Level Pretraining'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=marker_styles['Slide Level']['color'], markeredgecolor='black', markersize=10, label='Slide Level Pretraining')]
handles = bar_handles
labels = [h.get_label() for h in handles]
unique = OrderedDict()
for label, handle in zip(labels, handles):
    if label not in unique:
        unique[label] = handle
ax.legend(unique.values(), unique.keys(), loc='lower right', bbox_to_anchor=(1.0, 0), fontsize=16)
plt.tight_layout()
# plt.savefig(os.path.join(FIG_DIR, 'am_plot_with_design_blank.tiff'), dpi=300)
plt.show()
plt.close()


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import OrderedDict

# --- Data Setup ---
data = [
    {'Model': 'Lunit', 'Patches (M)': 33, 'WSIs (K)': 37, 'Training Strategy': 'ViT-S + DINO', 'Level': 'Tile Level'},
    {'Model': 'GigaPath', 'Patches (M)': 1300, 'WSIs (K)': 171, 'Training Strategy': 'ViT-G + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CTransPath', 'Patches (M)': 16, 'WSIs (K)': 32, 'Training Strategy': 'ViT-S + MoCoV3', 'Level': 'Tile Level'},
    {'Model': 'Hibou', 'Patches (M)': 1200, 'WSIs (K)': 1139, 'Training Strategy': 'ViT-L + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'HOptimus1', 'Patches (M)': 1000, 'WSIs (K)': 1000, 'Training Strategy': 'ViT-G + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'Virchow2', 'Patches (M)': 2000, 'WSIs (K)': 3100, 'Training Strategy': 'ViT-H + DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CONCH1.5', 'Patches (M)': 1.26, 'WSIs (K)': 335, 'Training Strategy': 'ViT-L+ CoCa', 'Level': 'Tile Level'},
    {'Model': 'UNI2-h', 'Patches (M)': 200, 'WSIs (K)': 350, 'Training Strategy': 'ViT-H+ DINOv2', 'Level': 'Tile Level'},
    {'Model': 'CHIEF*', 'Patches (M)': 16, 'WSIs (K)': 60, 'Training Strategy': 'Attention Aggregation', 'Level': 'Slide Level'},
    {'Model': 'PRISM*', 'Patches (M)': 2000, 'WSIs (K)': 1500, 'Training Strategy': 'Perceiver network', 'Level': 'Slide Level'},
    {'Model': 'TITAN*', 'Patches (M)': 1.26, 'WSIs (K)': 335, 'Training Strategy': 'Attention + linear bias', 'Level': 'Slide Level'}
]
df = pd.DataFrame(data)

# --- Data Processing ---
df['Patches (K)'] = df['Patches (M)'] * 1000
df['Total Size (K)'] = df['Patches (K)'] + df['WSIs (K)']
df = df.sort_values(by='Total Size (K)').reset_index(drop=True)

# --- Color and Style Mapping ---
model_color_map = {
    'Lunit': {'WSIs': '#1A5C14', 'Patches': '#99CC99'},
    'CTransPath': {'WSIs': '#8B0000', 'Patches': '#FFB6C1'},
    'CONCH1.5': {'WSIs': '#FF8C00', 'Patches': '#FFDAB9'},
    'Hibou': {'WSIs': '#003366', 'Patches': '#ADD8E6'},
    'HOptimus1': {'WSIs': '#005A31', 'Patches': '#A5D6A7'},
    'GigaPath': {'WSIs': '#8B4513', 'Patches': '#F4A460'},
    'UNI2-h': {'WSIs': '#800080', 'Patches': '#DA70D6'},
    'CHIEF*': {'WSIs': '#FF69B4', 'Patches': '#FFC0CB'},
    'TITAN*': {'WSIs': "#5F1B90", 'Patches': '#9370DB'},
    'PRISM*': {'WSIs': '#1E90FF', 'Patches': '#87CEFA'},
    'Virchow2': {'WSIs': '#696969', 'Patches': '#D3D3D3'}
}

# Generate color lists based on the sorted dataframe
wsi_colors = [model_color_map[model]['WSIs'] for model in df['Model']]
patch_colors = [model_color_map[model]['Patches'] for model in df['Model']]

marker_styles = {
    'Tile Level': {'marker': 's', 'color': "#0A1156"},
    'Slide Level': {'marker': 'D', 'color': "#CD8E1A"}
}

# --- Plotting ---5
plt.rcParams['hatch.linewidth'] = 1.0  # Make hatch lines thicker
fig, ax = plt.subplots(figsize=(8, 8))
bar_height = 0.9
# --- CORRECTION: Add vertical spacing between model groups to prevent overlap ---
group_spacing = 1.9
y_positions = np.arange(len(df)) * group_spacing

# Main loop to draw bars, text, and markers for each model
for i, row in df.iterrows():
    # WSIs bar (lower, hatched)
    ax.barh(y_positions[i] - bar_height / 2, row['WSIs (K)'], height=bar_height, color=wsi_colors[i], hatch='///',
            edgecolor='black', zorder=2)

    # Patches bar (upper, plain)
    ax.barh(y_positions[i] + bar_height / 2, row['Patches (K)'], height=bar_height, color=patch_colors[i],
            edgecolor='black', zorder=2)
    
    # Add Training Strategy text on the Patches bar
    y_text = y_positions[i] + bar_height / 2
    # Place text slightly inside the bar for padding, unless the bar is too small
    x_text = row['Patches (K)'] * 0.015  # 2% inside the Patches bar
    # Only add text if there's space
    if x_text > 0:
       ax.text(x_text, y_text, row['Training Strategy'], va='center', ha='left', fontsize=16, color='black', zorder=3)


# --- Formatting and Legend ---
ax.set_yticks(y_positions)
ax.set_yticklabels(df['Model'], fontsize=16)
ax.set_xscale('log')
ax.set_xlabel('Training Data Scale', fontsize=18)
ax.set_ylabel('Model', fontsize=18)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)

# Define handles for the legend
bar_handles = [
    mpatches.Patch(facecolor="#4B4949", hatch='///', edgecolor='black', label="Number of WSIs"),
    mpatches.Patch(facecolor="#CFCBCB", label="Number of Patches")
]


# Combine handles for a single legend
handles = bar_handles 
labels = [h.get_label() for h in handles]

ax.legend(handles, labels, loc='lower right', fontsize=16)
fig.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for markers
plt.savefig(os.path.join(FIG_DIR, 'am_plot_with_design_new.png'), dpi=300)
plt.show()
plt.close()
