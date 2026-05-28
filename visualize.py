import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI环境也能保存图片
import matplotlib.pyplot as plt


def plot_curves(train_losses, test_losses, train_accs, test_accs, num_epochs, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='crimson', linewidth=2)
    axes[0].plot(range(1, num_epochs + 1), test_losses, color='blue', ls='--', label='Test Loss', linewidth=2)
    axes[0].set_title('Loss over Epochs', fontsize=12)
    axes[0].set_xlabel('Epochs', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    train_acc_pct = [a * 100 for a in train_accs]
    test_acc_pct = [a * 100 for a in test_accs]
    axes[1].plot(range(1, num_epochs + 1), train_acc_pct, label='Train Accuracy', color='red', linewidth=2)
    axes[1].plot(range(1, num_epochs + 1), test_acc_pct, label='Test Accuracy', color='royalblue', linewidth=2)
    axes[1].set_title('Accuracy over Epochs', fontsize=12)
    axes[1].set_xlabel('Epochs', fontsize=10)
    axes[1].set_ylabel('Accuracy (%)', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_all_channels(model, single_image, save_path=None):
    import cupy as cp

    if len(single_image.shape) == 3:
        current_x = single_image.reshape(1, *single_image.shape)
    elif len(single_image.shape) == 2:
        current_x = single_image.reshape(1, 1, *single_image.shape)
    else:
        current_x = single_image

    for i, layer in enumerate(model.layers):
        current_x = layer.forward(current_x)
        layer_name = layer.__class__.__name__

        if len(current_x.shape) == 4:
            num_channels = current_x.shape[1]
            feat_maps = current_x[0].get() if hasattr(current_x, 'get') else current_x[0]

            cols = min(8, num_channels)
            rows = int(np.ceil(num_channels / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
            fig.suptitle(
                f"Layer {i+1}: {layer_name} | Channels: {num_channels} | Shape: {feat_maps.shape[1:]}",
                fontsize=12, fontweight='bold', y=0.98
            )

            if num_channels == 1:
                axes = np.array([axes])
            else:
                axes = axes.flatten()

            for c in range(num_channels):
                ax = axes[c]
                ax.imshow(feat_maps[c], cmap='viridis')
                ax.axis('off')
                ax.set_title(f"Ch {c}", fontsize=8)

            for extra in range(num_channels, len(axes)):
                axes[extra].axis('off')

            plt.tight_layout()
            if save_path:
                layer_path = save_path.replace('.png', f'_layer{i+1}_{layer_name}.png')
                plt.savefig(layer_path, dpi=150, bbox_inches='tight')
                print(f"特征图已保存: {layer_path}")
            else:
                plt.show()
            plt.close()
