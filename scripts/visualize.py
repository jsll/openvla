import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.gridspec as gridspec

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = plt.cm.jet(mask, cv2.COLORMAP_JET)[:,:,:3]
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def visualize_attention(
    attention_rollout,
    observation_image,
    observation_tokens,
    action_tokens,
    save_path: str = "attention_output/",
    title = ""
) -> plt.Figure:
    """
    Plots attention weights from readout tokens as a heatmap.
    
    Args:
        model: CrossFormerModel instance  
        observations: Dictionary of observations
        tasks: Dictionary of task specifications
        readout_name: Name of readout head to analyze
        save_path: Optional path to save visualization
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(15, 5))
    # Create a nested GridSpec
    batch_size = attention_rollout.shape[0]
    outer_grid = plt.GridSpec(batch_size, 3, width_ratios=[1, 1, 1.1])  # Slightly wider last column for colorbar
    axs = []
    for i in range(3):
        axs.append(fig.add_subplot(outer_grid[i]))


    images_per_readout = []
    for action_token in action_tokens:
        mask = attention_rollout[0, action_token, observation_tokens]
        mask = np.asarray(mask.reshape(14,14))
        mask = mask / np.max(mask)
        mask = cv2.resize(mask, (224, 224))
        images_per_readout.append(mask.copy())

    average_readout_image = np.asarray(images_per_readout).mean(0)

    #print(average_readout_image .shape)    
    #print(observation_image.shape)
      
    # Plot on the GridSpec subplots
    axs[0].imshow(observation_image, cmap='jet')
        
    axs[1].imshow(average_readout_image, cmap='jet')
        
    overlay = show_mask_on_image(observation_image, average_readout_image)
    im = axs[2].imshow(overlay, cmap='jet')

    for i in range(3):
        axs[i].axis('off')
 
    axs[0].set_title(title)
    axs[1].set_title('Attention Rollout')
    axs[2].set_title('Attention superimposed on image')

    # Add colorbar with specific spacing
    plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    plt.savefig(save_path+title+".png")

    return fig

