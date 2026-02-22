import numpy as np
import matplotlib.pyplot as plt

def plot_history(history, title='', axs=None, exp_name=""):
    """
    Plot training history of the model.
    """
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label='train' + exp_name)
    ax1.plot(history.history['val_loss'], label='val' + exp_name)
    ax1.set_title('loss')
    ax1.legend()

    return (ax1)

def plot_pred_rv(true_rv, predict_rv):
    """
    Plot the predicted vs true Realized Volatility values over time segments.
    """
    # Calculate the number of subplots needed
    num_plots = len(true_rv) // 50
    rows, cols = 7, 2  # Setting up a 7x2 grid
    total_plots = rows * cols

    # Calculate the number of rows needed based on the actual number of plots
    num_actual_rows = (num_plots + cols - 1) // cols  # Ceiling division

    # Adjust figure size based on the number of rows needed
    fig, axes = plt.subplots(num_actual_rows, cols, figsize=(15, num_actual_rows * 5+5))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Keep track of the current plot index
    plot_index = 0

    y_min = 0
    y_max = 0.03

    # Loop through the segments and create subplots
    for i in range(num_plots):
        # Skip segment 8 (index 7) or 6 (index 5) depending on original logic
        if i == 7 or i == 5:
            continue
        
        start_idx = i * 50
        end_idx = start_idx + 50
        indices = np.arange(start_idx, end_idx)

        # Determine the y-axis limits based on the data of the current segment
        segment_true_rv = true_rv[start_idx:end_idx]
        segment_predict_rv = predict_rv[start_idx:end_idx]

        # Define the y-ticks
        y_ticks = np.linspace(y_min, y_max, num=5)

        axes[plot_index].plot(indices, segment_true_rv, marker='o', linestyle='-', color='b', label='True RV')
        axes[plot_index].plot(indices, segment_predict_rv, marker='x', linestyle='-', color='r', label='Predict RV')
        axes[plot_index].set_xlabel('Day in Test Set')
        axes[plot_index].set_ylabel('Realized Volatility')
        axes[plot_index].set_title(f'Segment {i+1}')
        axes[plot_index].legend()
        axes[plot_index].grid(True)

        # Set the y-axis limits and ticks
        axes[plot_index].set_ylim([y_min, y_max])
        axes[plot_index].set_yticks(y_ticks)

        plot_index += 1

    # Hide any unused subplots
    for j in range(plot_index, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)  # Add space between subplots

    # Save the figure with higher resolution
    plt.savefig("amgen_good_results.png", dpi=300)
    plt.show()
