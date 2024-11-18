import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = 'eval/Lazy Pretraining - combined sheet'
PPL_COLS = [
    'lambada_openai',
    # '4chan',
    'C4 100',
    'C4',
    'Dolma',
    # '100 PLs',
    '100 Subred',
    'Falcon',
    # 'Gab',
    'M2D2 S2ORC',
    'M2D2 Wiki',
    'Manosphere',
    'mC4',
    'PTB',
    'RedPajama',
    'Twitter AAE',
    'Wikitext-103'
]


def three_lines(df, col, title, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.title(title)

    plt.xlabel('Training step')
    plt.ylabel(col)
    plt.grid(True)

    # Draw 3 superimposed line plots for each training type (Target modules)
    for target_module in df['Target modules'].unique():
        # ignore nan values
        if target_module != target_module:
            continue
        print(target_module)
        target_module_df = df[df['Target modules'] == target_module]

        plt.plot(target_module_df['Base ckpt'].str.slice(4).astype(int), 
                 target_module_df[col], label=target_module)

    plt.gca().invert_yaxis()  # Flip the y-axis
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()



def radar_plot(df, cols, title, save_path=None):

    # Number of tasks
    num_tasks = len(cols)

    # Compute the angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    plt.title(title, size=15, y=1.1)

    # Add each target module's line
    for target_module in df['Target modules'].unique():
        # Ignore NaN values
        if pd.isna(target_module):

            print('nan!!')
            
            # If Base Model == 'Pythia-70m', add this line
            target_module_df = df[(df['Base ckpt'] == 'Pythia-70m') & (df['Target modules'].isna())]

            print(target_module_df)
            values = target_module_df[cols].values.flatten().tolist()
            print(f'target_module: {target_module}')
            print(f'values: {target_module_df[cols].values.flatten().tolist()}')
            ax.plot(angles, values, label='Pythia-70m-base')
            ax.fill(angles, values, alpha=0.05)

        # Extract the PPL values for the target module
        target_module_df = df[df['Target modules'] == target_module]
        values = target_module_df[cols].values.flatten().tolist()
        print(f'target_module: {target_module}')
        print(f'values: {target_module_df[cols].values.flatten().tolist()}')
        values += values[:1]  # Close the circle

        # Plot the line and fill the area
        ax.plot(angles, values, label=target_module)
        ax.fill(angles, values, alpha=0.05)

    # Set the labels for each task
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols)

    # Show the legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()




if __name__ == '__main__':


    # The first two rows are headers, so if the fist row is nan for a   
    # column, then the second row is the actual header
    df = pd.read_csv(f'{CSV}.csv', header=[0])

    print(df['Base ckpt'])

    # Include only the rows where the Base ckpt is 'step143000' or the Target modules is NaN
    df_step143000 = df[(df['Base ckpt'] == 'step143000') | (df['Target modules'].isna())]
    df_step142000 = df[(df['Base ckpt'] == 'step142000') | (df['Target modules'].isna())]
    df_step141000 = df[(df['Base ckpt'] == 'step141000') | (df['Target modules'].isna())]

    
    radar_plot(df_step143000, PPL_COLS[:8], 'Perplexity for step143000', 'plots/radar_plot_step143000.png')
    radar_plot(df_step142000, PPL_COLS[:8], 'Perplexity for step142000', 'plots/radar_plot_step142000.png')
    radar_plot(df_step141000, PPL_COLS[:8], 'Perplexity for step141000', 'plots/radar_plot_step141000.png')

    # three_lines(df, 'lambada_openai', 'Lambada OpenAI', 'plots/lambada_openai.png')


    # Draw 3 superimposed line plots for each training type (Target modules)
