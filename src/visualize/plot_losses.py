import re
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Plot the losses of a pretraining run')

    parser.add_argument('--log', type=str, default='.slurm_logs/1b_full_70m_410m_step140000-1e-3.out',
                        help='Path to the log file')
    
    parser.add_argument('--plot_title', type=str, default='Pretraining loss for 1b_full_70m_410m_step140000-1e-3',
                        help='Title of the plot')

    parser.add_argument('--output', type=str, default='plots/1b_full_70m_410m_step140000-1e-3.png',
                        help='Path to save the plot')

    return parser.parse_args()

def read_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    losses = []
    for line in lines:
        if 'loss' in line:
            loss = re.search(r'loss: ([0-9]+\.[0-9]+)', line).group(1)
            losses.append(float(loss))

    return losses

def plot_losses(losses, output_path, title):
    plt.plot(losses, label='Pretraining loss')
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path)

def main():
    args = parse_args()
    losses = read_log(args.log)
    plot_losses(losses, args.output, args.plot_title)

if __name__ == '__main__':
    main()