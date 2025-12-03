import textwrap
import matplotlib.pyplot as plt
import seaborn as sns

def print_section(title: str):
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")

def wrap_print(text: str, width: int = 100):
    print(textwrap.fill(text, width=width))

def set_plot_style():
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 5)
