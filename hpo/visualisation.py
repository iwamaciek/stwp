import xarray as xr
import cfgrib
import numpy as np
import optuna
import sys
import matplotlib.pyplot as plt
from datetime import datetime

from tabulate import tabulate
from functools import partial
from sklearn.metrics import mean_squared_error
import sys

sys.path.append("..")

import json


class Visualization:
    def __init__(self):
        self.plots_data = {}
        self.feature_list = ["t2m", "sp", "tcc", "u10", "v10", "tp"]

    def read_plots_from_json(self):
        file_name = "modelsplots.json"

        try:
            with open(file_name, "r") as infile:
                self.plots_data = json.load(infile)
        except FileNotFoundError:
            pass

    def create_metrics_table(self):
        # Create an empty table
        table = []

        # Iterate over each baseline_type and create a row in the table
        for baseline_type in self.plots_data.keys():
            # Get the metrics for the current baseline_type
            metrics = self.plots_data[baseline_type]["metrics"]

            # Create a row with the baseline_type and metrics
            row = {"baseline_type": baseline_type, "metrics": metrics}

            # Add the row to the table
            table.append(row)
        
        print(tabulate(table, tablefmt="fancy_grid"))
    
    def create_metrics_for_scalers_table(self):
        # Create an empty table
        table = []

        # Iterate over each baseline_type and create a row in the table
        for baseline_type in self.plots_data.keys():
            # Get the metrics for the current baseline_type
            metrics = self.plots_data[baseline_type]["metrics_for_scalers"]

            # Create a row with the baseline_type and metrics
            row = {"baseline_type": baseline_type, "metrics_for_scalers": metrics}

            # Add the row to the table
            table.append(row)

        return table

    def plot_data_sequence(self, one_plot=False):
        if one_plot:
            # Create a single plot with multiple lines and legend
            fig, ax = plt.subplots(figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for baseline_type in self.plots_data.keys():
                # Get the sequence_plot_x and sequence_plot_y for the current baseline_type
                sequence_plot_x = self.plots_data[baseline_type]["sequence_plot_x"]
                sequence_plot_y = self.plots_data[baseline_type]["sequence_plot_y"]

                # Plot the data on the single plot
                ax.plot(sequence_plot_x, sequence_plot_y, label=baseline_type)

            # Set the title and legend
            ax.set_title("Data Sequence")
            ax.legend()

            # Show the plot
            plt.show()
        else:
            # Create a grid of subplots
            num_plots = len(self.plots_data)
            num_cols = 2  # Number of columns in the grid
            num_rows = (num_plots + num_cols - 1) // num_cols  # Number of rows in the grid
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for i, baseline_type in enumerate(self.plots_data.keys()):
                # Get the sequence_plot_x and sequence_plot_y for the current baseline_type
                sequence_plot_x = self.plots_data[baseline_type]["sequence_plot_x"]
                sequence_plot_y = self.plots_data[baseline_type]["sequence_plot_y"]

                # Determine the subplot position based on the current index
                row = i // num_cols
                col = i % num_cols

                # Plot the data on the current subplot
                axes[row, col].plot(sequence_plot_x, sequence_plot_y)
                axes[row, col].set_title(baseline_type)  # Set the title as the baseline_type

            # Adjust the layout and spacing of the subplots
            plt.tight_layout()

            # Show the plot
            plt.show()

    def plot_data_sequence_time(self, one_plot=False):
        if one_plot:
            # Create a single plot with multiple lines and legend
            fig, ax = plt.subplots(figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for baseline_type in self.plots_data.keys():
                # Get the sequence_plot_x and sequence_plot_time for the current baseline_type
                sequence_plot_x = self.plots_data[baseline_type]["sequence_plot_x"]
                sequence_plot_time = self.plots_data[baseline_type]["sequence_plot_time"]

                # Plot the data on the single plot
                ax.plot(sequence_plot_time, sequence_plot_x, label=baseline_type)

            # Set the title and legend
            ax.set_title("Data Sequence Time")
            ax.legend()

            # Show the plot
            plt.show()
        else:
            # Create a grid of subplots
            num_plots = len(self.plots_data)
            num_cols = 2  # Number of columns in the grid
            num_rows = (num_plots + num_cols - 1) // num_cols  # Number of rows in the grid
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for i, baseline_type in enumerate(self.plots_data.keys()):
                # Get the sequence_plot_x and sequence_plot_time for the current baseline_type
                sequence_plot_x = self.plots_data[baseline_type]["sequence_plot_x"]
                sequence_plot_time = self.plots_data[baseline_type]["sequence_plot_time"]

                # Determine the subplot position based on the current index
                row = i // num_cols
                col = i % num_cols

                # Plot the data on the current subplot
                axes[row, col].plot(sequence_plot_time, sequence_plot_x)
                axes[row, col].set_title(baseline_type)  # Set the title as the baseline_type

            # Adjust the layout and spacing of the subplots
            plt.tight_layout()

            # Show the plot
            plt.show()

    def plot_data_fh(self, one_plot=False):
        if one_plot:
            # Create a single plot with multiple lines and legend
            fig, ax = plt.subplots(figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for baseline_type in self.plots_data.keys():
                # Get the fh_plot_x and fh_plot_y for the current baseline_type
                fh_plot_x = self.plots_data[baseline_type]["fh_plot_x"]
                fh_plot_y = self.plots_data[baseline_type]["fh_plot_y"]

                # Plot the data on the single plot
                ax.plot(fh_plot_x, fh_plot_y, label=baseline_type)

            # Set the title and legend
            ax.set_title("Data FH")
            ax.legend()

            # Show the plot
            plt.show()
        else:
            # Create a grid of subplots
            num_plots = len(self.plots_data)
            num_cols = 2  # Number of columns in the grid
            num_rows = (num_plots + num_cols - 1) // num_cols  # Number of rows in the grid
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for i, baseline_type in enumerate(self.plots_data.keys()):
                # Get the fh_plot_x and fh_plot_y for the current baseline_type
                fh_plot_x = self.plots_data[baseline_type]["fh_plot_x"]
                fh_plot_y = self.plots_data[baseline_type]["fh_plot_y"]

                # Determine the subplot position based on the current index
                row = i // num_cols
                col = i % num_cols

                # Plot the data on the current subplot
                axes[row, col].plot(fh_plot_x, fh_plot_y)
                axes[row, col].set_title(baseline_type)  # Set the title as the baseline_type

            # Adjust the layout and spacing of the subplots
            plt.tight_layout()

            # Show the plot
            plt.show()


    def plot_data_fh_time(self, one_plot=False):
        if one_plot:
            # Create a single plot with multiple lines and legend
            fig, ax = plt.subplots(figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for baseline_type in self.plots_data.keys():
                # Get the fh_plot_x and fh_plot_time for the current baseline_type
                fh_plot_x = self.plots_data[baseline_type]["fh_plot_x"]
                fh_plot_time = self.plots_data[baseline_type]["fh_plot_time"]

                # Plot the data on the single plot
                ax.plot(fh_plot_time, fh_plot_x, label=baseline_type)

            # Set the title and legend
            ax.set_title("Data FH Time")
            ax.legend()

            # Show the plot
            plt.show()
        else:
            # Create a grid of subplots
            num_plots = len(self.plots_data)
            num_cols = 2  # Number of columns in the grid
            num_rows = (num_plots + num_cols - 1) // num_cols  # Number of rows in the grid
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for i, baseline_type in enumerate(self.plots_data.keys()):
                # Get the fh_plot_x and fh_plot_time for the current baseline_type
                fh_plot_x = self.plots_data[baseline_type]["fh_plot_x"]
                fh_plot_time = self.plots_data[baseline_type]["fh_plot_time"]

                # Determine the subplot position based on the current index
                row = i // num_cols
                col = i % num_cols

                # Plot the data on the current subplot
                axes[row, col].plot(fh_plot_time, fh_plot_x)
                axes[row, col].set_title(baseline_type)  # Set the title as the baseline_type

            # Adjust the layout and spacing of the subplots
            plt.tight_layout()

            # Show the plot
            plt.show()

        


