import sys
import json
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import cartopy.crs as ccrs
from utils.draw_functions import draw_poland
from models.data_processor import DataProcessor
import os


plt.style.use("ggplot")
sys.path.append("..")


class Visualization:
    def __init__(self):
        self.plots_data = {}
        self.feature_list = ["t2m", "sp", "tcc", "u10", "v10", "tp"]


        self.colors = {
            "simple-linear":'#377eb8',
            "linear": '#ff7f00',
            "lgbm": '#4daf4a',
            "gnn": 'black',
            "cnn": '#a65628',
        }

        self.error_maps = []

    def read_plots_from_json(self, file_name="modelsplots.json"):
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
    
    def plot_not_normalized_data_sequence(self, for_features=False, one_plot=False, save=None):
        if for_features == False:
            # Iterate over each baseline_type and plot the data
            for i, baseline_type in enumerate(self.plots_data.keys()):
                # Check if the baseline_type has the "not_normalized_plot_sequence" field
                if "not_normalized_plot_sequence" in self.plots_data[baseline_type]:
                    # Create a new plot for each baseline_type
                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Get the not_normalized_plot_sequence for the current baseline_type
                    not_normalized_plot_sequence = self.plots_data[baseline_type][
                        "not_normalized_plot_sequence"
                    ]

                # Plot the data on the subplot
                ax.plot(
                    list(not_normalized_plot_sequence.keys()),
                    list(not_normalized_plot_sequence.values()),
                    "-o",
                    # color=self.colors[baseline_type],
                )

                # Set the title and legend for each subplot
                ax.set_title(f"Not Normalized Data Sequence - {baseline_type}")
                ax.legend(self.feature_list)
                ax.set_xlabel("Sequence Length")
                ax.set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

                # Adjust layout for better spacing
                plt.tight_layout()

                if save is not None:
                    plt.savefig(f'{baseline_type}_{save}.pdf')

                    # Show the plot for each baseline_type
                plt.show()

                
        elif one_plot == True and for_features == True:
            num_plots = len(self.plots_data)
            num_cols = 2  # Number of columns in the grid
            num_rows = (
                num_plots + num_cols - 1
            ) // num_cols  # Number of rows in the grid
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 18))
            # initialize data structures
            baselines = []
            plot_dict = {}
            for feature in self.feature_list:
                plot_dict[feature] = []
            
            # transform data fromat from features per baseline to baselines per feature
            for i, baseline_type in enumerate(self.plots_data.keys()):
                for_feature_dict = {}
                baselines.append(baseline_type)

                for feature in self.feature_list:
                    for_feature_dict[feature] = []

                if "not_normalized_plot_sequence" in self.plots_data[baseline_type]:
                    not_normalized_plot_sequence = self.plots_data[baseline_type]["not_normalized_plot_sequence"]

                    for i, feature in enumerate(self.feature_list):
                        for key in not_normalized_plot_sequence.keys():
                            for_feature_dict[feature].append(list(not_normalized_plot_sequence[key])[i])
                    
                    for feature in self.feature_list:
                        plot_dict[feature].append(for_feature_dict[feature])

            # plot the transformed data
            for j, feature in enumerate(self.feature_list):
                row = j // num_cols
                col = j %num_cols
                legend = []
                for i in range(len(baselines)):
                    if "not_normalized_plot_sequence" in self.plots_data[baseline_type] and plot_dict[feature][i]:  # Check if not_normalized_plot_sequence is not empty
                        legend.append(baselines[i])
                        axes[row, col].plot(list(not_normalized_plot_sequence.keys()), plot_dict[feature][i], "-o", color=self.colors[baselines[i]])
                        axes[row, col].set_title(f"Not Normalized Data Sequence Length - {feature}", fontsize=10)
                        axes[row, col].legend(legend, fontsize=8)
                        axes[row, col].set_xlabel("Sequence Length", fontsize=8)
                        axes[row, col].set_ylabel("Rmse", fontsize=8)
            
            if save is not None:
                    plt.savefig(f'{save}.pdf')
            plt.show()

            
        else:
            # initialize data structures
            baselines = []
            plot_dict = {}
            for feature in self.feature_list:
                plot_dict[feature] = []

            # transform data fromat from features per baseline to baselines per feature
            for i, baseline_type in enumerate(self.plots_data.keys()):
                for_feature_dict = {}
                baselines.append(baseline_type)

                for feature in self.feature_list:
                    for_feature_dict[feature] = []

                if "not_normalized_plot_sequence" in self.plots_data[baseline_type]:
                    not_normalized_plot_sequence = self.plots_data[baseline_type][
                        "not_normalized_plot_sequence"
                    ]

                    for i, feature in enumerate(self.feature_list):
                        for key in not_normalized_plot_sequence.keys():
                            for_feature_dict[feature].append(
                                list(not_normalized_plot_sequence[key])[i]
                            )

                    for feature in self.feature_list:
                        plot_dict[feature].append(for_feature_dict[feature])

            # plot the transformed data
            for feature in self.feature_list:
                fig, ax = plt.subplots(figsize=(10, 8))
                for i in range(len(baselines)):
                    ax.plot(
                        list(not_normalized_plot_sequence.keys()), plot_dict[feature][i]
                    )
                    ax.set_title(f"Not Normalized Data Sequence - {feature}")
                ax.legend(baselines)
                ax.set_xlabel("Sequence Length")
                ax.set_ylabel("Rmse")

                if save is not None:
                    plt.savefig(f'{feature}_{save}.pdf')
                plt.show()

                


    def plot_not_normalized_data_fh(self, for_features=False, one_plot=False, save=None):
        if for_features == False:
            # Iterate over each baseline_type and plot the data
            for i, baseline_type in enumerate(self.plots_data.keys()):
                # Check if the baseline_type has the "not_normalized_plot_sequence" field
                if "not_normalized_plot_fh" in self.plots_data[baseline_type]:
                    # Create a new plot for each baseline_type
                    fig, ax = plt.subplots(figsize=(10, 8))

                # Get the not_normalized_plot_sequence for the current baseline_type
                not_normalized_plot_sequence = self.plots_data[baseline_type][
                    "not_normalized_plot_fh"
                ]

                # Plot the data on the single plot
                ax.plot(
                    list(not_normalized_plot_sequence.keys()),
                    list(not_normalized_plot_sequence.values()),
                    "-o",
                )

                # Set the title and legend for each plot
                ax.set_title(
                    f"Not Normalized Data Forcasting Horizon - {baseline_type}"
                )
                ax.legend(self.feature_list)
                ax.set_xlabel("Forcasting Horizon")
                ax.set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

                    # Show the plot for each baseline_type
                if save is not None:
                    plt.savefig(f'{baseline_type}_{save}.pdf')
                plt.show()

                
        elif one_plot == True and for_features == True:
            num_plots = len(self.plots_data)
            num_cols = 2  # Number of columns in the grid
            num_rows = (
                num_plots + num_cols - 1
            ) // num_cols  # Number of rows in the grid
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 18))
            # initialize data structures
            baselines = []
            plot_dict = {}
            for feature in self.feature_list:
                plot_dict[feature] = []
            
            # transform data fromat from features per baseline to baselines per feature
            for i, baseline_type in enumerate(self.plots_data.keys()):
                for_feature_dict = {}
                baselines.append(baseline_type)

                for feature in self.feature_list:
                    for_feature_dict[feature] = []

                if "not_normalized_plot_fh" in self.plots_data[baseline_type]:
                    
                    not_normalized_plot_sequence = self.plots_data[baseline_type]["not_normalized_plot_fh"]
                    for i, feature in enumerate(self.feature_list):
                        
                        for key in not_normalized_plot_sequence.keys():
                            # print(baseline_type)
                            for_feature_dict[feature].append(list(not_normalized_plot_sequence[key])[i])
                    
                    for feature in self.feature_list:
                        # print(baseline_type)
                        # print(for_feature_dict[feature])
                        plot_dict[feature].append(for_feature_dict[feature])

            # plot the transformed data
            for j, feature in enumerate(self.feature_list):
                row = j // num_cols
                col = j % num_cols
                legend = []
                for i in range(len(baselines)):
                    
                    if "not_normalized_plot_fh" in self.plots_data[baseline_type] and plot_dict[feature][i]:
                        legend.append(baselines[i])
                        axes[row, col].plot(list(not_normalized_plot_sequence.keys()), plot_dict[feature][i], "-o", color=self.colors[baselines[i]])
                        axes[row, col].set_title(f"Not Normalized Data Forcasting Horizon - {feature}", fontsize=10)
                        axes[row, col].legend(legend, fontsize=8)
                        axes[row, col].set_xlabel("Forcasting Horizon", fontsize=8)
                        axes[row, col].set_ylabel("Rmse", fontsize=8)

            if save is not None:
                    plt.savefig(f'{save}.pdf')

            plt.show()

            
        else:
            # initialize data structures
            baselines = []
            plot_dict = {}
            for feature in self.feature_list:
                plot_dict[feature] = []

            # transform data fromat from features per baseline to baselines per feature
            for i, baseline_type in enumerate(self.plots_data.keys()):
                for_feature_dict = {}
                baselines.append(baseline_type)

                for feature in self.feature_list:
                    for_feature_dict[feature] = []

                if "not_normalized_plot_fh" in self.plots_data[baseline_type]:
                    not_normalized_plot_sequence = self.plots_data[baseline_type][
                        "not_normalized_plot_fh"
                    ]

                    for i, feature in enumerate(self.feature_list):
                        for key in not_normalized_plot_sequence.keys():
                            for_feature_dict[feature].append(
                                list(not_normalized_plot_sequence[key])[i]
                            )

                    for feature in self.feature_list:
                        plot_dict[feature].append(for_feature_dict[feature])

            # plot the transformed data
            for feature in self.feature_list:
                fig, ax = plt.subplots(figsize=(10, 8))
                for i in range(len(baselines)):
                    ax.plot(
                        list(not_normalized_plot_sequence.keys()), plot_dict[feature][i]
                    )
                    ax.set_title(f"Not Normalized Data Forcasting Horizon - {feature}")
                ax.legend(baselines)
                ax.set_xlabel("Forcasting Horizon")
                ax.set_ylabel("Rmse")

                if save is not None:
                    plt.savefig(f'{feature}_{save}.pdf')
                plt.show()

                

    def plot_data_sequence(self, one_plot=False, save=None):
        if one_plot:
            # Create a single plot with multiple lines and legend
            fig, ax = plt.subplots(figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for baseline_type in self.plots_data.keys():
                # Get the sequence_plot_x and sequence_plot_y for the current baseline_type
                sequence_plot_x = self.plots_data[baseline_type]["sequence_plot_x"]
                sequence_plot_y = self.plots_data[baseline_type]["sequence_plot_y"]

                # Plot the data on the single plot
                ax.plot(sequence_plot_x, sequence_plot_y, "-o", label=baseline_type, color=self.colors[baseline_type])
                ax.set_xlabel("Sequence Length")
                ax.set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

            # Set the title and legend
            ax.set_title("Data Sequence")
            ax.legend()

            if save is not None:
                plt.savefig(f'{save}.pdf')

            # Show the plot
            plt.show()

            
        else:
            # Create a grid of subplots
            num_plots = len(self.plots_data)
            num_cols = 2  # Number of columns in the grid
            num_rows = (
                num_plots + num_cols - 1
            ) // num_cols  # Number of rows in the grid
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
                axes[row, col].plot(sequence_plot_x, sequence_plot_y, "-o", color=self.colors[baseline_type])
                axes[row, col].set_title(
                    baseline_type
                )  # Set the title as the baseline_type
                axes[row, col].set_xlabel("Sequence Length")
                axes[row, col].set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

            # Adjust the layout and spacing of the subplots
            plt.tight_layout()

            if save is not None:
                plt.savefig(f'{save}.pdf')

            # Show the plot
            plt.show()

            

    def plot_data_sequence_time(self, one_plot=False, save=None):
        if one_plot:
            # Create a single plot with multiple lines and legend
            fig, ax = plt.subplots(figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for baseline_type in self.plots_data.keys():
                if baseline_type in ["simple-linear", "linear", "lgbm"]:
                    # Get the sequence_plot_x and sequence_plot_time for the current baseline_type
                    sequence_plot_x = self.plots_data[baseline_type]["sequence_plot_x"]
                    sequence_plot_time = self.plots_data[baseline_type][
                        "sequence_plot_time"
                    ]

                    # Plot the data on the single plot
                    ax.plot(
                        sequence_plot_x, sequence_plot_time, "-o", label=baseline_type, color=self.colors[baseline_type]
                    )

            # Set the title and legend
            ax.set_title("Data Sequence Time (Baselines)")
            ax.legend()
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Time [s]")

            if save is not None:
                plt.savefig(f'baselines_{save}.pdf')

            # Show the plot
            plt.show()

            

            # Create a single plot with multiple lines and legend
            fig, ax = plt.subplots(figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for baseline_type in self.plots_data.keys():
                if baseline_type in ["gnn", "cnn"]:
                    # Get the sequence_plot_x and sequence_plot_time for the current baseline_type
                    sequence_plot_x = self.plots_data[baseline_type]["sequence_plot_x"]
                    sequence_plot_time = self.plots_data[baseline_type][
                        "sequence_plot_time"
                    ]

                    # Plot the data on the single plot
                    ax.plot(
                        sequence_plot_x, sequence_plot_time, "-o", label=baseline_type, color=self.colors[baseline_type]
                    )

            # Set the title and legend
            ax.set_title("Data Sequence Time (Nets)")
            ax.legend()
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Time [s]")


            if save is not None:
                plt.savefig(f'nets_{save}.pdf')

            # Show the plot
            plt.show()

            
        else:
            # Create a grid of subplots
            num_plots = len(self.plots_data)
            num_cols = 2  # Number of columns in the grid
            num_rows = (
                num_plots + num_cols - 1
            ) // num_cols  # Number of rows in the grid
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for i, baseline_type in enumerate(self.plots_data.keys()):
                # Get the sequence_plot_x and sequence_plot_time for the current baseline_type
                sequence_plot_x = self.plots_data[baseline_type]["sequence_plot_x"]
                sequence_plot_time = self.plots_data[baseline_type][
                    "sequence_plot_time"
                ]

                # Determine the subplot position based on the current index
                row = i // num_cols
                col = i % num_cols

                # Plot the data on the current subplot
                axes[row, col].plot(sequence_plot_x, sequence_plot_time, "-o", color=self.colors[baseline_type])
                axes[row, col].set_title(
                    baseline_type
                )  # Set the title as the baseline_type
                axes[row, col].set_xlabel("Sequence Length")
                axes[row, col].set_ylabel("Time [s]")

            # Adjust the layout and spacing of the subplots
            plt.tight_layout()

            if save is not None:
                plt.savefig(f'{save}.pdf')

            # Show the plot
            plt.show()

            

    def plot_data_fh(self, one_plot=False, save=None):
        if one_plot:
            # Create a single plot with multiple lines and legend
            fig, ax = plt.subplots(figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for baseline_type in self.plots_data.keys():
                # Get the fh_plot_x and fh_plot_y for the current baseline_type
                fh_plot_x = self.plots_data[baseline_type]["fh_plot_x"]
                fh_plot_y = self.plots_data[baseline_type]["fh_plot_y"]

                # Plot the data on the single plot
                ax.plot(fh_plot_x, fh_plot_y, "-o", label=baseline_type, color=self.colors[baseline_type])

            # Set the title and legend
            ax.set_title("Data FH")
            ax.legend()
            ax.set_xlabel("Forcasting Horizon")
            ax.set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

            if save is not None:
                plt.savefig(f'{save}.pdf')

            # Show the plot
            plt.show()

            
        else:
            # Create a grid of subplots
            num_plots = len(self.plots_data)
            num_cols = 2  # Number of columns in the grid
            num_rows = (
                num_plots + num_cols - 1
            ) // num_cols  # Number of rows in the grid
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
                axes[row, col].plot(fh_plot_x, fh_plot_y, "-o", color=self.colors[baseline_type])
                axes[row, col].set_title(
                    baseline_type
                )  # Set the title as the baseline_type
                axes[row, col].set_xlabel("Forcasting Horizon")
                axes[row, col].set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

            # Adjust the layout and spacing of the subplots
            plt.tight_layout()

            if save is not None:
                plt.savefig(f'{save}.pdf')

            # Show the plot
            plt.show()

            

    def plot_months(self, one_plot=False, save=None):
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        if one_plot:
            # Create a single plot with multiple bars and legend
            fig, ax = plt.subplots(figsize=(15, 8))  # Adjust the width of the plot
            n = 12
            width = 0.2  # Adjust the width of each bar
            i = 0
            for baseline_type in self.plots_data.keys():
                month_plot_x = list(range(1, 13))
                month_plot_y = list(
                    self.plots_data[baseline_type]["month_error"].values()
                )
                ind = np.arange(max(month_plot_x))
                if len(month_plot_y) == 12:
                    
                    ax.bar(ind+width*i, month_plot_y, width, label=baseline_type, color=self.colors[baseline_type])
                    i+=1

            # Set the title and legend
            plt.xticks(
                ind + width * (i - 1) / 2, months
            )  # Adjust the position of x-ticks
            ax.set_title("Monthly errors")
            ax.legend()
            ax.set_xlabel("Months")
            ax.set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

            # Show the plot
            plt.xticks(
                ind + width * (i - 1) / 2, months
            )  # Adjust the position of x-ticks


            if save is not None:
                plt.savefig(f'{save}.pdf')
            plt.show()

            

    def plot_data_fh_time(self, one_plot=False, save=None):
        if one_plot:
            # Create a single plot with multiple lines and legend
            fig, ax = plt.subplots(figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for baseline_type in self.plots_data.keys():
                if baseline_type in ["simple-linear", "linear", "lgbm"]:
                    # Get the fh_plot_x and fh_plot_time for the current baseline_type
                    fh_plot_x = self.plots_data[baseline_type]["fh_plot_x"]
                    fh_plot_time = self.plots_data[baseline_type]["fh_plot_time"]

                    # Plot the data on the single plot
                    ax.plot(fh_plot_x, fh_plot_time, "-o", label=baseline_type, color=self.colors[baseline_type])

            # Set the title and legend
            ax.set_title("Data FH Time (Baselines)")
            ax.legend()
            ax.set_xlabel("Forcasting Horizon")
            ax.set_ylabel("Time [s]")
            if save is not None:
                plt.savefig(f'baselines_{save}.pdf')
            # Show the plot
            plt.show()

            

            fig, ax = plt.subplots(figsize=(10, 8))

            # Iterate over each baseline_type and plot the data
            for baseline_type in self.plots_data.keys():
                if baseline_type in ["gnn", "cnn"]:
                    # Get the fh_plot_x and fh_plot_time for the current baseline_type
                    fh_plot_x = self.plots_data[baseline_type]["fh_plot_x"]
                    fh_plot_time = self.plots_data[baseline_type]["fh_plot_time"]

                    # Plot the data on the single plot
                    ax.plot(fh_plot_x, fh_plot_time, "-o", label=baseline_type, color=self.colors[baseline_type])

            # Set the title and legend
            ax.set_title("Data FH Time (Nets)")
            ax.legend()
            ax.set_xlabel("Forcasting Horizon")
            ax.set_ylabel("Time [s]")

            if save is not None:
                plt.savefig(f'nets_{save}.pdf')

            # Show the plot
            plt.show()

            
        else:
            # Create a grid of subplots
            num_plots = len(self.plots_data)
            num_cols = 2  # Number of columns in the grid
            num_rows = (
                num_plots + num_cols - 1
            ) // num_cols  # Number of rows in the grid
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
                axes[row, col].plot(fh_plot_x, fh_plot_time, "-o", color=self.colors[baseline_type])
                axes[row, col].set_title(
                    baseline_type
                )  # Set the title as the baseline_type
                axes[row, col].set_xlabel("Forcasting Horizon")
                axes[row, col].set_ylabel("Time [s]")
            # Adjust the layout and spacing of the subplots
            plt.tight_layout()
            if save is not None:
                plt.savefig(f'{save}.pdf')
            # Show the plot
            plt.show()

            

    def plot_gnn_alpha(self, save=None):
        # Create a single plot with multiple lines and legend
        fig, ax = plt.subplots(figsize=(10, 8))

        # Iterate over each baseline_type and plot the data
        for baseline_type in self.plots_data.keys():
            if baseline_type == "gnn":
                alpha_plot_x = self.plots_data[baseline_type]["gnn_alpha_plot_x"]
                alpha_plot_y = self.plots_data[baseline_type]["gnn_alpha_plot_y"]

                # Plot the data on the single plot
                ax.plot(alpha_plot_x, alpha_plot_y, "-o", label=baseline_type, color=self.colors[baseline_type])
                ax.set_xlabel("Alpha")
                ax.set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

        # Set the title and legend
        ax.set_title("Alpha for Gnn and Tigge mix")
        ax.legend()

        if save is not None:
            plt.savefig(f'{save}.pdf')

        # Show the plot
        plt.show()

        

    def plot_gnn_layers(self, save=None):
        # Create a single plot with multiple lines and legend
        fig, ax = plt.subplots(figsize=(10, 8))

        # Iterate over each baseline_type and plot the data
        for baseline_type in self.plots_data.keys():
            if baseline_type == "gnn":
                cell_plot_x = self.plots_data[baseline_type]["gnn_cell_plot_x"]
                cell_plot_y = self.plots_data[baseline_type]["gnn_cell_plot_y"]

                # Plot the data on the single plot
                ax.plot(cell_plot_x, cell_plot_y, "-o", label=baseline_type, color=self.colors[baseline_type])
                ax.set_xlabel("Number of Graph Cells")
                ax.set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

        # Set the title and legend
        ax.set_title("Gnn Graph Cells")
        ax.legend()

        if save is not None:
            plt.savefig(f'{save}.pdf')

        # Show the plot
        plt.show()

        

    def plot_error_maps(self):
        for baseline_type in self.plots_data.keys():
            if os.path.exists(f"./{baseline_type}/error_maps.npy"):
                error_maps = np.load(f"./{baseline_type}/error_maps.npy")


                lat_span, lon_span, spatial_limits = DataProcessor.get_spatial_info()
                spatial = {
                    "lat_span": lat_span,
                    "lon_span": lon_span,
                    "spatial_limits": spatial_limits,
                }
                fig, axs = plt.subplots(
                len(self.feature_list),
                # 1,
                figsize=(10, 12),
                subplot_kw={"projection": ccrs.Mercator(central_longitude=40)},
                # constrained_layout=True
                )
                for j, feature_name in enumerate(self.feature_list):
                    ax = axs[j]
                    title = rf"$|(X - \hat{{X}})^2|_{{{feature_name}}}$"
                    value = error_maps[j]
                    cmap = "binary"
                    draw_poland(ax, value, title, cmap, **spatial)
                fig.suptitle(f"{baseline_type} error maps", x=0.7, y=0.95, weight="bold")
                