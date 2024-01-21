import sys
import json
import matplotlib.pyplot as plt
from tabulate import tabulate

plt.style.use("ggplot")
sys.path.append("..")


class Visualization:
    def __init__(self):
        self.plots_data = {}
        self.feature_list = ["t2m", "sp", "tcc", "u10", "v10", "tp"]

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
    
    def plot_not_normalized_data_sequence(self, for_features=False):
        if for_features == False:
            # Iterate over each baseline_type and plot the data
            for i, baseline_type in enumerate(self.plots_data.keys()):
                # Check if the baseline_type has the "not_normalized_plot_sequence" field
                if "not_normalized_plot_sequence" in self.plots_data[baseline_type]:
                    # Create a new plot for each baseline_type
                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Get the not_normalized_plot_sequence for the current baseline_type
                    not_normalized_plot_sequence = self.plots_data[baseline_type]["not_normalized_plot_sequence"]

                # Plot the data on the subplot
                ax.plot(
                    list(not_normalized_plot_sequence.keys()),
                    list(not_normalized_plot_sequence.values()),
                    "-o",
                )

                # Set the title and legend for each subplot
                ax.set_title(f"Not Normalized Data Sequence - {baseline_type}")
                ax.legend(self.feature_list)
                ax.set_xlabel("Sequence Length")
                ax.set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

        # Adjust layout for better spacing
        plt.tight_layout()

                    # Show the plot for each baseline_type
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
                    not_normalized_plot_sequence = self.plots_data[baseline_type]["not_normalized_plot_sequence"]

                    for i, feature in enumerate(self.feature_list):
                        for key in not_normalized_plot_sequence.keys():
                            for_feature_dict[feature].append(list(not_normalized_plot_sequence[key])[i])
                
                    for feature in self.feature_list:
                        plot_dict[feature].append(for_feature_dict[feature])


            # plot the transformed data
            for feature in self.feature_list:
                fig, ax = plt.subplots(figsize=(10, 8)) 
                for i in range(len(baselines)):
                    ax.plot(list(not_normalized_plot_sequence.keys()), plot_dict[feature][i])
                    ax.set_title(f"Not Normalized Data Sequence - {feature}")
                ax.legend(baselines)
                ax.set_xlabel("Sequence Length")
                ax.set_ylabel("Rmse")
                plt.show()


    def plot_not_normalized_data_fh(self, for_features=False):
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
                    not_normalized_plot_sequence = self.plots_data[baseline_type]["not_normalized_plot_fh"]

                    for i, feature in enumerate(self.feature_list):
                        for key in not_normalized_plot_sequence.keys():
                            for_feature_dict[feature].append(list(not_normalized_plot_sequence[key])[i])
                    
                    for feature in self.feature_list:
                        plot_dict[feature].append(for_feature_dict[feature])


            # plot the transformed data
            for feature in self.feature_list:
                fig, ax = plt.subplots(figsize=(10, 8)) 
                for i in range(len(baselines)):
                    ax.plot(list(not_normalized_plot_sequence.keys()), plot_dict[feature][i])
                    ax.set_title(f"Not Normalized Data Forcasting Horizon - {feature}")
                ax.legend(baselines)
                ax.set_xlabel("Forcasting Horizon")
                ax.set_ylabel("Rmse")
                plt.show()

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
                ax.plot(sequence_plot_x, sequence_plot_y, "-o", label=baseline_type)
                ax.set_xlabel("Sequence Length")
                ax.set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

            # Set the title and legend
            ax.set_title("Data Sequence")
            ax.legend()

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
                axes[row, col].plot(sequence_plot_x, sequence_plot_y, "-o")
                axes[row, col].set_title(
                    baseline_type
                )  # Set the title as the baseline_type
                axes[row, col].set_xlabel("Sequence Length")
                axes[row, col].set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

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
                if baseline_type in ["simple-linear", "linear", "lgbm"]:
                    # Get the sequence_plot_x and sequence_plot_time for the current baseline_type
                    sequence_plot_x = self.plots_data[baseline_type]["sequence_plot_x"]
                    sequence_plot_time = self.plots_data[baseline_type][
                        "sequence_plot_time"
                    ]

                    # Plot the data on the single plot
                    ax.plot(
                        sequence_plot_x, sequence_plot_time, "-o", label=baseline_type
                    )

            # Set the title and legend
            ax.set_title("Data Sequence Time (Baselines)")
            ax.legend()
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Time [s]")

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
                        sequence_plot_x, sequence_plot_time, "-o", label=baseline_type
                    )

            # Set the title and legend
            ax.set_title("Data Sequence Time (Nets)")
            ax.legend()
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Time [s]")

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
                axes[row, col].plot(sequence_plot_x, sequence_plot_time, "-o")
                axes[row, col].set_title(
                    baseline_type
                )  # Set the title as the baseline_type
                axes[row, col].set_xlabel("Sequence Length")
                axes[row, col].set_ylabel("Time [s]")

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
                ax.plot(fh_plot_x, fh_plot_y, "-o", label=baseline_type)

            # Set the title and legend
            ax.set_title("Data FH")
            ax.legend()
            ax.set_xlabel("Forcasting Horizon")
            ax.set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

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
                axes[row, col].plot(fh_plot_x, fh_plot_y, "-o")
                axes[row, col].set_title(
                    baseline_type
                )  # Set the title as the baseline_type
                axes[row, col].set_xlabel("Forcasting Horizon")
                axes[row, col].set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

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
                ax.plot(fh_plot_x, fh_plot_time, "-o", label=baseline_type)

            # Set the title and legend
            ax.set_title("Data FH Time")
            ax.legend()
            ax.set_xlabel("Forcasting Horizon")
            ax.set_ylabel("Time [s]")

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
                axes[row, col].plot(fh_plot_x, fh_plot_time, "-o")
                axes[row, col].set_title(
                    baseline_type
                )  # Set the title as the baseline_type
                axes[row, col].set_xlabel("Forcasting Horizon")
                axes[row, col].set_ylabel("Time [s]")
            # Adjust the layout and spacing of the subplots
            plt.tight_layout()

            # Show the plot
            plt.show()

    def plot_gnn_alpha(self):
        # Create a single plot with multiple lines and legend
        fig, ax = plt.subplots(figsize=(10, 8))

        # Iterate over each baseline_type and plot the data
        for baseline_type in self.plots_data.keys():
            if baseline_type == "gnn":
                alpha_plot_x = self.plots_data[baseline_type]["gnn_alpha_plot_x"]
                alpha_plot_y = self.plots_data[baseline_type]["gnn_alpha_plot_y"]

                # Plot the data on the single plot
                ax.plot(alpha_plot_x, alpha_plot_y, "-o", label=baseline_type)
                ax.set_xlabel("Alpha")
                ax.set_ylabel(r"$\overline{\| \mathcal{L}_{RMSE} \|}$")

        # Set the title and legend
        ax.set_title("Alpha for Gnn and Tigge mix")
        ax.legend()

        # Show the plot
        plt.show()
