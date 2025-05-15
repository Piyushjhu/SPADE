# spall_analysis/__init__.py
"""
Spall Analysis Toolkit (`spall_analysis`) - Original Naming

A Python package for processing experimental data (primarily velocity traces)
from spallation or shock physics experiments to analyze material failure dynamics.
Provides tools for calculating spall strength, strain rate, visualizing results,
comparing with literature, and applying relevant material models.

(Corrected plotting function imports)
"""

# --- Package Version ---
__version__ = "0.1.0" # Initial version identifier

# --- Expose Key Functions and Classes for Easier Access ---
# Make core functionalities directly available when importing the package.
# Example: import spall_analysis as sa; sa.process_velocity_files(...)

# ** IMPORT UTILS FIRST to define constants early **
from .utils import (
    # Constants
    DENSITY_COPPER,
    ACOUSTIC_VELOCITY_COPPER,
    # Mappings (useful for customizing plots)
    MATERIAL_MAPPING,
    ENERGY_VELOCITY_MAPPING,
    COLOR_MAPPING,
    MARKER_MAPPING,
    # Core helper functions
    find_data_files,
    extract_legend_info,
    calculate_shock_stress,
    add_shock_stress_column
)

# From data_processing module:
from .data_processing import (
    process_velocity_files,
    calculate_spall_parameters
)

# From plotting module:
# *** CORRECTED & EXPANDED: Import all necessary plotting functions used in run_analysis...py ***
from .plotting import (
    plot_velocity_comparison,
    plot_spall_vs_strain_rate,
    plot_spall_vs_shock_stress,
    plot_wilkerson_comparison,
    plot_spall_vs_strain_rate_multi_wilkerson,
    plot_combined_mean_traces,
    plot_combined_mean_models, # Keep if used, otherwise optional
    plot_model_vs_feature,      # For potential future 2D model plots
    plot_model_3d_surface,      # For the per-material 3D plots
    plot_combined_material_surfaces, # For the combined 3D plot
    plot_interactive_spall_vs_strain_rate  # For interactive HTML plots
    # plot_elastic_net_results # This is just a placeholder, no need to import
)

# From models module:
from .models import (
    calculate_wilkerson_spall_complex,
    train_elastic_net, # Keep if still used elsewhere
    prepare_feature_matrix,
    train_and_plot_models_per_material
)

# From literature module:
from .literature import load_literature_data


print(f"Spall Analysis Toolkit (spall_analysis) version {__version__} loaded.")
