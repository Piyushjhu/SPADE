# spade_analysis/data_processing.py
"""
Functions for processing raw velocity-time data using a constrained
5-segment piecewise linear model, based on the user's reference script.
Includes logging for diagnostics. Returns model parameters.

(Reverted Line 2 to horizontal plateau; Added strict P1/P2 validity check)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import warnings
import traceback
import logging
from . import utils

# --- Helper Functions ---
# _get_interp_y, _fit_line_through_points, _fit_line_to_range,
# _find_intersection, _evaluate_piecewise_model,
# _plot_trace_with_5segment_model
# (Assume they exist as before)
# ... (Previous helper function code omitted for brevity) ...
def _get_interp_y(x_data, y_data, x_target, kind='linear'):
    if not isinstance(x_data, (pd.Series, np.ndarray)) or not isinstance(y_data, (pd.Series, np.ndarray)): return np.nan
    if len(x_data) < 2: return np.nan
    try:
        x_vals = x_data.values if isinstance(x_data, pd.Series) else np.asarray(x_data); y_vals = y_data.values if isinstance(y_data, pd.Series) else np.asarray(y_data)
        finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals);
        if finite_mask.sum() < 2: return np.nan
        x_vals = x_vals[finite_mask]; y_vals = y_vals[finite_mask]; sort_idx = np.argsort(x_vals)
        x_data_sorted = x_vals[sort_idx]; y_data_sorted = y_vals[sort_idx]
        unique_x_mask = np.concatenate(([True], np.diff(x_data_sorted) > 1e-9));
        if unique_x_mask.sum() < 2: return np.nan
        x_unique = x_data_sorted[unique_x_mask]; y_unique = y_data_sorted[unique_x_mask]
        interp_func = interp1d(x_unique, y_unique, kind=kind, bounds_error=False, fill_value=np.nan)
        y_target = interp_func(x_target); return float(y_target) if pd.notna(y_target) else np.nan
    except (ValueError, Exception): return np.nan

def _fit_line_through_points(p1, p2):
    x1, y1 = p1; x2, y2 = p2
    if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2): return np.nan, np.nan
    if np.isclose(x1, x2): return np.inf, x1
    m = (y2 - y1) / (x2 - x1); c = y1 - m * x1; return m, c

def _fit_line_to_range(x_data, y_data, x_start, x_end):
    if not isinstance(x_data, pd.Series): x_data = pd.Series(x_data)
    if not isinstance(y_data, pd.Series): y_data = pd.Series(y_data)
    if pd.isna(x_start) or pd.isna(x_end) or x_start >= x_end: return np.nan, np.nan
    mask = (x_data >= x_start) & (x_data <= x_end); x_subset = x_data[mask]; y_subset = y_data[mask]
    valid_subset = x_subset.notna() & y_subset.notna();
    if valid_subset.sum() < 2: return np.nan, np.nan
    try:
        coeffs = np.polyfit(x_subset[valid_subset], y_subset[valid_subset], 1)
        if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)): return np.nan, np.nan
        return coeffs[0], coeffs[1]
    except (np.linalg.LinAlgError, ValueError, Exception): return np.nan, np.nan

def _find_intersection(m1, c1, m2, c2):
    if np.isnan(m1) or np.isnan(c1) or np.isnan(m2) or np.isnan(c2): return np.nan, np.nan
    if np.isinf(m1) and np.isinf(m2): return np.nan, np.nan
    if not np.isinf(m1) and not np.isinf(m2) and np.isclose(m1, m2): return np.nan, np.nan
    if np.isinf(m1): x_intersect = c1; y_intersect = m2 * x_intersect + c2 if not np.isinf(m2) else np.nan
    elif np.isinf(m2): x_intersect = c2; y_intersect = m1 * x_intersect + c1 if not np.isinf(m1) else np.nan
    else: x_intersect = (c2 - c1) / (m1 - m2); y_intersect = m1 * x_intersect + c1
    if not (-100 < x_intersect < 500): return np.nan, np.nan
    return x_intersect, y_intersect

def _evaluate_piecewise_model(t, lines_info, intersections):
    if not lines_info or not intersections or len(lines_info) != 5 or len(intersections) != 4: return np.full_like(t, np.nan, dtype=float) if isinstance(t, np.ndarray) else np.nan
    try: m1, c1 = lines_info[0]; m2, c2 = lines_info[1]; m3, c3 = lines_info[2]; m4, c4 = lines_info[3]; m5, c5 = lines_info[4]; P1x, _ = intersections[0]; P2x, _ = intersections[1]; P3x, _ = intersections[2]; P4x, _ = intersections[3]
    except (TypeError, IndexError): return np.full_like(t, np.nan, dtype=float) if isinstance(t, np.ndarray) else np.nan
    b1 = P1x if pd.notna(P1x) else np.inf; b2 = P2x if pd.notna(P2x) and P2x >= b1 else b1; b3 = P3x if pd.notna(P3x) and P3x >= b2 else b2; b4 = P4x if pd.notna(P4x) and P4x >= b3 else b3; t = np.asarray(t)
    condlist = [(t >= 0) & (t < b1), (t >= b1) & (t < b2), (t >= b2) & (t < b3), (t >= b3) & (t < b4), (t >= b4)]
    funclist = [lambda t_val: m1*t_val+c1 if pd.notna(m1) and pd.notna(c1) and ~np.isinf(m1) else np.nan, lambda t_val: m2*t_val+c2 if pd.notna(m2) and pd.notna(c2) and ~np.isinf(m2) else np.nan, lambda t_val: m3*t_val+c3 if pd.notna(m3) and pd.notna(c3) and ~np.isinf(m3) else np.nan, lambda t_val: m4*t_val+c4 if pd.notna(m4) and pd.notna(c4) and ~np.isinf(m4) else np.nan, lambda t_val: m5*t_val+c5 if pd.notna(m5) and pd.notna(c5) and ~np.isinf(m5) else np.nan]
    return np.piecewise(t, condlist, funclist)

def _plot_trace_with_5segment_model(data_dict, lines_info, intersections, output_folder):
    import matplotlib.pyplot as plt
    x_trace = data_dict.get('x', pd.Series(dtype=float)); y_trace = data_dict.get('y', pd.Series(dtype=float)); filename = data_dict.get('filename', 'Unknown Filename'); base_filename = os.path.splitext(filename)[0]
    if x_trace.empty or y_trace.empty: logging.warning(f"Skipping plot for {filename}: No valid trace data."); return
    logging.debug(f"    Generating 5-segment plot for {base_filename}"); fig_single, ax_single = plt.subplots(figsize=(10, 6))
    try:
        ax_single.plot(x_trace, y_trace, label='Original Data', color='grey', alpha=0.7, linewidth=1.5); colors = ['blue', 'green', 'red', 'purple', 'brown']; labels = ['Line 1 (Rise)', 'Line 2 (Plateau)', 'Line 3 (Pullback)', 'Line 4 (Recomp Rise)', 'Line 5 (Recomp Tail)']
        intersections = intersections if isinstance(intersections, list) and len(intersections)==4 else [(np.nan, np.nan)]*4; P1x, P1y = intersections[0]; P2x, P2y = intersections[1]; P3x, P3y = intersections[2]; P4x, P4y = intersections[3]
        plot_ranges = [(0, P1x if pd.notna(P1x) else 9), (P1x if pd.notna(P1x) else 7, P2x if pd.notna(P2x) else 22), (P2x if pd.notna(P2x) else 20, P3x if pd.notna(P3x) else 38), (P3x if pd.notna(P3x) else 35, P4x if pd.notna(P4x) else 48), (P4x if pd.notna(P4x) else 45, (P4x + 10) if pd.notna(P4x) else 55)]
        lines_info = lines_info if isinstance(lines_info, list) and len(lines_info)==5 else [(np.nan, np.nan)]*5
        for i, ((m, c), (x_start, x_end), color, label) in enumerate(zip(lines_info, plot_ranges, colors, labels)):
             # ** Reverted: Plot Line 2 as horizontal if m=0 **
             if i == 1 and np.isclose(m, 0) and pd.notna(c):
                  x_line = np.linspace(x_start, x_end, 10); y_line = np.full_like(x_line, c)
                  ax_single.plot(x_line, y_line, label=f'{label} (m=0.00)', color=color, linestyle='--', linewidth=2)
             elif pd.notna(m) and pd.notna(c) and not np.isinf(m) and pd.notna(x_start) and pd.notna(x_end) and x_start < x_end:
                 if np.isclose(x_start, x_end): x_end += 0.1
                 x_line = np.linspace(x_start, x_end, 10); y_line = m * x_line + c; ax_single.plot(x_line, y_line, label=f'{label} (m={m:.2f})', color=color, linestyle='--', linewidth=2)
             elif np.isinf(m) and pd.notna(c): ax_single.axvline(c, label=f'{label} (Vertical)', color=color, linestyle='--', linewidth=2)
        int_labels = ['P1', 'P2', 'P3', 'P4 (Recomp Peak Proxy)']; int_colors = ['cyan', 'magenta', 'orange', 'lime']
        for i, (px, py) in enumerate(intersections):
            if pd.notna(px) and pd.notna(py): ax_single.scatter([px], [py], label=f'{int_labels[i]} ({px:.1f},{py:.1f})', color=int_colors[i], s=60, zorder=5, edgecolors='black')
        ax_single.set_xlabel('Time (ns)', fontsize=14); ax_single.set_ylabel('Velocity (m/s)', fontsize=14); ax_single.set_title(f'5-Segment Linear Model: {base_filename}', fontsize=16); ax_single.legend(fontsize=8, loc='best'); ax_single.grid(True, linestyle=':'); ax_single.tick_params(axis='both', which='major', labelsize=12)
        y_data_for_limits = [y for y in [y_trace.values] + [[p[1]] for p in intersections if pd.notna(p[1])] if y is not None and len(y)>0]
        if y_data_for_limits: all_y = np.concatenate(y_data_for_limits); all_y = all_y[np.isfinite(all_y)];
        if 'all_y' in locals() and len(all_y) > 0: min_y, max_y = np.nanmin(all_y), np.nanmax(all_y); ax_single.set_ylim(min_y - 50, max_y + 100)
        else: ax_single.set_ylim(0, 800)
        ax_single.set_xlim(-5, 60); fig_single.tight_layout(); plot_filename = os.path.join(output_folder, f"{base_filename}_5segment_model.png")
        logging.debug(f"    Attempting to save 5-segment plot to: {plot_filename}")
        try: plt.savefig(plot_filename, dpi=150); logging.debug(f"    Successfully saved 5-segment plot: {os.path.basename(plot_filename)}")
        except Exception as e: logging.error(f"    ERROR saving 5-segment plot {plot_filename}: {e}")
    except Exception as plot_err: logging.exception(f"    ERROR during 5-segment plot generation for {base_filename}: {plot_err}")
    finally:
        if 'fig_single' in locals() and fig_single is not None: logging.debug(f"    Closing 5-segment plot figure."); plt.close(fig_single)


# --- Main Processing Function (Horizontal L2, Strict P1/P2 Check) ---

def calculate_spall_parameters(time, velocity, y_err_data, filename, output_folder,
                               density=utils.DENSITY_COPPER,
                               acoustic_velocity=utils.ACOUSTIC_VELOCITY_COPPER,
                               plot_individual=True):
    """
    Processes a single velocity trace using the constrained 5-segment piecewise
    linear model approach. Calculates parameters based on segment intersections.
    Line 2 is horizontal at dynamic plateau level. Includes strict P1/P2 check.
    Returns calculated parameters and model info.
    """
    base_name = os.path.basename(filename)
    base_name_no_ext = os.path.splitext(base_name)[0]
    os.makedirs(output_folder, exist_ok=True)
    results = {'Filename': base_name_no_ext}
    logging.debug(f"  Starting 5-segment model analysis (Horizontal L2, Strict Check) for {base_name_no_ext}")

    # Initialize all output values to NaN or empty lists
    first_max_val, first_max_err = np.nan, np.nan
    min_val, min_err = np.nan, np.nan
    recomp_vel, time_recomp = np.nan, np.nan
    spall_str, spall_err = np.nan, np.nan
    strain_rate_val, strain_rate_err = np.nan, np.nan
    recomp_slope = np.nan
    p1x_val, p2x_val = np.nan, np.nan
    comp_shock_rate = np.nan
    lines_info = [(np.nan, np.nan)] * 5
    intersections = [(np.nan, np.nan)] * 4
    current_data_dict = {'filename': base_name}

    try:
        # --- Data Cleaning, Sorting, Shifting, Filtering ---
        X_orig = pd.Series(time); Y_orig = pd.Series(velocity); YErr_orig = pd.Series(y_err_data)
        X = pd.to_numeric(X_orig, errors='coerce'); Y = pd.to_numeric(Y_orig, errors='coerce'); YErr = pd.to_numeric(YErr_orig, errors='coerce')
        valid_indices = X.notna() & Y.notna() & YErr.notna()
        if not valid_indices.any(): raise ValueError("No valid numeric rows found.")
        X = X[valid_indices]; Y = Y[valid_indices]; YErr = YErr[valid_indices]
        if not X.is_monotonic_increasing: sort_idx = X.argsort(); X = X.iloc[sort_idx]; Y = Y.iloc[sort_idx]; YErr = YErr.iloc[sort_idx]
        X = X.reset_index(drop=True); Y = Y.reset_index(drop=True); YErr = YErr.reset_index(drop=True)
        if len(X) < 2: raise ValueError("< 2 valid rows after cleaning/sorting.")
        x_vals_for_grad = X.values; y_vals_for_grad = Y.values
        if len(x_vals_for_grad) < 2 : raise ValueError("Need >= 2 points for gradient.")
        with np.errstate(divide='ignore', invalid='ignore'): dY_dX = np.gradient(y_vals_for_grad, x_vals_for_grad)
        positive_slope_indices = np.where(dY_dX > 0)[0]
        idx_shift = 0
        if len(positive_slope_indices) > 0:
            try: candidate_values = Y.iloc[positive_slope_indices]; closest_label = np.abs(candidate_values - 30).idxmin(); idx_shift = Y.index.get_loc(closest_label)
            except Exception: idx_shift = 0
        if not (0 <= idx_shift < len(X)): idx_shift = 0
        t_shift = X.iloc[idx_shift]; X_shifted = (X - t_shift) * 1e9
        mask = X_shifted >= 0
        if not mask.any(): raise ValueError("No data >= 0 ns after shift.")
        X_shifted_filtered = X_shifted[mask].reset_index(drop=True); Y_filtered = Y[mask].reset_index(drop=True); YErr_filtered = YErr[mask].reset_index(drop=True)
        if len(X_shifted_filtered) < 2: raise ValueError("< 2 points after time filter.")
        current_data_dict['x'] = X_shifted_filtered.copy(); current_data_dict['y'] = Y_filtered.copy()

        # === Find Original Minimum Value & Error ===
        min_range_mask = (X_shifted_filtered >= 25) & (X_shifted_filtered <= 35)
        orig_min_val, orig_min_err = np.nan, np.nan
        if min_range_mask.sum() >= 1:
            Y_min_range = Y_filtered[min_range_mask]
            if not Y_min_range.empty:
                 min_idx_label = Y_min_range.idxmin()
                 if min_idx_label in Y_filtered.index and min_idx_label in YErr_filtered.index:
                     orig_min_val = Y_filtered.loc[min_idx_label]; orig_min_err = YErr_filtered.loc[min_idx_label]
        min_err = orig_min_err

        # === Piecewise Linear Model Construction (Horizontal L2, Strict Check) ===
        logging.debug(f"  Fitting 5-segment model (Horizontal L2, Strict Check)...")
        # --- Line 1 (Rise) ---
        l1_range = [0, 9]; m1, c1 = _fit_line_to_range(X_shifted_filtered, Y_filtered, l1_range[0], l1_range[1])
        if pd.notna(m1) and m1 <= 0: l1_range_adj = [0, 7.5]; m1_adj, c1_adj = _fit_line_to_range(X_shifted_filtered, Y_filtered, l1_range_adj[0], l1_range_adj[1])
        if 'm1_adj' in locals() and pd.notna(m1_adj): m1, c1 = m1_adj, c1_adj
        final_m1, final_c1 = m1, c1
        if pd.isna(final_m1) or pd.isna(final_c1): raise ValueError("Failed to fit Line 1.")

        # --- Determine Refined Plateau Level for Horizontal Line 2 ---
        initial_plat_mask = (X_shifted_filtered >= 0) & (X_shifted_filtered <= 15)
        c2_temp_initial = Y_filtered[initial_plat_mask].max() if initial_plat_mask.sum() > 0 else Y_filtered.iloc[0] if not Y_filtered.empty else 0
        p1_initial = _find_intersection(final_m1, final_c1, 0.0, c2_temp_initial)
        p1x_initial = p1_initial[0] if pd.notna(p1_initial[0]) else 5.0
        plateau_search_start = p1x_initial + 1.0; plateau_search_end = plateau_search_start + 10.0
        refined_plat_mask = (X_shifted_filtered >= plateau_search_start) & (X_shifted_filtered <= plateau_search_end)
        if refined_plat_mask.sum() > 0: c2_temp = Y_filtered[refined_plat_mask].max()
        else: c2_temp = c2_temp_initial; logging.warning(f"    Could not find data in dynamic plateau window. Using initial guess: {c2_temp:.1f}")
        # ** Line 2 is now defined as horizontal at this level **
        final_m2, final_c2 = 0.0, c2_temp
        if pd.isna(final_c2): raise ValueError("Failed to determine plateau level (c2).")

        # --- Line 3 (Pullback - Adaptive Start based on P1) ---
        p1_intersect = _find_intersection(final_m1, final_c1, final_m2, final_c2) # Find P1 using final L1 and L2
        if pd.isna(p1_intersect[0]): raise ValueError(f"Failed to find valid P1 intersection using plateau level {final_c2:.1f}.")
        p1x_val = p1_intersect[0] # Store P1 time

        L3_START_BUFFER = 2.0; L3_DEFAULT_START = 20.0; L3_DURATION = 14.0
        l3_start = max(p1x_val + L3_START_BUFFER, L3_DEFAULT_START)
        l3_end = l3_start + L3_DURATION
        logging.debug(f"    Adaptive Line 3 fit range: [{l3_start:.1f}, {l3_end:.1f}] ns")
        m3, c3 = _fit_line_to_range(X_shifted_filtered, Y_filtered, l3_start, l3_end)
        final_m3, final_c3 = m3, c3
        if pd.isna(final_m3) or pd.isna(final_c3): raise ValueError(f"Failed to fit Line 3 in range [{l3_start:.1f}, {l3_end:.1f}].")

        # --- Find P2 and perform STRICT CHECK ---
        p2_intersect = _find_intersection(final_m2, final_c2, final_m3, final_c3)
        p2x_val = p2_intersect[0] # Store P2 time
        logging.debug(f"    Calculated P1={p1_intersect}, P2={p2_intersect}")
        if pd.isna(p1x_val) or pd.isna(p2x_val) or p2x_val < p1x_val or p2x_val < 0:
             logging.error(f"Invalid P1/P2 sequence: P1x={p1x_val:.2f}, P2x={p2x_val:.2f}. Check fits/ranges.")
             raise ValueError(f"Invalid P1/P2 sequence (P2x={p2x_val:.2f} not valid relative to P1x={p1x_val:.2f})")

        # --- Line 4 (Recompaction Rise - Adaptive Start) ---
        l4_start_buffer = 2.0; l4_default_start = 35.0; l4_duration = 8.0
        l4_start = max(p2x_val + l4_start_buffer, l4_default_start)
        l4_end = l4_start + l4_duration
        logging.debug(f"    Adaptive Line 4 fit range: [{l4_start:.1f}, {l4_end:.1f}] ns")
        m4, c4 = _fit_line_to_range(X_shifted_filtered, Y_filtered, l4_start, l4_end)
        if pd.notna(m4) and m4 <= 0:
            l4_range_adj = [l4_start - 2, l4_end - 2];
            m4_adj, c4_adj = _fit_line_to_range(X_shifted_filtered, Y_filtered, l4_range_adj[0], l4_range_adj[1])
            if pd.notna(m4_adj): m4, c4 = m4_adj, c4_adj
        final_m4, final_c4 = m4, c4
        if pd.isna(final_m4) or pd.isna(final_c4): raise ValueError(f"Failed to fit Line 4 in range [{l4_start:.1f}, {l4_end:.1f}].")

        # --- Line 5 (Recompaction Tail - Adaptive Start) ---
        p3_intersect = _find_intersection(final_m3, final_c3, final_m4, final_c4)
        if pd.isna(p3_intersect[0]): raise ValueError("Failed to find P3 intersection.")
        p3x = p3_intersect[0]
        l5_start_buffer = 2.0; l5_default_start = max(p3x + l5_start_buffer, 45.0); l5_duration = 6.0
        l5_start = l5_default_start; l5_end = l5_start + l5_duration
        logging.debug(f"    Adaptive Line 5 fit range: [{l5_start:.1f}, {l5_end:.1f}] ns")
        m5, c5 = _fit_line_to_range(X_shifted_filtered, Y_filtered, l5_start, l5_end)
        final_m5, final_c5 = m5, c5
        # If L5 fit fails, don't raise error, P4 will just be NaN
        if pd.isna(final_m5) or pd.isna(final_c5): logging.warning("Failed to fit Line 5.")

        # Store final line parameters
        lines_info = [(final_m1, final_c1), (final_m2, final_c2), (final_m3, final_c3), (final_m4, final_c4), (final_m5, final_c5)]
        current_data_dict['lines_info'] = lines_info

        # === Find Final Intersections (P4) ===
        p4_intersect = _find_intersection(final_m4, final_c4, final_m5, final_c5)
        if pd.isna(p4_intersect[0]): logging.warning("Failed to find P4 intersection.")
        intersections = [p1_intersect, p2_intersect, p3_intersect, p4_intersect]
        current_data_dict['intersections'] = intersections

        # === Calculate Derived Parameters ===
        logging.debug(f"  Calculating derived parameters (P1={p1_intersect}, P2={p2_intersect}, P3={p3_intersect}, P4={p4_intersect})...")
        first_max_val = final_c2 # Plateau level is the max proxy now
        # Estimate error from raw data near P1.x
        if pd.notna(p1x_val):
             closest_idx_label = (X_shifted_filtered - p1x_val).abs().idxmin()
             if closest_idx_label in YErr_filtered.index: first_max_err = YErr_filtered.loc[closest_idx_label]
             else: first_max_err = np.nan
        else: first_max_err = np.nan

        min_proxy = p3_intersect[1]; min_val = min_proxy
        recomp_vel = p4_intersect[1]; p4x_ns = p4_intersect[0]; time_recomp = p4x_ns * 1e-9 if pd.notna(p4x_ns) else np.nan
        line4_slope_ns = final_m4 if pd.notna(final_m4) and not np.isinf(final_m4) else np.nan
        recomp_slope = line4_slope_ns * 1e9 if pd.notna(line4_slope_ns) else np.nan
        p2x, p2y = p2_intersect; p3x, p3y = p3_intersect
        strain_rate_val = np.nan; delta_v_pullback = np.nan
        if pd.notna(p2x) and pd.notna(p2y) and pd.notna(p3x) and pd.notna(p3y):
            delta_v_pullback = p2y - p3y; delta_t_pullback_ns = p2x - p3x
            if not np.isclose(delta_t_pullback_ns, 0) and acoustic_velocity > 0:
                strain_rate_val = -(delta_v_pullback) / (2 * acoustic_velocity * (delta_t_pullback_ns * 1e-9))

        spall_str = np.nan; spall_err = np.nan; strain_rate_err = np.nan
        max_proxy = first_max_val
        if pd.notna(max_proxy) and pd.notna(min_proxy):
            delta_v_spall = max_proxy - min_proxy
            if delta_v_spall > 0: spall_str = (0.5 * density * acoustic_velocity * delta_v_spall) * 1e-9
        if pd.notna(first_max_err) and pd.notna(min_err):
            spall_err = (0.5 * density * acoustic_velocity * 1e-9) * np.sqrt(first_max_err**2 + min_err**2)
        if pd.notna(strain_rate_val) and pd.notna(first_max_err) and pd.notna(min_err) and pd.notna(delta_v_pullback) and not np.isclose(delta_v_pullback, 0):
            strain_rate_err = abs(strain_rate_val) * np.sqrt(first_max_err**2 + min_err**2) / abs(delta_v_pullback)

        y_origin = _get_interp_y(X_shifted_filtered, Y_filtered, 0); m_comp, _ = _fit_line_through_points((0, y_origin), p1_intersect)
        comp_shock_rate = m_comp * 1e9 if pd.notna(m_comp) and not np.isinf(m_comp) else np.nan

        results['Processing Status'] = 'Success'

    except Exception as e:
        logging.error(f"  ERROR processing file {base_name}: {e}")
        # logging.debug(traceback.format_exc()) # Uncomment for detailed traceback
        results['Processing Status'] = f'Failed: {type(e).__name__} - {e}'
        first_max_val, first_max_err = np.nan, np.nan; min_val, min_err = np.nan, np.nan
        recomp_vel, time_recomp = np.nan, np.nan; spall_str, spall_err = np.nan, np.nan
        strain_rate_val, strain_rate_err = np.nan, np.nan; recomp_slope = np.nan
        p1x_val, p2x_val = np.nan, np.nan; comp_shock_rate = np.nan
        lines_info = [(np.nan, np.nan)] * 5; intersections = [(np.nan, np.nan)] * 4

    # --- Store results in dictionary ---
    results['First Maxima (m/s)'] = first_max_val
    results['First Maxima Err (m/s)'] = first_max_err
    results['Minima (m/s)'] = min_val
    results['Minima Err (m/s)'] = min_err
    results['Recompression Velocity (m/s)'] = recomp_vel
    results['Time at Recompression Peak (s)'] = time_recomp
    results['Spall Strength (GPa)'] = spall_str
    results['Spall Strength Err (GPa)'] = spall_err
    results['Strain Rate (s^-1)'] = strain_rate_val
    results['Strain Rate Err (s^-1)'] = strain_rate_err
    results['Recompression Slope (m/s^2)'] = recomp_slope
    results['P1.x (ns)'] = p1x_val
    results['P2.x (ns)'] = p2x_val
    results['Compression Shock Rate (m/s/s)'] = comp_shock_rate
    results['model_lines_info'] = tuple(lines_info) if lines_info else tuple([(np.nan, np.nan)] * 5)
    results['model_intersections'] = tuple(intersections) if intersections else tuple([(np.nan, np.nan)] * 4)

    # --- Plotting ---
    if plot_individual and results['Processing Status'] == 'Success':
        _plot_trace_with_5segment_model(current_data_dict, lines_info, intersections, output_folder)
    elif plot_individual:
        logging.warning(f"Skipping individual plot for {base_name} due to processing failure.")

    logging.debug(f"  Finished 5-segment analysis for {base_name_no_ext}")
    return results


def process_velocity_files(input_folder, file_pattern, output_folder,
                           save_summary_table=True, summary_table_name=None,
                           **kwargs):
    """
    Processes all velocity trace files using the 5-segment model approach
    via `calculate_spall_parameters` and aggregates results. Handles model info.
    """
    # (Function content remains the same as previous version)
    files = utils.find_data_files(input_folder, file_pattern)
    if not files: logging.warning(f"No files found matching pattern '{file_pattern}' in '{input_folder}'."); return pd.DataFrame()
    all_results = []; processed_count = 0; error_count = 0
    table_output_dir = os.path.join(output_folder, 'tables'); plot_output_dir = output_folder
    os.makedirs(table_output_dir, exist_ok=True); os.makedirs(plot_output_dir, exist_ok=True)
    logging.info(f"\nStarting 5-segment model processing of {len(files)} files in '{input_folder}'...")
    for f in files:
        base_name = os.path.basename(f); logging.info(f"Processing: {base_name}")
        try:
            data = pd.read_csv(f, header='infer', on_bad_lines='skip', engine='python')
            if data.shape[1] < 3: logging.warning(f"Skipping {base_name}: Expected >= 3 columns (T,V,Err), found {data.shape[1]}."); error_count += 1; all_results.append({'Filename': os.path.splitext(base_name)[0], 'Processing Status': 'Failed: Incorrect Columns'}); continue
            time_data = data.iloc[:, 0].to_numpy(); velocity_data = data.iloc[:, 1].to_numpy(); y_err_data = data.iloc[:, 2].to_numpy()
            calc_args = {'density': kwargs.get('density', utils.DENSITY_COPPER), 'acoustic_velocity': kwargs.get('acoustic_velocity', utils.ACOUSTIC_VELOCITY_COPPER), 'plot_individual': kwargs.get('plot_individual', True)}
            result = calculate_spall_parameters(time_data, velocity_data, y_err_data, f, plot_output_dir, **calc_args)
            if result:
                all_results.append(result)
                if result.get('Processing Status') == 'Success': processed_count += 1
                else: error_count += 1; logging.warning(f"  -> Processing failed/partially failed: {result.get('Processing Status')}")
            else: error_count += 1; logging.error(f"  -> Processing failed critically: No result returned."); all_results.append({'Filename': os.path.splitext(base_name)[0], 'Processing Status': 'Failed: Critical Error'})
        except pd.errors.EmptyDataError: logging.warning(f"Skipping {base_name}: File is empty."); error_count += 1; all_results.append({'Filename': os.path.splitext(base_name)[0], 'Processing Status': 'Failed: Empty File'})
        except ValueError as ve: logging.warning(f"Skipping {base_name}: Processing error - {ve}"); error_count += 1; all_results.append({'Filename': os.path.splitext(base_name)[0], 'Processing Status': f'Failed: ValueError - {ve}'})
        except Exception as e: logging.exception(f"Failed to load or process {base_name}: {e}"); error_count += 1; all_results.append({'Filename': os.path.splitext(base_name)[0], 'Processing Status': f'Failed: {type(e).__name__}'})
    logging.info(f"\n--- Processing Summary for Bin '{os.path.basename(input_folder)}' ---"); logging.info(f"- Traces processed (incl. failures): {len(all_results)}"); logging.info(f"- Traces with 'Success' status: {processed_count}"); logging.info(f"- Traces with 'Failed' status: {error_count}")
    if not all_results: return pd.DataFrame()
    results_df = pd.DataFrame(all_results)
    cols_order = ['Filename', 'First Maxima (m/s)', 'First Maxima Err (m/s)', 'Minima (m/s)', 'Minima Err (m/s)', 'Recompression Velocity (m/s)', 'Time at Recompression Peak (s)', 'Spall Strength (GPa)', 'Spall Strength Err (GPa)', 'Strain Rate (s^-1)', 'Strain Rate Err (s^-1)', 'Recompression Slope (m/s^2)', 'P1.x (ns)', 'P2.x (ns)', 'Compression Shock Rate (m/s/s)', 'model_lines_info', 'model_intersections', 'Processing Status']
    for col in cols_order:
        if col not in results_df.columns:
            if col in ['model_lines_info', 'model_intersections']: results_df[col] = pd.Series([None] * len(results_df), dtype=object)
            else: results_df[col] = np.nan
    results_df = results_df[cols_order]
    if save_summary_table:
        if summary_table_name is None: input_folder_basename = os.path.basename(os.path.normpath(input_folder)); summary_table_name = f"{input_folder_basename}_results_table_5segment.csv"
        results_output_filename = os.path.join(table_output_dir, summary_table_name)
        try:
            df_to_save = results_df.copy()
            for col in ['model_lines_info', 'model_intersections']:
                if col in df_to_save.columns: df_to_save[col] = df_to_save[col].apply(lambda x: str(x) if x is not None else '')
            df_to_save.to_csv(results_output_filename, index=False, float_format='%.4e'); logging.info(f"\nBin summary table saved to: {results_output_filename}")
        except Exception as e: logging.error(f"Could not save summary table {results_output_filename}: {e}")
    return results_df

