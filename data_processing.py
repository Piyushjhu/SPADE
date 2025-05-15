# spade_analysis/data_processing.py
"""
Functions for processing raw velocity-time data using a hybrid approach (V3).

(Improved recompaction peak search logic V2: adjusted window end and prominence)
(Fixed UnboundLocalError in Line 5 fitting warning)
(Fixed NameError by initializing time_recomp and recomp_vel before try block)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
import os
import warnings
import traceback
import logging
from . import utils # Import utils to access MATERIAL_MAPPING

# --- Helper Functions ---
def _get_interp_y(x_data, y_data, x_target, kind='linear'):
    """ Safely interpolates y value at x_target from x_data, y_data. """
    if not isinstance(x_data, (pd.Series, np.ndarray)) or not isinstance(y_data, (pd.Series, np.ndarray)): return np.nan
    if len(x_data) < 2: return np.nan
    try:
        x_vals = x_data.values if isinstance(x_data, pd.Series) else np.asarray(x_data)
        y_vals = y_data.values if isinstance(y_data, pd.Series) else np.asarray(y_data)
        finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if finite_mask.sum() < 2: return np.nan # Need at least 2 finite points

        x_vals = x_vals[finite_mask]
        y_vals = y_vals[finite_mask]
        sort_idx = np.argsort(x_vals) # Sort data by x values
        x_data_sorted = x_vals[sort_idx]
        y_data_sorted = y_vals[sort_idx]

        unique_x_mask = np.concatenate(([True], np.diff(x_data_sorted) > 1e-9)) # Remove duplicate x values
        if unique_x_mask.sum() < 2: return np.nan # Need at least 2 unique points
        x_unique = x_data_sorted[unique_x_mask]
        y_unique = y_data_sorted[unique_x_mask]

        interp_func = interp1d(x_unique, y_unique, kind=kind, bounds_error=False, fill_value="extrapolate") # Perform interpolation
        y_target = interp_func(x_target)
        if isinstance(y_target, np.ndarray): y_target = y_target.item(0) # Handle potential array output
        return float(y_target) if pd.notna(y_target) else np.nan
    except (ValueError, Exception) as e: return np.nan

def _fit_line_to_range(x_data, y_data, x_start, x_end):
    """ Fits a line to data within a specified x range. """
    if not isinstance(x_data, pd.Series): x_data = pd.Series(x_data)
    if not isinstance(y_data, pd.Series): y_data = pd.Series(y_data)
    if pd.isna(x_start) or pd.isna(x_end) or x_start >= x_end: return np.nan, np.nan
    x_min_data, x_max_data = x_data.min(), x_data.max() # Ensure start/end are within data range
    x_start = max(x_start, x_min_data); x_end = min(x_end, x_max_data)
    if x_start >= x_end: return np.nan, np.nan # Check again after clamping
    mask = (x_data >= x_start) & (x_data <= x_end)
    x_subset = x_data[mask]; y_subset = y_data[mask]
    valid_subset = x_subset.notna() & y_subset.notna() # Ensure enough valid points
    if valid_subset.sum() < 2: return np.nan, np.nan
    try:
        coeffs = np.polyfit(x_subset[valid_subset], y_subset[valid_subset], 1)
        if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)): return np.nan, np.nan # Check for NaN/inf coefficients
        return coeffs[0], coeffs[1] # slope, intercept
    except (np.linalg.LinAlgError, ValueError, Exception): return np.nan, np.nan

def _fit_line_through_points(p1, p2):
    """ Calculates slope and intercept of a line passing through two points. """
    x1, y1 = p1; x2, y2 = p2
    if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2): return np.nan, np.nan
    if np.isclose(x1, x2): return np.inf, x1 # Handle vertical line case
    m = (y2 - y1) / (x2 - x1); c = y1 - m * x1; # Calculate slope and intercept
    return m, c

def _find_intersection(m1, c1, m2, c2):
    """ Finds the intersection point (x, y) of two lines y=m1*x+c1 and y=m2*x+c2. """
    if np.isnan(m1) or np.isnan(c1) or np.isnan(m2) or np.isnan(c2): return np.nan, np.nan # Check for NaN/inf inputs
    if np.isinf(m1) and np.isinf(m2): return np.nan, np.nan # Parallel vertical lines
    if not np.isinf(m1) and not np.isinf(m2) and np.isclose(m1, m2): return np.nan, np.nan # Handle parallel non-vertical lines
    if np.isinf(m1): x_intersect = c1; y_intersect = m2 * x_intersect + c2 if not np.isinf(m2) else np.nan # Handle vertical lines
    elif np.isinf(m2): x_intersect = c2; y_intersect = m1 * x_intersect + c1 if not np.isinf(m1) else np.nan
    else: x_intersect = (c2 - c1) / (m1 - m2); y_intersect = m1 * x_intersect + c1 # Standard case
    if not (-500 < x_intersect < 1000): return np.nan, np.nan # Basic sanity check
    return x_intersect, y_intersect

# --- Plotting Function (Still using 5-Segment Style Plot) ---
def _plot_trace_with_hybrid_model(data_dict, lines_info, intersections, output_folder):
    """ Plots trace, smoothed data, 5 fitted lines, and intersections. """
    import matplotlib.pyplot as plt # Local import
    x_trace = data_dict.get('x_shifted', pd.Series(dtype=float)); y_trace = data_dict.get('y_original', pd.Series(dtype=float)); y_smooth = data_dict.get('y_smooth', pd.Series(dtype=float)); filename = data_dict.get('filename', 'Unknown Filename'); base_filename = os.path.splitext(filename)[0]
    if x_trace.empty or y_trace.empty: logging.warning(f"Skipping hybrid model plot for {filename}: No valid trace data."); return
    logging.debug(f"    Generating hybrid model plot for {base_filename}"); fig_single, ax_single = plt.subplots(figsize=(10, 6))
    try:
        ax_single.plot(x_trace, y_trace, label='Original Data', color='grey', alpha=0.6, linewidth=1.0)
        if not y_smooth.empty: ax_single.plot(x_trace, y_smooth, label='Smoothed Data', color='black', alpha=0.8, linewidth=1.0, linestyle=':')
        colors = ['blue', 'green', 'red', 'purple', 'brown']; labels = ['Line 1 (Rise)', 'Line 2 (Plateau)', 'Line 3 (Pullback)', 'Line 4 (Recomp Rise)', 'Line 5 (Recomp Tail)']
        intersections = intersections if isinstance(intersections, list) and len(intersections)==4 else [(np.nan, np.nan)]*4; P1x, P1y = intersections[0]; P2x, P2y = intersections[1]; P3x, P3y = intersections[2]; P4x, P4y = intersections[3]
        plot_ranges = [(0, P1x if pd.notna(P1x) else 9), (P1x if pd.notna(P1x) else 7, P2x if pd.notna(P2x) else 22), (P2x if pd.notna(P2x) else 20, P3x if pd.notna(P3x) else 38), (P3x if pd.notna(P3x) else 35, P4x if pd.notna(P4x) else 48), (P4x if pd.notna(P4x) else 45, (P4x + 10) if pd.notna(P4x) else 55)]
        lines_info = lines_info if isinstance(lines_info, list) and len(lines_info)==5 else [(np.nan, np.nan)]*5
        plotted_handles = []; plotted_labels = []
        h_orig, = ax_single.plot([], [], label='Original Data', color='grey', alpha=0.6, linewidth=1.0); plotted_handles.append(h_orig); plotted_labels.append('Original Data')
        if not y_smooth.empty: h_smooth, = ax_single.plot([], [], label='Smoothed Data', color='black', alpha=0.8, linewidth=1.0, linestyle=':'); plotted_handles.append(h_smooth); plotted_labels.append('Smoothed Data')
        for i, ((m, c), (x_start, x_end), color, label) in enumerate(zip(lines_info, plot_ranges, colors, labels)):
             label_text = f'{label} (m={m:.2f})' if pd.notna(m) and not np.isinf(m) else label
             if i == 1 and np.isclose(m, 0) and pd.notna(c): x_line = np.linspace(x_start, x_end, 10); y_line = np.full_like(x_line, c); handle, = ax_single.plot(x_line, y_line, color=color, linestyle='--', linewidth=2); plotted_handles.append(handle); plotted_labels.append(label_text)
             elif pd.notna(m) and pd.notna(c) and not np.isinf(m) and pd.notna(x_start) and pd.notna(x_end) and x_start < x_end:
                 if np.isclose(x_start, x_end): x_end += 0.1
                 x_line = np.linspace(x_start, x_end, 10); y_line = m * x_line + c; handle, = ax_single.plot(x_line, y_line, color=color, linestyle='--', linewidth=2); plotted_handles.append(handle); plotted_labels.append(label_text)
             elif np.isinf(m) and pd.notna(c): handle = ax_single.axvline(c, label=label_text, color=color, linestyle='--', linewidth=2); plotted_handles.append(handle); plotted_labels.append(label_text)
        int_labels = ['P1', 'P2', 'P3', 'P4']; int_colors = ['cyan', 'magenta', 'orange', 'lime']
        for i, (px, py) in enumerate(intersections):
            if pd.notna(px) and pd.notna(py): handle = ax_single.scatter([px], [py], label=f'{int_labels[i]} ({px:.1f},{py:.1f})', color=int_colors[i], s=80, zorder=5, edgecolors='black'); plotted_handles.append(handle); plotted_labels.append(f'{int_labels[i]} ({px:.1f},{py:.1f})')
        ax_single.set_xlabel('Time (ns)', fontsize=20)
        ax_single.set_ylabel('Velocity (m/s)', fontsize=20)
        ax_single.set_title(f'Hybrid Dynamic/5-Segment Model: {base_filename}', fontsize=20)
        ax_single.legend(handles=plotted_handles, labels=plotted_labels, fontsize=14, loc='best')
        ax_single.grid(True, linestyle=':')
        ax_single.tick_params(axis='both', which='major', labelsize=20)
        fig_single.patch.set_facecolor('white')
        ax_single.set_facecolor('white')
        all_y = np.concatenate([arr for arr in [y_trace.values, y_smooth.values] if arr is not None and len(arr)>0]); all_y = all_y[np.isfinite(all_y)]
        if len(all_y) > 0: min_y, max_y = np.nanmin(all_y), np.nanmax(all_y); ax_single.set_ylim(min_y - 50, max_y + 100)
        else: ax_single.set_ylim(0, 800)
        ax_single.set_xlim(-5, x_trace.max() + 5 if not x_trace.empty else 60); fig_single.tight_layout(); plot_filename = os.path.join(output_folder, f"{base_filename}_hybrid_model_v3.png")
        logging.debug(f"    Attempting to save hybrid model plot to: {plot_filename}")
        try: plt.savefig(plot_filename, dpi=150); logging.debug(f"    Successfully saved hybrid model plot: {os.path.basename(plot_filename)}")
        except Exception as e: logging.error(f"    ERROR saving hybrid model plot {plot_filename}: {e}")
    except Exception as plot_err: logging.exception(f"    ERROR during hybrid model plot generation for {base_filename}: {plot_err}")
    finally:
        if 'fig_single' in locals() and fig_single is not None: logging.debug(f"    Closing hybrid model plot figure."); plt.close(fig_single)


# --- Main Processing Function (Hybrid V3 + P1 Constraint + Improved Recomp V2) ---

def calculate_spall_parameters(time, velocity, y_err_data, filename, output_folder,
                               density=utils.DENSITY_COPPER,
                               acoustic_velocity=utils.ACOUSTIC_VELOCITY_COPPER,
                               smooth_window=7, polyorder=3, prominence_factor=0.05,
                               peak_distance_ns=5.0,
                               plot_individual=True):
    """
    Processes a single velocity trace using a hybrid approach with user constraints:
    1. Finds dynamic minimum/recompaction peak (improved search V2).
    2. Defines Line 1 using (0,0) and interpolated point (trying t=6, 5, 4ns)
       to ensure P1 intersection < 10 ns.
    3. Defines Line 2 using mean velocity in fixed window [9, 21] ns.
    4. Defines Line 3 using dynamic minimum and interpolated point at t=28.5ns.
    5. Fits Lines 4, 5 based on dynamic features.
    6. Calculates intersections and parameters.
    """
    base_name = os.path.basename(filename)
    base_name_no_ext = os.path.splitext(base_name)[0]
    os.makedirs(output_folder, exist_ok=True)
    results = {'Filename': base_name_no_ext}
    logging.debug(f"  Starting hybrid analysis (V3 + P1 Constraint + Imp Recomp V2) for {base_name_no_ext}")

    # Initialize results - ** Ensure ALL variables used in final dict are initialized **
    u_fs_max_dyn, u_pullback_min_dyn, u_recomp_max_dyn = np.nan, np.nan, np.nan
    t_peak_ns, t_min_ns, t_recomp_ns = np.nan, np.nan, np.nan
    peak_idx, min_idx, recomp_idx = None, None, None
    plateau_mean_vel, plateau_std_dev = np.nan, np.nan
    t_plateau_start_fixed, t_plateau_end_fixed = 9.0, 21.0
    first_max_val, first_max_err = np.nan, np.nan
    min_val, min_err = np.nan, np.nan
    recomp_vel = np.nan # **** Initialize recomp_vel ****
    time_recomp = np.nan # **** Initialize time_recomp ****
    spall_str, spall_err = np.nan, np.nan
    strain_rate_val, strain_rate_err = np.nan, np.nan
    hugoniot_stress = np.nan
    rise_slope, pullback_slope, recomp_slope_val = np.nan, np.nan, np.nan
    lines_info = [(np.nan, np.nan)] * 5
    intersections = [(np.nan, np.nan)] * 4
    current_data_dict = {'filename': base_name}

    try:
        # --- Data Cleaning, Sorting, Shifting ---
        X_orig=pd.Series(time); Y_orig=pd.Series(velocity); YErr_orig=pd.Series(y_err_data); X=pd.to_numeric(X_orig, errors='coerce'); Y=pd.to_numeric(Y_orig, errors='coerce'); YErr=pd.to_numeric(YErr_orig, errors='coerce'); valid_indices=X.notna()&Y.notna();
        if not valid_indices.any(): raise ValueError("No valid numeric time/velocity rows.")
        X=X[valid_indices]; Y=Y[valid_indices]; YErr=YErr[valid_indices];
        if not X.is_monotonic_increasing: sort_idx=X.argsort(); X=X.iloc[sort_idx]; Y=Y.iloc[sort_idx]; YErr=YErr.iloc[sort_idx]
        X=X.reset_index(drop=True); Y=Y.reset_index(drop=True); YErr=YErr.reset_index(drop=True);
        if len(X)<2: raise ValueError("< 2 valid rows after cleaning/sorting.")
        x_vals_for_grad=X.values; y_vals_for_grad=Y.values;
        if len(x_vals_for_grad)<2 : raise ValueError("Need >= 2 points for gradient.")
        with np.errstate(divide='ignore', invalid='ignore'): dY_dX=np.gradient(y_vals_for_grad, x_vals_for_grad)
        positive_slope_indices=np.where(dY_dX > 0)[0]; idx_shift = 0
        if len(positive_slope_indices)>0:
            try: candidate_values=Y.iloc[positive_slope_indices]; closest_label=np.abs(candidate_values - 30).idxmin(); idx_shift=Y.index.get_loc(closest_label)
            except Exception: idx_shift = 0
        if not (0 <= idx_shift < len(X)): idx_shift = 0
        t_shift=X.iloc[idx_shift]; X_shifted=(X - t_shift)*1e9; mask=X_shifted >= 0;
        if not mask.any(): raise ValueError("No data >= 0 ns after shift.")
        X_shifted_ns=X_shifted[mask].reset_index(drop=True); Y_filtered=Y[mask].reset_index(drop=True); YErr_filtered=YErr[mask].reset_index(drop=True); min_required_len = smooth_window if 'smooth_window' in locals() else 5
        if len(X_shifted_ns)<min_required_len: raise ValueError(f"Not enough points ({len(X_shifted_ns)}) after shift for smoothing window {min_required_len}.")
        current_data_dict['x_shifted']=X_shifted_ns.copy(); current_data_dict['y_original']=Y_filtered.copy();

        # --- Smoothing ---
        logging.debug(f"  Applying Savitzky-Golay filter: window={smooth_window}, polyorder={polyorder}")
        if smooth_window % 2 == 0: smooth_window += 1
        min_win = polyorder + 1 + (polyorder % 2); smooth_window = max(smooth_window, min_win)
        if len(Y_filtered)<smooth_window: raise ValueError(f"Data length ({len(Y_filtered)}) too short for smoothing window {smooth_window}.")
        Y_smooth=savgol_filter(Y_filtered, window_length=smooth_window, polyorder=polyorder); current_data_dict['y_smooth']=pd.Series(Y_smooth, index=Y_filtered.index);

        # --- Step 1: Dynamic Feature Detection (Min & Recomp) ---
        logging.debug("  Detecting dynamic features (Min/Recomp)...")
        dt_ns = np.mean(np.diff(X_shifted_ns)) if len(X_shifted_ns)>1 else 1.0; min_dist_samples = max(1, int(peak_distance_ns / dt_ns));
        vel_range = np.ptp(Y_smooth) if len(Y_smooth)>1 else 0; prominence_threshold = vel_range * prominence_factor if vel_range > 0 else 1.0;
        logging.debug(f"    Prominence threshold: {prominence_threshold:.1f} m/s")

        # Find Peak Time (needed for L1/L3 range definition)
        search_end_idx_peak = X_shifted_ns[X_shifted_ns <= 30].index.max() if any(X_shifted_ns <= 30) else len(X_shifted_ns)//2; search_end_idx_peak = max(search_end_idx_peak, 1)
        peaks, _ = find_peaks(Y_smooth[:search_end_idx_peak+1], prominence=prominence_threshold*0.5, distance=min_dist_samples//2)
        if len(peaks)>0: peak_idx=peaks[np.argmax(Y_smooth[peaks])]; u_fs_max_dyn=Y_smooth[peak_idx]; t_peak_ns=X_shifted_ns[peak_idx]; logging.debug(f"    Found Dynamic Peak: Idx={peak_idx}, Time={t_peak_ns:.1f} ns, Vel={u_fs_max_dyn:.1f} m/s")
        else: peak_idx=np.argmax(Y_smooth[:search_end_idx_peak+1]); u_fs_max_dyn=Y_smooth[peak_idx]; t_peak_ns=X_shifted_ns[peak_idx]; logging.warning(f"    No prominent peak found, using max value: Idx={peak_idx}, Time={t_peak_ns:.1f} ns, Vel={u_fs_max_dyn:.1f} m/s")
        if peak_idx is None or pd.isna(t_peak_ns): raise ValueError("Could not determine dynamic peak velocity time.")

        # Find Pullback Minimum Time and Value
        search_start_idx_min_dyn = peak_idx + min_dist_samples
        if search_start_idx_min_dyn < len(Y_smooth): minima, props = find_peaks(-Y_smooth[search_start_idx_min_dyn:], prominence=prominence_threshold, distance=min_dist_samples)
        if 'minima' in locals() and len(minima)>0: min_idx_rel=minima[0]; min_idx=search_start_idx_min_dyn+min_idx_rel; u_pullback_min_dyn=Y_smooth[min_idx]; t_min_ns=X_shifted_ns[min_idx]; logging.debug(f"    Found Dynamic Pullback Minimum: Idx={min_idx}, Time={t_min_ns:.1f} ns, Vel={u_pullback_min_dyn:.1f} m/s")
        else: logging.warning("    No prominent pullback minimum found dynamically.")
        if min_idx is None or pd.isna(t_min_ns): raise ValueError("Could not determine dynamic pullback minimum time.")
        min_val = u_pullback_min_dyn
        if min_idx is not None and min_idx < len(YErr_filtered): min_err = YErr_filtered.iloc[min_idx]
        else: min_err = np.nan

        # Find Recompaction Peak Time and Value (Improved Search V2)
        search_start_idx_recomp = min_idx + min_dist_samples
        search_end_idx_recomp = len(X_shifted_ns) - 1 # Search till end of data
        recomp_prominence = prominence_threshold * 0.75 # Try slightly lower prominence
        logging.debug(f"    Searching for recomp peak between index {search_start_idx_recomp} and {search_end_idx_recomp} with prominence >= {recomp_prominence:.2f}")
        if search_start_idx_recomp < search_end_idx_recomp:
             recomp_peaks, recomp_props = find_peaks(Y_smooth[search_start_idx_recomp:search_end_idx_recomp+1], prominence=recomp_prominence, distance=min_dist_samples)
             logging.debug(f"      find_peaks results: indices={recomp_peaks}, props={recomp_props}")
             if len(recomp_peaks) > 0:
                 recomp_idx_rel = recomp_peaks[0]; recomp_idx = search_start_idx_recomp + recomp_idx_rel
                 u_recomp_max_dyn = Y_smooth[recomp_idx]; t_recomp_ns = X_shifted_ns[recomp_idx]
                 logging.debug(f"    Found Dynamic Recompaction Peak: Idx={recomp_idx}, Time={t_recomp_ns:.1f} ns, Vel={u_recomp_max_dyn:.1f} m/s")
             else: logging.warning("    No prominent recompaction peak found after minimum in search window.")
        else: logging.warning("    Not enough data after minimum to search for recompaction peak.")
        recomp_vel = u_recomp_max_dyn if pd.notna(u_recomp_max_dyn) else np.nan


        # --- Step 2: Define Line 2 ---
        logging.debug("  Defining Line 2...")
        t_plateau_start_fixed = 9.0; t_plateau_end_fixed = 21.0; plateau_mask = (X_shifted_ns >= t_plateau_start_fixed) & (X_shifted_ns <= t_plateau_end_fixed)
        if plateau_mask.sum() > 1: Y_plateau_smooth = Y_smooth[plateau_mask]; plateau_mean_vel = np.nanmean(Y_plateau_smooth); plateau_std_dev = np.nanstd(Y_plateau_smooth); logging.debug(f"    Plateau Calculated (Fixed Window [{t_plateau_start_fixed:.1f},{t_plateau_end_fixed:.1f}]): Mean={plateau_mean_vel:.1f}, StdDev={plateau_std_dev:.1f}")
        else: logging.warning(f"    Not enough points ({plateau_mask.sum()}) in fixed plateau window [{t_plateau_start_fixed:.1f}, {t_plateau_end_fixed:.1f}] ns. Using NaN."); plateau_mean_vel = np.nan; plateau_std_dev = np.nan;
        final_m2, final_c2 = 0.0, plateau_mean_vel
        if pd.isna(final_c2): raise ValueError("Failed to determine plateau level (c2).")
        logging.debug(f"    Line 2 defined: m={final_m2:.2f}, c={final_c2:.2f}"); first_max_val = plateau_mean_vel; first_max_err = plateau_std_dev;

        # --- Step 3: Define Line 1 and Calculate P1 (Iterative with Constraint) ---
        logging.debug("  Defining Line 1 and P1 (Constraint: P1.x < 10 ns)..."); p1_l1 = (0.0, 0.0); t_l1_options = [6.0, 5.0, 4.0]; p1_intersect = (np.nan, np.nan); final_m1, final_c1 = np.nan, np.nan; p1_found_valid = False
        for t_l1_try in t_l1_options:
            logging.debug(f"    Trying Line 1 with point at t={t_l1_try} ns..."); y_p2_l1 = _get_interp_y(X_shifted_ns, Y_filtered, t_l1_try);
            if pd.isna(y_p2_l1): logging.warning(f"      Could not interpolate raw data at t={t_l1_try}ns. Skipping this attempt."); continue
            p2_l1 = (t_l1_try, y_p2_l1); m1_try, c1_try = _fit_line_through_points(p1_l1, p2_l1);
            if pd.isna(m1_try) or pd.isna(c1_try): logging.warning(f"      Failed to fit Line 1 through (0,0) and {p2_l1}. Skipping this attempt."); continue
            logging.debug(f"      Line 1 (try t={t_l1_try}): m={m1_try:.2f}, c={c1_try:.2f}"); p1_try = _find_intersection(m1_try, c1_try, final_m2, final_c2); logging.debug(f"      Calculated P1 (try t={t_l1_try}): {p1_try}")
            if pd.notna(p1_try[0]) and p1_try[0] < 10.0 and p1_try[0] >= 0: final_m1, final_c1 = m1_try, c1_try; p1_intersect = p1_try; p1_found_valid = True; logging.debug(f"    VALID P1 found with t={t_l1_try}ns: {p1_intersect}"); break
            else: logging.debug(f"      P1 with t={t_l1_try}ns is invalid or >= 10ns.")
        if not p1_found_valid: raise ValueError("Could not find a valid P1 intersection with P1.x < 10 ns after trying multiple Line 1 definitions.")
        logging.debug(f"    Final Line 1: m={final_m1:.2f}, c={final_c1:.2f}"); logging.debug(f"    Final P1: {p1_intersect}")

        # --- Step 4: Define Line 3 ---
        logging.debug("  Defining Line 3..."); p_min_l3 = (t_min_ns, u_pullback_min_dyn); t_mid_l3 = 28.5; y_mid_l3 = _get_interp_y(X_shifted_ns, Y_filtered, t_mid_l3);
        if pd.isna(y_mid_l3): raise ValueError(f"Could not interpolate raw data at t={t_mid_l3}ns for Line 3.")
        p_mid_l3 = (t_mid_l3, y_mid_l3); logging.debug(f"    Line 3 points: {p_min_l3}, {p_mid_l3}"); final_m3, final_c3 = _fit_line_through_points(p_min_l3, p_mid_l3);
        if pd.isna(final_m3) or pd.isna(final_c3): raise ValueError("Failed to fit Line 3 through specified points.")
        logging.debug(f"    Line 3 fitted: m={final_m3:.2f}, c={final_c3:.2f}"); pullback_slope = final_m3;

        # --- Step 5: Calculate P2 ---
        p2_intersect = _find_intersection(final_m2, final_c2, final_m3, final_c3);
        if pd.isna(p2_intersect[0]): raise ValueError("Failed to find P2 intersection.")
        if pd.notna(p1_intersect[0]) and p2_intersect[0] < p1_intersect[0]: logging.warning(f"Calculated P2 time ({p2_intersect[0]:.1f}) is before P1 time ({p1_intersect[0]:.1f}). Check definitions.")
        logging.debug(f"    Calculated P2: {p2_intersect}")

        # --- Step 6: Fit Lines 4 & 5 ---
        logging.debug("  Fitting Lines 4 & 5...")
        # Initialize L4/L5 params to NaN before attempting fit
        final_m4, final_c4 = np.nan, np.nan
        final_m5, final_c5 = np.nan, np.nan
        if pd.notna(t_recomp_ns):
            final_m4, final_c4 = _fit_line_to_range(X_shifted_ns, Y_smooth, t_min_ns, t_recomp_ns)
            if pd.isna(final_m4) or pd.isna(final_c4):
                logging.warning(f"Failed to fit Line 4 ({t_min_ns:.1f} to {t_recomp_ns:.1f} ns). Setting L4/L5 params to NaN.")
                final_m4, final_c4 = np.nan, np.nan # Ensure NaN if fit failed
            logging.debug(f"    Line 4 fitted: m={final_m4}, c={final_c4}")

            if pd.notna(final_m4): # Only attempt L5 if L4 was successful
                l5_duration = 5.0
                l5_start = t_recomp_ns # Define l5_start here
                l5_end = t_recomp_ns + l5_duration # Define l5_end here
                final_m5, final_c5 = _fit_line_to_range(X_shifted_ns, Y_smooth, l5_start, l5_end)
                if pd.isna(final_m5) or pd.isna(final_c5):
                    # **** Corrected Warning Message (Removed undefined variables) ****
                    logging.warning(f"Failed to fit Line 5 (starting near {t_recomp_ns:.1f} ns).")
                    final_m5, final_c5 = np.nan, np.nan # Ensure NaN if failed
        else:
             logging.warning("Recompaction peak time (t_recomp_ns) not found. Skipping Line 4 and 5 fitting.")
             # Ensure L4/L5 are NaN if t_recomp_ns is NaN
             final_m4, final_c4 = np.nan, np.nan
             final_m5, final_c5 = np.nan, np.nan

        logging.debug(f"    Line 5 fitted: m={final_m5}, c={final_c5}");
        lines_info = [(final_m1, final_c1), (final_m2, final_c2), (final_m3, final_c3), (final_m4, final_c4), (final_m5, final_c5)];

        # --- Step 7: Calculate P3 & P4 Intersections ---
        p3_intersect = _find_intersection(final_m3, final_c3, final_m4, final_c4); p4_intersect = _find_intersection(final_m4, final_c4, final_m5, final_c5); intersections = [p1_intersect, p2_intersect, p3_intersect, p4_intersect];
        if pd.isna(p3_intersect[0]): logging.warning("Failed to find P3 intersection.")
        if pd.isna(p4_intersect[0]): logging.warning("Failed to find P4 intersection.")
        logging.debug(f"    Calculated P3={p3_intersect}, P4={p4_intersect}")

        # --- Step 8: Parameter Calculation ---
        logging.debug("  Calculating derived parameters...")
        # recomp_vel and time_recomp already assigned using dynamic values if available
        # ** Calculate time_recomp (in seconds) for results **
        time_recomp = t_recomp_ns * 1e-9 if pd.notna(t_recomp_ns) else p4_intersect[0] * 1e-9 if pd.notna(p4_intersect[0]) else np.nan

        if pd.notna(first_max_val) and pd.notna(min_val): delta_v_spall = first_max_val - min_val;
        if 'delta_v_spall' in locals() and delta_v_spall > 0: spall_str = (0.5 * density * acoustic_velocity * delta_v_spall) * 1e-9
        else: logging.warning(f"    Calculated delta_v_spall is not positive ({delta_v_spall:.2f}). Spall strength is NaN.")
        if pd.notna(first_max_err) and pd.notna(min_err): first_max_err_sq = max(0, first_max_err**2); min_err_sq = max(0, min_err**2); spall_err = (0.5 * density * acoustic_velocity * 1e-9) * np.sqrt(first_max_err_sq + min_err_sq)

        # Strain Rate Calculation (Syntax Fixed)
        if pd.notna(pullback_slope) and acoustic_velocity > 0:
            strain_rate_val = -pullback_slope * 1e9 / (2 * acoustic_velocity) # pullback_slope is in m/s/ns
            # Error propagation for strain rate (using simple proxy)
            if pd.notna(first_max_val) and pd.notna(min_val) and not np.isclose(first_max_val, min_val) \
               and pd.notna(first_max_err) and pd.notna(min_err):
                 first_max_err_sq = max(0, first_max_err**2)
                 min_err_sq = max(0, min_err**2)
                 strain_rate_err = abs(strain_rate_val) * np.sqrt(first_max_err_sq + min_err_sq) / abs(first_max_val - min_val)
            else:
                 strain_rate_err = np.nan
        else: # This else corresponds to the outer if: pullback_slope is NaN or acoustic_velocity <= 0
             # **** CORRECTED SYNTAX: NO SEMICOLON, CORRECT INDENTATION ****
             strain_rate_val, strain_rate_err = np.nan, np.nan
             if pd.isna(pullback_slope):
                  logging.warning("    Pullback slope (Line 3) is NaN, cannot calculate Strain Rate.")
        # **** END OF CORRECTED BLOCK ****

        # Hugoniot Stress
        if pd.notna(first_max_val): hugoniot_stress=(0.5*density*acoustic_velocity*first_max_val)*1e-9
        rise_slope = final_m1; recomp_slope_val = final_m4; results['Processing Status']='Success'
    except Exception as e:
        logging.error(f"  ERROR processing file {base_name}: {e}"); results['Processing Status']=f'Failed: {type(e).__name__} - {e}'
        # Ensure ALL variables used in the final results dict are defined here
        plateau_mean_vel, plateau_std_dev = np.nan, np.nan; spall_str, spall_err = np.nan, np.nan
        strain_rate_val, strain_rate_err = np.nan, np.nan; hugoniot_stress = np.nan
        rise_slope, pullback_slope, recomp_slope_val = np.nan, np.nan, np.nan
        first_max_err, min_err = np.nan, np.nan
        u_fs_max_dyn, u_pullback_min_dyn, u_recomp_max_dyn = np.nan, np.nan, np.nan
        t_peak_ns, t_min_ns, t_recomp_ns = np.nan, np.nan, np.nan
        lines_info = [(np.nan, np.nan)] * 5; intersections = [(np.nan, np.nan)] * 4
        time_recomp = np.nan # **** Initialize time_recomp in except block ****
        recomp_vel = np.nan # **** Initialize recomp_vel in except block ****

    # Populate results dictionary - ensure all keys use defined variables
    results['Peak Velocity (m/s)']=u_fs_max_dyn
    results['Plateau Mean Velocity (m/s)']=plateau_mean_vel
    results['Plateau Velocity StdDev (m/s)']=plateau_std_dev
    results['Pullback Minimum (m/s)']=u_pullback_min_dyn
    results['Pullback Minimum Err (m/s)']=min_err
    results['Recompression Peak (m/s)']=u_recomp_max_dyn
    results['Time at Peak (ns)']=t_peak_ns
    results['Time at Minimum (ns)']=t_min_ns
    results['Time at Recompression Peak (ns)']=t_recomp_ns
    results['Plateau Start Time (ns)']=t_plateau_start_fixed
    results['Plateau End Time (ns)']=t_plateau_end_fixed
    results['Spall Strength (GPa)']=spall_str
    results['Spall Strength Err (GPa)']=spall_err
    results['Strain Rate (s^-1)']=strain_rate_val
    results['Strain Rate Err (s^-1)']=strain_rate_err
    results['Hugoniot Stress (GPa)']=hugoniot_stress
    results['Rise Slope (m/s/s)']=rise_slope*1e18 if pd.notna(rise_slope) else np.nan
    results['Pullback Slope (m/s/s)']=pullback_slope*1e18 if pd.notna(pullback_slope) else np.nan
    results['Recompression Slope (m/s/s)']=recomp_slope_val*1e18 if pd.notna(recomp_slope_val) else np.nan
    # Aliases for compatibility
    results['First Maxima (m/s)']=plateau_mean_vel
    results['First Maxima Err (m/s)']=plateau_std_dev
    results['Minima (m/s)']=u_pullback_min_dyn
    results['Minima Err (m/s)']=min_err
    results['Recompression Velocity (m/s)']=recomp_vel # Use initialized/calculated recomp_vel
    results['Time at Recompression Peak (s)']=time_recomp # Use initialized/calculated time_recomp
    # Fitted model info (optional to keep)
    results['model_lines_info']=tuple(lines_info) if lines_info else tuple([(np.nan, np.nan)]*5)
    results['model_intersections']=tuple(intersections) if intersections else tuple([(np.nan, np.nan)]*4)

    # Plotting call
    if plot_individual and results['Processing Status']=='Success':
        current_data_dict['features']={'Peak Idx': peak_idx, 'Peak Velocity (m/s)': u_fs_max_dyn, 'Min Idx': min_idx, 'Min Vel (m/s)': u_pullback_min_dyn, 'Recomp Idx': recomp_idx, 'Recomp Vel (m/s)': u_recomp_max_dyn}
        _plot_trace_with_hybrid_model(current_data_dict, lines_info, intersections, output_folder)
    elif plot_individual:
        logging.warning(f"Skipping individual plot for {base_name} due to processing failure.")

    logging.debug(f"  Finished hybrid analysis for {base_name_no_ext}")
    return results


def process_velocity_files(input_folder, file_pattern, output_folder,
                           save_summary_table=True, summary_table_name=None,
                           **kwargs):
    """
    Processes velocity files using the hybrid approach V3 + P1 Constraint via
    `calculate_spall_parameters` and aggregates results.
    Passes smoothing/peak finding parameters via kwargs. Adds 'Material' column.
    """
    # (Function content remains the same as previous version - already cleaned up)
    files = utils.find_data_files(input_folder, file_pattern)
    if not files: logging.warning(f"No files found matching pattern '{file_pattern}' in '{input_folder}'."); return pd.DataFrame()
    all_results = []; processed_count = 0; error_count = 0
    table_output_dir = os.path.join(output_folder, 'tables'); plot_output_dir = output_folder
    os.makedirs(table_output_dir, exist_ok=True); os.makedirs(plot_output_dir, exist_ok=True)
    logging.info(f"\nStarting hybrid processing (V3 + P1 Constraint) of {len(files)} files in '{input_folder}'...")
    subfolder_name = os.path.basename(os.path.normpath(input_folder));
    logging.debug(f"DEBUG: Extracting material for folder: '{subfolder_name}' using mapping: {utils.MATERIAL_MAPPING}")
    material_label_tuple = utils.extract_legend_info(subfolder_name, utils.MATERIAL_MAPPING, utils.ENERGY_VELOCITY_MAPPING) # Pass correct energy map
    logging.debug(f"DEBUG: Raw result from extract_legend_info: {material_label_tuple}")
    material_label = material_label_tuple[0] if material_label_tuple is not None else None # Safely access tuple

    if material_label is None: material_label = "Unknown"; logging.warning(f"Could not determine material label for folder '{subfolder_name}'. Using '{material_label}'.")
    logging.info(f"Determined material label for this bin: '{material_label}'")
    for f in files:
        base_name = os.path.basename(f); logging.info(f"Processing: {base_name}")
        try:
            data = pd.read_csv(f, header='infer', on_bad_lines='skip', engine='python')
            if data.shape[1] < 3: logging.warning(f"Skipping {base_name}: Expected >= 3 columns (T,V,Err), found {data.shape[1]}."); error_count += 1; all_results.append({'Filename': os.path.splitext(base_name)[0], 'Material': material_label, 'Processing Status': 'Failed: Incorrect Columns'}); continue
            time_data = data.iloc[:, 0].to_numpy(); velocity_data = data.iloc[:, 1].to_numpy(); y_err_data = data.iloc[:, 2].to_numpy()
            calc_args = {'density': kwargs.get('density', utils.DENSITY_COPPER), 'acoustic_velocity': kwargs.get('acoustic_velocity', utils.ACOUSTIC_VELOCITY_COPPER), 'plot_individual': kwargs.get('plot_individual', True), 'smooth_window': kwargs.get('smooth_window', 7), 'polyorder': kwargs.get('polyorder', 3), 'prominence_factor': kwargs.get('prominence_factor', 0.05), 'peak_distance_ns': kwargs.get('peak_distance_ns', 5.0)}
            result = calculate_spall_parameters(time_data, velocity_data, y_err_data, f, plot_output_dir, **calc_args)
            if result:
                result['Material'] = material_label; all_results.append(result)
                if result.get('Processing Status') == 'Success': processed_count += 1
                else: error_count += 1; logging.warning(f"  -> Processing failed/partially failed: {result.get('Processing Status')}")
            else: error_count += 1; logging.error(f"  -> Processing failed critically: No result returned."); all_results.append({'Filename': os.path.splitext(base_name)[0], 'Material': material_label, 'Processing Status': 'Failed: Critical Error'})
        except pd.errors.EmptyDataError: logging.warning(f"Skipping {base_name}: File is empty."); error_count += 1; all_results.append({'Filename': os.path.splitext(base_name)[0], 'Material': material_label, 'Processing Status': 'Failed: Empty File'})
        except ValueError as ve: logging.warning(f"Skipping {base_name}: Processing error - {ve}"); error_count += 1; all_results.append({'Filename': os.path.splitext(base_name)[0], 'Material': material_label, 'Processing Status': f'Failed: ValueError - {ve}'})
        except Exception as e: logging.exception(f"Failed to load or process {base_name}: {e}"); error_count += 1; all_results.append({'Filename': os.path.splitext(base_name)[0], 'Material': material_label, 'Processing Status': f'Failed: {type(e).__name__}'})
    logging.info(f"\n--- Processing Summary for Bin '{os.path.basename(input_folder)}' ---"); logging.info(f"- Traces processed (incl. failures): {len(all_results)}"); logging.info(f"- Traces with 'Success' status: {processed_count}"); logging.info(f"- Traces with 'Failed' status: {error_count}")
    if not all_results: return pd.DataFrame()
    results_df = pd.DataFrame(all_results)
    # ** UPDATED Column Order for Hybrid V3 Output (Added Material) **
    cols_order = ['Filename', 'Material', 'Peak Velocity (m/s)', 'Plateau Mean Velocity (m/s)', 'Plateau Velocity StdDev (m/s)', 'Pullback Minimum (m/s)', 'Pullback Minimum Err (m/s)', 'Recompression Peak (m/s)', 'Time at Peak (ns)', 'Time at Minimum (ns)', 'Time at Recompression Peak (ns)', 'Plateau Start Time (ns)', 'Plateau End Time (ns)', 'Spall Strength (GPa)', 'Spall Strength Err (GPa)', 'Strain Rate (s^-1)', 'Strain Rate Err (s^-1)', 'Hugoniot Stress (GPa)', 'Rise Slope (m/s/s)', 'Pullback Slope (m/s/s)', 'Recompression Slope (m/s/s)', 'First Maxima (m/s)', 'First Maxima Err (m/s)', 'Minima (m/s)', 'Minima Err (m/s)', 'Recompression Velocity (m/s)', 'Time at Recompression Peak (s)', 'model_lines_info', 'model_intersections', 'Processing Status']
    # Add aliases if they don't exist
    if 'Plateau Mean Velocity (m/s)' in results_df.columns: results_df['First Maxima (m/s)'] = results_df['Plateau Mean Velocity (m/s)']
    if 'Plateau Velocity StdDev (m/s)' in results_df.columns: results_df['First Maxima Err (m/s)'] = results_df['Plateau Velocity StdDev (m/s)']
    if 'Pullback Minimum (m/s)' in results_df.columns: results_df['Minima (m/s)'] = results_df['Pullback Minimum (m/s)']
    if 'Pullback Minimum Err (m/s)' in results_df.columns: results_df['Minima Err (m/s)'] = results_df['Pullback Minimum Err (m/s)']
    if 'Recompression Peak (m/s)' in results_df.columns: results_df['Recompression Velocity (m/s)'] = results_df['Recompression Peak (m/s)']
    if 'Time at Recompression Peak (ns)' in results_df.columns: results_df['Time at Recompression Peak (s)'] = results_df['Time at Recompression Peak (ns)'] * 1e-9
    # Ensure all requested columns exist, adding NaNs if necessary
    for col in cols_order:
        if col not in results_df.columns:
            if col in ['model_lines_info', 'model_intersections']: results_df[col] = pd.Series([None] * len(results_df), dtype=object)
            else: results_df[col] = np.nan
    results_df = results_df[cols_order] # Reorder
    if save_summary_table:
        if summary_table_name is None: input_folder_basename = os.path.basename(os.path.normpath(input_folder)); summary_table_name = f"{input_folder_basename}_results_table_hybrid_v3.csv" # ** New default name **
        results_output_filename = os.path.join(table_output_dir, summary_table_name)
        try:
            df_to_save = results_df.copy()
            for col in ['model_lines_info', 'model_intersections']:
                if col in df_to_save.columns: df_to_save[col] = df_to_save[col].apply(lambda x: str(x) if x is not None else '')
            df_to_save.to_csv(results_output_filename, index=False, float_format='%.4e');
            logging.info(f"\nBin summary table (hybrid V3) saved to: {results_output_filename}")
        except Exception as e: logging.error(f"Could not save summary table {results_output_filename}: {e}")
    return results_df

