To gather, organize and analyze ALPSS outputs for final combined analysis 
Each code to be run sequentially

Analysis pipeline

ALPSS out puts 

Velocity_smooth, velocity and velocity uncert files

All files are combined in respective folders created according to this naming sequence
XX_YYYYmJ_shots
XX- grain size
YYYY is laser energy during shot
 

Run: 1_cocatenate_velocity_smooth_and_vel_uncertpy
This files takes file input of velocity_smooth and vel_uncert and creates a new file velocity which has first column as time, second as velocity (smooth) and 3rd as uncertainity (from ALPSS)

Example:
# Specify the directory path
directory = '/Users/piyushwanchoo/Library/CloudStorage/OneDrive-JohnsHopkins/Malon_PDV_scope/Cu_Polycrystals/Cu_100nm/100nm_1600mJ_shots'

# Get all files with the specified patterns
smooth_files = glob.glob(os.path.join(directory, '*--velocity--smooth.csv'))
uncert_files = glob.glob(os.path.join(directory, '*--vel--uncert.csv'))



use 2_spall_data_table_creation_v_3_with_unc.py to create spall table for each data 

First Maxima (m/s)  First Maxima Err (m/s)  Minima (m/s)  \
0             312.493650                0.941706    140.020773   
1             309.509505                2.000971    128.749249   
2             318.470180                2.076266    134.674939   
3             313.318594                1.405302    125.940035   
4             309.272567                2.268261    121.451760   
...                  ...                     ...           ...   
4659                 NaN                     NaN           NaN   
4660                 NaN                     NaN           NaN   
4661                 NaN                     NaN           NaN   
4662                 NaN                     NaN           NaN   
4663                 NaN                     NaN           NaN   

      Minima Err (m/s)  Recompression Velocity (m/s)  \
0             1.389506                    265.603820   
1             3.487822                    284.322122   
2             1.966222                    291.011269   
3             2.020064                    279.255044   
4             1.995153                    276.589651 


3_u_t_plot_comparison
Plot free surface velocities of all experimental data in the study 
Has a custom legend which requires user specification and requires data from free surface velocity acquired through velocity shots.

4_spall_strength_f_strain_rate_w_lit_review 
Gather and plot all data in the study with literature data (if available or can be commented out)

5_spall_strength_f_shock_stress_w_lit_review 



6_Effect_of_strain_rate_with_error/ 6_Effect_of_strain_rate_with_error_2

Plots with errors propagated and compared at different laser energy levels


8_LASSO regression






