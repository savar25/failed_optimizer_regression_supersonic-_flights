# %%
import matplotlib.pyplot as plt

import openmdao.api as om

import dymos as dm
from dymos.examples.plotting import plot_results
from min_time_climb_ode import MinTimeClimbODE
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import normalize
from scipy.interpolate import interp1d

# %%

data_num=1
os.makedirs("Data_Files", exist_ok=True)
os.makedirs("Data_Images", exist_ok=True)

def interpolate_arrays(x: np.ndarray, x_exp: np.ndarray):
    target_length = max(len(x.flatten()), len(x_exp.flatten()))
    
    x_indices = np.linspace(0, 1, len(x.flatten()))
    x_exp_indices = np.linspace(0, 1, len(x_exp.flatten()))
    
    interpolator_x = interp1d(x_indices, x.flatten(), kind='linear', fill_value='extrapolate')
    interpolator_x_exp = interp1d(x_exp_indices, x_exp.flatten(), kind='linear', fill_value='extrapolate')
    
    new_indices = np.linspace(0, 1, target_length)
    x_resampled = interpolator_x(new_indices)
    x_exp_resampled = interpolator_x_exp(new_indices)
    
    return x_resampled, x_exp_resampled


#max_iter,nsegments,order, tol
def run_sim(max_iter=4,n_segments=15,order=6,tol=5,data_num=1):
    p = om.Problem(model=om.Group())

    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.declare_coloring()
    p.driver.options['disp'] = False
    # p.driver.opt_settings['maxtime'] = 10  # seconds
    # p.driver.opt_settings['iters'] = 3
    p.driver.options['maxiter'] = max_iter
    p.driver.options['tol'] = 1.0/(10**tol)


    #
    # Instantiate the trajectory and phase
    #
    traj = dm.Trajectory()

  
    phase = dm.Phase(ode_class=MinTimeClimbODE,
                    transcription=dm.Radau(num_segments=n_segments, order=order,compressed=False))
    
    traj.add_phase('phase0', phase)

    # %%
    p.model.add_subsystem('traj', traj)

    #
    # Set the options on the optimization variables
    # Note the use of explicit state units here since much of the ODE uses imperial units
    # and we prefer to solve this problem using metric units.
    #
    phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                        duration_ref=100.0)

    phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6, units='m',
                    ref=1.0E3, defect_ref=1.0E3,
                    rate_source='flight_dynamics.r_dot')

    phase.add_state('h', fix_initial=True, lower=0, upper=20000.0, units='m',
                    ref=1.0E2, defect_ref=1.0E2,
                    rate_source='flight_dynamics.h_dot')

    phase.add_state('v', fix_initial=True, lower=10.0, units='m/s',
                    ref=1.0E2, defect_ref=1.0E2,
                    rate_source='flight_dynamics.v_dot')

    phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5, units='rad',
                    ref=1.0, defect_ref=1.0,
                    rate_source='flight_dynamics.gam_dot')

    phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5, units='kg',
                    ref=1.0E3, defect_ref=1.0E3,
                    rate_source='prop.m_dot')

    phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                    rate_continuity=True, rate_continuity_scaler=100.0,
                    rate2_continuity=False)

    phase.add_parameter('S', val=49.2386, units='m**2', opt=False, targets=['S'])
    phase.add_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
    phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

    #
    # Setup the boundary and path constraints
    #
    phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3)
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0)
    phase.add_boundary_constraint('gam', loc='final', equals=0.0)

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', ref=1.0)

    p.model.linear_solver = om.DirectSolver()

    #
    # Setup the problem and set the initial guess
    #
    p.setup(check=True)

    phase.set_time_val(initial=0.0, duration=350)
    phase.set_state_val('r', [0.0, 50000.0])
    phase.set_state_val('h', [100.0, 20000.0])
    phase.set_state_val('v', [135.964, 283.159])
    phase.set_state_val('gam', [0.0, 0.0])
    phase.set_state_val('m', [19030.468, 10000.])
    phase.set_control_val('alpha', [2.0, 5.0])

    #
    # Solve for the optimal trajectory
    #
    sol=p.run_driver()
    sim=traj.simulate()

    # %%
    # %matplotlib inline
    x = p.get_val('traj.phase0.timeseries.h')
    y = p.get_val('traj.phase0.timeseries.v')
    t = p.get_val('traj.phase0.timeseries.time')


    x_exp = sim.get_val('traj.phase0.timeseries.h')
    y_exp = sim.get_val('traj.phase0.timeseries.v')
    t_exp = sim.get_val('traj.phase0.timeseries.time')

    combined = np.concatenate([x.flatten(), y.flatten(), x_exp.flatten(), y_exp.flatten()]).reshape(1, -1)

# Normalize the combined vector
    combined_n = normalize(combined, norm='l2', axis=1)

   # Reshape back into original shapes
    split_indices = [x.size, y.size, x_exp.size]  # Indices to split at
    x_n, y_n, x_exp_n, y_exp_n = np.split(combined_n.flatten(), np.cumsum(split_indices))

   # Reshape back to original shapes
    x_n = x_n.reshape(x.shape)
    y_n = y_n.reshape(y.shape)
    x_exp_n = x_exp_n.reshape(x_exp.shape)
    y_exp_n = y_exp_n.reshape(y_exp.shape)

    x_n,x_exp_n=interpolate_arrays(x_n,x_exp_n)
    y_n,y_exp_n=interpolate_arrays(y_n,y_exp_n)
    t,t_exp=interpolate_arrays(t,t_exp)

    fig, axes = plt.subplots(nrows=2, ncols=1)

    axes[0].plot(t, x_n, 'o')
    axes[0].plot(t_exp, x_exp_n, '-')
    axes[0].set_ylabel('h_norm (m)')
    axes[0].set_xlabel('time (s)')

    axes[1].plot(t, y_n, 'o')
    axes[1].plot(t_exp, y_exp_n, '-')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('v_norm (m/s)')

    plot_filename = os.path.join(os.path.dirname(__file__), f"Data_Images/trajectory_plot_{data_num}.png")
    print(plot_filename)
    # plot_filename = f"Data_Images/trajectory_plot_{data_num}.png"
    plt.savefig(plot_filename, dpi=300)

      # Change this as needed
    csv_filename = os.path.join(os.path.dirname(__file__), f"Data_Files/data_case_{data_num}.csv")
   #  # csv_filename = f"Data_Files/data_case_{data_num}.csv"
   #  print(len(x.flatten()))
   #  print(len(x_exp.flatten()))
   #  length_var=len(x.flatten())-len(x_exp.flatten())

    
    data = pd.DataFrame({
         "time": t.flatten(),
         "h": x_n.flatten(),
         "v": y_n.flatten(),
         "h_sim": x_exp_n.flatten(),
         "v_sim": y_exp_n.flatten(),
         })

    data.to_csv(csv_filename, index=False)
    


if __name__ == "__main__":
    
  data_num=1

  for max_iter in range(2,20):
     run_sim(max_iter=max_iter,data_num=data_num)
     print(f"Simulation_{data_num} completed")
     data_num+=1
     
  print("************************************************* MAX ITER COMPLETED**************************************")
  for n_segments in range(5,40,5):
     run_sim(n_segments=n_segments,data_num=data_num)
     print(f"Simulation_{data_num} completed")
     data_num+=1
  print("************************************************* SEGMENTS COMPLETED**************************************")
  for order in range(2,15):
     run_sim(order=order,data_num=data_num)
     print(f"Simulation_{data_num} completed")
     data_num+=1

  print("************************************************* ORDER COMPLETED**************************************")
  for tol in range(0,12):
     run_sim(tol=tol,data_num=data_num)
     print(f"Simulation_{data_num} completed")
     data_num+=1
  
  print("************************************************* TOL COMPLETED**************************************")
  


    




