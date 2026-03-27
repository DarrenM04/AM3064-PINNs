"""
Physics-Informed Neural Network for PDE Discovery (Navier-Stokes Cylinder Wake)
Upgraded with GPU Acceleration, Adam -> L-BFGS Hybrid Optimization,
Loss Tracking, Enhanced Plotting, and Accuracy Export.
"""

import os
import sys

# --- WINDOWS GPU DLL BYPASS (For Lenovo LOQ / Anaconda) ---
# If running on Google Colab, this block safely ignores itself.
bin_path = r"C:\Users\dnmoy\anaconda3\envs\pinn_gpu\Library\bin"
if os.path.exists(bin_path):
    os.add_dll_directory(bin_path)
    os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("=======================================")
print("Executing with GPU Acceleration & L-BFGS.")
print("=======================================")

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.optimize
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

np.random.seed(1234)
tf.set_random_seed(1234)

# =============================================================================
# CUSTOM L-BFGS WRAPPER
# =============================================================================
class ScipyLBFGS:
    def __init__(self, loss, lambda_1, lambda_2, variables, sess, feed_dict):
        self.loss = loss
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.variables = variables
        self.sess = sess
        self.feed_dict = feed_dict
        
        # --- THE FIX: Replace 'None' gradients with Zeros ---
        raw_grads = tf.gradients(self.loss, self.variables)
        self.grads =[g if g is not None else tf.zeros_like(v) for g, v in zip(raw_grads, self.variables)]
        # ---------------------------------------------------
        
        self.assign_ops =[]
        self.placeholders =[]
        for var in self.variables:
            shape = var.shape.as_list() 
            ph = tf.placeholder(var.dtype, shape=shape)
            self.placeholders.append(ph)
            self.assign_ops.append(tf.assign(var, ph))
            
        self.iteration = 0
        self.loss_history =[]
        self.current_loss = None

    def _get_flat_vars(self):
        vals = self.sess.run(self.variables)
        return np.concatenate([v.flatten() for v in vals])

    def _set_flat_vars(self, flat_vars):
        idx = 0
        feed = {}
        for var, ph in zip(self.variables, self.placeholders):
            shape = var.shape.as_list()           
            size = int(np.prod(shape))            
            feed[ph] = flat_vars[idx : idx + size].reshape(shape)
            idx += size
        self.sess.run(self.assign_ops, feed_dict=feed)

    def loss_and_grads(self, flat_vars):
        self._set_flat_vars(flat_vars)
        loss_val, grads_val = self.sess.run([self.loss, self.grads], feed_dict=self.feed_dict)
        flat_grads = np.concatenate([g.flatten() for g in grads_val])
        self.current_loss = loss_val.astype(np.float64)
        return self.current_loss, flat_grads.astype(np.float64)

    def callback(self, flat_vars):
        self.iteration += 1
        self.loss_history.append(self.current_loss)
        if self.iteration % 100 == 0:
            l_tot, l1, l2 = self.sess.run([self.loss, self.lambda_1, self.lambda_2], feed_dict=self.feed_dict)
            print(f'L-BFGS It: {self.iteration:05d} | Loss: {l_tot:.3e} | l1: {l1[0]:.4f} | l2: {l2[0]:.5f}')

    def minimize(self, maxiter=50000):
        initial_vars = self._get_flat_vars()
        print(f"\nStarting L-BFGS optimization (Max Iterations: {maxiter})...")
        results = scipy.optimize.minimize(
            fun=self.loss_and_grads,
            x0=initial_vars,
            method='L-BFGS-B',
            jac=True,
            callback=self.callback,
            options={'maxiter': maxiter, 'maxcor': 50, 'maxfun': maxiter, 'ftol': 1.0 * np.finfo(float).eps, 'gtol': 1e-8}
        )  
        self._set_flat_vars(results.x)
        msg = results.message.decode("utf-8") if isinstance(results.message, bytes) else results.message
        print(f"L-BFGS Terminated. Success: {results.success}, Message: {msg}")
        return self.loss_history

# =============================================================================
# PHYSICS-INFORMED NEURAL NETWORK
# =============================================================================
class PhysicsInformedNN:
    def __init__(self, x, y, t, u, v, layers):
        X = np.concatenate([x, y, t], 1)
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.X = X
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]
        self.u = u
        self.v = v
        self.layers = layers
        
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # Initialize PDE parameters to be discovered
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
        
        # GPU VRAM Configuration
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf, self.t_tf)
        
        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
                    tf.reduce_sum(tf.square(self.f_v_pred))
        
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.piecewise_constant(self.global_step, [5000],[1e-2, 1e-3])
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, global_step=self.global_step)                            
        
        self.sess.run(tf.global_variables_initializer())
        
        self.loss_history_adam = []
        self.loss_history_lbfgs =[]

    def initialize_NN(self, layers):        
        weights =[]; biases =[]
        for l in range(0,len(layers)-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W); biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]; out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]; b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]; b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1; lambda_2 = self.lambda_2
        
        psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
        psi = psi_and_p[:,0:1]
        p = psi_and_p[:,1:2]
        
        u = tf.gradients(psi, y)[0]; v = -tf.gradients(psi, x)[0]  
        
        u_t = tf.gradients(u, t)[0]; u_x = tf.gradients(u, x)[0]; u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]; u_yy = tf.gradients(u_y, y)[0]
        
        v_t = tf.gradients(v, t)[0]; v_x = tf.gradients(v, x)[0]; v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]; v_yy = tf.gradients(v_y, y)[0]
        
        p_x = tf.gradients(p, x)[0]; p_y = tf.gradients(p, y)[0]

        f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
        f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
        
        return u, v, p, f_u, f_v
      
    def train(self, nIter_adam, nIter_lbfgs): 
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                   self.u_tf: self.u, self.v_tf: self.v}
        
        print(f"Starting Adam Optimization for {nIter_adam} iterations...")
        start_time = time.time()
        for it in range(nIter_adam):
            _, loss_val = self.sess.run([self.train_op_Adam, self.loss], tf_dict)
            self.loss_history_adam.append(loss_val)
            
            if it % 100 == 0:
                elapsed = time.time() - start_time
                l1 = self.sess.run(self.lambda_1).item() 
                l2 = self.sess.run(self.lambda_2).item()
                print(f'Adam It: {it:05d} | Loss: {loss_val:.3e} | l1: {l1:.4f} | l2: {l2:.5f} | Time: {elapsed:.2f}s')
                start_time = time.time()
                
        if nIter_lbfgs > 0:
            optimization_vars = self.weights + self.biases + [self.lambda_1, self.lambda_2]
            self.lbfgs_optimizer = ScipyLBFGS(self.loss, self.lambda_1, self.lambda_2, optimization_vars, self.sess, tf_dict)
            self.loss_history_lbfgs = self.lbfgs_optimizer.minimize(maxiter=nIter_lbfgs)
            
    def predict(self, x_star, y_star, t_star):
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        return u_star, v_star, p_star

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

# =============================================================================
# DATA PREP, TRAINING & VISUALIZATION
# =============================================================================
if __name__ == "__main__": 
      
    N_train = 5000
    layers =[3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    
    # --- AUTOMATIC PATH FINDER FOR DATA ---
    possible_paths =[
        'cylinder_nektar_wake.mat',
        '../Data/cylinder_nektar_wake.mat',
        'C:/Users/dnmoy/OneDrive/Desktop/AM3064/cylinder_nektar_wake.mat'
    ]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
            
    if data_path is None:
        print("\nERROR: Could not find 'cylinder_nektar_wake.mat'. Please check the path!")
        sys.exit()
        
    data = scipy.io.loadmat(data_path)
           
    U_star = data['U_star'] 
    P_star = data['p_star'] 
    t_star = data['t'] 
    X_star = data['X_star'] 
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    XX = np.tile(X_star[:,0:1], (1,T)) 
    YY = np.tile(X_star[:,1:2], (1,T)) 
    TT = np.tile(t_star, (1,N)).T 
    
    UU = U_star[:,0,:] 
    VV = U_star[:,1,:] 
    PP = P_star 
    
    x = XX.flatten()[:,None] 
    y = YY.flatten()[:,None] 
    t = TT.flatten()[:,None] 
    
    u = UU.flatten()[:,None] 
    v = VV.flatten()[:,None] 
    p = PP.flatten()[:,None] 
    
    ######################################################################
    ######################## CLEAN DATA TRAINING #########################
    ######################################################################
    print("\n=======================================================")
    print("--- Starting Clean Data Training (Adam -> L-BFGS) ---")
    print("=======================================================\n")
    
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]

    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    model.train(nIter_adam=10000, nIter_lbfgs=20000)
    
    snap = np.array([100])
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star_snap = TT[:,snap]
    
    u_star_val = U_star[:,0,snap]
    v_star_val = U_star[:,1,snap]
    p_star_val = P_star[:,snap]
    
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star_snap)
    
    lambda_1_value = model.sess.run(model.lambda_1).item()
    lambda_2_value = model.sess.run(model.lambda_2).item()
    
    error_u = np.linalg.norm(u_star_val-u_pred,2)/np.linalg.norm(u_star_val,2)
    error_v = np.linalg.norm(v_star_val-v_pred,2)/np.linalg.norm(v_star_val,2)
    error_p = np.linalg.norm(p_star_val-p_pred,2)/np.linalg.norm(p_star_val,2)

    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
    
    print('\n[Clean Data Results]')
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))                  
    
    loss_history_clean_adam = model.loss_history_adam.copy()
    loss_history_clean_lbfgs = model.loss_history_lbfgs.copy()

    # CRITICAL: Close session and reset graph before starting Noisy Data
    model.sess.close()
    tf.reset_default_graph()

    ######################################################################
    ######################## NOISY DATA TRAINING #########################
    ######################################################################
    print("\n\n=======================================================")
    print("--- Starting Noisy Data Training (Adam -> L-BFGS) ---")
    print("=======================================================\n")
    noise = 0.01        
    u_train_noisy = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    v_train_noisy = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])    

    model_noisy = PhysicsInformedNN(x_train, y_train, t_train, u_train_noisy, v_train_noisy, layers)
    model_noisy.train(nIter_adam=10000, nIter_lbfgs=20000)
    
    # Calculate accuracy metrics for noisy data predictions
    u_pred_noisy, v_pred_noisy, p_pred_noisy = model_noisy.predict(x_star, y_star, t_star_snap)
    error_u_noisy = np.linalg.norm(u_star_val-u_pred_noisy,2)/np.linalg.norm(u_star_val,2)
    error_v_noisy = np.linalg.norm(v_star_val-v_pred_noisy,2)/np.linalg.norm(v_star_val,2)
    error_p_noisy = np.linalg.norm(p_star_val-p_pred_noisy,2)/np.linalg.norm(p_star_val,2)

    lambda_1_value_noisy = model_noisy.sess.run(model_noisy.lambda_1).item()
    lambda_2_value_noisy = model_noisy.sess.run(model_noisy.lambda_2).item()
      
    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)*100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.01)/0.01 * 100
        
    print('\n[Noisy Data Results]')
    print('Error u: %e' % (error_u_noisy))    
    print('Error v: %e' % (error_v_noisy))    
    print('Error p: %e' % (error_p_noisy)) 
    print('Error l1: %.5f%%' % (error_lambda_1_noisy))                             
    print('Error l2: %.5f%%' % (error_lambda_2_noisy))     

    loss_history_noisy_adam = model_noisy.loss_history_adam.copy()
    loss_history_noisy_lbfgs = model_noisy.loss_history_lbfgs.copy()

    ######################################################################
    ###################### EXPORT ACCURACY RESULTS #######################
    ######################################################################
    accuracy_text = f"""==================================================
PINN NAVIER-STOKES PDE DISCOVERY - ACCURACY REPORT
==================================================

[CLEAN DATA RESULTS]
Velocity Error (u):       {error_u:.6e}
Velocity Error (v):       {error_v:.6e}
Pressure Error (p):       {error_p:.6e}
Lambda_1 Predicted:       {lambda_1_value:.6f} (True: 1.0, Error: {error_lambda_1:.4f}%)
Lambda_2 Predicted:       {lambda_2_value:.6f} (True: 0.01, Error: {error_lambda_2:.4f}%)[NOISY DATA RESULTS (1% Noise)]
Velocity Error (u):       {error_u_noisy:.6e}
Velocity Error (v):       {error_v_noisy:.6e}
Pressure Error (p):       {error_p_noisy:.6e}
Lambda_1 Predicted:       {lambda_1_value_noisy:.6f} (True: 1.0, Error: {error_lambda_1_noisy:.4f}%)
Lambda_2 Predicted:       {lambda_2_value_noisy:.6f} (True: 0.01, Error: {error_lambda_2_noisy:.4f}%)
"""
    with open("accuracy_results.txt", "w") as f:
        f.write(accuracy_text)
    print("\n[+] Detailed accuracy metrics successfully exported to 'accuracy_results.txt'")

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    print("\nTraining complete. Building final visualizations...")
    
    lb_plot = X_star.min(0)
    ub_plot = X_star.max(0)
    nn = 200
    x_grid = np.linspace(lb_plot[0], ub_plot[0], nn)
    y_grid = np.linspace(lb_plot[1], ub_plot[1], nn)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    UU_g = griddata(X_star, u_pred_noisy.flatten(), (X_grid, Y_grid), method='cubic')
    VV_g = griddata(X_star, v_pred_noisy.flatten(), (X_grid, Y_grid), method='cubic')
    PP_g = griddata(X_star, p_pred_noisy.flatten(), (X_grid, Y_grid), method='cubic')
    P_ex = griddata(X_star, p_star_val.flatten(), (X_grid, Y_grid), method='cubic')

    # -------------------------------------------------------------------
    # PLOT 0: Iterations vs Loss Curve
    # -------------------------------------------------------------------
    fig0, (ax_loss1, ax_loss2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Clean Data Loss
    iter_c_adam = range(len(loss_history_clean_adam))
    iter_c_lbfgs = range(len(loss_history_clean_adam), len(loss_history_clean_adam) + len(loss_history_clean_lbfgs))
    ax_loss1.plot(iter_c_adam, loss_history_clean_adam, label='Adam', color='blue')
    ax_loss1.plot(iter_c_lbfgs, loss_history_clean_lbfgs, label='L-BFGS', color='red')
    ax_loss1.axvline(x=len(loss_history_clean_adam), color='k', linestyle='--', label='Optimizer Switch')
    ax_loss1.set_yscale('log')
    ax_loss1.set_xlabel('Iterations', fontsize=11)
    ax_loss1.set_ylabel('Loss (Log Scale)', fontsize=11)
    ax_loss1.set_title('Training Loss vs Iterations (Clean Data)', fontsize=13)
    ax_loss1.legend()
    ax_loss1.grid(True, which='both', linestyle='--', alpha=0.5)

    # Noisy Data Loss
    iter_n_adam = range(len(loss_history_noisy_adam))
    iter_n_lbfgs = range(len(loss_history_noisy_adam), len(loss_history_noisy_adam) + len(loss_history_noisy_lbfgs))
    ax_loss2.plot(iter_n_adam, loss_history_noisy_adam, label='Adam', color='blue')
    ax_loss2.plot(iter_n_lbfgs, loss_history_noisy_lbfgs, label='L-BFGS', color='red')
    ax_loss2.axvline(x=len(loss_history_noisy_adam), color='k', linestyle='--', label='Optimizer Switch')
    ax_loss2.set_yscale('log')
    ax_loss2.set_xlabel('Iterations', fontsize=11)
    ax_loss2.set_ylabel('Loss (Log Scale)', fontsize=11)
    ax_loss2.set_title('Training Loss vs Iterations (Noisy Data)', fontsize=13)
    ax_loss2.legend()
    ax_loss2.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()

    # -------------------------------------------------------------------
    # PLOT 1: Vorticity & 3D Velocity (Enhanced Labels)
    # -------------------------------------------------------------------
    fig1 = plt.figure(figsize=(12, 10))
    gs0 = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5])

    # Top Row: Vorticity
    ax_vort = fig1.add_subplot(gs0[0])
    vort_path = None
    possible_vort_paths =[
        'cylinder_nektar_t0_vorticity.mat',
        '../Data/cylinder_nektar_t0_vorticity.mat',
        'C:/Users/dnmoy/OneDrive/Desktop/AM3064/cylinder_nektar_t0_vorticity.mat'
    ]
    for vp in possible_vort_paths:
        if os.path.exists(vp):
            vort_path = vp
            break
            
    if vort_path:
        data_v = scipy.io.loadmat(vort_path)
        m = data_v['modes'].item(); nel = data_v['nel'].item()
        xx_v = np.reshape(data_v['x'], (m+1,m+1,nel), order='F')
        yy_v = np.reshape(data_v['y'], (m+1,m+1,nel), order='F')
        ww_v = np.reshape(data_v['w'], (m+1,m+1,nel), order='F')
        for i in range(nel): 
            h = ax_vort.pcolormesh(xx_v[:,:,i], yy_v[:,:,i], ww_v[:,:,i], cmap='seismic', shading='gouraud', vmin=-3, vmax=3)
        
        box_lb = [1.0, -2.0]; box_ub =[8.0, 2.0]
        ax_vort.plot([box_lb[0],box_ub[0],box_ub[0],box_lb[0],box_lb[0]], [box_lb[1],box_lb[1],box_ub[1],box_ub[1],box_lb[1]], 'k', linewidth=1.5)
        ax_vort.set_aspect('equal')
        ax_vort.set_title("Vorticity Field (t=0) over Cylinder Wake", fontsize=14, fontweight='bold')
        ax_vort.set_xlabel("x-coordinate", fontsize=12); ax_vort.set_ylabel("y-coordinate", fontsize=12)
        divider = make_axes_locatable(ax_vort)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        fig1.colorbar(h, cax=cax, label='Vorticity (1/s)')
    else:
        ax_vort.text(0.5, 0.5, "Vorticity file not found. Skipping plot...", ha='center', va='center', fontsize=12)

    # Bottom Row: 3D Velocity Slices
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1])
    r1 = [X_star[:,0].min(), X_star[:,0].max()]
    r2 = [data['t'].min(), data['t'].max()]       
    r3 = [X_star[:,1].min(), X_star[:,1].max()] 

    def draw_box(ax):
        for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
            if np.sum(np.abs(s-e)) in [r1[1]-r1[0], r2[1]-r2[0], r3[1]-r3[0]]:
                ax.plot3D(*zip(s,e), color="black", linewidth=0.5, alpha=0.5)

    ax_u = fig1.add_subplot(gs1[0], projection='3d')
    ax_u.axis('on') # Turned axis ON to see labels clearly
    draw_box(ax_u)
    ax_u.scatter(x_train, t_train, y_train, s=0.1, alpha=0.3, label='Collocation Points')
    ax_u.contourf(X_grid, UU_g, Y_grid, zdir='y', offset=t_star_snap.mean(), cmap='rainbow', alpha=0.8) 
    ax_u.set_title("3D Spatiotemporal Plot:\n u-velocity", fontweight='bold', fontsize=13)
    ax_u.set_xlabel("x-coordinate")
    ax_u.set_ylabel("Time (t)")
    ax_u.set_zlabel("y-coordinate")
    ax_u.set_xlim3d(r1); ax_u.set_ylim3d(r2); ax_u.set_zlim3d(r3)
    axisEqual3D(ax_u)

    ax_v = fig1.add_subplot(gs1[1], projection='3d')
    ax_v.axis('on') 
    draw_box(ax_v)
    ax_v.scatter(x_train, t_train, y_train, s=0.1, alpha=0.3)
    ax_v.contourf(X_grid, VV_g, Y_grid, zdir='y', offset=t_star_snap.mean(), cmap='rainbow', alpha=0.8)
    ax_v.set_title("3D Spatiotemporal Plot:\n v-velocity", fontweight='bold', fontsize=13)
    ax_v.set_xlabel("x-coordinate")
    ax_v.set_ylabel("Time (t)")
    ax_v.set_zlabel("y-coordinate")
    ax_v.set_xlim3d(r1); ax_v.set_ylim3d(r2); ax_v.set_zlim3d(r3)
    axisEqual3D(ax_v)
    
    plt.tight_layout()

    # -------------------------------------------------------------------
    # PLOT 2: Predicted vs Exact Pressure (Enhanced Labels)
    # -------------------------------------------------------------------
    fig2, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(14, 5))
    im1 = ax_p1.imshow(PP_g, interpolation='nearest', cmap='rainbow', extent=[r1[0], r1[1], r3[0], r3[1]], origin='lower', aspect='auto')
    ax_p1.set_title("Predicted Pressure p(x,y)", fontsize=14)
    ax_p1.set_xlabel("x-coordinate", fontsize=12)
    ax_p1.set_ylabel("y-coordinate", fontsize=12)
    fig2.colorbar(im1, ax=ax_p1, label='Pressure')

    im2 = ax_p2.imshow(P_ex, interpolation='nearest', cmap='rainbow', extent=[r1[0], r1[1], r3[0], r3[1]], origin='lower', aspect='auto')
    ax_p2.set_title("Exact Pressure p(x,y)", fontsize=14)
    ax_p2.set_xlabel("x-coordinate", fontsize=12)
    ax_p2.set_ylabel("y-coordinate", fontsize=12)
    fig2.colorbar(im2, ax=ax_p2, label='Pressure')
    
    plt.tight_layout()
    plt.show()