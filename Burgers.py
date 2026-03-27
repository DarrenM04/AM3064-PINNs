"""
Physics-Informed Neural Network (Burgers' Equation)
FORWARD PROBLEM: Solving the PDE given Initial and Boundary Conditions.
Upgraded with GPU Acceleration, Adam -> L-BFGS Hybrid Optimization, & Convergence Tracking.
"""

import os
# Suppress TF logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

np.random.seed(1234)
tf.set_random_seed(1234)

# =============================================================================
# CUSTOM L-BFGS WRAPPER
# =============================================================================
class ScipyLBFGS:
    def __init__(self, loss, loss_u, loss_f, variables, sess, feed_dict):
        self.loss = loss
        self.loss_u = loss_u
        self.loss_f = loss_f
        self.variables = variables
        self.sess = sess
        self.feed_dict = feed_dict
        
        self.grads = tf.gradients(self.loss, self.variables)
        
        self.assign_ops = []
        self.placeholders =[]
        for var in self.variables:
            shape = var.shape.as_list() 
            ph = tf.placeholder(var.dtype, shape=shape)
            self.placeholders.append(ph)
            self.assign_ops.append(tf.assign(var, ph))
            
        self.iteration = 0
        self.lbfgs_iter_history = []
        self.lbfgs_bc_history =[]
        self.lbfgs_f_history =[]

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
        return loss_val.astype(np.float64), flat_grads.astype(np.float64)

    def callback(self, flat_vars):
        self.iteration += 1
        l_tot, l_u, l_f = self.sess.run([self.loss, self.loss_u, self.loss_f], feed_dict=self.feed_dict)
        
        self.lbfgs_iter_history.append(self.iteration)
        self.lbfgs_bc_history.append(l_u)
        self.lbfgs_f_history.append(l_f)
        
        if self.iteration % 100 == 0:
            print(f'L-BFGS It: {self.iteration:05d} | Tot Loss: {l_tot:.3e} | BC Loss: {l_u:.3e} | PDE Loss: {l_f:.3e}')

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

# =============================================================================
# PHYSICS-INFORMED NEURAL NETWORK (FORWARD PROBLEM)
# =============================================================================
class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):
        self.lb = lb
        self.ub = ub
        
        # Initial & Boundary Data (Supervised)
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        self.u = u
        
        # Collocation Points (Unsupervised PDE enforcement)
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.nu = nu # Fixed physics parameter
        self.layers = layers
        
        self.weights, self.biases = self.initialize_NN(layers)
        
        # --- GPU VRAM Configuration ---
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
                
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
        
        # Forward Problem Losses: Fit BC/IC + Enforce Physical Laws
        self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_f = tf.reduce_mean(tf.square(self.f_pred))
        self.loss = self.loss_u + self.loss_f
        
        # Dynamic Learning Rate for Adam
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.piecewise_constant(self.global_step,[5000, 10000],[1e-3, 1e-4, 1e-5])
        
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, global_step=self.global_step)
        
        # History lists for convergence plotting
        self.adam_iter_history = []
        self.adam_bc_history =[]
        self.adam_f_history =[]
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases =[]
        num_layers = len(layers) 
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
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
            H = tf.tanh(tf.add(tf.matmul(H, W), b)) # Tanh is optimal for smooth Burgers'
        W = weights[-1]; b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_u(self, x, t):  
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u
    
    def net_f(self, x, t):
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        # Known Forward Problem PDE 
        f = u_t + u*u_x - self.nu*u_xx
        return f
        
    def train(self, nIter_adam, nIter_lbfgs):
        tf_dict = {
            self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
            self.x_f_tf: self.x_f, self.t_f_tf: self.t_f
        }
        
        print(f"Starting Adam Optimization for {nIter_adam} iterations...")
        start_time = time.time()
        for it in range(nIter_adam):
            _, l_tot, l_u, l_f = self.sess.run([self.train_op_Adam, self.loss, self.loss_u, self.loss_f], tf_dict)
            
            # Record tracking data
            self.adam_iter_history.append(it)
            self.adam_bc_history.append(l_u)
            self.adam_f_history.append(l_f)
            
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                print(f'Adam It: {it:05d} | Tot Loss: {l_tot:.3e} | BC: {l_u:.3e} | PDE: {l_f:.3e} | Time: {elapsed:.2f}s')
                start_time = time.time()
                
        if nIter_lbfgs > 0:
            optimization_vars = self.weights + self.biases
            self.lbfgs_optimizer = ScipyLBFGS(self.loss, self.loss_u, self.loss_f, 
                                              optimization_vars, self.sess, tf_dict)
            self.lbfgs_optimizer.minimize(maxiter=nIter_lbfgs)
        
    def predict(self, X_star):
        tf_dict = {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

# =============================================================================
# DATA PREP, TRAINING & VISUALIZATION
# =============================================================================
if __name__ == "__main__": 
     
    nu = 0.01/np.pi
    N_u = 100       # Number of IC/BC training points
    N_f = 10000     # Number of collocation points evaluating PDE
    layers =[2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    # Check current directory (Colab standard) or local paths
    possible_paths =[
        'burgers_shock.mat',
        '../Data/burgers_shock.mat',
        'C:/Users/dnmoy/OneDrive/Desktop/AM3064/burgers_shock.mat'
    ]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
            
    if data_path is None:
        print("\nERROR: Could not find 'burgers_shock.mat'. Please upload it!")
        sys.exit()
        
    data = scipy.io.loadmat(data_path)
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    lb = X_star.min(0)
    ub = X_star.max(0)    
    
    # -------------------------------------------------------------------------
    # EXTRACT BOUNDARY / INITIAL CONDITIONS FOR FORWARD PROBLEM
    # -------------------------------------------------------------------------
    # Initial Condition (t=0)
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    
    # Boundary Condition (x=-1)
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    
    # Boundary Condition (x=1)
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]
    
    # Combine IC and BCs
    X_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])
    
    # Randomly select N_u points for supervised learning on the boundaries
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    # Generate random Collocation points N_f in the domain
    X_f_train = lb + (ub-lb)*np.random.rand(N_f, 2)
    X_f_train = np.vstack((X_f_train, X_u_train)) # Append boundary points to collocation points
    
    print("\n=======================================================")
    print("--- Starting Forward Problem Training ---")
    print(f"Supervised Boundary Points: {N_u}")
    print(f"Collocation (Physics) Points: {N_f}")
    print("=======================================================\n")
    
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)
    
    # Run Adam, then polish with L-BFGS
    model.train(nIter_adam=10000, nIter_lbfgs=20000) 
    
    # Predictions over entire grid
    u_pred = model.predict(X_star)
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    
    print('\n[Final Model Results]')
    print(f'Relative L2 Error u: {error_u:e}')    
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    print("\nTraining complete. Building plots...")
    
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.5, 1, 0.5])
    
    ####### Row 0: u(t,x) Heatmap ##################    
    ax0 = fig.add_subplot(gs[0, :])
    h = ax0.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    # Plot Boundary training points on the heatmap
    ax0.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = f'IC/BC Data ({N_u} pts)', markersize=4, clip_on=False)
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax0.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax0.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax0.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
    ax0.set_xlabel('Time (t)')
    ax0.set_ylabel('Position (x)')
    ax0.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.8)
    ax0.set_title("Forward Prediction: u(t,x)", fontsize=14)
    
    ####### Row 1: u(t,x) slices ##################    
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(x, Exact[25,:], 'b-', linewidth=2, label='Exact')       
    ax1.plot(x, U_pred[25,:], 'r--', linewidth=2, label='Prediction')
    ax1.set_xlabel('x'); ax1.set_ylabel('u(t,x)')    
    ax1.set_title('t = 0.25', fontsize=12)
    ax1.set_xlim([-1.1,1.1]); ax1.set_ylim([-1.1,1.1])
    
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(x, Exact[50,:], 'b-', linewidth=2, label='Exact')       
    ax2.plot(x, U_pred[50,:], 'r--', linewidth=2, label='Prediction')
    ax2.set_xlabel('x'); ax2.set_ylabel('u(t,x)')
    ax2.set_title('t = 0.50', fontsize=12)
    ax2.set_xlim([-1.1,1.1]); ax2.set_ylim([-1.1,1.1])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(x, Exact[75,:], 'b-', linewidth=2, label='Exact')       
    ax3.plot(x, U_pred[75,:], 'r--', linewidth=2, label='Prediction')
    ax3.set_xlabel('x'); ax3.set_ylabel('u(t,x)')
    ax3.set_title('t = 0.75', fontsize=12)
    ax3.set_xlim([-1.1,1.1]); ax3.set_ylim([-1.1,1.1])    
    
    ####### Row 3: Identified PDE text block ##################    
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    text_result = (
        f"BURGERS EQUATION FORWARD PROBLEM\n"
        f"Kinematic Viscosity (ν): {nu:.6f}\n"
        f"Network Layers: {layers}\n"
        f"Final Rel. L2 Error: {error_u:.4e}"
    )
    
    ax4.text(0.5, 0.5, text_result, ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#f0f0f0', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    plt.show()

    # =============================================================================
    # LOSS VS ITERATION PLOTTING
    # =============================================================================
    print("\nGenerating Loss vs Iteration plot...")
    
    # 1. Extract Adam History
    adam_iters = np.array(model.adam_iter_history)
    adam_bc_loss = np.array(model.adam_bc_history)
    adam_f_loss = np.array(model.adam_f_history)
    
    # 2. Extract L-BFGS History (Offset the X-axis by Adam steps)
    n_adam = len(adam_iters)
    lbfgs_iters = np.array(model.lbfgs_optimizer.lbfgs_iter_history) + n_adam
    lbfgs_bc_loss = np.array(model.lbfgs_optimizer.lbfgs_bc_history)
    lbfgs_f_loss = np.array(model.lbfgs_optimizer.lbfgs_f_history)
    
    # 3. Create the Plot
    os.makedirs('training_plots', exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot Boundary Condition (BC) Loss
    ax.plot(adam_iters, adam_bc_loss, label='Data (IC/BC) Loss (Adam)', color='#1f77b4', alpha=0.8)
    if len(lbfgs_iters) > 0:
        ax.plot(lbfgs_iters, lbfgs_bc_loss, label='Data (IC/BC) Loss (L-BFGS)', color='#00d2ff', linewidth=2.5)
        
    # Plot Physics/PDE (f) Loss
    ax.plot(adam_iters, adam_f_loss, label='PDE Physics Loss (Adam)', color='#d62728', alpha=0.8)
    if len(lbfgs_iters) > 0:
        ax.plot(lbfgs_iters, lbfgs_f_loss, label='PDE Physics Loss (L-BFGS)', color='#ff7f0e', linewidth=2.5)
        
    # Styling and Labels
    ax.axvline(x=n_adam, color='gray', linestyle=':', linewidth=2, label='Adam -> L-BFGS Switch')

    ax.set_yscale('log')
    ax.set_title('Training Convergence: Boundary Condition vs Physics Loss', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Total Training Iterations', fontsize=14)
    ax.set_ylabel('Loss Component Value (Log Scale)', fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plot_path = 'training_plots/final_loss_history.png'
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Loss plot successfully saved to '{plot_path}'!")