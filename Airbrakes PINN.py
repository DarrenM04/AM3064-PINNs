import os
import sys

gpu_lib_path = r"C:\Users\dnmoy\anaconda3\envs\pinn_gpu\Library\bin"
if os.path.exists(gpu_lib_path):
    os.add_dll_directory(gpu_lib_path)
    os.environ['PATH'] = gpu_lib_path + os.pathsep + os.environ['PATH']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pandas as pd
import scipy.optimize

np.random.seed(1234)
tf.set_random_seed(1234)

if not os.path.exists('training_plots'):
    os.makedirs('training_plots')

# CUSTOM L-BFGS WRAPPER (With NoneType & Memory Protection)

class ScipyLBFGS:
    def __init__(self, loss, loss_bc, loss_f, loss_p, variables, sess, feed_dict):
        self.loss = loss
        self.loss_bc = loss_bc
        self.loss_f = loss_f
        self.loss_p = loss_p
        self.variables = variables
        self.sess = sess
        self.feed_dict = feed_dict
        
        # --- THE FIX: Replace 'None' gradients with Zeros ---
        raw_grads = tf.gradients(self.loss, self.variables)
        self.grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(raw_grads, self.variables)]
        
        self.assign_ops = []
        self.placeholders = []
        for var in self.variables:
            shape = var.shape.as_list() 
            ph = tf.placeholder(var.dtype, shape=shape)
            self.placeholders.append(ph)
            self.assign_ops.append(tf.assign(var, ph))
            
        self.iteration = 0
        self.lbfgs_iter_history = []
        self.lbfgs_bc_history = []
        self.lbfgs_f_history = []

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
        if self.iteration % 10 == 0:
            l_total, l_bc, l_f = self.sess.run([self.loss, self.loss_bc, self.loss_f], feed_dict=self.feed_dict)
            self.lbfgs_iter_history.append(self.iteration)
            self.lbfgs_bc_history.append(l_bc)
            self.lbfgs_f_history.append(l_f)
            print(f'L-BFGS It: {self.iteration:05d} | Tot: {l_total:.3e} | BC: {l_bc:.3e} | PDE: {l_f:.3e}')

    def minimize(self, maxiter=20000):
        initial_vars = self._get_flat_vars()
        print(f"\nStarting L-BFGS optimization...")
        results = scipy.optimize.minimize(
            fun=self.loss_and_grads,
            x0=initial_vars,
            method='L-BFGS-B',
            jac=True,
            callback=self.callback,
            
            options={'maxiter': maxiter, 'maxcor': 10, 'maxfun': maxiter, 'ftol': 1.0 * np.finfo(float).eps, 'gtol': 1e-6}
        )  
        self._set_flat_vars(results.x)
        print(f"L-BFGS Terminated. Message: {results.message}")

# PHYSICS-INFORMED NEURAL NETWORK

class PhysicsInformedNN_Forward:
    def __init__(self, X_wall, X_inlet, X_far, X_f, X_out, layers, x_raw, y_raw, x_mask, y_mask):
        self.x_wall = X_wall[:, 0:1]; self.y_wall = X_wall[:, 1:2]
        self.x_inlet = X_inlet[:, 0:1]; self.y_inlet = X_inlet[:, 1:2]
        self.x_far = X_far[:, 0:1]; self.y_far = X_far[:, 1:2]
        self.x_f = X_f[:, 0:1]; self.y_f = X_f[:, 1:2]
        self.x_out = X_out[:, 0:1]; self.y_out = X_out[:, 1:2]
        
        self.x_raw = x_raw; self.y_raw = y_raw
        self.x_mask = x_mask; self.y_mask = y_mask

        self.layers = layers
        self.lambda_1 = 1.0       
        self.Re_tf = tf.placeholder(tf.float32, shape=[])
        
        X_all = np.vstack([X_wall, X_inlet, X_far, X_f, X_out])
        self.lb = X_all.min(0); self.ub = X_all.max(0)
        
        self.weights, self.biases = self.initialize_NN(layers)        

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        self.x_wall_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_wall_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_inlet_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_inlet_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_far_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_far_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_out_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_out_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.u_wall_pred, self.v_wall_pred, _, _, _ = self.net_NS(self.x_wall_tf, self.y_wall_tf)
        self.u_inlet_pred, self.v_inlet_pred, _, _, _ = self.net_NS(self.x_inlet_tf, self.y_inlet_tf)
        self.u_far_pred, self.v_far_pred, _, _, _ = self.net_NS(self.x_far_tf, self.y_far_tf)
        _, _, _, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_f_tf, self.y_f_tf)
        _, _, self.p_pred_out, _, _ = self.net_NS(self.x_out_tf, self.y_out_tf)
        self.u_pred, self.v_pred, self.p_pred, _, _ = self.net_NS(self.x_pred_tf, self.y_pred_tf)

        self.loss_wall_val = tf.reduce_mean(tf.square(self.u_wall_pred)) + tf.reduce_mean(tf.square(self.v_wall_pred))
        self.loss_inlet_val = tf.reduce_mean(tf.square(self.u_inlet_pred - 1.0)) + tf.reduce_mean(tf.square(self.v_inlet_pred))
        self.loss_far_val = tf.reduce_mean(tf.square(self.u_far_pred - 1.0)) + tf.reduce_mean(tf.square(self.v_far_pred))
        self.loss_bc_val = self.loss_wall_val + self.loss_inlet_val + self.loss_far_val
        self.loss_p_val  = tf.reduce_mean(tf.square(self.p_pred_out))
        
        dist_to_ab = tf.sqrt(tf.square(self.x_f_tf - 0.7) + tf.square(self.y_f_tf - 0.1))
        weight_mask = 1.0 + 10.0 * tf.exp(-dist_to_ab / 0.05) 
        self.loss_f_val = tf.reduce_mean(weight_mask * (tf.square(self.f_u_pred) + tf.square(self.f_v_pred)))
                       
        self.loss = 100.0 * self.loss_wall_val + \
                    10.0  * self.loss_inlet_val + \
                    2.0   * self.loss_far_val + \
                    1.0   * self.loss_p_val + \
                    1.0   * self.loss_f_val
        
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.piecewise_constant(self.global_step,[10000, 20000],[1e-3, 1e-4, 1e-5])
        self.optimizer_adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_adam = self.optimizer_adam.minimize(self.loss, global_step=self.global_step)                            
        
        self.adam_iter_history = []; self.adam_bc_history = []; self.adam_f_history = []
        self.sess.run(tf.global_variables_initializer())

    def initialize_NN(self, layers):        
        weights = []; biases = []
        for l in range(0, len(layers)-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32))
            weights.append(W); biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        xavier_stddev = np.sqrt(2/(size[0] + size[1]))
        return tf.Variable(tf.truncated_normal([size[0], size[1]], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0, len(weights)-1):
            H = tf.nn.swish(tf.add(tf.matmul(H, weights[l]), biases[l]))
        return tf.add(tf.matmul(H, weights[-1]), biases[-1])
        
    def net_NS(self, x, y):
        psi_and_p = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
        psi = psi_and_p[:, 0:1]; p = psi_and_p[:, 1:2]
        u = tf.gradients(psi, y)[0]; v = -tf.gradients(psi, x)[0]  
        u_x = tf.gradients(u, x)[0]; u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]; u_yy = tf.gradients(u_y, y)[0]
        v_x = tf.gradients(v, x)[0]; v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]; v_yy = tf.gradients(v_y, y)[0]
        p_x = tf.gradients(p, x)[0]; p_y = tf.gradients(p, y)[0]
        f_u = 1.0*(u*u_x + v*u_y) + p_x - (1.0/self.Re_tf)*(u_xx + u_yy) 
        f_v = 1.0*(u*v_x + v*v_y) + p_y - (1.0/self.Re_tf)*(v_xx + v_yy)
        return u, v, p, f_u, f_v

    def save_snapshot(self, iteration, current_Re, name_prefix="Adam"):
        nx, ny = 250, 125
        x_grid = np.linspace(self.lb[0], self.ub[0], nx)
        y_grid = np.linspace(self.lb[1], self.ub[1], ny)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        X_star = np.column_stack((X_mesh.flatten(), Y_mesh.flatten()))
        u_pred, v_pred, p_pred = self.predict(X_star[:,0:1], X_star[:,1:2])
        U_plot = u_pred.reshape(ny, nx); P_plot = p_pred.reshape(ny, nx)
        
        rock_radius_grid = np.interp(X_mesh, self.x_mask, self.y_mask)
        mask = (X_mesh >= 0) & (X_mesh <= 1.0) & (np.abs(Y_mesh) <= rock_radius_grid)
        U_plot[mask] = np.nan; P_plot[mask] = np.nan

        x_raw_full = np.concatenate([self.x_raw, self.x_raw[::-1]])
        y_raw_full = np.concatenate([self.y_raw, -self.y_raw[::-1]])

        fig, ax = plt.subplots(2, 1, figsize=(14, 10))
        c1 = ax[0].contourf(X_mesh, Y_mesh, U_plot, 100, cmap='jet')
        fig.colorbar(c1, ax=ax[0]); ax[0].fill(x_raw_full, y_raw_full, color='black')
        ax[0].set_title(f'It {iteration} | Re={current_Re} - Velocity')
        
        c2 = ax[1].contourf(X_mesh, Y_mesh, P_plot, 100, cmap='seismic')
        fig.colorbar(c2, ax=ax[1]); ax[1].fill(x_raw_full, y_raw_full, color='black')
        ax[1].set_title(f'It {iteration} | Re={current_Re} - Pressure')
        
        plt.tight_layout()
        plt.savefig(f'training_plots/{name_prefix}_it_{iteration:05d}.png')
        plt.close(fig)

    def train(self, nIter_adam): 
        start_time = time.time()
        current_Re = 1000000
    
        for it in range(nIter_adam):
            if it == 10000: current_Re = 2000000.0
            if it == 20000: current_Re = 10000000.0
            idx_f = np.random.choice(self.x_f.shape[0], 8192, replace=False) 
            tf_dict = {
                self.x_wall_tf: self.x_wall, self.y_wall_tf: self.y_wall, 
                self.x_inlet_tf: self.x_inlet, self.y_inlet_tf: self.y_inlet,
                self.x_far_tf: self.x_far, self.y_far_tf: self.y_far,
                self.x_f_tf: self.x_f[idx_f,:], self.y_f_tf: self.y_f[idx_f,:], 
                self.x_out_tf: self.x_out, self.y_out_tf: self.y_out,
                self.Re_tf: current_Re 
            }
            self.sess.run(self.train_op_adam, tf_dict)
            
            if it % 50 == 0:
                l_total = self.sess.run(self.loss, tf_dict)
                print(f'Adam It: {it:05d} | Re: {current_Re} | Loss: {l_total:.3e} | Time: {time.time()-start_time:.2f}s')
                start_time = time.time()

            if it % 10 == 0:
                l_total, l_bc, l_f = self.sess.run([self.loss, self.loss_bc_val, self.loss_f_val], tf_dict)
                self.adam_iter_history.append(it); self.adam_bc_history.append(l_bc); self.adam_f_history.append(l_f)
            
            if it > 0 and it % 1000 == 0: self.save_snapshot(it, current_Re, name_prefix="Adam")

        print(f"\nSwitching to L-BFGS at Re = {current_Re}...")
        
       
        idx_lbfgs = np.random.choice(self.x_f.shape[0], min(25000, self.x_f.shape[0]), replace=False)
        
        tf_dict_full = {
            self.x_wall_tf: self.x_wall, self.y_wall_tf: self.y_wall, 
            self.x_inlet_tf: self.x_inlet, self.y_inlet_tf: self.y_inlet,
            self.x_far_tf: self.x_far, self.y_far_tf: self.y_far,
            self.x_f_tf: self.x_f[idx_lbfgs,:], self.y_f_tf: self.y_f[idx_lbfgs,:], 
            self.x_out_tf: self.x_out, self.y_out_tf: self.y_out,
            self.Re_tf: current_Re
        }
        
        network_vars = self.weights + self.biases
        self.lbfgs_optimizer = ScipyLBFGS(self.loss, self.loss_bc_val, self.loss_f_val, self.loss_p_val, network_vars, self.sess, tf_dict_full)
        
        original_callback = self.lbfgs_optimizer.callback
        def custom_callback(flat_vars):
            original_callback(flat_vars)
            if self.lbfgs_optimizer.iteration % 1000 == 0:
                self.save_snapshot(self.lbfgs_optimizer.iteration + nIter_adam, current_Re, name_prefix="LBFGS")
                
        self.lbfgs_optimizer.callback = custom_callback
        self.lbfgs_optimizer.minimize(maxiter=20000)
            
    def predict(self, x_star, y_star):
        return self.sess.run([self.u_pred, self.v_pred, self.p_pred], {self.x_pred_tf: x_star, self.y_pred_tf: y_star})

# DATA GENERATION

if __name__ == "__main__": 
    layers = [2, 80, 80, 80, 80, 80, 80, 80, 80, 2] 
    
    print("Loading Geometry...")
    csv_path = "pinn_boundary_points.csv" 
    if not os.path.exists(csv_path):
        x_raw = np.array([0, 1.0]); y_raw = np.array([0.05, 0.05])
    else:
        geom_data = pd.read_csv(csv_path)
        x_raw = geom_data['x'].values; y_raw = geom_data['y'].values


    x_wall_dense, y_wall_dense = [], []
    for i in range(len(x_raw) - 1):
        xs = np.linspace(x_raw[i], x_raw[i+1], 50)
        ys = np.linspace(y_raw[i], y_raw[i+1], 50)
        x_wall_dense.extend(xs); y_wall_dense.extend(ys)
        
    x_rock = np.array(x_wall_dense); y_rock = np.array(y_wall_dense)
    x_mask = []; y_mask = []
    for i in range(len(x_raw)):
        if i > 0 and x_raw[i] == x_raw[i-1]: y_mask[-1] = max(y_mask[-1], y_raw[i])
        else: x_mask.append(x_raw[i]); y_mask.append(y_raw[i])
    x_mask = np.array(x_mask); y_mask = np.array(y_mask)
    
    x_rock_full = np.concatenate([x_rock, x_rock])
    y_rock_full = np.concatenate([y_rock, -y_rock])
    X_wall = np.column_stack((x_rock_full, y_rock_full))
    
    x_min, x_max = -1, 2.0; y_min, y_max = -0.6, 0.6
    X_inlet = np.column_stack((np.full(200, x_min), np.linspace(y_min, y_max, 200)))
    X_far = np.vstack([np.column_stack((np.linspace(x_min, x_max, 400), np.full(400, y_max))),
                       np.column_stack((np.linspace(x_min, x_max, 400), np.full(400, y_min)))])
    X_out = np.column_stack((np.full(200, x_max), np.linspace(y_min, y_max, 200)))
    
    # --- POINT DISTRIBUTION (~62k total) ---
    x_f = np.concatenate([np.random.uniform(x_min, x_max, 10000), 
                          np.random.uniform(0.6, 2.0, 25000), 
                          np.random.uniform(0.65, 0.98, 25000), 
                          np.random.uniform(-0.05, 0.05, 2000),
                          np.random.uniform(0.7, 1.5, 1000)])
    y_f = np.concatenate([np.random.uniform(y_min, y_max, 10000), 
                          np.random.uniform(-0.3, 0.3, 25000), 
                          np.random.uniform(-0.18, 0.18, 25000), 
                          np.random.uniform(-0.05, 0.05, 2000),
                          np.random.uniform(0.01, 0.25, 1000)])
    
    valid_f = []
    for xp, yp in zip(x_f, y_f):
        if 0 <= xp <= 1.0:
            if abs(yp) < np.interp(xp, x_mask, y_mask): continue 
        valid_f.append([xp, yp])
    X_f = np.array(valid_f)
    
    print("Initializing PINN...")
    model = PhysicsInformedNN_Forward(X_wall, X_inlet, X_far, X_f, X_out, layers, x_raw, y_raw, x_mask, y_mask)
    model.train(30000)