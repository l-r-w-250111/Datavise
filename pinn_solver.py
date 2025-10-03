import tensorflow as tf
import numpy as np
import sympy

class PINNSolver:
    """
    A generalized Physics-Informed Neural Network (PINN) solver that can handle
    user-defined ordinary differential equations (ODEs).
    """
    def __init__(self, model=None, num_hidden_layers=3, num_neurons_per_layer=20):
        """
        Initializes the PINN model.
        If a Keras model is provided, it's used; otherwise, a new model is built.
        """
        if model:
            self.model = model
        else:
            self.model = self._build_model(num_hidden_layers, num_neurons_per_layer)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.history = []
        self.physics_loss_fn = None
        self.ode_string = None
        self.needs_second_derivative = False

    def _build_model(self, num_hidden_layers, num_neurons_per_layer):
        """
        Builds a simple feed-forward neural network.
        Assumes 1 input variable (x) and 1 output variable (y).
        """
        inputs = tf.keras.Input(shape=(1,))
        x = inputs
        for _ in range(num_hidden_layers):
            x = tf.keras.layers.Dense(num_neurons_per_layer, activation='tanh')(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _create_physics_loss_function(self, ode_string):
        """
        Parses the ODE string using SymPy and creates a function to compute the physics residual.
        The ODE should be provided as its residual form, e.g., for dy/dx + y = 0, the string should be "dy/dx + y".
        """
        self.ode_string = ode_string
        x_sym = sympy.Symbol('x')
        y_sym = sympy.Function('y')(x_sym)
        
        dydx_sym = sympy.diff(y_sym, x_sym)
        d2ydx2_sym = sympy.diff(y_sym, x_sym, 2)

        local_dict = {'x': x_sym, 'y': y_sym}
        temp_ode_string = ode_string
        
        # Replace derivative terms with temporary names for safe parsing
        self.needs_second_derivative = 'd2y/dx2' in temp_ode_string
        if self.needs_second_derivative:
            temp_ode_string = temp_ode_string.replace('d2y/dx2', 'd2ydx2')
            local_dict['d2ydx2'] = d2ydx2_sym
        
        needs_first_derivative = 'dy/dx' in temp_ode_string
        if needs_first_derivative:
            temp_ode_string = temp_ode_string.replace('dy/dx', 'dydx')
            local_dict['dydx'] = dydx_sym

        try:
            parsed_ode = sympy.sympify(temp_ode_string, locals=local_dict)
        except Exception as e:
            raise ValueError(f"Error parsing the ODE string: '{ode_string}'. Error: {e}")

        lambda_args_sym = [x_sym, y_sym]
        if needs_first_derivative:
            lambda_args_sym.append(dydx_sym)
        if self.needs_second_derivative:
            lambda_args_sym.append(d2ydx2_sym)

        numeric_residual_fn = sympy.lambdify(lambda_args_sym, parsed_ode, 'tensorflow')

        def residual_function_wrapper(x, y_pred, dy_dx_pred, d2y_dx2_pred):
            lambda_args_tensors = [x, y_pred]
            if needs_first_derivative:
                lambda_args_tensors.append(dy_dx_pred)
            if self.needs_second_derivative:
                if d2y_dx2_pred is None:
                     raise ValueError("Second derivative (d2y/dx2) is required by the ODE but was not computed.")
                lambda_args_tensors.append(d2y_dx2_pred)
            
            return numeric_residual_fn(*lambda_args_tensors)

        self.physics_loss_fn = residual_function_wrapper
        print(f"Successfully parsed ODE: {ode_string}")

    @tf.function
    def _loss_fn(self, x_domain, x_data, y_data):
        """
        Calculates the total loss (data loss + physics loss).
        """
        # 1. Data loss (from boundary/initial conditions)
        y_pred_data = self.model(x_data)
        data_loss = tf.reduce_mean(tf.square(y_pred_data - y_data))

        # 2. Physics loss (enforcing the ODE)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_domain)
            y_pred_domain = self.model(x_domain)
            dy_dx = tape.gradient(y_pred_domain, x_domain)
        
        d2y_dx2 = None
        if self.needs_second_derivative:
            if dy_dx is not None:
                d2y_dx2 = tape.gradient(dy_dx, x_domain)
        
        del tape

        if self.physics_loss_fn is None:
            physics_loss = tf.constant(0.0, dtype=tf.float32)
        else:
            residual = self.physics_loss_fn(x_domain, y_pred_domain, dy_dx, d2y_dx2)
            physics_loss = tf.reduce_mean(tf.square(residual))

        total_loss = data_loss + physics_loss
        return total_loss, data_loss, physics_loss

    @tf.function
    def _train_step(self, x_domain, x_data, y_data):
        """
        Performs a single training step.
        """
        with tf.GradientTape() as tape:
            total_loss, data_loss, physics_loss = self._loss_fn(x_domain, x_data, y_data)
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, data_loss, physics_loss

    def train(self, ode_string, domain_points, condition_data, epochs=2000):
        """
        Trains the PINN model.
        - ode_string: The residual of the differential equation (e.g., "dy/dx + y").
        - domain_points: A NumPy array of collocation points for physics loss.
        - condition_data: A Pandas DataFrame with columns for 'x' and 'y' for boundary/initial conditions.
        - epochs: Number of training epochs.
        """
        self._create_physics_loss_function(ode_string)

        x_domain_tf = tf.constant(domain_points, dtype=tf.float32)
        x_data_tf = tf.constant(condition_data['x'].values.reshape(-1, 1), dtype=tf.float32)
        y_data_tf = tf.constant(condition_data['y'].values.reshape(-1, 1), dtype=tf.float32)

        self.history = []
        for epoch in range(epochs):
            total_loss, data_loss, physics_loss = self._train_step(x_domain_tf, x_data_tf, y_data_tf)
            self.history.append((total_loss.numpy(), data_loss.numpy(), physics_loss.numpy()))
            if epoch % 200 == 0:
                print(f"Epoch {epoch}: Total Loss={total_loss.numpy():.4e}, "
                      f"Data Loss={data_loss.numpy():.4e}, Physics Loss={physics_loss.numpy():.4e}")
    
    def predict(self, x_values):
        """
        Makes predictions using the trained model.
        """
        x_tf = tf.constant(x_values, dtype=tf.float32)
        return self.model(x_tf)

    def save_model(self, filepath):
        """
        Saves the trained Keras model in the recommended .keras format.
        """
        if not filepath.endswith(".keras"):
            filepath += ".keras"
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """
        Loads a Keras model and returns a PINNSolver instance.
        """
        try:
            # compile=False is used as we are using a custom training loop.
            loaded_model = tf.keras.models.load_model(filepath, compile=False)
            print(f"Model loaded successfully from {filepath}")
            return PINNSolver(model=loaded_model)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None