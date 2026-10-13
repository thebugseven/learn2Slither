import numpy as np
import os


# ==============================
# Activation Functions
# ==============================

class activation_function:
    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, grad_output):
        raise NotImplementedError("Backward method not implemented.")


class ReLU(activation_function):
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)

    def __name__(self):
        return f"ReLU"


class Sigmoid(activation_function):
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

    def __name__(self):
        return f"Sigmoid"


class Tanh(activation_function):
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)

    def __name__(self):
        return f"Tanh"


class Softmax(activation_function):
    def __init__(self):
        self.output = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        x = x - np.max(x, axis=-1, keepdims=True)

        exp_x = np.exp(x)
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        return self.output

    def backward(self, grad_output):
        return grad_output

    def __name__(self):
        return "Softmax"


# ==============================
# Layers
# ==============================

class Layer:
    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, grad_output):
        raise NotImplementedError("Backward method not implemented.")

    def update(self, lr):
        pass


class Dense(Layer):
    def __init__(
            self, input_size, output_size,
            weights_init="xavier", activation=None
            ):
        if weights_init == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(
                -limit, limit, (output_size, input_size)
            )
        elif weights_init == "he":
            limit = np.sqrt(2 / input_size)
            self.W = np.random.randn(output_size, input_size) * limit
        else:
            self.W = np.random.randn(output_size, input_size) * 0.01

        match activation:
            case "ReLU":
                self.activation = ReLU()
            case "Sigmoid":
                self.activation = Sigmoid()
            case "Tanh":
                self.activation = Tanh()
            case "Softmax":
                self.activation = Softmax()
            case _:
                self.activation = None

        self.b = np.zeros(output_size)
        self.activation_name = activation
        self.input = None
        self.dW = None
        self.db = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        self.input = x
        out = x @ self.W.T + self.b

        if self.activation:
            out = self.activation.forward(out)

        return out

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=float)

        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)

        if self.activation and not isinstance(self.activation, Softmax):
            grad_output = self.activation.backward(grad_output)

        batch_size = self.input.shape[0]

        self.dW = grad_output.T @ self.input / batch_size
        self.db = np.mean(grad_output, axis=0)
    
        grad_input = grad_output @ self.W

        return grad_input

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def load_weights(self, W, b):
        self.W = W
        self.b = b


class Flatten(Layer):
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(-1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation_name = activation

        limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels))
        self.W = np.random.uniform(
            -limit, limit,
            (out_channels, in_channels, kernel_size, kernel_size)
        )
        self.b = np.zeros(out_channels)

        match activation:
            case "ReLU":
                self.activation = ReLU()
            case "Sigmoid":
                self.activation = Sigmoid()
            case "Tanh":
                self.activation = Tanh()
            case "Softmax":
                self.activation = Softmax()
            case _:
                self.activation = None

    def forward(self, x):
        self.input = x
        c, H, W = x.shape
        K = self.kernel_size
        out_H = H - K + 1
        out_W = W - K + 1

        out = np.zeros((self.out_channels, out_H, out_W))
        
        for f in range(self.out_channels): 
            for i in range(out_H):
                for j in range(out_W):
                    region = x[:, i:i+K, j:j+K]
                    out[f, i, j] = np.sum(region * self.W[f] + self.b[f])

        self.pre_activation = out
        if self.activation:
            out = self.activation.forward(out)

        return out
    
    def backward(self, grad_output):
        if self.activation and not isinstance(self.activation, Softmax):
            grad_output = self.activation.backward(grad_output)

        c, H, W = self.input.shape
        K = self.kernel_size
        _, out_H, out_W = grad_output.shape

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        grad_input = np.zeros_like(self.input)

        for f in range(self.out_channels):
            self.db[f] = np.sum(grad_output[f])
            for i in range(out_H):
                for j in range(out_W):
                    region = self.input[:, i:i+K, j:j+K]
                    self.dW[f] += grad_output[f, i, j] * region
                    grad_input[:, i:i+K, j:j+K] += grad_output[f, i, j] * self.W[f]

        return grad_input

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class MaxPool2D(Layer):
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, x):
        self.input = x
        C, H, W = x.shape
        p = self.pool_size

        out_H = H // p
        out_W = W // p
        out = np.zeros((C, out_H, out_W))

        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    region = x[c, i*p:(i+1)*p, j*p:(j+1)*p]
                    out[c, i, j] = np.max(region)

        return out

    def backward(self, grad_output):
        C, H, W = self.input.shape
        P = self.pool_size
        _, out_H, out_W = grad_output.shape

        grad_input = np.zeros_like(self.input)

        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    region = self.input[c, i*P:(i+1)*P, j*P:(j+1)*P]
                    max_pos = np.unravel_index(np.argmax(region), region.shape)
                    grad_input[c, i*P + max_pos[0], j*P + max_pos[1]] += grad_output[c, i, j]

        return grad_input


# ==============================
# Optimizers
# ==============================


class Optimizer:
    def update(self, layer):
        raise NotImplementedError("Update method not implemented.")


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layer):
        layer.W -= self.lr * layer.dW
        layer.b -= self.lr * layer.db


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def init_layer(self, layer):
        if not hasattr(layer, "mW"):
            layer.mW = np.zeros_like(layer.W)
            layer.vW = np.zeros_like(layer.W)
            layer.mb = np.zeros_like(layer.b)
            layer.vb = np.zeros_like(layer.b)

    def update(self, layer):
        self.init_layer(layer)
        self.t += 1

        layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * layer.dW
        layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * (layer.dW ** 2)

        layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * layer.db
        layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * (layer.db ** 2)

        mW_hat = layer.mW / (1 - self.beta1 ** self.t)
        vW_hat = layer.vW / (1 - self.beta2 ** self.t)

        mb_hat = layer.mb / (1 - self.beta1 ** self.t)
        vb_hat = layer.vb / (1 - self.beta2 ** self.t)

        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


# ==============================
# Losses
# ==============================

class CrossEntropyLoss:
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(
            np.asarray(y_pred, dtype=float),
            self.eps,
            1 - self.eps
        )
        self.y_true = np.asarray(y_true, dtype=float)
        return -np.mean(np.sum(self.y_true * np.log(self.y_pred), axis=1))

    def backward(self):
        return self.y_pred - self.y_true


class BinaryCrossEntropyLoss:
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(np.asarray(
            y_pred, dtype=float), self.eps, 1 - self.eps
        )
        self.y_true = np.asarray(y_true, dtype=float)
        return -np.sum(
            self.y_true * np.log(self.y_pred)
            + (1 - self.y_true) * np.log(1 - self.y_pred)
        ) / len(self.y_true)

    def backward(self):
        return (self.y_pred - self.y_true) / (
            self.y_pred * (1 - self.y_pred) + self.eps
        )


# ==============================
# Neural Network
# ==============================

class Sequential:
    def __init__(self, optimizer=None, lr=0.001):
        self.layers = []
        if optimizer is None or optimizer == "sgd":
            print("Using SGD optimizer with learning rate:", lr)
            self.optimizer = SGD(lr=lr)
        elif optimizer == 'adam':
            print("Using Adam optimizer with learning rate:", lr)
            self.optimizer = Adam(lr=lr)
        else:
            raise ValueError("Unsupported optimizer. Please choose 'sgd' or 'adam'.")

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def update(self):
        for layer in self.layers:
            if isinstance(layer, Dense):
                self.optimizer.update(layer)

    def predict_proba(self, x):
        return self.forward(x)

    def predict(self, x):
        prob = self.forward(x)
        return np.argmax(prob, axis=1)

    def save_weights(self, filename, mean=None, std=None):
        filename = "model/" + filename
        os.makedirs("model", exist_ok=True)
        w_all = []
        b_all = []
        activation = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                w_all.append(layer.W)
                b_all.append(layer.b)
                activation.append(layer.activation_name)
        np.savez(
            filename,
            nb_layers=len(w_all),
            W=np.array(w_all, dtype=object),
            b=np.array(b_all, dtype=object),
            mean=mean,
            std=std,
            activation=np.array(activation, dtype=object)
        )

    def load_weights(self, filename):
        filename = "model/" + filename

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file '{filename}' not found.")

        data = np.load(filename, allow_pickle=True)

        w_all = data['W']
        b_all = data['b']
        activation = data["activation"]
        idx = int(data["nb_layers"])

        for i in range(idx):
            W = w_all[i]
            b = b_all[i]
            activation_name = activation[i]

            layer = Dense(W.shape[1], W.shape[0], activation=activation_name)
            layer.load_weights(W, b)

            self.layers.append(layer)
        return data.get('mean', None), data.get('std', None)
