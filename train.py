import numpy as np
import matplotlib.pyplot as plt

class MultilayerPerceptron:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=500):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        
        # инициализация весов
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        z_values = []
        
        current_input = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)
            current_input = self.relu(z)
            activations.append(current_input)
            
        z_out = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        z_values.append(z_out)
        out = self.softmax(z_out)
        activations.append(out)
        
        return out, activations, z_values

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def backward(self, y_true_onehot, activations, z_values):
        m = y_true_onehot.shape[0]
        d_weights = []
        d_biases = []
        
        delta = activations[-1] - y_true_onehot
        
        for i in range(len(self.weights) - 1, -1, -1):
            d_w = np.dot(activations[i].T, delta) / m
            d_b = np.sum(delta, axis=0, keepdims=True) / m
            
            d_weights.insert(0, d_w)
            d_biases.insert(0, d_b)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(z_values[i-1])
                
        return d_weights, d_biases

    def train(self, X_train, y_train, X_val, y_val):
        y_train_oh = np.zeros((y_train.size, 2))
        y_train_oh[np.arange(y_train.size), y_train.astype(int)] = 1
        
        y_val_oh = np.zeros((y_val.size, 2))
        y_val_oh[np.arange(y_val.size), y_val.astype(int)] = 1

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(self.epochs):
            train_pred, train_activations, train_z = self.forward(X_train)
            
            d_w, d_b = self.backward(y_train_oh, train_activations, train_z)
            
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * d_w[i]
                self.biases[i] -= self.learning_rate * d_b[i]
            
            train_loss = self.compute_loss(y_train_oh, train_pred)
            train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            val_pred, _, _ = self.forward(X_val)
            val_loss = self.compute_loss(y_val_oh, val_pred)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
                
            if (epoch + 1) % 50 == 0:
                print(f"Эпоха {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        return train_losses, val_losses, train_accs, val_accs

    def save_model(self, filename='model.npz'):
        save_dict = {'num_layers': len(self.weights),
                     'layer_sizes': np.array(self.layer_sizes)}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            save_dict[f'weight_{i}'] = w
            save_dict[f'bias_{i}'] = b
        
        np.savez(filename, **save_dict)
        print(f"Модель сохранена в {filename}")

if __name__ == "__main__":
    train_data = np.load('train_data.npz')
    val_data = np.load('val_data.npz')
    
    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    
    model = MultilayerPerceptron(layer_sizes=[30, 16, 8, 2], learning_rate=0.1, epochs=300)
    
    print("Начало обучения...")
    train_losses, val_losses, train_accs, val_accs = model.train(X_train, y_train, X_val, y_val)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Кривая функции потерь (Cross-Entropy)')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Кривая точности (Accuracy)')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Графики обучения сохранены как 'learning_curves.png'")

    model.save_model()