import numpy as np

def load_model(filename='model.npz'):
    data = np.load(filename, allow_pickle=True)
    num_layers = int(data['num_layers'])
    layer_sizes = data['layer_sizes'].tolist()
    
    weights = []
    biases = []
    for i in range(num_layers):
        weights.append(data[f'weight_{i}'])
        biases.append(data[f'bias_{i}'])
    
    return weights, biases, layer_sizes

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X, weights, biases):
    current_input = X
    for i in range(len(weights) - 1):
        z = np.dot(current_input, weights[i]) + biases[i]
        current_input = relu(z)
    
    z_out = np.dot(current_input, weights[-1]) + biases[-1]
    return softmax(z_out)

def binary_cross_entropy(y_true, y_pred_prob_m):
    # Бинарная кросс-энтропия для вероятности класса M (индекс 1)
    eps = 1e-15
    y_pred_prob_m = np.clip(y_pred_prob_m, eps, 1 - eps)
    bce = -np.mean(y_true * np.log(y_pred_prob_m) + (1 - y_true) * np.log(1 - y_pred_prob_m))
    return bce

if __name__ == "__main__":
    weights, biases, layer_sizes = load_model('model.npz')
    print(f"Модель загружена. Архитектура: {layer_sizes}")
    
    val_data = np.load('val_data.npz')
    X_val, y_val = val_data['X'], val_data['y']
    
    predictions_prob = forward(X_val, weights, biases)
    
    # Получаем предсказанный класс
    predicted_classes = np.argmax(predictions_prob, axis=1)
    
    prob_m = predictions_prob[:, 1]
    
    accuracy = np.mean(predicted_classes == y_val)
    bce = binary_cross_entropy(y_val, prob_m)
    
    print("\n--- Результаты на проверочной выборке ---")
    print(f"Количество примеров: {X_val.shape[0]}")
    print(f"Точность (Accuracy): {accuracy * 100:.2f}%")
    print(f"Бинарная кросс-энтропия (BCE): {bce:.4f}")
    
    print("\nПримеры предсказаний (первые 5):")
    for i in range(min(5, len(y_val))):
        true_label = 'M' if y_val[i] == 1 else 'B'
        pred_label = 'M' if predicted_classes[i] == 1 else 'B'
        print(f"Истинный: {true_label}, Предсказанный: {pred_label} (Вероятность M: {prob_m[i]:.4f})")