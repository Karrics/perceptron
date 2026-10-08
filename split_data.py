import pandas as pd
import numpy as np

def prepare_and_split_data(seed=42, test_size=0.2):
    np.random.seed(seed)
    
    df = pd.read_csv('data.csv', header=None)
    df.columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(30)]
    
    y = (df['diagnosis'] == 'M').astype(int).values
    X = df.iloc[:, 2:].values 
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1e-8
    X_scaled = (X - mean) / std
    
    # разделение на обучающую и проверочную выборки
    m = X_scaled.shape[0]
    indices = np.random.permutation(m)
    test_count = int(m * test_size)
    
    train_indices = indices[test_count:]
    test_indices = indices[:test_count]
    
    X_train, y_train = X_scaled[train_indices], y[train_indices]
    X_val, y_val = X_scaled[test_indices], y[test_indices]
    
    assert X_train.shape[0] == y_train.shape[0], "Критическая ошибка: размер X_train и y_train не совпадает!"
    assert X_val.shape[0] == y_val.shape[0], "Критическая ошибка: размер X_val и y_val не совпадает!"
    
    np.savez('train_data.npz', X=X_train, y=y_train, mean=mean, std=std)
    np.savez('val_data.npz', X=X_val, y=y_val)
    
    print("Данные успешно обработаны и сохранены.")
    print(f"Размер обучающей выборки: X={X_train.shape}, y={y_train.shape}")
    print(f"Размер проверочной выборки: X={X_val.shape}, y={y_val.shape}")

if __name__ == "__main__":
    prepare_and_split_data()