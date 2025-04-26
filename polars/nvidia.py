# %% [markdown]
# ### Import Libraries
import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import Parallel, delayed
from datetime import timedelta
import matplotlib.pyplot as plt
import os

# Configuración de matplotlib
import matplotlib
matplotlib.use('TkAgg')

# Definición de la ruta del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
SUBJECTS_RELATIVE_PATH = "data/Subjects"
SUBJECTS_PATH = os.path.join(PROJECT_ROOT, SUBJECTS_RELATIVE_PATH)

# Crear directorios para resultados
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "various_models")
os.makedirs(FIGURES_DIR, exist_ok=True)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Número de trabajadores para paralelizar
NUM_WORKERS = os.cpu_count()

subject_files = [f for f in os.listdir(SUBJECTS_PATH) if f.startswith("Subject") and f.endswith(".xlsx")]
print(f"Total sujetos: {len(subject_files)}")

# %% [markdown]
# ### Verify GPU Availability with PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

# Small test operation on GPU
if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device=device)
    y = torch.matmul(x, x)
    print("GPU test successful")

# %% [markdown]
# ### Data Preprocessing

def get_cgm_window(bolus_time, cgm_df: pl.DataFrame, window_hours: int = 2) -> np.ndarray:
    """
    Obtiene la ventana de datos CGM para un tiempo de bolo específico.

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina
    cgm_df : pl.DataFrame
        DataFrame con datos CGM
    window_hours : int, opcional
        Horas de la ventana de datos (default: 2)

    Retorna:
    --------
    np.ndarray
        Ventana de datos CGM o None si no hay suficientes datos
    """
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df.filter(
        (pl.col("date") >= window_start) & (pl.col("date") <= bolus_time)
    ).sort("date").tail(24)
    
    if window.height < 24:
        return None
    return window.get_column("mg/dl").to_numpy()

def calculate_iob(bolus_time, basal_df: pl.DataFrame, half_life_hours: float = 4.0) -> float:
    """
    Calcula la insulina activa en el cuerpo (IOB).

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina
    basal_df : pl.DataFrame
        DataFrame con datos de insulina basal
    half_life_hours : float, opcional
        Vida media de la insulina en horas (default: 4.0)

    Retorna:
    --------
    float
        Cantidad de insulina activa
    """
    if basal_df is None or basal_df.is_empty():
        return 0.0
    
    iob = 0.0
    for row in basal_df.iter_rows(named=True):
        start_time = row["date"]
        duration_hours = row["duration"] / (1000 * 3600)
        end_time = start_time + timedelta(hours=duration_hours)
        rate = row["rate"] if row["rate"] is not None else 0.9
        
        if start_time <= bolus_time <= end_time:
            time_since_start = (bolus_time - start_time).total_seconds() / 3600
            remaining = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0.0, remaining)
    return iob

def process_subject(subject_path: str, idx: int) -> list:
    """
    Procesa los datos de un sujeto.

    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo del sujeto
    idx : int
        Índice del sujeto

    Retorna:
    --------
    list
        Lista de diccionarios con características procesadas
    """
    print(f"Procesando {os.path.basename(subject_path)} ({idx+1})...")
    
    try:
        cgm_df = pl.read_excel(subject_path, sheet_name="CGM")
        bolus_df = pl.read_excel(subject_path, sheet_name="Bolus")
        try:
            basal_df = pl.read_excel(subject_path, sheet_name="Basal")
        except Exception:
            basal_df = None
    except Exception as e:
        print(f"Error al cargar {os.path.basename(subject_path)}: {e}")
        return []

    # Conversión de fechas
    cgm_df = cgm_df.with_columns(pl.col("date").cast(pl.Datetime))
    bolus_df = bolus_df.with_columns(pl.col("date").cast(pl.Datetime))
    if basal_df is not None:
        basal_df = basal_df.with_columns(pl.col("date").cast(pl.Datetime))
    
    cgm_df = cgm_df.sort("date")

    processed_data = []
    for row in bolus_df.iter_rows(named=True):
        bolus_time = row["date"]
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        
        if cgm_window is not None:
            iob = calculate_iob(bolus_time, basal_df)
            hour_of_day = bolus_time.hour / 23.0
            bg_input = row["bgInput"] if row["bgInput"] is not None else cgm_window[-1]
            
            features = {
                'subject_id': idx,
                'cgm_window': cgm_window,
                'carbInput': row["carbInput"] if row["carbInput"] is not None else 0.0,
                'bgInput': bg_input,
                'insulinCarbRatio': row["insulinCarbRatio"] if row["insulinCarbRatio"] is not None else 10.0,
                'insulinSensitivityFactor': 50.0,
                'insulinOnBoard': iob,
                'hour_of_day': hour_of_day,
                'normal': row["normal"] if row["normal"] is not None else 0.0
            }
            processed_data.append(features)
    
    return processed_data

all_processed_data = Parallel(n_jobs=-1)(
    delayed(process_subject)(
        os.path.join(SUBJECTS_PATH, f), 
        idx
    ) for idx, f in enumerate(subject_files)
)

# Aplanar lista de listas
all_processed_data = [item for sublist in all_processed_data for item in sublist]

# Convertir a DataFrame 
df_processed = pl.DataFrame(all_processed_data)
print("Muestra de datos procesados combinados:")
print(df_processed.head())
print(f"Total de muestras: {len(df_processed)}")

# %%
# Inspección de outliers en 'normal' por sujeto
print("\nEstadísticas de 'normal' por sujeto (antes de normalización):")
for subject_id in df_processed.get_column('subject_id').unique():
    subject_data = df_processed.filter(pl.col('subject_id') == subject_id).get_column('normal')
    stats = subject_data.describe()
    print(f"Sujeto {subject_id}: "
          f"min={stats['min']:.2f}, "
          f"max={stats['max']:.2f}, "
          f"mean={stats['mean']:.2f}, "
          f"std={stats['std']:.2f}")

# %%
# Dividir ventana CGM y otras características
cgm_columns = [f'cgm_{i}' for i in range(24)]
df_cgm = pl.DataFrame({
    col: [row['cgm_window'][i] for row in all_processed_data]
    for i, col in enumerate(cgm_columns)
}, schema={col: pl.Float64 for col in cgm_columns})

# Combinar con otras características
df_final = pl.concat([
    df_cgm,
    df_processed.drop('cgm_window')
], how="horizontal")

# Verificar valores nulos
print("Verificación de valores nulos en df_final:")
df_final = df_final.drop_nulls()
print(df_final.null_count())

# %%
# Normalizar características
scaler_cgm = MinMaxScaler(feature_range=(0, 1))
scaler_other = StandardScaler()
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Normalizar CGM
X_cgm = scaler_cgm.fit_transform(df_final.select(cgm_columns).to_numpy())
X_cgm = X_cgm.reshape(X_cgm.shape[0], X_cgm.shape[1], 1)

# Extraer subject_id como tensor separado
X_subject = df_final.get_column('subject_id').to_numpy()

# Normalizar otras características
other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                 'insulinSensitivityFactor', 'hour_of_day']
X_other = scaler_other.fit_transform(df_final.select(other_features).to_numpy())

# Normalizar etiquetas
y = df_final.get_column('normal').to_numpy().reshape(-1, 1)
y = scaler_y.fit_transform(y).flatten()

# Verificar NaN
print("NaN en X_cgm:", np.isnan(X_cgm).sum())
print("NaN en X_other:", np.isnan(X_other).sum())
print("NaN en X_subject:", np.isnan(X_subject).sum())
print("NaN en y:", np.isnan(y).sum())

if np.isnan(X_cgm).sum() > 0 or np.isnan(X_other).sum() > 0 or \
   np.isnan(X_subject).sum() > 0 or np.isnan(y).sum() > 0:
    raise ValueError("Valores NaN detectados en X_cgm, X_other, X_subject o y")

# %%
# División por sujeto
subject_ids = df_final.get_column('subject_id').unique().to_numpy()
train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

# %%
# Crear máscaras 
train_mask = df_final.get_column('subject_id').is_in(train_subjects).to_numpy()
val_mask = df_final.get_column('subject_id').is_in(val_subjects).to_numpy()
test_mask = df_final.get_column('subject_id').is_in(test_subjects).to_numpy()

X_cgm_train, X_cgm_val, X_cgm_test = X_cgm[train_mask], X_cgm[val_mask], X_cgm[test_mask]
X_other_train, X_other_val, X_other_test = X_other[train_mask], X_other[val_mask], X_other[test_mask]
X_subject_train, X_subject_val, X_subject_test = X_subject[train_mask], X_subject[val_mask], X_subject[test_mask]
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
subject_test = df_final.filter(pl.col('subject_id').is_in(test_subjects)).get_column('subject_id').to_numpy()

print(f"Entrenamiento CGM: {X_cgm_train.shape}, Validación CGM: {X_cgm_val.shape}, Prueba CGM: {X_cgm_test.shape}")
print(f"Entrenamiento Otros: {X_other_train.shape}, Validación Otros: {X_other_val.shape}, Prueba Otros: {X_other_test.shape}")
print(f"Entrenamiento Subject: {X_subject_train.shape}, Validación Subject: {X_subject_val.shape}, Prueba Subject: {X_subject_test.shape}")
print(f"Sujetos de prueba: {test_subjects}")

# %%
# Convertir a tensores PyTorch
X_cgm_train = torch.tensor(X_cgm_train, dtype=torch.float32).to(device)
X_cgm_val = torch.tensor(X_cgm_val, dtype=torch.float32).to(device)
X_cgm_test = torch.tensor(X_cgm_test, dtype=torch.float32).to(device)
X_other_train = torch.tensor(X_other_train, dtype=torch.float32).to(device)
X_other_val = torch.tensor(X_other_val, dtype=torch.float32).to(device)
X_other_test = torch.tensor(X_other_test, dtype=torch.float32).to(device)
X_subject_train = torch.tensor(X_subject_train, dtype=torch.long).to(device)
X_subject_val = torch.tensor(X_subject_val, dtype=torch.long).to(device)
X_subject_test = torch.tensor(X_subject_test, dtype=torch.long).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# %%
# Crear DataLoaders
train_dataset = TensorDataset(X_cgm_train, X_other_train, X_subject_train, y_train)
val_dataset = TensorDataset(X_cgm_val, X_other_val, X_subject_val, y_val)
test_dataset = TensorDataset(X_cgm_test, X_other_test, X_subject_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

# %% [markdown]
# ### Define Models in PyTorch

class EnhancedLSTM(nn.Module):
    """
    Modelo LSTM mejorado con embeddings de sujeto.
    
    Parámetros:
    -----------
    num_subjects : int
        Número total de sujetos para la capa de embedding
    embedding_dim : int
        Dimensión del embedding de sujetos (default: 8)
    """
    def __init__(self, num_subjects: int, embedding_dim: int = 8):
        super(EnhancedLSTM, self).__init__()
        self.subject_embedding = nn.Embedding(num_subjects, embedding_dim)
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.4)
        self.concat_dense = nn.Linear(64 + 6 + embedding_dim, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, cgm_input, other_input, subject_ids):
        subject_embed = self.subject_embedding(subject_ids)
        lstm_out, _ = self.lstm1(cgm_input)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.batch_norm1(lstm_out)
        lstm_out = self.dropout1(lstm_out)
        combined = torch.cat((lstm_out, other_input, subject_embed), dim=1)
        dense_out = torch.relu(self.concat_dense(combined))
        dense_out = self.batch_norm2(dense_out)
        dense_out = self.dropout2(dense_out)
        output = self.output_layer(dense_out)
        return output

class TCNTransformer(nn.Module):
    """
    Modelo híbrido TCN-Transformer con embeddings de sujeto.
    
    Parámetros:
    -----------
    num_subjects : int
        Número total de sujetos para la capa de embedding
    embedding_dim : int
        Dimensión del embedding de sujetos (default: 8)
    """
    def __init__(self, num_subjects: int, embedding_dim: int = 8):
        super(TCNTransformer, self).__init__()
        self.subject_embedding = nn.Embedding(num_subjects, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.ln1 = nn.LayerNorm(128)
        self.transformer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=2)
        self.ln2 = nn.LayerNorm(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.concat_dense1 = nn.Linear(128 + 6 + embedding_dim, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        self.dense2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, cgm_input, other_input, subject_ids):
        subject_embed = self.subject_embedding(subject_ids)
        cgm_input = cgm_input.permute(0, 2, 1)
        tcn_out = torch.relu(self.conv1(cgm_input))
        tcn_out = self.pool1(tcn_out)
        tcn_out = torch.relu(self.conv2(tcn_out))
        tcn_out = self.pool2(tcn_out)
        tcn_out = torch.relu(self.conv3(tcn_out))
        tcn_out = tcn_out.permute(0, 2, 1)
        tcn_out = self.ln1(tcn_out)
        transformer_out = self.transformer_encoder(tcn_out)
        transformer_out = transformer_out + tcn_out
        transformer_out = self.ln2(transformer_out)
        transformer_out = transformer_out.permute(0, 2, 1)
        pooled = self.global_pool(transformer_out).squeeze(-1)
        combined = torch.cat((pooled, other_input, subject_embed), dim=1)
        dense_out = torch.relu(self.concat_dense1(combined))
        dense_out = self.batch_norm1(dense_out)
        dense_out = self.dropout1(dense_out)
        dense_out = torch.relu(self.dense2(dense_out))
        dense_out = self.batch_norm2(dense_out)
        dense_out = self.dropout2(dense_out)
        output = self.output_layer(dense_out)
        return output

def custom_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Función de pérdida MSE personalizada que penaliza más las sobrepredicciones.

    Parámetros:
    -----------
    y_true : torch.Tensor
        Etiquetas verdaderas
    y_pred : torch.Tensor
        Etiquetas predichas

    Retorna:
    --------
    torch.Tensor
        Pérdida MSE
    """
    error = y_true - y_pred
    overprediction_penalty = torch.where(error < 0, 2 * error**2, error**2)
    return torch.mean(overprediction_penalty)

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                epochs: int = 200, 
                patience: int = 30) -> tuple[list, list]:
    """
    Entrena el modelo con early stopping y learning rate scheduling.
    
    Parámetros:
    -----------
    model : nn.Module
        Modelo a entrenar
    train_loader : DataLoader
        DataLoader de entrenamiento
    val_loader : DataLoader
        DataLoader de validación
    epochs : int
        Número máximo de épocas
    patience : int
        Épocas a esperar antes de early stopping
        
    Retorna:
    --------
    tuple[list, list]
        Historiales de pérdida de entrenamiento y validación
    """
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for cgm_batch, other_batch, subject_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(cgm_batch, other_batch, subject_batch).squeeze()
            loss = custom_mse(y_batch, y_pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for cgm_batch, other_batch, subject_batch, y_batch in val_loader:
                y_pred = model(cgm_batch, other_batch, subject_batch).squeeze()
                loss = custom_mse(y_batch, y_pred)
                val_loss += loss.item() * len(y_batch)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_model_state)
    return train_losses, val_losses

# %% [markdown]
# ### Initialize and Train Models

# Número total de sujetos para embeddings
num_subjects = len(subject_ids)
# Initialize models
model_lstm = EnhancedLSTM(num_subjects=num_subjects, embedding_dim=8).to(device)
model_tcn = TCNTransformer(num_subjects=num_subjects, embedding_dim=8).to(device)

# Train models
print("\nEntrenando LSTM Mejorado...")
lstm_train_losses, lstm_val_losses = train_model(model_lstm, train_loader, val_loader)

print("\nEntrenando Transformer con TCN...")
tcn_train_losses, tcn_val_losses = train_model(model_tcn, train_loader, val_loader)

# %% [markdown]
# ### Plot Training History
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(lstm_train_losses, label='Pérdida de Entrenamiento (LSTM)')
plt.plot(lstm_val_losses, label='Pérdida de Validación (LSTM)')
plt.xlabel('Época')
plt.ylabel('Pérdida MSE Personalizada')
plt.legend()
plt.title('Historial de Entrenamiento - LSTM Mejorado')

plt.subplot(1, 2, 2)
plt.plot(tcn_train_losses, label='Pérdida de Entrenamiento (TCN)')
plt.plot(tcn_val_losses, label='Pérdida de Validación (TCN)')
plt.xlabel('Época')
plt.ylabel('Pérdida MSE Personalizada')
plt.legend()
plt.title('Historial de Entrenamiento - Transformer con TCN')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

# %% [markdown]
# ### Evaluate Models
def evaluate_model(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """
    Evalúa el modelo en el conjunto de datos proporcionado.
    
    Parámetros:
    -----------
    model : nn.Module
        Modelo a evaluar
    loader : DataLoader
        DataLoader con datos de evaluación
        
    Retorna:
    --------
    tuple[np.ndarray, np.ndarray]
        Predicciones y valores reales desnormalizados
    """
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for cgm_batch, other_batch, subject_batch, y_batch in loader:
            y_pred = model(cgm_batch, other_batch, subject_batch).squeeze()
            predictions.append(y_pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # Desnormalizar predicciones y etiquetas
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()
    return predictions, targets

def rule_based_prediction(X_other: torch.Tensor, target_bg: float = 100.0) -> np.ndarray:
    """
    Realiza predicciones basadas en reglas.
    
    Parámetros:
    -----------
    X_other : torch.Tensor
        Tensor con otras características
    target_bg : float
        Valor objetivo de glucosa en sangre
        
    Retorna:
    --------
    np.ndarray
        Predicciones basadas en reglas
    """
    inverse_transformed = scaler_other.inverse_transform(X_other.cpu().numpy())
    carb_input = inverse_transformed[:, 0]
    bg_input = inverse_transformed[:, 1]
    icr = inverse_transformed[:, 3]
    isf = inverse_transformed[:, 4]
    
    # Evitar división por cero
    icr = np.where(icr == 0, 1e-6, icr)
    isf = np.where(isf == 0, 1e-6, isf)
    
    carb_component = np.divide(carb_input, icr, out=np.zeros_like(carb_input), where=icr!=0)
    bg_component = np.divide(bg_input - target_bg, isf, out=np.zeros_like(bg_input), where=isf!=0)
    prediction = carb_component + bg_component
    
    return np.clip(prediction, 0, 30)

# Evaluate LSTM
y_pred_lstm, y_test_lstm = evaluate_model(model_lstm, test_loader)
mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))
r2_lstm = r2_score(y_test_lstm, y_pred_lstm)

# Evaluate TCN
y_pred_tcn, y_test_tcn = evaluate_model(model_tcn, test_loader)
mae_tcn = mean_absolute_error(y_test_tcn, y_pred_tcn)
rmse_tcn = np.sqrt(mean_squared_error(y_test_tcn, y_pred_tcn))
r2_tcn = r2_score(y_test_tcn, y_pred_tcn)

# Rule-based prediction
y_rule = rule_based_prediction(X_other_test)
y_test_denorm = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()

mae_rule = mean_absolute_error(y_test_denorm, y_rule)
rmse_rule = np.sqrt(mean_squared_error(y_test_denorm, y_rule))
r2_rule = r2_score(y_test_denorm, y_rule)

# Print results
print("\nResultados generales:")
print(f"LSTM Mejorado - MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}, R²: {r2_lstm:.2f}")
print(f"Transformer con TCN - MAE: {mae_tcn:.2f}, RMSE: {rmse_tcn:.2f}, R²: {r2_tcn:.2f}")
print(f"Basado en reglas - MAE: {mae_rule:.2f}, RMSE: {rmse_rule:.2f}, R²: {r2_rule:.2f}")

# %% [markdown]
# ### Per-Subject Evaluation and Visualization

# Evaluación por sujeto
print("\nRendimiento por sujeto:")
for subject_id in test_subjects:
    mask = subject_test == subject_id
    y_test_sub = y_test_denorm[mask]
    y_pred_lstm_sub = y_pred_lstm[mask]
    y_pred_tcn_sub = y_pred_tcn[mask]
    y_rule_sub = y_rule[mask]
    
    if len(y_test_sub) > 0:
        mae_lstm_sub = mean_absolute_error(y_test_sub, y_pred_lstm_sub)
        rmse_lstm_sub = np.sqrt(mean_squared_error(y_test_sub, y_pred_lstm_sub))
        r2_lstm_sub = r2_score(y_test_sub, y_pred_lstm_sub)
        
        mae_tcn_sub = mean_absolute_error(y_test_sub, y_pred_tcn_sub)
        rmse_tcn_sub = np.sqrt(mean_squared_error(y_test_sub, y_pred_tcn_sub))
        r2_tcn_sub = r2_score(y_test_sub, y_pred_tcn_sub)
        
        mae_rule_sub = mean_absolute_error(y_test_sub, y_rule_sub)
        print(f"Sujeto {subject_id}:")
        print(f"  LSTM - MAE={mae_lstm_sub:.2f}, RMSE={rmse_lstm_sub:.2f}, R²={r2_lstm_sub:.2f}")
        print(f"  TCN - MAE={mae_tcn_sub:.2f}, RMSE={rmse_tcn_sub:.2f}, R²={r2_tcn_sub:.2f}")
        print(f"  Reglas - MAE={mae_rule_sub:.2f}")

# Visualización
plt.figure(figsize=(15, 10))

# Predicciones vs Real
plt.subplot(2, 2, 1)
plt.scatter(y_test_denorm, y_pred_lstm, label='LSTM', alpha=0.5, color='blue')
plt.scatter(y_test_denorm, y_pred_tcn, label='TCN', alpha=0.5, color='green')
plt.scatter(y_test_denorm, y_rule, label='Basado en Reglas', alpha=0.5, color='orange')
plt.plot([0, 15], [0, 15], 'r--')
plt.xlabel('Dosis Real (u. de insulina)')
plt.ylabel('Dosis Predicha (u. de insulina)')
plt.legend()
plt.title('Predicciones vs Real (Todos los Sujetos)')

# Distribución de Residuos
plt.subplot(2, 2, 2)
plt.hist(y_test_denorm - y_pred_lstm, bins=20, label='LSTM', alpha=0.5, color='blue')
plt.hist(y_test_denorm - y_pred_tcn, bins=20, label='TCN', alpha=0.5, color='green')
plt.hist(y_test_denorm - y_rule, bins=20, label='Basado en Reglas', alpha=0.5, color='orange')
plt.xlabel('Residuo (u. de insulina)')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de Residuos')

# MAE por Sujeto
plt.subplot(2, 2, 3)
mae_lstm_subjects = []
mae_tcn_subjects = []
mae_rule_subjects = []

for sid in test_subjects:
    mask = subject_test == sid
    if np.sum(mask) > 0:
        mae_lstm_subjects.append(mean_absolute_error(y_test_denorm[mask], y_pred_lstm[mask]))
        mae_tcn_subjects.append(mean_absolute_error(y_test_denorm[mask], y_pred_tcn[mask]))
        mae_rule_subjects.append(mean_absolute_error(y_test_denorm[mask], y_rule[mask]))

x = np.arange(len(test_subjects))
width = 0.2
plt.bar(x - width, mae_lstm_subjects, width, label='LSTM', color='blue')
plt.bar(x, mae_tcn_subjects, width, label='TCN', color='green')
plt.bar(x + width, mae_rule_subjects, width, label='Basado en Reglas', color='orange')
plt.xlabel('Sujeto')
plt.ylabel('MAE (u. de insulina)')
plt.xticks(x, test_subjects)
plt.legend()
plt.title('MAE por Sujeto')

# R² por Sujeto
plt.subplot(2, 2, 4)
r2_lstm_subjects = []
r2_tcn_subjects = []
r2_rule_subjects = []

for sid in test_subjects:
    mask = subject_test == sid
    if np.sum(mask) > 0:
        r2_lstm_subjects.append(r2_score(y_test_denorm[mask], y_pred_lstm[mask]))
        r2_tcn_subjects.append(r2_score(y_test_denorm[mask], y_pred_tcn[mask]))
        r2_rule_subjects.append(r2_score(y_test_denorm[mask], y_rule[mask]))

plt.bar(x - width, r2_lstm_subjects, width, label='LSTM', color='blue')
plt.bar(x, r2_tcn_subjects, width, label='TCN', color='green')
plt.bar(x + width, r2_rule_subjects, width, label='Basado en Reglas', color='orange')
plt.xlabel('Sujeto')
plt.ylabel('R²')
plt.xticks(x, test_subjects)
plt.legend()
plt.title('R² por Sujeto')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Guardar modelos
torch.save(model_lstm.state_dict(), os.path.join(MODELS_DIR, 'lstm.pt'))
torch.save(model_tcn.state_dict(), os.path.join(MODELS_DIR, 'tcn.pt'))

print("\nModelos guardados en:", MODELS_DIR)
print("Figuras guardadas en:", FIGURES_DIR)