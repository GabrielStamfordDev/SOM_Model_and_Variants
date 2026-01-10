import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 

# -------------------- FUNÇÃO AUXILIAR DE RESULTADOS --------------------
def print_results(beta_matrix, labels, map_dims):
    print("\n--- Afiliação dos Neurônios e Coeficientes ---")
    map_h, map_w = map_dims
    n_neurons = len(beta_matrix)
    
    for j in range(n_neurons):
        cluster_indices = np.where(labels == j)[0]
        row, col = divmod(j, map_w)
        
        # Removemos o 'if' para mostrar também neurônios vazios
        print(f" Neurônio {j} (Pos {row},{col}) - {len(cluster_indices)} amostras.")
        # Se quiser ver os betas (coeficientes da reta):
        # print(f"   Beta: {np.round(beta_matrix[j], 4)}")

# -------------------- CARREGAMENTO ADAPTADO --------------------
def load_data_initial(data_file_path, headers_file_path, delimiter, class_label_col=None):
    """
    Carrega os dados e separa APENAS o rótulo de classificação (se houver).
    Retorna o DataFrame numérico bruto e os labels de classificação.
    """
    try:
        headers_df = pd.read_csv(headers_file_path, header=None)
        headers = headers_df.squeeze().tolist()
        df = pd.read_csv(data_file_path, header=None, names=headers, sep=delimiter)
    except Exception as e:
        print(f"Erro ao ler arquivos: {e}")
        return None, None

    y_class_labels = None
    
    # 1. Separa o rótulo de CLASSIFICAÇÃO (ex: nome da flor), se informado
    if class_label_col and class_label_col in df.columns:
        y_class_labels = df[class_label_col]
        df_numeric = df.drop(columns=[class_label_col])
    else:
        df_numeric = df

    return df_numeric, y_class_labels

# -------------------- CLASSE SOM-WISE REGRESSION --------------------
class SOMWiseRegression:
    def __init__(self, map_height=5, map_width=5, max_iter=100, h0=0.001, hf=0.5, random_state=None):
        self.map_height = map_height
        self.map_width = map_width
        self.n_neurons = map_height * map_width 
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.betas = None 
        self.labels_ = None
        
        self.neuron_coordinates = np.array([
            [i, j] for i in range(map_height) for j in range(map_width)
        ])
        
        self.grid_distances_sq = np.zeros((self.n_neurons, self.n_neurons))
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                self.grid_distances_sq[i, j] = np.sum((self.neuron_coordinates[i] - self.neuron_coordinates[j])**2)

        max_dist_sq = np.max(self.grid_distances_sq)
        
        # Proteção matemática
        h0 = np.clip(h0, 1e-10, 0.999)
        hf = np.clip(hf, 1e-10, 0.999)
        
        self.sigma_0 = np.sqrt(-max_dist_sq / (2 * np.log(h0)))
        self.sigma_f = np.sqrt(-1 / (2 * np.log(hf)))

        print(f"  [Config] Sigma_0={self.sigma_0:.4f} | Sigma_f={self.sigma_f:.4f}")

    def _calculate_neighborhood_matrix(self, current_sigma):
        denom = 2 * (current_sigma ** 2)
        H = np.exp(-self.grid_distances_sq / denom)
        return H

    def _calculate_regression_errors_sq(self, X_prime, y, betas):
        # Predição: X' @ Betas.T
        predictions = np.dot(X_prime, betas.T)
        y_col = y.reshape(-1, 1)
        # Erro quadrático: (y - y_pred)^2
        errors_sq = (y_col - predictions) ** 2
        return errors_sq

    def _calculate_generalized_error(self, errors_sq, H):
        # Erro generalizado = Erro_Sq @ H.T (Soma ponderada dos erros vizinhos)
        return np.dot(errors_sq, H.T)

    def calculate_cost_function(self, generalized_errors, labels):
        chosen = generalized_errors[np.arange(len(labels)), labels]
        return np.sum(chosen)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        N_samples, n_features = X.shape
        
        # X' (matriz estendida) [cite: 8]
        ones_col = np.ones((N_samples, 1))
        X_prime = np.hstack((ones_col, X)) 
        y_vec = y.values if hasattr(y, 'values') else y
        
        # --- INICIALIZAÇÃO [cite: 50-55] ---
        t = 0
        current_sigma = self.sigma_0
        H = self._calculate_neighborhood_matrix(current_sigma) # 
        
        initialization_success = False
        attempt_counter = 0
        self.betas = np.zeros((self.n_neurons, n_features + 1))

        while not initialization_success:
            attempt_counter += 1
            
            # 1. "Afete aleatoriamente os exemplos xk aos grupos Pr" 
            current_labels = np.random.randint(0, self.n_neurons, size=N_samples)
            
            # 2. "Calcular H'r" 
            # No código, G representa a matriz H' 
            # G[k, r] é o h_{f(x_k), r}
            G = H[current_labels, :] 
            
            all_neurons_invertible = True
            temp_betas = np.zeros((self.n_neurons, n_features + 1))
            
            for r in range(self.n_neurons):
                # Extraindo a diagonal de H'r 
                # A coluna r de G contém os elementos da diagonal de H'r
                diag_Hr = G[:, r] 
                
                # 3. "if X'T H'r X' for inversível" [cite: 52]
                # Implementação otimizada: (X * sqrt(W))^T * (X * sqrt(W))
                sqrt_w = np.sqrt(diag_Hr + 1e-12).reshape(-1, 1)
                X_weighted = X_prime * sqrt_w
                y_weighted = y_vec * sqrt_w.flatten()
                
                XTX = np.dot(X_weighted.T, X_weighted) # Isso é X'T H'r X'
                XTy = np.dot(X_weighted.T, y_weighted)
                
                XTX_reg = XTX + np.eye(XTX.shape[0]) * 1e-9 # Estabilidade numérica
                
                try:
                    beta_r = np.linalg.solve(XTX_reg, XTy) # [cite: 54] calcular Beta_r
                    temp_betas[r] = beta_r
                except np.linalg.LinAlgError:
                    # "else... afeta-se novamente aleatoriamente" [cite: 53-55]
                    all_neurons_invertible = False
                    break 
            
            if all_neurons_invertible:
                self.labels_ = current_labels
                self.betas = temp_betas
                initialization_success = True
                # "Quando conseguir, calcula-se J" [cite: 55]
                raw_err = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
                gen_err = self._calculate_generalized_error(raw_err, H)
                init_cost = self.calculate_cost_function(gen_err, self.labels_)
                print(f">> Inicialização SUCESSO ({attempt_counter} tentativas). J Inicial = {init_cost:.2f}")
                print(self.betas) # <--- IMPRIME OS BETAS AQUI
                print("-" * 40)
            else:
                print(f"   Tentativa {attempt_counter} falhou (matriz singular)...")
        #Loop Principal:#
        while t < self.max_iter:
            t += 1
            current_sigma = self.sigma_0 * ((self.sigma_f / self.sigma_0) ** (t / self.max_iter))
            H = self._calculate_neighborhood_matrix(current_sigma)
            
            # [cite_start]--- STEP 2: ADAPTAÇÃO (Cálculo dos Betas) [cite: 66-74] ---
            # Resolvemos Mínimos Quadrados Ponderados (WLS) para cada neurônio
            
            G = H[self.labels_, :] 
            
            for r in range(self.n_neurons):
                weights = G[:, r] # Influência de cada dado no neurônio r
                
                # Otimização WLS: X^T W X beta = X^T W y
                sqrt_w = np.sqrt(weights + 1e-12).reshape(-1, 1)
                X_weighted = X_prime * sqrt_w
                y_weighted = y_vec * sqrt_w.flatten()
                
                XTX = np.dot(X_weighted.T, X_weighted)
                XTy = np.dot(X_weighted.T, y_weighted)
                
                # Regulaização Ridge (Tikhonov) para evitar matriz singular
                XTX_reg = XTX + np.eye(XTX.shape[0]) * 1e-6
                
                try:
                    self.betas[r] = np.linalg.solve(XTX_reg, XTy)
                except np.linalg.LinAlgError:
                    pass # Se falhar, mantém beta anterior

            # [cite_start]--- STEP 1: COMPETIÇÃO (Atualizar Labels) [cite: 61-64] ---
            # Escolhe o neurônio que minimiza o erro de regressão ponderado
            raw_errors_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
            gen_errors = self._calculate_generalized_error(raw_errors_sq, H)
            self.labels_ = np.argmin(gen_errors, axis=1)
            
            new_cost = self.calculate_cost_function(gen_errors, self.labels_)
            
            if t % 10 == 0 or t == self.max_iter:
                print(f" Iteração {t}: Sigma = {current_sigma:.4f} -> Custo J = {new_cost:.2f}")

        return self.betas, self.labels_, new_cost

# -------------------- BLOCO PRINCIPAL --------------------

if __name__ == "__main__":
    print("--- SOM-Wise Regression (Seleção Dinâmica de Target) ---")
    data_file = input("Arquivo CSV Dados: ")
    header_file = input("Arquivo CSV Headers: ")
    delim = input("Delimitador: ")
    
    # 1. Pergunta sobre Rótulo de Classificação (para descartar/analisar depois)
    class_label = input("Nome da coluna de CLASSE/RÓTULO (ex: Iris-setosa) para separar: ")
    class_label = class_label.strip() if class_label.strip() else None

    # 2. Carrega Dados Brutos
    df_full, y_class_labels = load_data_initial(data_file, header_file, delim, class_label)
    
    if df_full is None: exit()

    # 3. Seleção da Variável de Regressão (y)
    print("\n--- Variáveis Numéricas Disponíveis ---")
    cols = df_full.columns.tolist()
    for idx, col_name in enumerate(cols):
        print(f" [{idx}] {col_name}")

    try:
        target_idx = int(input("\nDigite o ÍNDICE da variável numérica que será o alvo (y) da regressão: "))
        target_col_name = cols[target_idx]
        
        # Separa X (Features) e y (Alvo Regressão)
        y_regression = df_full[target_col_name]
        X_raw = df_full.drop(columns=[target_col_name])
        
        feature_names = X_raw.columns.tolist()
        print(f"\n>> Alvo da Regressão (y): {target_col_name}")
        print(f">> Features de Entrada (X): {feature_names}")
        
    except (ValueError, IndexError):
        print("Seleção inválida.")
        exit()

    # 4. Normalização (Apenas no X)
    norm_choice = input("Normalizar X (features)? (S/N): ").upper()
    if norm_choice == 'S':
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X_raw)
        print("X normalizado.")
    else:
        X_final = X_raw.values

    # 5. Configuração do Mapa
    try:
        print("\n--- Configuração SOM ---")
        mh = int(input("Altura: "))
        mw = int(input("Largura: "))
        h0 = float(input("h0 (0-1): "))
        hf = float(input("hf (0-1): "))
        reps = int(input("Repetições: "))
        iters = int(input("Max Iterações: "))
    except:
        print("Erro nos inputs numéricos.")
        exit()

    best_cost = float('inf')
    best_model = None
    '''
    # 6. Loop de Treinamento
    for i in range(reps):
        print(f"\n--- Repetição {i+1}/{reps} ---")
        som = SOMWiseRegression(mh, mw, iters, h0, hf, random_state=i)
        
        # Treina com X (features) e y_regression (alvo numérico)
        betas, labels, cost = som.fit(X_final, y_regression)
        
        print(f"  > Custo Final: {cost:.2f}")
        
        if cost < best_cost:
            best_cost = cost
            best_model = {'betas': betas, 'labels': labels, 'obj': som}

    # 7. Resultados Finais
    print("\n\n=== RESULTADO FINAL (Melhor Custo) ===")
    print(f"Melhor J: {best_cost:.2f}")
    
    print_results(best_model['betas'], best_model['labels'], (mh, mw))

    # Cálculo do RMSE Global
    # Recostrói X' para predição
    N = X_final.shape[0]
    X_prime = np.hstack((np.ones((N, 1)), X_final))
    # Pega os betas do neurônio vencedor de cada dado
    chosen_betas = best_model['betas'][best_model['labels']]
    # Predição = produto escalar linha a linha
    y_pred = np.sum(X_prime * chosen_betas, axis=1)
    
    rmse = np.sqrt(np.mean((y_regression.values - y_pred)**2))
    print(f"\n>> RMSE Global (Erro de Regressão): {rmse:.4f}")

    # Análise Cruzada: Neurônios vs Classe Original (se houver)
    if y_class_labels is not None:
        print("\n--- Matriz de Confusão: Neurônios de Regressão vs Classes Originais ---")
        # Função auxiliar simples para matriz de confusão
        unique_classes = np.unique(y_class_labels)
        n_neurons = mh * mw
        cm = np.zeros((n_neurons, len(unique_classes)), dtype=int)
        class_map = {c: i for i, c in enumerate(unique_classes)}
        
        for n_id in range(n_neurons):
            idx_in_neuron = np.where(best_model['labels'] == n_id)[0]
            classes_in_neuron = y_class_labels.iloc[idx_in_neuron]
            for cls in classes_in_neuron:
                cm[n_id, class_map[cls]] += 1
        
        df_cm = pd.DataFrame(cm, 
                             index=[f"N{i}" for i in range(n_neurons)], 
                             columns=unique_classes)
        print(df_cm)
        df_cm.to_csv("somwise_confusion_matrix.csv")

    # Salvando CSVs
    print("\nSalvnado arquivos...")
    df_out = pd.DataFrame(X_raw, columns=feature_names) # Salva X original (sem normalizar)
    df_out[target_col_name] = y_regression.values # Salva y original
    df_out['y_pred'] = y_pred
    df_out['neuronio_id'] = best_model['labels']
    if y_class_labels is not None:
        df_out['classe_original'] = y_class_labels.values
        
    df_out.to_csv("somwise_resultado_final.csv", index=False)
    
    # Salva Betas
    beta_cols = ['Intercept'] + feature_names
    pd.DataFrame(best_model['betas'], columns=beta_cols).to_csv("somwise_betas.csv", index_label="neuronio")
    print("Concluído.")'''