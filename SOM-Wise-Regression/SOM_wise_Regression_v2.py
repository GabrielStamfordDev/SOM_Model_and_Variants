import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler 

# -------------------- CLASSE PARA SALVAR LOG (NOVO) --------------------
'''class DualLogger:
    """
    Classe auxiliar para redirecionar o print tanto para o console
    quanto para um arquivo de texto simultaneamente.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Necessário para compatibilidade com o sistema de arquivos
        self.terminal.flush()
        self.log.flush()'''

# -------------------- FUNÇÃO AUXILIAR DE RESULTADOS --------------------

# -------------------- FUNÇÃO PARA MATRIZ DE CONFUSÃO --------------------
def generate_confusion_matrix_clustering(y_true, cluster_labels, n_clusters):
    """
    Gera a matriz de confusão para o problema de clustering.
    """
    unique_true_classes = np.unique(y_true)
    class_to_index = {cls: i for i, cls in enumerate(unique_true_classes)}
    n_true_classes = len(unique_true_classes)
    
    confusion_matrix = np.zeros((n_clusters, n_true_classes), dtype=int)
    
    for i in range(n_clusters):
        true_labels_in_cluster = y_true[cluster_labels == i]
        for true_class in true_labels_in_cluster:
            j = class_to_index[true_class]
            confusion_matrix[i, j] += 1
            
    col_names = [f'{cls}' for cls in unique_true_classes] # Simplifiquei o nome da coluna
    row_names = [f'Neurônio {i}' for i in range(n_clusters)] 
    df_cm = pd.DataFrame(confusion_matrix, index=row_names, columns=col_names)
    
    return df_cm

# -------------------- CARREGAMENTO ADAPTADO --------------------
def load_data_initial(data_file_path, headers_file_path, delimiter, class_label_col=None):
    try:
        headers_df = pd.read_csv(headers_file_path, header=None)
        headers = headers_df.squeeze().tolist()
        df = pd.read_csv(data_file_path, header=None, names=headers, sep=delimiter)
    except Exception as e:
        print(f"Erro ao ler arquivos: {e}")
        return None, None

    y_class_labels = None
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
        
        self.neuron_coordinates = np.array([[i, j] for i in range(map_height) for j in range(map_width)])
        self.grid_distances_sq = np.zeros((self.n_neurons, self.n_neurons))
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                self.grid_distances_sq[i, j] = np.sum((self.neuron_coordinates[i] - self.neuron_coordinates[j])**2)

        max_dist_sq = np.max(self.grid_distances_sq)
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
        # epsilon^2 = (y - X'Beta)^2
        predictions = np.dot(X_prime, betas.T)
        y_col = y.reshape(-1, 1)
        errors_sq = (y_col - predictions) ** 2
        return errors_sq

    def _calculate_generalized_error(self, errors_sq, H):
        # Soma ponderada dos erros vizinhos
        return np.dot(errors_sq, H.T)

    def calculate_cost_function(self, generalized_errors, labels):
        chosen = generalized_errors[np.arange(len(labels)), labels]
        return np.sum(chosen)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        N_samples, n_features = X.shape
        
        ones_col = np.ones((N_samples, 1))
        X_prime = np.hstack((ones_col, X)) 
        y_vec = y.values if hasattr(y, 'values') else y
        
        # [cite_start]--- INICIALIZAÇÃO [cite: 50-55] ---
        t = 0
        current_sigma = self.sigma_0
        H = self._calculate_neighborhood_matrix(current_sigma)
        
        initialization_success = False
        attempt_counter = 0
        self.betas = np.zeros((self.n_neurons, n_features + 1))

        while not initialization_success:
            attempt_counter += 1
            
            # [cite_start]1. Afetar aleatoriamente [cite: 51]
            current_labels = np.random.randint(0, self.n_neurons, size=N_samples)
            
            # 2. Calcular H' (Matriz G)
            G = H[current_labels, :] 
            
            all_neurons_invertible = True
            temp_betas = np.zeros((self.n_neurons, n_features + 1))
            
            # 3. Calcular Beta_r para todos os neurônios
            for r in range(self.n_neurons):
                diag_Hr = G[:, r] 
                sqrt_w = np.sqrt(diag_Hr + 1e-12).reshape(-1, 1)
                X_weighted = X_prime * sqrt_w
                y_weighted = y_vec * sqrt_w.flatten()
                
                XTX = np.dot(X_weighted.T, X_weighted)
                XTy = np.dot(X_weighted.T, y_weighted)
                XTX_reg = XTX + np.eye(XTX.shape[0]) * 1e-9 
                
                try:
                    temp_betas[r] = np.linalg.solve(XTX_reg, XTy)
                except np.linalg.LinAlgError:
                    all_neurons_invertible = False
                    break 
            
            if all_neurons_invertible:
                self.labels_ = current_labels
                self.betas = temp_betas
                initialization_success = True
                
                # Cálculo do J Inicial usando o erro Epsilon
                raw_errors_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
                gen_err = self._calculate_generalized_error(raw_errors_sq, H)
                init_cost = self.calculate_cost_function(gen_err, self.labels_)
                print(f">> Inicialização OK ({attempt_counter} tentativas). Custo J = {init_cost:.2f}")

            else:
                print(f"   Tentativa {attempt_counter} falhou (matriz singular)...")

        # --- LOOP PRINCIPAL (Repeat until t_max) ---
        while t < self.max_iter:
            t += 1
            current_sigma = self.sigma_0 * ((self.sigma_f / self.sigma_0) ** (t / self.max_iter))
            H = self._calculate_neighborhood_matrix(current_sigma)
            
            #Step 1: Competição (PRIMEIRO)
            # Atualiza labels usando betas da iteração anterior (ou inicialização)
            #raw_errors_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
            gen_errors = self._calculate_generalized_error(raw_errors_sq, H)
            self.labels_ = np.argmin(gen_errors, axis=1)

            #Step 2: Adaptação (SEGUNDO)
            # Atualiza betas usando os novos labels definidos acima
            G = H[self.labels_, :] 
            
            for r in range(self.n_neurons):
                #Guarda beta anterior caso precise reverter [cite: 69]
                # (Como já estamos escrevendo em self.betas, se falhar na inversão, 
                # basta não sobrescrever, mantendo o valor antigo)
                
                diag_Hr = G[:, r]
                sqrt_w = np.sqrt(diag_Hr + 1e-12).reshape(-1, 1)
                X_weighted = X_prime * sqrt_w
                y_weighted = y_vec * sqrt_w.flatten()
                
                XTX = np.dot(X_weighted.T, X_weighted)
                XTy = np.dot(X_weighted.T, y_weighted)
                XTX_reg = XTX + np.eye(XTX.shape[0]) * 1e-9
                
                try:
                    self.betas[r] = np.linalg.solve(XTX_reg, XTy)
                except np.linalg.LinAlgError:
                    pass 

            # Calcula J Final da iteração [cite: 75]
            # Recalcula erro com os betas novos para o J
            raw_errors_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
            gen_errors = self._calculate_generalized_error(raw_errors_sq, H)
            new_cost = self.calculate_cost_function(gen_errors, self.labels_)
            
            print(f" Iteração {t}: Sigma = {current_sigma:.4f} -> Custo J = {new_cost:.2f}")

        return self.betas, self.labels_, new_cost

# -------------------- BLOCO PRINCIPAL --------------------

if __name__ == "__main__":
    print("--- SOM-Wise Regression (Seleção Dinâmica de Target) ---")
    data_file = input("Digite o nome do ARQUIVO DE DADOS CSV: ")
    header_file = input("Digite o nome do ARQUIVO DE HEADERS CSV:")
    delim = input("Digite o delimitador do CSV: ")
    
    class_label = input("Digite o NOME da coluna de rótulos/target para remover:  ")
    class_label = class_label.strip() if class_label.strip() else None

    # 1. Carrega Dados
    df_full, y_class_labels = load_data_initial(data_file, header_file, delim, class_label)
    if df_full is None: exit()

    print("\n--- Variáveis Numéricas Disponíveis ---")
    cols = df_full.columns.tolist()
    for idx, col_name in enumerate(cols):
        print(f" [{idx}] {col_name}")

    try:
        target_idx = int(input("\nDigite o ÍNDICE da variável numérica que será o alvo (y) da regressão: "))
        target_col_name = cols[target_idx]
        
        y_regression = df_full[target_col_name]
        X_raw = df_full.drop(columns=[target_col_name])
        
        feature_names = X_raw.columns.tolist()
        print(f"\n>> Alvo da Regressão (y): {target_col_name}")
        print(f">> Features de Entrada (X): {feature_names}")
        
    except (ValueError, IndexError):
        print("Seleção inválida.")
        exit()

    norm_choice = input("Normalizar X (features)? (S/N): ").upper()
    if norm_choice == 'S':
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X_raw)
        print("X normalizado.")
    else:
        X_final = X_raw.values

    try:
        print("\n--- Configuração SOM ---")
        mh = int(input("Digite a Altura do mapa (linhas de neurônios): "))
        mw = int(input("Digite a Largura do mapa (colunas de neurônios):"))
        h0 = float(input("Digite o valor inicial de h0 usado para calcular sigma0:"))
        hf = float(input("Digite o valor de hf usado para calcular sigmaf: "))
        reps = int(input("Digite a quantidade de repetições do Treinamento:"))
        iters = int(input("Digite o número máximo de iterações (tmax):"))
    except:
        print("Erro nos inputs numéricos.")
        exit()

    best_cost = float('inf')
    best_model = None
    
    # 2. Loop de Treinamento
    for i in range(reps):
        print(f"\n--- Repetição {i+1}/{reps} ---")
        som = SOMWiseRegression(mh, mw, iters, h0, hf, random_state=i)
        
        betas, labels, cost = som.fit(X_final, y_regression)
        
        print(f"  > Custo Final: {cost:.2f}")
        
        if cost < best_cost:
            best_cost = cost
            best_model = {'betas': betas, 'labels': labels, 'obj': som}

# =========================================================
    #            SEÇÃO DE OUTPUTS (MODIFICADA)
    # =========================================================

    # ATIVA LOG PARA ARQUIVO APENAS AGORA (Melhor Resultado)

    #sys.stdout = DualLogger("somwise_melhor_resultado.txt")
    #np.set_printoptions(threshold=sys.maxsize, linewidth=200, suppress=True)
    # --- Resultado Final ---
    print("\n\n--- Resultado Final (Melhor Execução) ---")
    print(f"Menor custo J: {best_cost:.2f}")
    
    # --- Afiliação dos Neurônios ---
    print("\n--- Afiliação dos Neurônios ---")
    n_total_neurons = mh * mw
    best_labels = best_model['labels']
    
    for j in range(n_total_neurons):
        cluster_indices = np.where(best_labels == j)[0]
        row, col = divmod(j, mw)
        print(f" Neurônio {j} (Pos {row},{col}) - {len(cluster_indices)} elementos: {cluster_indices.tolist()}")

    # --- Visualização Topológica do Mapa ---
    print("\n--- Visualização Topológica do Mapa (Linha, Coluna)---")
    for r in range(mh - 1, -1, -1): 
        row_str = ""
        for c in range(mw):
            neuron_id = r * mw + c
            row_str += f"{neuron_id:^6}" 
        print(row_str)

    # --- Matriz de Betas (Substitui W) ---
    print("\n--- Matriz de Coeficientes Beta (W_equivalente) ---")
    for j, beta_vec in enumerate(best_model['betas']):
        print(f" Beta_{j}: {beta_vec}")
        
    # --- NOVOS PRINTS SOLICITADOS (y, Mu e Erros) ---
    # Precisamos reconstruir as matrizes para calcular Mu e Erros
    N = X_final.shape[0]
    ones_col = np.ones((N, 1))
    X_prime = np.hstack((ones_col, X_final))
    y_vec = y_regression.values.reshape(-1, 1)
    
    # Matriz Mu (Estimativas de todos os neurônios para todos os dados)
    # Mu[k, r] = x_k^T * beta_r
    Mu_matrix = np.dot(X_prime, best_model['betas'].T)
    
    # Matriz de Erros Quadráticos (Epsilon^2)
    Error_sq_matrix = (y_vec - Mu_matrix) ** 2
    
    print("\n--- Vetor Alvo (y) ---")
    # Transforma em array 1D para printar como lista
    print(y_vec.flatten()) 
    
    print("\n--- Matriz Mu (Estimativas de cada neurônio para cada dado) ---")
    print(f"Shape: {Mu_matrix.shape} (Linhas=Dados, Colunas=Neurônios)")
    print(Mu_matrix)
    
    print("\n--- Matriz de Erros Quadráticos (Epsilon^2) ---")
    print(f"Shape: {Error_sq_matrix.shape} (Linhas=Dados, Colunas=Neurônios)")
    print(Error_sq_matrix)

    # --- Matriz de Confusão ---
    print("\n\n--- Matriz de Confusão (Neurônios vs. Classes Verdadeiras) ---")
    if y_class_labels is not None:
        best_cm_df = generate_confusion_matrix_clustering(y_class_labels.values, best_labels, n_total_neurons)
        print(best_cm_df)
        best_cm_df.to_csv("somwise_matriz_confusao.csv", index=True)
    else:
        print("Sem coluna de classe original para gerar matriz de confusão.")

    # --- Salvando CSVs ---
    print("\n--- Salvando Resultados ---")
    
    # Salva Betas
    beta_cols = ['Intercept'] + feature_names
    df_B = pd.DataFrame(best_model['betas'], columns=beta_cols)
    df_B.to_csv("somwise_betas_finais.csv", index_label="neuronio_id")
    
    # Salva Resultados por Amostra
    # Predição FINAL (apenas do vencedor) para o CSV
    chosen_betas = best_model['betas'][best_labels]
    y_pred_final = np.sum(X_prime * chosen_betas, axis=1)

    df_results = pd.DataFrame(X_raw, columns=feature_names) # Salva X original
    df_results['y_real'] = y_regression.values
    df_results['y_pred'] = y_pred_final
    df_results['neuronio_vencedor'] = best_labels
    
    if y_class_labels is not None:
        df_results['classe_original'] = y_class_labels.values
        
    df_results.to_csv("somwise_resultados_finais.csv", index=False)
    
    print("Arquivos salvos com sucesso.")
    #print("Log do melhor resultado salvo em: somwise_melhor_resultado.txt")
