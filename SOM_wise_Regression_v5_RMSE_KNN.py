import numpy as np
import pandas as pd
import sys 
from collections import Counter
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import KFold 

# -------------------- FUNÇÕES AUXILIARES --------------------

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

def calculate_global_euclidean_matrix(X):
    """
    Calcula a matriz de distâncias quadradas euclidianas entre TODOS os pontos do dataset.
    Otimização: (a-b)^2 = a^2 + b^2 - 2ab
    Retorna matriz (N, N)
    """
    # Soma dos quadrados de cada linha (N,)
    sum_sq = np.sum(X**2, axis=1)
    
    # 2*a*b (N, N)
    dot_product = np.dot(X, X.T)
    
    # Broadcasting para montar a matriz:
    # sum_sq[:, None] é coluna (N, 1)
    # sum_sq[None, :] é linha (1, N)
    dist_sq = sum_sq[:, np.newaxis] + sum_sq[np.newaxis, :] - 2 * dot_product
    
    # Garante não negativos (erros numéricos podem dar -0.0000...)
    return np.maximum(dist_sq, 0)

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
        # Nota: Não precisamos calcular protótipos W explicitamente aqui
        # pois a predição será feita via k-NN nos dados de treino.
        
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

    def _calculate_neighborhood_matrix(self, current_sigma):
        denom = 2 * (current_sigma ** 2)
        H = np.exp(-self.grid_distances_sq / denom)
        return H

    def _calculate_regression_errors_sq(self, X_prime, y, betas):
        predictions = np.dot(X_prime, betas.T)
        y_col = y.reshape(-1, 1)
        errors_sq = (y_col - predictions) ** 2
        return errors_sq

    def _calculate_generalized_error(self, errors_sq, H):
        return np.dot(errors_sq, H.T)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        N_samples, n_features = X.shape
        
        ones_col = np.ones((N_samples, 1))
        X_prime = np.hstack((ones_col, X)) 
        y_vec = y.values if hasattr(y, 'values') else y
        
        t = 0
        current_sigma = self.sigma_0
        H = self._calculate_neighborhood_matrix(current_sigma)
        
        initialization_success = False
        attempt_counter = 0
        self.betas = np.zeros((self.n_neurons, n_features + 1))

        # --- INICIALIZAÇÃO ---
        while not initialization_success:
            attempt_counter += 1
            current_labels = np.random.randint(0, self.n_neurons, size=N_samples)
            G = H[current_labels, :] 
            
            all_neurons_invertible = True
            temp_betas = np.zeros((self.n_neurons, n_features + 1))
            
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
            else:
                if attempt_counter > 100: return None, None, None

        # --- LOOP DE TREINAMENTO ---
        while t < self.max_iter:
            t += 1
            current_sigma = self.sigma_0 * ((self.sigma_f / self.sigma_0) ** (t / self.max_iter))
            H = self._calculate_neighborhood_matrix(current_sigma)
            
            # 1. Competição
            raw_errors_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
            gen_errors = self._calculate_generalized_error(raw_errors_sq, H)
            self.labels_ = np.argmin(gen_errors, axis=1)

            # 2. Adaptação
            G = H[self.labels_, :] 
            for r in range(self.n_neurons):
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

        return self.betas, self.labels_, None

# -------------------- BLOCO PRINCIPAL --------------------

if __name__ == "__main__":
    print("--- SOM-Wise Regression (CV com k-NN Otimizado e Matriz Global) ---")
    
    # Inputs
    data_file = input("Digite o nome do ARQUIVO DE DADOS CSV: ")
    header_file = input("Digite o nome do ARQUIVO DE HEADERS CSV: ")
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
        y_regression = df_full[target_col_name].values
        X_raw = df_full.drop(columns=[target_col_name])
        feature_names = X_raw.columns.tolist()
        print(f"\n>> Alvo da Regressão (y): {target_col_name}")
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
        mh = int(input("Altura do mapa: "))
        mw = int(input("Largura do mapa: "))
        h0 = float(input("h0 (0-1): "))
        hf = float(input("hf (0-1): "))
        iters = int(input("Max Iterações por Fold: "))
        
        print("\n--- Configuração da Validação Cruzada ---")
        n_folds_input = int(input("Número de Folds (k): "))
        n_cv_reps = int(input("Número de Repetições da CV (N): "))
    except:
        print("Erro nos inputs numéricos.")
        exit()

    # =========================================================
    #    CÁLCULO DA MATRIZ DE DISTÂNCIAS GLOBAL (OTIMIZAÇÃO)
    # =========================================================
    print("\nCalculando Matriz de Distâncias Global...")
    # Como não temos Lambda (peso por feature), a distância Euclidiana é estática.
    # Podemos calcular N x N uma única vez.
    GLOBAL_DIST_MATRIX = calculate_global_euclidean_matrix(X_final)
    print(f"Matriz calculada. Shape: {GLOBAL_DIST_MATRIX.shape}")

    # Configura Grid de k (vizinhos)
    N_total = len(X_final)
    limit_k = int(np.sqrt(N_total))
    if limit_k % 2 == 0: limit_k -= 1 
    k_grid = list(range(1, limit_k + 2, 2))
    print(f"Grid de Vizinhos (k): {k_grid}")

    # =========================================================
    #            VALIDAÇÃO CRUZADA REPETIDA
    # =========================================================
    
    global_rmse_list = []
    
    for rep in range(n_cv_reps):
        print(f"\n>>> REPETIÇÃO {rep+1}/{n_cv_reps}")
        
        kf = KFold(n_splits=n_folds_input, shuffle=True, random_state=rep * 100)
        best_squared_errors_rep = [] # Acumula o menor erro de cada dado
        
        fold_count = 1
        
        for train_index, test_index in kf.split(X_final):
            # 1. Separa Dados
            X_train, X_test = X_final[train_index], X_final[test_index]
            y_train, y_test = y_regression[train_index], y_regression[test_index]
            
            # 2. Treina Modelo
            som = SOMWiseRegression(mh, mw, iters, h0, hf, random_state=(rep * 1000) + fold_count)
            som.fit(X_train, y_train)
            
            # Rótulos (vencedores) dos dados de treino
            train_labels = som.labels_
            
            # 3. Predição k-NN usando Matriz Global
            # Recortamos a submatriz correspondente aos índices deste fold
            # Linhas: índices do teste atual | Colunas: índices do treino atual
            dists_matrix_fold = GLOBAL_DIST_MATRIX[np.ix_(test_index, train_index)]
            
            # Para cada dado de teste no fold
            for i in range(len(test_index)):
                dists_to_train = dists_matrix_fold[i]
                
                # Ordena índices dos vizinhos (no contexto do X_train deste fold)
                sorted_neighbor_indices = np.argsort(dists_to_train)
                
                errors_for_sample = []
                
                # Testa cada k do grid
                for k_neighbors in k_grid:
                    nearest_indices = sorted_neighbor_indices[:k_neighbors]
                    neighbor_neurons = train_labels[nearest_indices]
                    
                    # Votação
                    winner_neuron = Counter(neighbor_neurons).most_common(1)[0][0]
                    
                    # Predição com Beta do Vencedor
                    x_prime = np.hstack(([1], X_test[i]))
                    beta_vec = som.betas[winner_neuron]
                    y_pred_k = np.dot(x_prime, beta_vec)
                    
                    sq_error = (y_test[i] - y_pred_k) ** 2
                    errors_for_sample.append(sq_error)
                
                # Seleciona o melhor k para este dado (menor erro)
                min_sq_error = min(errors_for_sample)
                best_squared_errors_rep.append(min_sq_error)
            
            fold_count += 1
        
        # Média dos melhores erros quadráticos desta repetição
        mean_mse_rep = np.mean(best_squared_errors_rep)
        rep_rmse = np.sqrt(mean_mse_rep)
        global_rmse_list.append(rep_rmse)
        
        print(f"   [Concluído] RMSE Otimizado da Repetição {rep+1}: {rep_rmse:.4f}")

    # =========================================================
    #            ESTATÍSTICAS FINAIS
    # =========================================================
    
    mean_rmse = np.mean(global_rmse_list)
    std_rmse = np.std(global_rmse_list)
    
    print("\n" + "="*50)
    print("   RESULTADO FINAL (SOM-Wise k-NN Otimizado)")
    print("="*50)
    print(f"Configuração: {n_cv_reps} reps de {n_folds_input}-folds")
    print(f"Grid de Vizinhos: {k_grid}")
    print("-" * 30)
    print(f"RMSE Médio (Best-per-sample): {mean_rmse:.6f}")
    print(f"Desvio Padrão RMSE: {std_rmse:.6f}")
    print("-" * 30)
    
    with open("somwise_knn_cv_stats.txt", "w", encoding="utf-8") as f_out:
        f_out.write(f"--- RELATORIO SOM-WISE KNN CV ---\n")
        f_out.write(f"Map={mh}x{mw}, Iters={iters}, Folds={n_folds_input}, Reps={n_cv_reps}\n")
        f_out.write(f"Grid k: {k_grid}\n\n")
        f_out.write(f"RMSE Medio (Otimizado): {mean_rmse:.6f}\n")
        f_out.write(f"Desvio Padrao: {std_rmse:.6f}\n")

    print("\nRelatório salvo em 'somwise_knn_cv_stats.txt'.")