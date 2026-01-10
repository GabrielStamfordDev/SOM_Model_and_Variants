import numpy as np
import pandas as pd
import sys 
import math
from collections import Counter
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score 
from sklearn.model_selection import KFold 

# -------------------- FUNÇÕES AUXILIARES --------------------

def generate_confusion_matrix_clustering(y_true, cluster_labels, n_clusters):
    unique_true_classes = np.unique(y_true)
    class_to_index = {cls: i for i, cls in enumerate(unique_true_classes)}
    n_true_classes = len(unique_true_classes)
    
    confusion_matrix = np.zeros((n_clusters, n_true_classes), dtype=int)
    
    for i in range(n_clusters):
        true_labels_in_cluster = y_true[cluster_labels == i]
        for true_class in true_labels_in_cluster:
            j = class_to_index[true_class]
            confusion_matrix[i, j] += 1
            
    col_names = [f'{cls}' for cls in unique_true_classes] 
    row_names = [f'Neurônio {i}' for i in range(n_clusters)] 
    df_cm = pd.DataFrame(confusion_matrix, index=row_names, columns=col_names)
    return df_cm

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

# Função Otimizada para Distância Ponderada entre duas matrizes X1 e X2
def calculate_weighted_distance_matrix(X_test, X_train, lambdas):
    """
    Calcula a matriz de distâncias quadradas ponderadas entre cada ponto de X_test e X_train.
    Dist(a, b) = Sum( lambda_j * (a_j - b_j)^2 )
    
    Retorna: Matriz (N_test, N_train)
    """
    # Expansão de dimensões para broadcasting:
    # X_test: (N_test, 1, P)
    # X_train: (1, N_train, P)
    # Diff: (N_test, N_train, P)
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    
    # Eleva ao quadrado, multiplica por lambda e soma nas features (eixo 2)
    dist_sq = np.sum((diff ** 2) * lambdas, axis=2)
    
    return dist_sq

# -------------------- CLASSE SOM HÍBRIDO --------------------
class SOMHybrid:
    def __init__(self, map_height=5, map_width=5, max_iter=100, h0=0.001, hf=0.5, random_state=None):
        self.map_height = map_height
        self.map_width = map_width
        self.n_neurons = map_height * map_width 
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.betas = None       
        self.prototypes = None  
        self.lambdas = None     
        self.alpha = 1.0        
        self.labels_ = None # Armazena os vencedores do TREINO
        
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

    def _calculate_weighted_quantization_dist_sq(self, X, W, lambdas):
        X_sq_weighted = np.dot(X**2, lambdas).reshape(-1, 1)
        W_sq_weighted = np.dot(W**2, lambdas)
        Cross_term = 2 * np.dot(X * lambdas, W.T)
        dist_sq = X_sq_weighted + W_sq_weighted - Cross_term
        return np.maximum(dist_sq, 0)

    def _calculate_total_generalized_error(self, reg_errors_sq, quant_dists_sq, H, alpha):
        total_error_per_neuron = (alpha * reg_errors_sq) + quant_dists_sq
        return np.dot(total_error_per_neuron, H.T)

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
        self.prototypes = np.zeros((self.n_neurons, n_features))
        self.lambdas = np.ones(n_features) 
        self.alpha = 1.0                   
        
        while not initialization_success:
            attempt_counter += 1
            random_idx = np.random.choice(N_samples, self.n_neurons, replace=False)
            self.prototypes = X[random_idx].copy()
            self.lambdas = np.ones(n_features)
            self.alpha = 1.0
            
            quant_dists = self._calculate_weighted_quantization_dist_sq(X, self.prototypes, self.lambdas)
            current_labels = np.argmin(quant_dists, axis=1)
            
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

        while t < self.max_iter:
            t += 1
            current_sigma = self.sigma_0 * ((self.sigma_f / self.sigma_0) ** (t / self.max_iter))
            H = self._calculate_neighborhood_matrix(current_sigma)
            
            # 1) W
            G = H[self.labels_, :] 
            numerator = np.dot(G.T, X)
            denominator = np.sum(G, axis=0).reshape(-1, 1)
            denominator = np.where(denominator == 0, 1e-10, denominator)
            self.prototypes = numerator / denominator
            
            # 2) Competição
            reg_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
            quant_sq = self._calculate_weighted_quantization_dist_sq(X, self.prototypes, self.lambdas)
            gen_err = self._calculate_total_generalized_error(reg_sq, quant_sq, H, self.alpha)
            self.labels_ = np.argmin(gen_err, axis=1)
            
            # 3) H'
            G = H[self.labels_, :]
            
            # 4) Alpha/Lambda
            A = np.sum(G * reg_sq)
            G_sum_r = np.sum(G, axis=0) 
            G_X = np.dot(G.T, X)        
            G_X2 = np.dot(G.T, X**2)    
            term_diffs = G_X2 - 2 * self.prototypes * G_X + (self.prototypes**2) * G_sum_r.reshape(-1,1)
            B = np.sum(term_diffs, axis=0) 
            
            A = max(A, 1e-10)
            B = np.maximum(B, 1e-10)
            prod_B = np.prod(B)
            numerator_factor = (A * prod_B) ** (1 / (1 + n_features))
            
            self.alpha = numerator_factor / A
            self.lambdas = numerator_factor / B
            
            # 5) Betas
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
            
            reg_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
            quant_sq = self._calculate_weighted_quantization_dist_sq(X, self.prototypes, self.lambdas)
            gen_err = self._calculate_total_generalized_error(reg_sq, quant_sq, H, self.alpha)
            final_cost = np.sum(gen_err[np.arange(N_samples), self.labels_])

        return self.betas, self.labels_, final_cost

# -------------------- BLOCO PRINCIPAL --------------------

if __name__ == "__main__":
    print("--- SOM Híbrido (CV com k-NN Ponderado e Seleção Ótima) ---")
    
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

    # Define o Grid de Vizinhos (k) com base no tamanho TOTAL do dataset
    # Conforme solicitado: números ímpares de 1 até raiz(N)
    N_total = len(X_final)
    limit_k = int(np.sqrt(N_total))
    if limit_k % 2 == 0:
        limit_k -= 1 # Garante ímpar
    
    # Grid de vizinhos a testar (Ex: 1, 3, 5, ..., 11)
    k_grid = list(range(1, limit_k + 2, 2))
    print(f"\nGrid de Vizinhos (k) que serão testados: {k_grid}")

    # =========================================================
    #            VALIDAÇÃO CRUZADA COM KNN PONDERADO
    # =========================================================
    
    global_rmse_list = []
    
    for rep in range(n_cv_reps):
        print(f"\n>>> REPETIÇÃO {rep+1}/{n_cv_reps}")
        kf = KFold(n_splits=n_folds_input, shuffle=True, random_state=rep * 42)
        
        # Armazena os "Melhores Erros Quadráticos" para cada dado desta repetição
        best_squared_errors_rep = []
        
        fold_count = 1
        
        for train_index, test_index in kf.split(X_final):
            # 1. Separa Dados
            X_train, X_test = X_final[train_index], X_final[test_index]
            y_train, y_test = y_regression[train_index], y_regression[test_index]
            
            # 2. Treina SOM
            som = SOMHybrid(mh, mw, iters, h0, hf, random_state=(rep * 100) + fold_count)
            som.fit(X_train, y_train)
            
            # Recupera os "rótulos" (neurônios vencedores) dos dados de TREINO
            train_labels = som.labels_
            
            # 3. Predição baseada em k-Vizinhos (Lazy Learning)
            
            # Passo 3.1: Calcular matriz de distâncias ponderadas (N_test x N_train)
            # Usa os lambdas aprendidos pelo SOM
            dists_matrix = calculate_weighted_distance_matrix(X_test, X_train, som.lambdas)
            
            # Para cada dado de teste no fold atual
            for i in range(len(X_test)):
                dists_to_train = dists_matrix[i] # Distâncias deste teste para todos os treinos
                
                # Obtém índices dos vizinhos ordenados por distância
                # argsort é crescente (menor distância primeiro)
                sorted_neighbor_indices = np.argsort(dists_to_train)
                
                # Lista para guardar o erro quadrático deste dado para cada k do grid
                errors_for_sample = []
                
                # Passo 3.2: Testar cada k do grid
                for k_neighbors in k_grid:
                    # Pega os índices dos k vizinhos mais próximos
                    nearest_indices = sorted_neighbor_indices[:k_neighbors]
                    
                    # Vê a qual neurônio esses vizinhos pertencem (no treino)
                    neighbor_neurons = train_labels[nearest_indices]
                    
                    # Votação Majoritária (Moda)
                    # Qual o neurônio mais frequente entre os vizinhos?
                    # Em caso de empate, Counter/most_common pega o que apareceu primeiro (mais próximo)
                    winner_neuron = Counter(neighbor_neurons).most_common(1)[0][0]
                    
                    # Passo 3.3: Predição (Usando Betas do Vencedor Votado)
                    # Mu = [1, x_test] @ Beta_vencedor
                    x_prime = np.hstack(([1], X_test[i])) # Adiciona intercepto
                    beta_vec = som.betas[winner_neuron]
                    y_pred_k = np.dot(x_prime, beta_vec)
                    
                    # Calcula erro quadrático para este k
                    sq_error = (y_test[i] - y_pred_k) ** 2
                    errors_for_sample.append(sq_error)
                
                # Passo 3.4: Seleção Ótima ("Oracle")
                # Escolhe o menor erro entre todas as tentativas de k para este dado
                min_sq_error = min(errors_for_sample)
                best_squared_errors_rep.append(min_sq_error)
            
            fold_count += 1
        
        # Fim da repetição: Calcula RMSE Global desta rodada (Média dos Melhores Erros)
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
    print("   RESULTADO FINAL (k-NN CV Otimizado)")
    print("="*50)
    print(f"Configuração: {n_cv_reps} reps de {n_folds_input}-folds")
    print(f"Grid de Vizinhos testado: {k_grid}")
    print("-" * 30)
    print(f"RMSE Médio (Best-per-sample): {mean_rmse:.6f}")
    print(f"Desvio Padrão RMSE: {std_rmse:.6f}")
    print("-" * 30)
    
    with open("somhybrid_knn_cv_stats.txt", "w", encoding="utf-8") as f_out:
        f_out.write(f"--- RELATORIO SOM-HYBRID KNN CV ---\n")
        f_out.write(f"Map={mh}x{mw}, Iters={iters}, Folds={n_folds_input}, Reps={n_cv_reps}\n")
        f_out.write(f"Grid k: {k_grid}\n\n")
        f_out.write(f"RMSE Medio (Otimizado): {mean_rmse:.6f}\n")
        f_out.write(f"Desvio Padrao: {std_rmse:.6f}\n")

    print("\nRelatório salvo em 'somhybrid_knn_cv_stats.txt'.")