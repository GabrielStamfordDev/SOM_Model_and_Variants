import numpy as np
import pandas as pd
import sys 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import KFold 

# -------------------- FUNÇÃO PARA MATRIZ DE CONFUSÃO --------------------
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

# -------------------- CARREGAMENTO --------------------
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

# -------------------- CLASSE SOM-WISE REGRESSION (ATUALIZADA) --------------------
class SOMWiseRegression:
    def __init__(self, map_height=5, map_width=5, max_iter=100, h0=0.001, hf=0.5, random_state=None):
        self.map_height = map_height
        self.map_width = map_width
        self.n_neurons = map_height * map_width 
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.betas = None 
        self.labels_ = None
        self.prototypes = None # Armazenará os centróides calculados pós-treino
        
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

    def calculate_cost_function(self, generalized_errors, labels):
        chosen = generalized_errors[np.arange(len(labels)), labels]
        return np.sum(chosen)

    # --- NOVO MÉTODO AUXILIAR PARA DISTÂNCIA EUCLIDIANA SIMPLES ---
    def _calculate_euclidean_dists_sq(self, X, W):
        """
        Calcula a distância Euclidiana Quadrada entre dados X e protótipos W.
        (X - W)^2 = X^2 + W^2 - 2XW^T
        """
        X_sq = np.sum(X**2, axis=1).reshape(-1, 1)
        W_sq = np.sum(W**2, axis=1)
        cross_term = 2 * np.dot(X, W.T)
        dist_sq = X_sq + W_sq - cross_term
        return np.maximum(dist_sq, 0)

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
                if attempt_counter > 100:
                    print("Falha na inicialização.")
                    return None, None, None

        # --- LOOP DE TREINAMENTO ---
        while t < self.max_iter:
            t += 1
            current_sigma = self.sigma_0 * ((self.sigma_f / self.sigma_0) ** (t / self.max_iter))
            H = self._calculate_neighborhood_matrix(current_sigma)
            
            # 1. Competição (Baseada no Erro de Regressão Generalizado)
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

            # Custo Final (Opcional, para debug)
            raw_errors_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
            gen_errors = self._calculate_generalized_error(raw_errors_sq, H)
            new_cost = self.calculate_cost_function(gen_errors, self.labels_)
        
        # --- CÁLCULO DOS PROTÓTIPOS W (PÓS-TREINO) ---
        # Como o SOM-Wise não atualiza W iterativamente, calculamos agora 
        # como o centróide dos dados que "venceram" em cada neurônio.
        self.prototypes = np.zeros((self.n_neurons, n_features))
        
        for r in range(self.n_neurons):
            # Filtra os dados de treinamento mapeados para o neurônio r
            cluster_data = X[self.labels_ == r]
            
            if len(cluster_data) > 0:
                # O protótipo é a média (centróide) desses dados
                self.prototypes[r] = np.mean(cluster_data, axis=0)
            else:
                # Neurônio vazio (dead unit): permanece como zeros ou não é usado
                pass

        return self.betas, self.labels_, new_cost

    # --- MÉTODO PREDICT ---
    def predict(self, X_test):
        """
        Prevê valores para X_test.
        1. Encontra o protótipo W mais próximo (Distância Euclidiana).
        2. Usa os Betas desse vencedor para calcular a predição.
        """
        N_test = X_test.shape[0]
        
        # 1. Encontrar Vencedor (Nearest Prototype)
        # Calcula distância euclidiana simples para os protótipos calculados pós-treino
        dist_sq = self._calculate_euclidean_dists_sq(X_test, self.prototypes)
        winner_indices = np.argmin(dist_sq, axis=1)
        
        # 2. Calcular Previsão (Regressão Linear Local)
        ones_col = np.ones((N_test, 1))
        X_prime_test = np.hstack((ones_col, X_test))
        
        # Seleciona betas do vencedor
        chosen_betas = self.betas[winner_indices] 
        
        # Mu = Produto Escalar
        y_pred = np.sum(X_prime_test * chosen_betas, axis=1)
        
        return y_pred, winner_indices

# -------------------- BLOCO PRINCIPAL --------------------

if __name__ == "__main__":
    print("--- SOM-Wise Regression (Validação Cruzada Repetida) ---")
    
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
    #            VALIDAÇÃO CRUZADA REPETIDA
    # =========================================================
    
    global_rmse_list = []
    
    print(f"\nIniciando {n_cv_reps} rodadas de {n_folds_input}-Fold Cross Validation...")
    
    for rep in range(n_cv_reps):
        print(f"\n>>> REPETIÇÃO {rep+1}/{n_cv_reps}")
        
        # Garante mistura diferente a cada repetição
        kf = KFold(n_splits=n_folds_input, shuffle=True, random_state=rep * 100)
        
        current_rep_y_true = []
        current_rep_y_pred = []
        
        fold_count = 1
        
        for train_index, test_index in kf.split(X_final):
            # print(f"   Fold {fold_count}/{n_folds_input}...", end='\r')
            
            # 1. Separa Dados
            X_train, X_test = X_final[train_index], X_final[test_index]
            y_train, y_test = y_regression[train_index], y_regression[test_index]
            
            # 2. Treina Modelo (SOM-Wise Regression)
            som_seed = (rep * 1000) + fold_count 
            som = SOMWiseRegression(mh, mw, iters, h0, hf, random_state=som_seed)
            som.fit(X_train, y_train) 
            # Nota: O fit agora calcula 'som.prototypes' ao final internamente
            
            # 3. Prediz (Usa protótipos calculados para achar vencedor e betas para prever)
            y_pred_fold, _ = som.predict(X_test)
            
            # 4. Acumula
            current_rep_y_true.extend(y_test)
            current_rep_y_pred.extend(y_pred_fold)
            
            fold_count += 1
        
        # Calcula RMSE Global desta repetição
        current_rep_y_true = np.array(current_rep_y_true)
        current_rep_y_pred = np.array(current_rep_y_pred)
        
        rep_rmse = np.sqrt(np.mean((current_rep_y_true - current_rep_y_pred)**2))
        global_rmse_list.append(rep_rmse)
        
        print(f"   [Concluído] RMSE Global da Repetição {rep+1}: {rep_rmse:.4f}")

    # =========================================================
    #            ESTATÍSTICAS FINAIS
    # =========================================================
    
    mean_rmse = np.mean(global_rmse_list)
    std_rmse = np.std(global_rmse_list)
    
    print("\n" + "="*50)
    print("   RESULTADO FINAL (Múltiplas Repetições CV)")
    print("="*50)
    print(f"Número de Folds por rodada: {n_folds_input}")
    print(f"Número de Repetições (Rodadas): {n_cv_reps}")
    print("-" * 30)
    print(f"RMSE Médio: {mean_rmse:.6f}")
    print(f"Desvio Padrão RMSE: {std_rmse:.6f}")
    print("-" * 30)
    print(f"Lista de RMSEs: {[float(round(x, 4)) for x in global_rmse_list]}")
    
    # Salva relatório estatístico
    with open("somwise_cv_stats.txt", "w", encoding="utf-8") as f_out:
        f_out.write(f"--- RELATORIO ESTATISTICO SOM-WISE CV ---\n")
        f_out.write(f"Config: Map={mh}x{mw}, Iters={iters}, Folds={n_folds_input}, Reps={n_cv_reps}\n\n")
        f_out.write(f"RMSE Medio: {mean_rmse:.6f}\n")
        f_out.write(f"Desvio Padrao: {std_rmse:.6f}\n\n")
        f_out.write(f"Todos os RMSEs globais:\n")
        for idx, val in enumerate(global_rmse_list):
            f_out.write(f"Repeticao {idx+1}: {val:.6f}\n")

    print("\nRelatório estatístico salvo em 'somwise_cv_stats.txt'.")