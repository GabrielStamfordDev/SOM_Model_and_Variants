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
        # Mostra todos, inclusive vazios
        print(f" Neurônio {j} (Pos {row},{col}) - {len(cluster_indices)} amostras.")

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
        predictions = np.dot(X_prime, betas.T)
        y_col = y.reshape(-1, 1)
        errors_sq = (y_col - predictions) ** 2
        return errors_sq

    def _calculate_generalized_error(self, errors_sq, H):
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
        
        # Prints de inspeção das matrizes (como pedido anteriormente)
        print("\n" + "="*40)
        print("   DADOS PREPARADOS (Somente Inicialização)")
        print("="*40)
        # print(f">> Matriz Alvo y (primeiras 5 linhas):\n{y_vec[:5]}")
        # print(f">> Matriz X' (primeiras 5 linhas):\n{X_prime[:5]}")
        
        # [cite_start]--- INICIALIZAÇÃO [cite: 50-55] ---
        t = 0
        current_sigma = self.sigma_0
        H = self._calculate_neighborhood_matrix(current_sigma)
        
        initialization_success = False
        attempt_counter = 0
        self.betas = np.zeros((self.n_neurons, n_features + 1))

        print("\n>> Iniciando busca por partição inicial válida...")

        while not initialization_success:
            attempt_counter += 1
            
            # 1. Sorteio aleatório de grupos
            current_labels = np.random.randint(0, self.n_neurons, size=N_samples)
            
            # 2. Cálculo da matriz de influência G (H')
            G = H[current_labels, :] 
            
            all_neurons_invertible = True
            temp_betas = np.zeros((self.n_neurons, n_features + 1))
            
            # 3. Teste de inversibilidade para TODOS os neurônios
            for r in range(self.n_neurons):
                diag_Hr = G[:, r] 
                sqrt_w = np.sqrt(diag_Hr + 1e-12).reshape(-1, 1)
                X_weighted = X_prime * sqrt_w
                y_weighted = y_vec * sqrt_w.flatten()
                
                XTX = np.dot(X_weighted.T, X_weighted)
                XTy = np.dot(X_weighted.T, y_weighted)
                
                # Regularização minúscula apenas para evitar singularidade numérica perfeita
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
                
                # Calcula o Custo da configuração inicial encontrada
                raw_err = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
                gen_err = self._calculate_generalized_error(raw_err, H)
                init_cost = self.calculate_cost_function(gen_err, self.labels_)
                
                print(f">> INICIALIZAÇÃO SUCESSO após {attempt_counter} tentativas.")
                print(f">> Custo Inicial J = {init_cost:.2f}")
                print("\n--- Matriz Beta Inicial (Intercept + Coefs) ---")
                print(self.betas)
                print("-" * 40)
            else:
                # Printa falha a cada 10 tentativas para não floodar o console
                if attempt_counter % 10 == 0:
                    print(f"   Tentativa {attempt_counter} falhou (matrizes singulares)... tentando nova partição.")

        # ==============================================================================
        # O TREINAMENTO FOI COMENTADO ABAIXO PARA RODAR APENAS A INICIALIZAÇÃO
        # ==============================================================================
        '''
        while t < self.max_iter:
            t += 1
            current_sigma = self.sigma_0 * ((self.sigma_f / self.sigma_0) ** (t / self.max_iter))
            H = self._calculate_neighborhood_matrix(current_sigma)
            
            # Step 1: Competição
            raw_errors_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
            gen_errors = self._calculate_generalized_error(raw_errors_sq, H)
            self.labels_ = np.argmin(gen_errors, axis=1)
            
            # Step 2: Adaptação
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

            raw_errors_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
            gen_errors = self._calculate_generalized_error(raw_errors_sq, H)
            new_cost = self.calculate_cost_function(gen_errors, self.labels_)
            
            if t % 10 == 0 or t == self.max_iter:
                print(f" Iteração {t}: Sigma = {current_sigma:.4f} -> Custo J = {new_cost:.2f}")
        
        return self.betas, self.labels_, new_cost
        '''
        
        # Retorna os valores da inicialização apenas
        return self.betas, self.labels_, init_cost

# -------------------- BLOCO PRINCIPAL --------------------
if __name__ == "__main__":
    print("--- SOM-Wise Regression (SOMENTE INICIALIZAÇÃO) ---")
    data_file = input("Arquivo CSV Dados: ")
    header_file = input("Arquivo CSV Headers: ")
    delim = input("Delimitador: ")
    class_label = input("Coluna de CLASSE (para remover): ").strip()
    class_label = class_label if class_label else None

    df_full, y_class_labels = load_data_initial(data_file, header_file, delim, class_label)
    if df_full is None: exit()

    print("\n--- Variáveis Numéricas ---")
    cols = df_full.columns.tolist()
    for idx, col_name in enumerate(cols):
        print(f" [{idx}] {col_name}")

    try:
        target_idx = int(input("\nDigite o ÍNDICE da variável numérica que será o alvo (y) da regressão:  "))
        target_col_name = cols[target_idx]
        y_regression = df_full[target_col_name]
        X_raw = df_full.drop(columns=[target_col_name])
        feature_names = X_raw.columns.tolist()
    except:
        print("Erro na seleção.")
        exit()

    norm_choice = input("Normalizar X? (S/N): ").upper()
    X_final = StandardScaler().fit_transform(X_raw) if norm_choice == 'S' else X_raw.values

    try:
        mh = int(input("Altura: "))
        mw = int(input("Largura: "))
        h0 = float(input("h0 (0-1): "))
        hf = float(input("hf (0-1): "))
        reps = int(input("Repetições: ")) # Coloque 1 se quiser ver apenas uma vez
        iters = int(input("Max Iterações (Não será usado): "))
    except:
        print("Erro inputs.")
        exit()

    best_cost = float('inf')
    best_model = None
    
    for i in range(reps):
        print(f"\n--- Repetição {i+1}/{reps} ---")
        som = SOMWiseRegression(mh, mw, iters, h0, hf, random_state=i)
        
        # Vai rodar SOMENTE a inicialização
        betas, labels, cost = som.fit(X_final, y_regression)
        
        if cost < best_cost:
            best_cost = cost
            best_model = {'betas': betas, 'labels': labels}

    # Resultados Finais da Inicialização
    if best_model:
        print(f"\n\n=== MELHOR RESULTADO INICIAL ===")
        print(f"Melhor J Inicial: {best_cost:.2f}")
        print_results(best_model['betas'], best_model['labels'], (mh, mw))