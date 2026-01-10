import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


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
        self.log.flush()
'''
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
    
# --- MÉTODOS DE AVALIAÇÃO (ADICIONADOS) ---

    def calculate_quantization_error(self, X, y):
        """
        Calcula o QE (Erro de Quantização) para Regressão.
        QE = Média dos erros quadráticos do vencedor.
        Fonte: PDF [cite: 186]
        """
        N_samples = X.shape[0]
        ones_col = np.ones((N_samples, 1))
        X_prime = np.hstack((ones_col, X))
        y_vec = y.values if hasattr(y, 'values') else y
        y_vec = y_vec.reshape(-1, 1)

        # 1. Calcula todos os erros quadráticos (Epsilon^2)
        all_errors_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
        
        # 2. Seleciona apenas o erro do neurônio vencedor (f(x_k)) para cada dado
        winner_errors_sq = all_errors_sq[np.arange(N_samples), self.labels_]
        
        # 3. Média (1/N * Sum)
        return np.mean(winner_errors_sq)

    def calculate_topographic_error(self, X, y):
        """
        Calcula o Erro Topográfico (TE).
        Proporção de dados onde o 1º e 2º neurônios (baseados no custo generalizado)
        não são vizinhos adjacentes.
        Fonte: PDF 
        """
        N_samples = X.shape[0]
        ones_col = np.ones((N_samples, 1))
        X_prime = np.hstack((ones_col, X))
        y_vec = y.values if hasattr(y, 'values') else y
        y_vec = y_vec.reshape(-1, 1)

        # 1. Recalcula o Custo Generalizado (J) para encontrar 1º e 2º vencedores
        # Usa sigma final para avaliar a topologia final
        H_final = self._calculate_neighborhood_matrix(self.sigma_f)
        
        raw_errors_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
        gen_errors = self._calculate_generalized_error(raw_errors_sq, H_final)

        # 2. Encontra os índices dos 2 menores custos (1º e 2º vencedores) 
        sorted_indices = np.argsort(gen_errors, axis=1)
        bmu_1 = sorted_indices[:, 0] # i
        bmu_2 = sorted_indices[:, 1] # d

        # 3. Verifica Adjacência
        coords_1 = self.neuron_coordinates[bmu_1]
        coords_2 = self.neuron_coordinates[bmu_2]
        
        # Distância física no grid
        grid_dist_sq = np.sum((coords_1 - coords_2)**2, axis=1)
        
        # Se dist^2 > 1, não são vizinhos imediatos (adjacentes) -> Erro = 1
        errors = (grid_dist_sq > 1).astype(float)
        
        return np.mean(errors)

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
        # --- CÁLCULO DAS MÉTRICAS (ADICIONADO) ---
        qe = som.calculate_quantization_error(X_final, y_regression)
        te = som.calculate_topographic_error(X_final, y_regression)
        print(f"  > Custo Final: {cost:.2f}")
        print(f"  > QE: {qe:.4f} | TE: {te:.4f}")
        if cost < best_cost:
            best_cost = cost
            best_model = {'betas': betas, 'labels': labels, 'obj': som, 'qe': qe,'te': te}

# =========================================================
    #            OUTPUTS (CONSOLE RESUMIDO, TXT COMPLETO)
    # =========================================================
    
    # Abre o arquivo para escrita
    with open("somwise_melhor_resultado.txt", "w", encoding="utf-8") as f_out:
        
        # --- FUNÇÕES AUXILIARES DE LOG ---
        def log(text):
            """Printa no console (normal) e salva no arquivo."""
            print(text)
            f_out.write(str(text) + "\n")

        def log_matrix_full(title, matrix):
            """
            Printa no console resumido (default do numpy).
            Salva no arquivo COMPLETO (threshold=maxsize).
            """
            log(f"\n--- {title} ---")
            
            # 1. Console (Resumido)
            print(f"Shape: {matrix.shape}")
            print(matrix) 
            
            # 2. Arquivo (Completo)
            f_out.write(f"Shape: {matrix.shape}\n")
            # Converte a matriz inteira para string
            full_str = np.array2string(matrix, threshold=sys.maxsize, max_line_width=200)
            f_out.write(full_str + "\n")

        def log_matrix_with_indices(title, matrix, prefix="Row"):
            """
            Printa no console resumido.
            Salva no arquivo ITERANDO LINHA A LINHA com índices (Row_0, Row_1...)
            """
            log(f"\n--- {title} ---")
            
            # Console: Resumido (padrão numpy)
            print(f"Shape: {matrix.shape}")
            print(matrix)
            
            # Arquivo: Detalhado linha por linha
            f_out.write(f"Shape: {matrix.shape}\n")
            rows = matrix.shape[0]
            for i in range(rows):
                # Converte apenas a linha atual para string completa
                row_val = np.array2string(matrix[i], threshold=sys.maxsize, max_line_width=300)
                f_out.write(f"{prefix}_{i}: {row_val}\n")

        # --- INÍCIO DO RELATÓRIO ---
        log("\n\n" + "="*40)
        log("   RELATÓRIO FINAL (Melhor Execução)")
        log("="*40)
        log(f"Menor custo J: {best_cost:.2f}")
        log(f"Quantization Error (QE): {best_model['qe']:.4f}") # Novo Output
        log(f"Topographic Error (TE): {best_model['te']:.4f}")  # Novo Output
        # --- CÁLCULO DE RMSE GLOBAL (NOVO) ---
        # 1. Recupera as predições do neurônio vencedor para cada dado
        chosen_betas = best_model['betas'][best_model['labels']]
        ones = np.ones((X_final.shape[0], 1))
        X_p = np.hstack((ones, X_final))
        y_pred_final = np.sum(X_p * chosen_betas, axis=1)
        
        # 2. Fórmula do RMSE: sqrt(mean((y_true - y_pred)^2))
        rmse = np.sqrt(np.mean((y_regression.values - y_pred_final)**2))
        log(f"RMSE Global (Root Mean Squared Error): {rmse:.4f}")
        
        # --- Afiliação ---
        log("\n--- Afiliação dos Neurônios ---")
        n_total_neurons = mh * mw
        best_labels = best_model['labels']
        for j in range(n_total_neurons):
            cluster_indices = np.where(best_labels == j)[0]
            row, col = divmod(j, mw)
            log(f" Neurônio {j} (Pos {row},{col}) - {len(cluster_indices)} elementos: {cluster_indices.tolist()}")

        # --- Visualização Topológica ---
        log("\n--- Visualização Topológica do Mapa (Linha, Coluna) ---")
        for r in range(mh - 1, -1, -1): 
            row_str = ""
            for c in range(mw):
                neuron_id = r * mw + c
                row_str += f"{neuron_id:^6}" 
            log(row_str)

        # --- Betas ---
        log("\n--- Matriz de Coeficientes Beta (W_equivalente) ---")
        for j, beta_vec in enumerate(best_model['betas']):
            log(f" Beta_{j}: {beta_vec}")

        # --- CÁLCULO DAS MATRIZES ---
        N = X_final.shape[0]
        ones_col = np.ones((N, 1))
        X_prime = np.hstack((ones_col, X_final))
        y_vec = y_regression.values.reshape(-1, 1)
        
        Mu_matrix = np.dot(X_prime, best_model['betas'].T)
        Error_matrix = (y_vec - Mu_matrix)

        # --- PRINT HÍBRIDO (Console Resumido / TXT Linha a Linha) ---
        # Aqui usamos a nova função para colocar os índices
        log_matrix_full("Vetor Alvo (y)", y_vec.flatten()) # Vetor y não precisa de índice de linha complexo
                # --- Matriz de Confusão ---
        log("\n\n--- Matriz de Confusão ---")
        if y_class_labels is not None:
            best_cm_df = generate_confusion_matrix_clustering(y_class_labels.values, best_labels, n_total_neurons)
            
            # Console: pandas padrão
            print(best_cm_df)
            
            # Arquivo: string completa
            f_out.write(best_cm_df.to_string() + "\n")
            
            best_cm_df.to_csv("somwise_matriz_confusao.csv", index=True)

                        # --- CÁLCULO DE ARI E NMI ---
            ari = adjusted_rand_score(y_class_labels.values, best_labels)
            nmi = normalized_mutual_info_score(y_class_labels.values, best_labels)
            
            log(f"\n--- Métricas de Validação de Clustering ---")
            log(f"Adjusted Rand Index (ARI): {ari:.4f}")
            log(f"Normalized Mutual Information (NMI): {nmi:.4f}")
        else:
            log("Sem coluna de classe original para gerar matriz de confusão.")
        log_matrix_with_indices("Matriz Mu (Estimativas)", Mu_matrix, prefix="Sample")
        log_matrix_with_indices("Matriz de Erros (Epsilon)", Error_matrix, prefix="Sample")

    # --- Salvando CSVs ---
    print("\n--- Salvando Resultados CSV ---")
    
    beta_cols = ['Intercept'] + feature_names
    df_B = pd.DataFrame(best_model['betas'], columns=beta_cols)
    df_B.to_csv("somwise_betas_finais.csv", index_label="neuronio_id")
    
    chosen_betas = best_model['betas'][best_labels]
    y_pred_final = np.sum(X_prime * chosen_betas, axis=1)

    df_results = pd.DataFrame(X_raw, columns=feature_names) 
    df_results['y_real'] = y_regression.values
    df_results['y_pred'] = y_pred_final
    df_results['neuronio_vencedor'] = best_labels
    
    if y_class_labels is not None:
        df_results['classe_original'] = y_class_labels.values
        
    df_results.to_csv("somwise_resultados_finais.csv", index=False)
    
    print("Arquivos salvos com sucesso.")
    print("Log completo (matrizes inteiras) salvo em: somwise_melhor_resultado.txt")