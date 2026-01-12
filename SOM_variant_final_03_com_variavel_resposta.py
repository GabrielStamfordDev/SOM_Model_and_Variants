import numpy as np
import pandas as pd
import sys 
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler 

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

# -------------------- CLASSE SOM HÍBRIDO (Nova Versão) --------------------
class SOMHybrid:
    def __init__(self, map_height=5, map_width=5, max_iter=100, h0=0.001, hf=0.5, random_state=None):
        self.map_height = map_height
        self.map_width = map_width
        self.n_neurons = map_height * map_width 
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Parâmetros do Modelo Híbrido
        self.betas = None       # Coeficientes de Regressão (C x P+1)
        self.prototypes = None  # Protótipos W (C x P)
        self.lambdas = None     # Pesos das features (P,)
        self.alpha = 1.0        # Peso do erro de regressão
        self.labels_ = None
        self.history = []       # Histórico para a linha do tempo
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
        """Calcula epsilon^2 = (y - X'Beta)^2 para todos os neurônios."""
        predictions = np.dot(X_prime, betas.T)
        y_col = y.reshape(-1, 1)
        errors_sq = (y_col - predictions) ** 2
        return errors_sq

    def _calculate_weighted_quantization_dist_sq(self, X, W, lambdas):
        """
        Calcula a distância Euclidiana ponderada por Lambda:
        Sum(lambda_j * (x_kj - w_mj)^2)
        Otimizado matricialmente: Lambda*X^2 + Lambda*W^2 - 2*(Lambda*X)*W^T
        """
        # Termo 1: Soma ponderada de X^2 (N, 1)
        X_sq_weighted = np.dot(X**2, lambdas).reshape(-1, 1)
        
        # Termo 2: Soma ponderada de W^2 (C, )
        W_sq_weighted = np.dot(W**2, lambdas)
        
        # Termo 3: Produto cruzado ponderado
        # (X * lambdas) @ W.T
        Cross_term = 2 * np.dot(X * lambdas, W.T)
        
        # Distância (N, C)
        # Broadcasting: (N,1) + (C,) - (N,C)
        dist_sq = X_sq_weighted + W_sq_weighted - Cross_term
        return np.maximum(dist_sq, 0)

    def _calculate_total_generalized_error(self, reg_errors_sq, quant_dists_sq, H, alpha):
        """
        Calcula o erro combinado ponderado pela vizinhança:
        Sum_m h_sm [ alpha * epsilon_km^2 + quant_dist_km^2 ]
        """
        # Erro Total por dado k e neurônio m
        total_error_per_neuron = (alpha * reg_errors_sq) + quant_dists_sq
        
        # Ponderação pela vizinhança (Generalized Error)
        # Resultado (N, C) -> Custo se o neurônio s for o vencedor
        return np.dot(total_error_per_neuron, H.T)
    
# --- MÉTODOS DE AVALIAÇÃO (NOVO) ---
    
    def calculate_quantization_error(self, X, y):
        """
        Calcula o QE baseado na fórmula do Slide 5.
        QE = Média( Alpha * (ErroRegressãoVencedor)^2 + (DistanciaW_Vencedor)^2 )
        """
        N_samples = X.shape[0]
        ones_col = np.ones((N_samples, 1))
        X_prime = np.hstack((ones_col, X))
        y_vec = y.values if hasattr(y, 'values') else y
        y_vec = y_vec.reshape(-1, 1)

        # 1. Calcula erros individuais para todos
        reg_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
        quant_sq = self._calculate_weighted_quantization_dist_sq(X, self.prototypes, self.lambdas)

        # 2. Seleciona apenas os erros do Vencedor (BMU)
        # Fancy indexing: Linha k, Coluna labels_[k]
        winner_reg_sq = reg_sq[np.arange(N_samples), self.labels_]
        winner_quant_sq = quant_sq[np.arange(N_samples), self.labels_]

        # 3. Combina ponderado por Alpha
        total_error = (self.alpha * winner_reg_sq) + winner_quant_sq
        
        # 4. Média
        return np.mean(total_error)

    def calculate_topographic_error(self, X, y):
        """
        Calcula o TE baseado no Slide 6 [cite: 93-107].
        Proporção de dados onde o 1º e 2º vencedores (baseados no custo generalizado)
        NÃO são vizinhos adjacentes.
        """
        N_samples = X.shape[0]
        ones_col = np.ones((N_samples, 1))
        X_prime = np.hstack((ones_col, X))
        y_vec = y.values if hasattr(y, 'values') else y
        
        # 1. Recalcula o Custo Generalizado Total (usando sigma final ou atual)
        # O PDF define o vencedor usando a vizinhança[cite: 94].
        reg_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
        quant_sq = self._calculate_weighted_quantization_dist_sq(X, self.prototypes, self.lambdas)
        
        # Usamos sigma_f para a avaliação final de topologia
        H_final = self._calculate_neighborhood_matrix(self.sigma_f)
        gen_err = self._calculate_total_generalized_error(reg_sq, quant_sq, H_final, self.alpha)

        # 2. Encontra 1º e 2º Vencedores
        # argsort retorna os índices ordenados do menor custo para o maior
        sorted_indices = np.argsort(gen_err, axis=1)
        bmu_1 = sorted_indices[:, 0]
        bmu_2 = sorted_indices[:, 1]

        # 3. Verifica Adjacência
        coords_1 = self.neuron_coordinates[bmu_1]
        coords_2 = self.neuron_coordinates[bmu_2]
        
        # Distância Euclidiana Quadrada no Grid
        grid_dist_sq = np.sum((coords_1 - coords_2)**2, axis=1)
        
        # Erro = 1 se dist > 1 (não adjacentes), 0 se adjacentes ou mesmo neurônio
        # Adjacentes em grid retangular tem dist^2 = 1.
        errors = (grid_dist_sq > 1).astype(float)
        
        return np.mean(errors)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        N_samples, n_features = X.shape
        
        # X' (matriz estendida para regressão)
        ones_col = np.ones((N_samples, 1))
        X_prime = np.hstack((ones_col, X)) 
        y_vec = y.values if hasattr(y, 'values') else y
        
        # --- INICIALIZAÇÃO [Conforme PDF Híbrido] ---
        t = 0
        current_sigma = self.sigma_0
        H = self._calculate_neighborhood_matrix(current_sigma)
        
        initialization_success = False
        attempt_counter = 0
        
        self.betas = np.zeros((self.n_neurons, n_features + 1))
        self.prototypes = np.zeros((self.n_neurons, n_features))
        self.lambdas = np.ones(n_features) # Inicializa Lambda com 1s
        self.alpha = 1.0                   # Inicializa Alpha com 1
        self.history = []
        print(f"\n>> Inicialização Híbrida (Tentativa e Erro)...")

        while not initialization_success:
            attempt_counter += 1
            
            # [cite_start]1. Afetação Aleatória de Protótipos W [cite: 132]
            random_idx = np.random.choice(N_samples, self.n_neurons, replace=False)
            self.prototypes = X[random_idx].copy()
            
            # Reset de lambdas e alpha
            self.lambdas = np.ones(n_features)
            self.alpha = 1.0
            
            # 2. Definição inicial dos grupos (baseado apenas em W pois beta ainda não existe)
            # [cite_start]"rho_r = ... argmin Sum lambda(x-w)^2" [cite: 133]
            quant_dists = self._calculate_weighted_quantization_dist_sq(X, self.prototypes, self.lambdas)
            current_labels = np.argmin(quant_dists, axis=1)
            
            # 3. Calcular H' (Matriz G)
            G = H[current_labels, :] 
            
            all_neurons_invertible = True
            temp_betas = np.zeros((self.n_neurons, n_features + 1))
            
            # [cite_start]4. Calcular Beta_r inicial [cite: 134-137]
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
                
                # [cite_start]J Inicial [cite: 140]
                reg_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
                quant_sq = self._calculate_weighted_quantization_dist_sq(X, self.prototypes, self.lambdas)
                gen_err = self._calculate_total_generalized_error(reg_sq, quant_sq, H, self.alpha)
                # Soma custos dos vencedores
                init_cost = np.sum(gen_err[np.arange(N_samples), self.labels_])
                
                print(f">> Inicialização OK ({attempt_counter} tentativas). Custo J = {init_cost:.2f}")
            else:
                print(f"   Tentativa {attempt_counter} falhou (matriz singular)...")

        # --- LOOP PRINCIPAL ---
        while t < self.max_iter:
            t += 1
            current_sigma = self.sigma_0 * ((self.sigma_f / self.sigma_0) ** (t / self.max_iter))
            H = self._calculate_neighborhood_matrix(current_sigma)
            
            # [cite_start]1) Compute W (Update Protótipos - Batch SOM tradicional) [cite: 144]
            G = H[self.labels_, :] # Matriz de vizinhança dos vencedores
            numerator = np.dot(G.T, X)
            denominator = np.sum(G, axis=0).reshape(-1, 1)
            denominator = np.where(denominator == 0, 1e-10, denominator)
            self.prototypes = numerator / denominator
            
            # [cite_start]2) Competição (Update Labels) [cite: 145]
            # Usa alpha e lambdas da iteração anterior
            reg_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
            quant_sq = self._calculate_weighted_quantization_dist_sq(X, self.prototypes, self.lambdas)
            
            gen_err = self._calculate_total_generalized_error(reg_sq, quant_sq, H, self.alpha)
            self.labels_ = np.argmin(gen_err, axis=1)
            
            # [cite_start]3) Compute H' (Atualiza G com novos labels) [cite: 148]
            G = H[self.labels_, :]
            
            # [cite_start]4) Calcule Alpha e Lambda [cite: 150-154]
            # Precisamos calcular os termos A e B_j
            # A = Sum_r Sum_k h_f(k),r * epsilon_kr^2 (Erro Regressão Ponderado Total)
            # A é equivalente a np.sum(np.dot(reg_sq, H.T)[range(N), labels]) 
            # Mas vamos calcular de forma otimizada usando G.
            # O peso do dado k no neuronio r é G[k,r]
            
            # A (Termo de Regressão)
            # Sum_k Sum_r G[k,r] * error_sq[k,r]
            # Otimização: Element-wise multiply e soma tudo
            A = np.sum(G * reg_sq)
            
            # B_j (Termo de Quantização por feature j)
            # B_j = Sum_k Sum_r G[k,r] * (x_kj - w_rj)^2
            # Expandir: Sum_k Sum_r G_kr * (x_kj^2 - 2 x_kj w_rj + w_rj^2)
            # Cálculo vetorizado para todas as features de uma vez
            G_sum_r = np.sum(G, axis=0) # (C,) Soma das colunas de G
            G_X = np.dot(G.T, X)        # (C, P) Soma ponderada de X para cada neuronio
            G_X2 = np.dot(G.T, X**2)    # (C, P) Soma ponderada de X^2
            
            # Para cada feature j e neuronio r:
            # Termo = G_X2[r,j] - 2*W[r,j]*G_X[r,j] + W[r,j]^2 * G_sum_r[r]
            # Somar sobre r para obter B_j
            term_diffs = G_X2 - 2 * self.prototypes * G_X + (self.prototypes**2) * G_sum_r.reshape(-1,1)
            B = np.sum(term_diffs, axis=0) # (P,) Vetor B_j
            
            # Evitar divisão por zero
            A = max(A, 1e-10)
            B = np.maximum(B, 1e-10)
            
            # Produto de B_h
            prod_B = np.prod(B)
            
            # Numerador comum
            numerator_factor = (A * prod_B) ** (1 / (1 + n_features))
            
            # Atualiza Parâmetros
            self.alpha = numerator_factor / A
            self.lambdas = numerator_factor / B
            
            # [cite_start]5) Update Betas (Adaptação Regressão) [cite: 156-162]
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
                    pass # Mantém beta anterior

            # Calcular J Final da iteração
            reg_sq = self._calculate_regression_errors_sq(X_prime, y_vec, self.betas)
            quant_sq = self._calculate_weighted_quantization_dist_sq(X, self.prototypes, self.lambdas)
            gen_err = self._calculate_total_generalized_error(reg_sq, quant_sq, H, self.alpha)
            new_cost = np.sum(gen_err[np.arange(N_samples), self.labels_])
            self.history.append({
                't': t,
                'sigma': current_sigma,
                'alpha': self.alpha,
                'lambdas': self.lambdas.copy(),
                'cost': new_cost
            })
            #if t % 10 == 0 or t == self.max_iter:
            print(f" Iter {t} | Sigma: {current_sigma:.4f} | Alpha: {self.alpha:.6f} | J: {new_cost:.2f}")
            print(f"    Lambdas: {np.round(self.lambdas, 5)}")

        return self.betas, self.labels_, new_cost

# -------------------- BLOCO PRINCIPAL --------------------

if __name__ == "__main__":
    print("--- SOM Híbrido (Batch + Regression) ---")
    
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
        y_regression = df_full[target_col_name]
        # --- ALTERAÇÃO AQUI ---
        # Antes: X_raw = df_full.drop(columns=[target_col_name])
        # Agora: Mantemos o dataframe completo
        X_raw = df_full.copy()
        feature_names = X_raw.columns.tolist()
        print(f"\n>> Alvo da Regressão (y): {target_col_name}")
        print(f">> Features de Entrada (X - INCLUI O ALVO): {feature_names}")
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
        mw = int(input("Digite a Largura do mapa (colunas de neurônios): "))
        h0 = float(input("Digite o valor inicial de h0 usado para calcular sigma0: "))
        hf = float(input("Digite o valor de hf usado para calcular sigmaf: "))
        reps = int(input("Digite a quantidade de repetições do Treinamento: "))
        iters = int(input("Digite o número máximo de iterações (tmax): "))
    except:
        print("Erro nos inputs numéricos.")
        exit()

    best_cost = float('inf')
    best_model = None
    
    # 2. Loop de Treinamento
    for i in range(reps):
        print(f"\n--- Repetição {i+1}/{reps} ---")
        som = SOMHybrid(mh, mw, iters, h0, hf, random_state=i)
        
        betas, labels, cost = som.fit(X_final, y_regression)
        # --- CÁLCULO DAS MÉTRICAS NOVAS ---
        qe = som.calculate_quantization_error(X_final, y_regression)
        te = som.calculate_topographic_error(X_final, y_regression)
        print(f"  > Custo Final: {cost:.2f}")
        print(f"  > QE: {qe:.4f} | TE: {te:.4f}")

        if cost < best_cost:
            best_cost = cost
            best_model = {
                'betas': betas, 
                'labels': labels, 
                'obj': som, 
                'prototypes': som.prototypes,
                'alpha': som.alpha,
                'lambdas': som.lambdas,
                'qe': qe,
                'te': te
            }

    # =========================================================
    #            OUTPUTS E LOG (ABORDAGEM SEGURA)
    # =========================================================
    
    # Abre o arquivo uma única vez para escrita segura
    with open("somhybrid_melhor_resultado.txt", "w", encoding="utf-8") as f_out:
        
        # --- Funções Auxiliares LOCAIS ---
        def log(text):
            """Escreve no console e no arquivo."""
            print(text)
            f_out.write(str(text) + "\n")

        def log_matrix_full(title, matrix):
            """
            Console: Resumido (...)
            Arquivo: Bloco completo
            """
            log(f"\n--- {title} ---")
            
            # Console
            print(f"Shape: {matrix.shape}")
            print(matrix)
            
            # Arquivo
            f_out.write(f"Shape: {matrix.shape}\n")
            full_str = np.array2string(matrix, threshold=sys.maxsize, max_line_width=200)
            f_out.write(full_str + "\n")

        def log_matrix_with_indices(title, matrix, prefix="Row"):
            """
            Console: Resumido (...)
            Arquivo: Iterado linha a linha (Sample_0: [...])
            """
            log(f"\n--- {title} ---")
            
            # Console
            print(f"Shape: {matrix.shape}")
            print(matrix)
            
            # Arquivo
            f_out.write(f"Shape: {matrix.shape}\n")
            rows = matrix.shape[0]
            for i in range(rows):
                # Converte apenas a linha atual
                row_val = np.array2string(matrix[i], threshold=sys.maxsize, max_line_width=300)
                f_out.write(f"{prefix}_{i}: {row_val}\n")

        # --- GERAÇÃO DO RELATÓRIO ---
        log("\n\n" + "="*40)
        log("   RELATÓRIO FINAL (Melhor Execução Híbrida)")
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
        # --- Parâmetros Híbridos ---
        log("\n--- Parâmetros de Ponderação ---")
        log(f"Alpha (Peso Regressão): {best_model['alpha']:.6f}")
        log(f"Lambdas (Pesos Features W):")
        log(best_model['lambdas'])
        
# --- LINHA DO TEMPO (NOVO) ---
        log("\n" + "="*80)
        log("LINHA DO TEMPO DOS PARÂMETROS (Evolução durante o treino)")
        log("="*80)
        # Cabeçalho da tabela
        # Ajustar a largura dependendo de quantas features existirem
        log(f"{'Iter':<5} | {'Sigma':<8} | {'Custo J':<12} | {'Alpha':<10} | {'Lambdas (Vetor)'}")
        log("-" * 80)
        
        for entry in best_model['obj'].history:
            # Formata vetor lambdas em string compacta
            l_str = " ".join([f"{val:.3f}" for val in entry['lambdas']])
            row = f"{entry['t']:<5} | {entry['sigma']:<8.4f} | {entry['cost']:<12.2f} | {entry['alpha']:<10.6f} | [ {l_str} ]"
            log(row)
        log("-" * 80)

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
        log("\n--- Matriz de Coeficientes Beta (Regressão) ---")
        for j, beta_vec in enumerate(best_model['betas']):
            log(f" Beta_{j}: {beta_vec}")

        # --- Protótipos W ---
        log("\n--- Matriz de Protótipos W (Clustering) ---")
        for j, w_vec in enumerate(best_model['prototypes']):
            log(f" W_{j}: {w_vec}")

        # --- CÁLCULO DE MU E ERROS ---
        N = X_final.shape[0]
        ones_col = np.ones((N, 1))
        X_prime = np.hstack((ones_col, X_final))
        y_vec = y_regression.values.reshape(-1, 1)
        
        # Matriz Mu e Erros
        Mu_matrix = np.dot(X_prime, best_model['betas'].T)
        Error_matrix = (y_vec - Mu_matrix)

        # --- PRINTS DETALHADOS ---
        # 1. Vetor Y (Pode ser bloco completo, pois é um vetor simples)
        log_matrix_full("Vetor Alvo (y)", y_vec.flatten())
        
        # --- Matriz de Confusão ---
        log("\n\n--- Matriz de Confusão ---")
        if y_class_labels is not None:
            best_cm_df = generate_confusion_matrix_clustering(y_class_labels.values, best_labels, n_total_neurons)
            
            # Console
            print(best_cm_df)
            
            # Arquivo
            f_out.write(best_cm_df.to_string() + "\n")
            
            best_cm_df.to_csv("somhybrid_matriz_confusao.csv", index=True)

            # --- CÁLCULO DE ARI E NMI ---
            ari = adjusted_rand_score(y_class_labels.values, best_labels)
            nmi = normalized_mutual_info_score(y_class_labels.values, best_labels)
            
            log(f"\n--- Métricas de Validação de Clustering ---")
            log(f"Adjusted Rand Index (ARI): {ari:.4f}")
            log(f"Normalized Mutual Information (NMI): {nmi:.4f}")

        else:
            log("Sem coluna de classe original para gerar matriz de confusão.")


        # 2. Mu (Linha a linha com prefixo Sample)
        log_matrix_with_indices("Matriz Mu (Estimativas)", Mu_matrix, prefix="Sample")
        
        # 3. Erro (Linha a linha com prefixo Sample)
        log_matrix_with_indices("Matriz de Erros (Epsilon)", Error_matrix, prefix="Sample")

    # --- Salvando CSVs ---
    print("\n--- Salvando Resultados CSV ---")
    
    # Betas
    beta_cols = ['Intercept'] + feature_names
    pd.DataFrame(best_model['betas'], columns=beta_cols).to_csv("somhybrid_betas_finais.csv", index_label="neuronio_id")
    
    # Protótipos W
    pd.DataFrame(best_model['prototypes'], columns=feature_names).to_csv("somhybrid_prototipos_finais.csv", index_label="neuronio_id")
    
    # Resultados por Amostra
    chosen_betas = best_model['betas'][best_labels]
    y_pred_final = np.sum(X_prime * chosen_betas, axis=1)

    df_results = pd.DataFrame(X_raw, columns=feature_names) 
    df_results['y_real'] = y_regression.values
    df_results['y_pred'] = y_pred_final
    df_results['neuronio_vencedor'] = best_labels
    
    if y_class_labels is not None:
        df_results['classe_original'] = y_class_labels.values
        
    df_results.to_csv("somhybrid_resultados_finais.csv", index=False)
    
    print("Arquivos salvos com sucesso.")
    print("Log completo salvo em: somhybrid_melhor_resultado.txt")