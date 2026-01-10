import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 

# -------------------- FUNÇÃO PARA MATRIZ DE CONFUSÃO (MANTIDA) --------------------
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
            
    col_names = [f'Classe {cls}' for cls in unique_true_classes]
    row_names = [f'Neurônio {i}' for i in range(n_clusters)] 
    df_cm = pd.DataFrame(confusion_matrix, index=row_names, columns=col_names)
    
    return df_cm

# -------------------- CARREGAMENTO E PRÉ-PROCESSAMENTO (MANTIDO) --------------------
def load_and_preprocess_data(data_file_path, headers_file_path, delimiter, target_column_name=None, perform_normalization=True):
    try:
        headers_df = pd.read_csv(headers_file_path, header=None)
        headers = headers_df.squeeze().tolist()
        df = pd.read_csv(data_file_path, header=None, names=headers, sep=delimiter)
    except FileNotFoundError:
        print(f"Erro: Um dos arquivos não foi encontrado.")
        return None, None, None
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None, None, None

    y_true = None
    if target_column_name and target_column_name in df.columns:
        y_true = df[target_column_name]
        X_df = df.drop(columns=[target_column_name])
    else:
        X_df = df
    
    if perform_normalization:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_df)
        print("As features foram normalizadas.")
    else:
        X_processed = X_df.values
        print("Nenhuma normalização foi aplicada.")

    print(f"Dataset carregado: {X_processed.shape[0]} amostras e {X_processed.shape[1]} features.")
    return X_processed, y_true, X_df.columns.tolist()

# -------------------- CLASSE BATCH SOM (ATUALIZADA) --------------------

class BatchSOM:
    def __init__(self, map_height=5, map_width=5, max_iter=100, random_state=None, h0=0.001, hf=0.5):
        self.map_height = map_height
        self.map_width = map_width
        self.n_neurons = map_height * map_width 
        self.max_iter = max_iter
        
        #self.sigma_0 = sigma_0 if sigma_0 is not None else max(map_height, map_width) / 2
        #self.sigma_f = sigma_f

        self.random_state = random_state
        self.prototypes = None 
        self.labels_ = None
        
        self.neuron_coordinates = np.array([
            [i, j] for i in range(map_height) for j in range(map_width)
        ])
        
        self.grid_distances_sq = np.zeros((self.n_neurons, self.n_neurons))
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                self.grid_distances_sq[i, j] = np.sum((self.neuron_coordinates[i] - self.neuron_coordinates[j])**2)
 # 3. CÁLCULO DO SIGMA BASEADO EM h0 e hf (NOVA LÓGICA)
        # Nota: h0 e hf devem estar entre 0 e 1 (exclusivo).
        
        # Sigma Inicial (Sigma_0):
        # Baseado na distância MÁXIMA do mapa.
        # Fórmula: sqrt( -dist_max^2 / (2 * ln(h0)) )
        max_dist_sq = np.max(self.grid_distances_sq)
        
        # Pequena proteção para evitar divisão por zero ou log inválido
        if h0 <= 0 or h0 >= 1: raise ValueError("h0 deve estar entre 0 e 1.")
        if hf <= 0 or hf >= 1: raise ValueError("hf deve estar entre 0 e 1.")
        
        self.sigma_0 = np.sqrt(-max_dist_sq / (2 * np.log(h0)))
        
        # Sigma Final (Sigma_f):
        # Baseado na distância unitária (vizinho adjacente, dist^2 = 1).
        # Fórmula: sqrt( -1 / (2 * ln(hf)) )
        self.sigma_f = np.sqrt(-1 / (2 * np.log(hf)))

        print(f"  [Config] h0={h0} -> Sigma_0={self.sigma_0:.4f}")
        print(f"  [Config] hf={hf} -> Sigma_f={self.sigma_f:.4f}")
        
    
    def _calculate_neighborhood_matrix(self, current_sigma):
        denom = 2 * (current_sigma ** 2)
        H = np.exp(-self.grid_distances_sq / denom)
        return H

    def _calculate_euclidean_dists_sq(self, X, W):
        """
        Calcula apenas a distância Euclidiana Quadrada (sem ponderação H).
        Útil para as métricas de avaliação e base para a dissimilaridade.
        """
        dist_sq = np.sum(X**2, axis=1, keepdims=True) + \
                  np.sum(W**2, axis=1) - \
                  2 * np.dot(X, W.T)
        return np.maximum(dist_sq, 0)

    def _calculate_generalized_dissimilarity(self, dist_sq, H):
        """
        Calcula a Dissimilaridade Generalizada (Delta).
        Delta = Dist_Sq @ H.T
        """
        generalized_dists = np.dot(dist_sq, H.T)
        return generalized_dists

    def calculate_cost_function(self, generalized_dists, labels):
        chosen_dists = generalized_dists[np.arange(len(labels)), labels]
        return np.sum(chosen_dists)

    # --- NOVOS MÉTODOS DE AVALIAÇÃO (Slides 19 e 20) ---

    def calculate_quantization_error(self, X):
        """
        Slide 19: QE = (1/N) * Sum(||x_k - w_f(x_k)||^2)
        Média das distâncias quadradas dos dados para seus BMUs.
        """
        # 1. Calcula distância para todos os protótipos
        dists_sq = self._calculate_euclidean_dists_sq(X, self.prototypes)
        
        # 2. Encontra a menor distância para cada dado (distância para o BMU)
        min_dists_sq = np.min(dists_sq, axis=1)
        
        # 3. Média
        qe = np.mean(min_dists_sq)
        return qe

    def calculate_topographic_error(self, X):
        """
        Slide 20: Proporção de dados onde o 1º e 2º BMUs não são vizinhos adjacentes.
        """
        # 1. Calcula todas as distâncias
        dists_sq = self._calculate_euclidean_dists_sq(X, self.prototypes)
        
        # 2. Encontra os índices dos 2 neurônios mais próximos (argsort)
        # axis=1 ordena por linha. Pegamos as colunas 0 e 1 (1º e 2º menores)
        bmu_indices = np.argsort(dists_sq, axis=1)[:, :2]
        
        bmu_1_idx = bmu_indices[:, 0]
        bmu_2_idx = bmu_indices[:, 1]
        
        # 3. Verifica se são adjacentes no grid
        # Pega as coordenadas (x,y) do 1º e do 2º
        coords_1 = self.neuron_coordinates[bmu_1_idx]
        coords_2 = self.neuron_coordinates[bmu_2_idx]
        
        # Calcula a distância física quadrada entre eles
        phys_dist_sq = np.sum((coords_1 - coords_2)**2, axis=1)
        
        # Slide 20 define u(x) = 1 se não adjacentes.
        # Adjacente em grid retangular significa distância física = 1 (cima, baixo, esq, dir).
        # Se dist > 1 (diagonal ou longe), é erro.
        errors = (phys_dist_sq > 1).astype(int)
        
        # 4. Média dos erros (Topographic Error Total)
        te = np.mean(errors)
        return te

    def fit(self, X):
        np.random.seed(self.random_state)
        N_samples, n_features = X.shape
        
        random_idx = np.random.choice(N_samples, self.n_neurons, replace=False)
        self.prototypes = X[random_idx].copy() 
        
        t = 0
        current_sigma = self.sigma_0
        
        H = self._calculate_neighborhood_matrix(current_sigma)
        
        # Inicialização
        raw_dists_sq = self._calculate_euclidean_dists_sq(X, self.prototypes)
        gen_dists = self._calculate_generalized_dissimilarity(raw_dists_sq, H)
        self.labels_ = np.argmin(gen_dists, axis=1)
        
        current_cost = self.calculate_cost_function(gen_dists, self.labels_)
        
        print(f"Inicialização: Custo J = {current_cost:.2f}, Sigma = {current_sigma:.4f}")
        
        while t < self.max_iter:
            t += 1
            current_sigma = self.sigma_0 * ((self.sigma_f / self.sigma_0) ** (t / self.max_iter))
            
            H = self._calculate_neighborhood_matrix(current_sigma)
            
            # Step 1: Adaptação
            G = H[self.labels_, :] 
            numerator = np.dot(G.T, X)
            denominator = np.sum(G, axis=0).reshape(-1, 1)
            denominator = np.where(denominator == 0, 1e-10, denominator)
            self.prototypes = numerator / denominator
            
            # Step 2: Competição
            raw_dists_sq = self._calculate_euclidean_dists_sq(X, self.prototypes)
            gen_dists = self._calculate_generalized_dissimilarity(raw_dists_sq, H)
            self.labels_ = np.argmin(gen_dists, axis=1)
            
            new_cost = self.calculate_cost_function(gen_dists, self.labels_)
            
            print(f" Iteração {t}: Sigma = {current_sigma:.4f} -> Custo J = {new_cost:.2f}")
                
        return self.prototypes, self.labels_, new_cost

# -------------------- BLOCO PRINCIPAL --------------------

if __name__ == "__main__":
    print("--- Configuração do Dataset (Batch SOM) ---")
    data_file_name = input("Digite o nome do ARQUIVO DE DADOS CSV: ")
    headers_file_name = input("Digite o nome do ARQUIVO DE HEADERS CSV: ")
    delimiter = input("Digite o delimitador do CSV: ")
    target_name = input("Digite o NOME da coluna de rótulos/target para remover: ")
    normalize_choice = input("Deseja realizar a normalização das features? (S/N): ").upper()
    h0_input = float(input("Digite o valor inicial de h0 usado para calcular sigma0: "))
    hf_input = float(input("Digite o valor de hf usado para calcular sigmaf: "))
    target_name = target_name if target_name.strip() else None
    perform_normalization = (normalize_choice == 'S')
    
    X, y_true, feature_names = load_and_preprocess_data(data_file_name, headers_file_name, delimiter, target_name, perform_normalization)

    if X is None:
        exit()
        
    try:
        print("\n--- Configuração do Mapa SOM ---")
        map_h = int(input("Digite a Altura do mapa (linhas de neurônios): "))
        map_w = int(input("Digite a Largura do mapa (colunas de neurônios): "))
        n_repetitions = int(input("Digite a quantidade de repetições do Treinamento: "))
        max_iters = int(input("Digite o número máximo de iterações (tmax): "))
    except ValueError:
        print("Entrada inválida. Use números inteiros.")
        exit()

    n_total_neurons = map_h * map_w
    print(f"\nO mapa terá {n_total_neurons} neurônios no total.")

    best_cost = float('inf')
    best_results = {} # Dicionário para guardar o melhor modelo completo
    
    for i in range(n_repetitions):
        print(f"\n--- Repetição {i+1}/{n_repetitions} ---")
        
        som = BatchSOM(map_height=map_h, map_width=map_w, max_iter=max_iters, random_state=i, h0=h0_input, hf=hf_input)
        prototypes, labels, final_cost = som.fit(X)
        
        # Calcular Métricas de Avaliação
        qe = som.calculate_quantization_error(X)
        te = som.calculate_topographic_error(X)
        
        print(f"  > Custo J: {final_cost:.2f}")
        print(f"  > Quantization Error (QE): {qe:.4f}")
        print(f"  > Topographic Error (TE): {te:.4f}")
        
        # Critério de "Melhor": Usamos o Custo J (energia), mas salvamos as métricas
        if final_cost < best_cost:
            best_cost = final_cost
            best_results = {
                'prototypes': prototypes,
                'labels': labels,
                'qe': qe,
                'te': te,
                'som_obj': som # Salva objeto para consultas futuras
            }

    # --- Resultados Finais ---
    print("\n\n--- Resultado Final (Melhor Execução Batch SOM) ---")
    print(f"Menor custo J: {best_cost:.2f}")
    print(f"Quantization Error (QE): {best_results['qe']:.4f}")
    print(f"Topographic Error (TE): {best_results['te']:.4f}")

    print("\n--- Afiliação dos Neurônios ---")
    best_labels = best_results['labels']
    # Criamos uma matriz auxiliar para guardar a contagem de hits para o print do grid
    # hits_grid = np.zeros((map_h, map_w), dtype=int)
    for j in range(n_total_neurons):
        cluster_indices = np.where(best_labels == j)[0]
        row, col = divmod(j, map_w)
        # if len(cluster_indices) > 0:
        print(f" Neurônio {j} (Pos {row},{col}) - {len(cluster_indices)} elementos: {cluster_indices.tolist()}")

    # --- VISUALIZAÇÃO DO MAPA (GRID) ---
    print("\n--- Visualização Topológica do Mapa (Linha, Coluna)---")
    # Itera de cima para baixo (map_h-1 até 0) para imprimir visualmente correto
    '''for r in range(map_h - 1, -1, -1):
        row_str = ""
        for c in range(map_w):
            row_str += f"({r},{c})\t" 
        print(row_str)'''
    for r in range(map_h - 1, -1, -1): 
        row_str = ""
        for c in range(map_w):
            # Recalcula o ID linear baseado na linha e coluna
            neuron_id = r * map_w + c
            # Formatação: :^6 centraliza o número em 6 espaços para alinhar colunas
            row_str += f"{neuron_id:^6}" 
        print(row_str)

    # --- NOVO: PRINT DA MATRIZ W ---
    print("\n--- Matriz de Protótipos (W) ---")
    # Imprime a matriz completa. Cada linha é um neurônio, cada coluna uma feature.
    # print(best_results['prototypes'])
    
    # Se quiser ver formatado por neurônio:
    for j, w_vec in enumerate(best_results['prototypes']):
        print(f" W_{j}: {w_vec}")

    # Matriz de Confusão
    print("\n\n--- Matriz de Confusão (Neurônios vs. Classes Verdadeiras) ---")
    if y_true is not None:
        best_cm_df = generate_confusion_matrix_clustering(y_true.values, best_labels, n_total_neurons)
        print(best_cm_df)
        best_cm_df.to_csv("som_matriz_confusao.csv", index=True)
    else:
        print("Sem targets para gerar matriz de confusão.")

    # Salvando CSVs
    print("\n--- Salvando Resultados ---")
    df_Q = pd.DataFrame(best_results['prototypes'], columns=feature_names)
    df_Q.to_csv("som_prototipos_finais.csv", index_label="neuronio_id")
    
    df_results = pd.DataFrame(X, columns=feature_names)
    df_results['neuronio_vencedor'] = best_labels
    if y_true is not None:
        df_results['original_target'] = y_true.values
    df_results.to_csv("som_resultados_finais.csv", index=False)
    
    print("Arquivos salvos com sucesso.")