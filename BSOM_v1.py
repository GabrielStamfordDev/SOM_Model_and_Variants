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
    row_names = [f'Neurônio {i}' for i in range(n_clusters)] # Ajustado para "Neurônio"
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

# -------------------- CLASSE BATCH SOM (Slide 18) --------------------

class BatchSOM:
    def __init__(self, map_height=5, map_width=5, max_iter=100, sigma_0=None, sigma_f=0.01, random_state=None):
        self.map_height = map_height
        self.map_width = map_width
        self.n_neurons = map_height * map_width  # Total de clusters (K)
        self.max_iter = max_iter
        
        # Define sigma inicial como metade da maior dimensão do mapa se não for informado
        self.sigma_0 = sigma_0 if sigma_0 is not None else max(map_height, map_width) / 2
        self.sigma_f = sigma_f
        
        self.random_state = random_state
        self.prototypes = None # Matriz W
        self.labels_ = None
        
        # Coordenadas físicas dos neurônios no grid (matriz 'a' dos slides)
        self.neuron_coordinates = np.array([
            [i, j] for i in range(map_height) for j in range(map_width)
        ])
        
        # Pré-calcula a matriz de distâncias FÍSICAS no grid entre neurônios
        # Slide 11/18: ||a_s - a_r||^2
        self.grid_distances_sq = np.zeros((self.n_neurons, self.n_neurons))
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                self.grid_distances_sq[i, j] = np.sum((self.neuron_coordinates[i] - self.neuron_coordinates[j])**2)

    def _calculate_neighborhood_matrix(self, current_sigma):
        """
        Calcula a matriz H (h_sr) baseada no sigma atual.
        Fórmula Slide 9/18: h_bk = exp(-||ab - ak||^2 / 2sigma^2)
        """
        denom = 2 * (current_sigma ** 2)
        # H[s, r] é a vizinhança entre neurônio s e neurônio r
        H = np.exp(-self.grid_distances_sq / denom)
        return H

    def _calculate_generalized_dissimilarity(self, X, W, H):
        """
        Calcula a Dissimilaridade Generalizada (Delta) para a competição.
        Fórmula Slide 14/18: Delta = Sum(h_sr * d^2(x, w_r))
        Retorna matriz (N_amostras, N_neuronios) onde o valor é o custo se aquele neurônio vencer.
        """
        # 1. Distância Euclidiana Quadrada entre cada dado X e cada protótipo W
        # (X - W)^2 = X^2 - 2XW + W^2
        # Forma eficiente vetorizada:
        dist_sq = np.sum(X**2, axis=1, keepdims=True) + \
                  np.sum(W**2, axis=1) - \
                  2 * np.dot(X, W.T)
        
        # Garante não negativo por erros de ponto flutuante
        dist_sq = np.maximum(dist_sq, 0)
        
        # 2. Aplica a ponderação pela vizinhança
        # O custo de escolher o neurônio 's' como vencedor é a soma ponderada das distâncias
        # dos vizinhos 'r' aos dados. Matematicamente: GeneralizedDist = Dist_Sq @ H.T
        generalized_dists = np.dot(dist_sq, H.T)
        
        return generalized_dists, dist_sq

    def calculate_cost_function(self, generalized_dists, labels):
        """
        Calcula J(W, P). Slide 14.
        Soma das dissimilaridades generalizadas apenas para os vencedores escolhidos.
        """
        # Pega o valor da dissimilaridade correspondente ao vencedor (label) de cada amostra
        chosen_dists = generalized_dists[np.arange(len(labels)), labels]
        return np.sum(chosen_dists)

    def fit(self, X):
        np.random.seed(self.random_state)
        N_samples, n_features = X.shape
        
        # 1. Initialization (Slide 18)
        # Seleciona aleatoriamente C protótipos
        random_idx = np.random.choice(N_samples, self.n_neurons, replace=False)
        self.prototypes = X[random_idx].copy() # W
        
        t = 0
        current_sigma = self.sigma_0
        
        # Calcula vizinhança inicial
        H = self._calculate_neighborhood_matrix(current_sigma)
        
        # Partição Inicial (Step 2 do Initialization)
        # Calcula BMU baseado na dissimilaridade generalizada
        gen_dists, raw_dists_sq = self._calculate_generalized_dissimilarity(X, self.prototypes, H)
        self.labels_ = np.argmin(gen_dists, axis=1)
        
        # Custo Inicial (Step 3 do Initialization)
        current_cost = self.calculate_cost_function(gen_dists, self.labels_)
        
        print(f"Inicialização: Custo J = {current_cost:.2f}, Sigma = {current_sigma:.4f}")
        
        # Loop Principal (Repeat until t_max)
        while t < self.max_iter:
            # Atualiza t e Sigma (Slide 18, linha 5)
            t += 1
            # Decaimento exponencial do raio
            current_sigma = self.sigma_0 * ((self.sigma_f / self.sigma_0) ** (t / self.max_iter))
            
            # Recalcula Vizinhança com novo Sigma
            H = self._calculate_neighborhood_matrix(current_sigma)
            
            # --- STEP 1: ADAPTAÇÃO / COOPERAÇÃO (Slide 18) ---
            # w_rj = Sum(h_f(xk),r * x_kj) / Sum(h_f(xk),r)
            # Precisamos somar a influência de todos os dados 'k' em cada neurônio 'r'.
            # A influência depende de quem venceu para o dado 'k'.
            
            # Cria matriz de Influência G (N_amostras x N_neuronios)
            # G[k, r] = H[vencedor_de_k, r]
            G = H[self.labels_, :] 
            
            # Numerador: X ponderado pela influência (Matricial: G.T @ X)
            numerator = np.dot(G.T, X)
            
            # Denominador: Soma das influências (Soma das colunas de G)
            denominator = np.sum(G, axis=0).reshape(-1, 1)
            
            # Evita divisão por zero (caso raro onde um neurônio e seus vizinhos não capturam nada)
            denominator = np.where(denominator == 0, 1e-10, denominator)
            
            # Atualiza Protótipos
            self.prototypes = numerator / denominator
            
            # --- STEP 2: COMPETIÇÃO (Slide 18) ---
            # Atualiza a partição P (novos vencedores) minimizando dissimilaridade generalizada
            gen_dists, _ = self._calculate_generalized_dissimilarity(X, self.prototypes, H)
            self.labels_ = np.argmin(gen_dists, axis=1)
            
            # --- CÁLCULO DO CUSTO J (Slide 18, linha 8) ---
            new_cost = self.calculate_cost_function(gen_dists, self.labels_)
            
            # Print de progresso
            # if t % 10 == 0 or t == self.max_iter:
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
    
    target_name = target_name if target_name.strip() else None
    perform_normalization = (normalize_choice == 'S')
    
    X, y_true, feature_names = load_and_preprocess_data(data_file_name, headers_file_name, delimiter, target_name, perform_normalization)

    if X is None:
        exit()
        
    try:
        # SOM pede dimensões do mapa, não apenas K
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
    best_prototypes = None
    best_labels = None
    
    for i in range(n_repetitions):
        print(f"\n--- Repetição {i+1}/{n_repetitions} ---")
        
        som = BatchSOM(map_height=map_h, map_width=map_w, max_iter=max_iters, random_state=i)
        prototypes, labels, final_cost = som.fit(X)
        
        print(f"Custo Final J: {final_cost:.2f}")
        
        if final_cost < best_cost:
            best_cost = final_cost
            best_prototypes = prototypes
            best_labels = labels

    # --- Resultados Finais ---
    print("\n\n--- Resultado Final (Melhor Execução Batch SOM) ---")
    print(f"Menor custo J encontrado: {best_cost:.2f}")

    print("\n--- Afiliação dos Neurônios ---")
    for j in range(n_total_neurons):
        cluster_indices = np.where(best_labels == j)[0]
        
        # Converte índice linear j para coordenada (row, col) para visualização
        row, col = divmod(j, map_w)
        
        if len(cluster_indices) > 0:
            # CORREÇÃO AQUI: usar .tolist() em vez de list()
            print(f" Neurônio {j} (Pos {row},{col}) - {len(cluster_indices)} elementos: {cluster_indices.tolist()}")

    # Matriz de Confusão
    print("\n\n--- Matriz de Confusão (Neurônios vs. Classes Verdadeiras) ---")
    if y_true is not None:
        best_cm_df = generate_confusion_matrix_clustering(y_true.values, best_labels, n_total_neurons)
        print(best_cm_df)
        best_cm_df.to_csv("som_matriz_confusao.csv", index=True)
        print("Salvo em 'som_matriz_confusao.csv'")
    else:
        print("Sem targets para gerar matriz de confusão.")

    # Salvando CSVs
    print("\n--- Salvando Resultados ---")
    
    # Protótipos (Pesos finais)
    df_Q = pd.DataFrame(best_prototypes, columns=feature_names)
    df_Q.to_csv("som_prototipos_finais.csv", index_label="neuronio_id")
    
    # Dataset com Labels
    df_results = pd.DataFrame(X, columns=feature_names)
    df_results['neuronio_vencedor'] = best_labels
    if y_true is not None:
        df_results['original_target'] = y_true.values
    df_results.to_csv("som_resultados_finais.csv", index=False)
    
    print("Arquivos salvos com sucesso.")