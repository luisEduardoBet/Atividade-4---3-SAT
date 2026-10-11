import numpy as np
import matplotlib.pyplot as plt
import random
import math
from time import sleep
import os

def read_cnf(filepath):
    clauses = []
    num_vars = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('p'):
                parts = line.strip().split()
                num_vars = int(parts[2])
            else:
                clause = [int(x) for x in line.strip().split() if int(x) != 0]
                if clause:
                    clauses.append(clause)
    return num_vars, clauses

def evaluate(solution, clauses):
    satisfied = 0
    for clause in clauses:
        for literal in clause:
            var_idx = abs(literal) - 1
            val = solution[var_idx]
            if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                satisfied += 1
                break
    return satisfied

def get_neighbor(solution):
    neighbor = solution.copy()
    num_vars = len(solution)
    num_flips = max(1, int(num_vars * 0.05)) 
    
    indices_to_flip = random.sample(range(num_vars), num_flips)
    for idx in indices_to_flip:
        neighbor[idx] = 1 - neighbor[idx]
    return neighbor


def cooling_schedule_1(T0, TN, N, i):
    if T0 <= 0: return TN
    return T0 * ((TN / T0) ** (i / N))

def cooling_schedule_7(T0, TN, N, i):
    try:
        return ((T0 - TN) / math.cosh(10 * i / N)) + TN
    except OverflowError:
        return TN

# Simulated Annealing (SA)
def simulated_annealing(num_vars, clauses, max_evals, schedule_func, SAmax, T0=100.0, TN=0.0001):
    
    current_solution = [random.randint(0, 1) for _ in range(num_vars)]
    current_eval = evaluate(current_solution, clauses)

    
    best_solution = current_solution.copy() # guarda a mehlro solução absoluta
    best_eval = current_eval
    
    # Passo 2: Cálculo dos degraus de temperatura
    N_steps = max_evals // SAmax # Controla a queda de temperatura
    if N_steps == 0: N_steps = 1

    evals = 0 # Conta quantas vezes testamos a função objetivo
    history = []
    temp =  []
    i = 0 

    # Passo 3: Resfriamento -> O algoritmo vai esfriando o sistema passo a passo
    while (evals < max_evals):

        T = schedule_func(T0, TN, N_steps, i)
        if T < TN: T = TN
        i+=1

       
        for _ in range(SAmax):  

            temp.append(T)
                        
            if evals >= max_evals:
                break
                
            neighbor = get_neighbor(current_solution)
            neighbor_eval = evaluate(neighbor, clauses)
            evals += 1
            
            delta = neighbor_eval - current_eval
            
            if delta > 0:
                current_solution = neighbor
                current_eval = neighbor_eval
                # Atualiza a melhor global encontrada
                if neighbor_eval > best_eval:
                    best_solution = neighbor.copy()
                    best_eval = neighbor_eval
            else:
                try:
                    prob = math.exp(delta / T)
                except OverflowError:
                    prob = 0

                if random.random() < prob:
                    current_solution = neighbor
                    current_eval = neighbor_eval
        
            history.append(current_eval)
          
            
    return best_solution, best_eval, history, temp

def run_experiments(filepath, max_evals):
    print(f"\n{'='*60}")
    print(f"Iniciando testes para: {filepath} | Iterações: {max_evals}")
    print(f"{'='*60}")
    
    num_vars, clauses = read_cnf(filepath)
    max_possible = len(clauses)
    
    configs = [
        {"nome": "Schedule 1 (SAmax=1)", "func": cooling_schedule_1, "SAmax": 1},
        {"nome": "Schedule 1 (SAmax=100)", "func": cooling_schedule_1, "SAmax": 100},
        {"nome": "Schedule 7 (SAmax=1)", "func": cooling_schedule_7, "SAmax": 1},
        {"nome": "Schedule 7 (SAmax=100)", "func": cooling_schedule_7, "SAmax": 100}
    ]
    
    T0 = 1
    TN = 0.2
    
    plt.figure(figsize=(16, 12))
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuração: {config['nome']} ---")
        best_results = []
        best_history_sample = []
        best_temp_sample = []
        
        melhor_vetor_config = []
        melhor_fo_config = -1
        
        for run in range(10):
            sol, fo, hist, temp= simulated_annealing(
                num_vars, clauses, max_evals, config["func"], config["SAmax"], T0, TN
            )
            best_results.append(fo)
            
            if run == 0:
                best_history_sample = hist
                best_temp_sample = temp
                
            if fo > melhor_fo_config:
                melhor_fo_config = fo
                melhor_vetor_config = sol


        
        media = np.mean(best_results)
        dp = np.std(best_results)
        print(f"Resultados (10 execuções): {best_results}")
        print(f"Média: {media:.2f} | Desvio-Padrão: {dp:.2f} | Máximo Possível: {max_possible}")
        
        print(f"\n>>> MELHOR SOLUÇÃO ENCONTRADA: {melhor_fo_config} / {max_possible} cláusulas")
        vetor_str = "".join(str(bit) for bit in melhor_vetor_config)
        print(f"Vetor Final Binário:\n{vetor_str}\n")
        
        ax1 = plt.subplot(2, 2, i+1)
        ax1.plot(best_history_sample, color='blue', alpha=0.7, label='FO (cláusulas satisfeitas)')
        ax1.set_title(f"{config['nome']}\nResultado ilustrado: {best_history_sample[-1]}/{max_possible} cláusulas")
        ax1.set_xlabel("Iterações")
        ax1.set_ylabel("Cláusulas Satisfeitas (FO)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        # Gráfico da temperatura em outra cor (vermelho), em um eixo Y secundário

        ax2 = ax1.twinx()
        n_pts = min(len(best_temp_sample), len(best_history_sample))
        ax2.plot(best_temp_sample[:n_pts], color='red', linewidth=1.5, label='Temperatura')
        ax2.set_ylabel("Temperatura", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        
    plt.tight_layout()
    plt.savefig(f"convergencia_{os.path.basename(filepath)}.png")
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"Processamento concluído. Gráfico de convergência salvo com sucesso!")
    
if __name__ == "__main__":
    instancias = [
        # ("uf20-01.cnf", 1500),         
        # ("uf100-01.cnf", 500000) 
        ("uf250-01.cnf", 5000000)    
    ]
    
    for arquivo, max_evals in instancias:
        if os.path.exists(arquivo):
            run_experiments(arquivo, max_evals)
        else:
            print(f"Arquivo não encontrado: {arquivo}. Certifique-se de estar na mesma pasta.")