import re
import matplotlib.pyplot as plt

ARQUIVO = "aux3.txt"          # caminho do arquivo de dados
SAIDA = "boxplots_acertividade.png"


def ler_dados(caminho):
    """Lê o arquivo e devolve uma lista de (valores, max_possivel) por linha."""
    conjuntos = []
    with open(caminho, encoding="utf-8") as f:
        next(f)  
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            lista = re.search(r"\[(.*?)\]", linha).group(1)
            valores = [float(v) for v in lista.split(",")]
            resto = linha.split("]")[1].strip(", ").split(",")
            max_possivel = float(resto[-1])
            conjuntos.append((valores, max_possivel))
    return conjuntos


def main():
    conjuntos = ler_dados(ARQUIVO)

    # Taxa de acertividade = valor / máximo possível
    taxas = [[v / maximo for v in valores] for valores, maximo in conjuntos]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)

    for i, (ax, dados) in enumerate(zip(axes, taxas), start=1):
        ax.boxplot(dados, showmeans=False)
        ax.set_title(f"Dado {i}")
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Taxa de acertividade")
    fig.suptitle("Box plots da taxa de acertividade (valor / máximo possível)")
    plt.tight_layout()
    plt.savefig(SAIDA, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()