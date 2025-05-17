import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from collections import Counter

# Tabla de valores críticos de Wilcoxon bilateral para alfa=0.05 (dos colas) para n=5 a 30
valores_criticos = {
    5: 0, 6: 2, 7: 3, 8: 5, 9: 8, 10: 10,
    11: 13, 12: 17, 13: 21, 14: 25, 15: 30,
    16: 35, 17: 41, 18: 47, 19: 54, 20: 60,
    21: 66, 22: 73, 23: 79, 24: 86, 25: 93,
    26: 101, 27: 108, 28: 116, 29: 124, 30: 132
}

def obtener_valor_critico(n):
    return valores_criticos.get(n, None)

def distribucion_wilcoxon(n):
    """Distribución exacta de W para n pares."""
    distrib = Counter({0: 1})
    for i in range(1, n + 1):
        nueva = Counter()
        for suma, count in distrib.items():
            nueva[suma + i] += count
            nueva[suma] += count
        distrib = nueva
    total = 2 ** n
    return {w: c / total for w, c in distrib.items()}

def graficar_distribucion_wilcoxon_dos_colas(stat, p_valor, valor_critico, n, nombre, alfa=0.05, decision=""):
    """Gráfica SVG de la distribución exacta de Wilcoxon para dos colas, con leyenda de hipótesis."""
    dist = distribucion_wilcoxon(n)
    xs = np.array(sorted(dist.keys()))
    ps = np.array([dist[x] for x in xs])
    Wmax = n * (n + 1) // 2

    # Zonas de rechazo y aceptación
    zona_rechazo_izq = xs <= valor_critico
    zona_rechazo_der = xs >= (Wmax - valor_critico)
    zona_aceptacion = ~(zona_rechazo_izq | zona_rechazo_der)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(xs[zona_aceptacion], ps[zona_aceptacion], color='lightgreen', label='Zona de aceptación')
    ax.bar(xs[zona_rechazo_izq], ps[zona_rechazo_izq], color='salmon', label=f'Rechazo (W ≤ {valor_critico})')
    ax.bar(xs[zona_rechazo_der], ps[zona_rechazo_der], color='salmon', label=f'Rechazo (W ≥ {Wmax - valor_critico})')

    # Línea del estadístico observado
    ax.axvline(stat, color='blue', linestyle='--', linewidth=2)

    # Etiqueta del estadístico observado: SIEMPRE debajo del eje X, centrada
    ylim = ax.get_ylim()
    y_min = ylim[0]
    y_offset = (ylim[1] - ylim[0]) * 0.06
    ax.text(stat, y_min - y_offset, f'W obs = {stat:.0f}',
            color='blue', fontsize=12, ha='center', va='top',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='blue'))

    # Valores críticos
    ax.axvline(valor_critico, color='red', linestyle='-.', linewidth=2)
    ax.text(valor_critico, ylim[1]*0.7, f'Crítico izq\n{valor_critico}', color='red', rotation=90, va='top', ha='center',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))
    ax.axvline(Wmax - valor_critico, color='red', linestyle='-.', linewidth=2)
    ax.text(Wmax - valor_critico, ylim[1]*0.7, f'Crítico der\n{Wmax - valor_critico}', color='red', rotation=90, va='top', ha='center',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    # Anotaciones separadas
    ax.text(0.01, 0.97, f'n = {n}', transform=ax.transAxes, ha='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    ax.text(0.01, 0.87, f'α = {alfa:.2f}\nDos colas', transform=ax.transAxes, ha='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    ax.text(0.99, 0.97, f'Valor p = {p_valor:.4f}', transform=ax.transAxes, ha='right', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Leyenda de hipótesis nula (esquina inferior derecha)
    ax.text(0.99, 0.05, f'Validación de H₀:\n{decision}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=13,
            bbox=dict(facecolor='lightyellow', alpha=0.95, edgecolor='orange', boxstyle='round,pad=0.5'))

    ax.set_title(f'Distribución exacta de Wilcoxon para {nombre}\nZonas de rechazo (dos colas)', fontsize=15)
    ax.set_xlabel('Estadístico W')
    ax.set_ylabel('Probabilidad')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f'wilcoxon_{nombre}_2colas.svg', format='svg')
    plt.close()


def analizar_muestra(df, nombre_muestra, alfa=0.05):
    # Seleccionar columnas relevantes y eliminar filas con NaN
    df_clean = df[['Puntaje_Pretest', 'Puntaje_Postest']].dropna().copy()
    # Calcular diferencias y eliminar empates (diferencia = 0)
    df_clean['Diferencia'] = df_clean['Puntaje_Postest'] - df_clean['Puntaje_Pretest']
    df_validos = df_clean[df_clean['Diferencia'] != 0]
    n = len(df_validos)
    if n < 5:
        return {
            "Muestra": nombre_muestra,
            "N pares válidos": n,
            "Mensaje": "Tamaño de muestra insuficiente para Wilcoxon (n < 5, excluyendo empates)"
        }
    # Aplicar prueba de Wilcoxon
    stat, p_valor = wilcoxon(df_validos['Puntaje_Pretest'], df_validos['Puntaje_Postest'], zero_method="wilcox", alternative='two-sided')
    valor_critico = obtener_valor_critico(n)
    Wmax = n * (n + 1) // 2
    if valor_critico is not None:
        # Generar SVG
        graficar_distribucion_wilcoxon_dos_colas(stat, p_valor, valor_critico, n, nombre_muestra, alfa)
        # Decisión dos colas
        decision = "Rechazar H0" if (stat <= valor_critico or stat >= (Wmax - valor_critico)) else "No rechazar H0"
        interpretacion = (
            f"Con un estadístico W = {stat:.2f} y p = {p_valor:.4f}, "
            f"{'se rechaza' if (stat <= valor_critico or stat >= (Wmax - valor_critico)) else 'no se rechaza'} la hipótesis nula "
            f"a un nivel de significancia de {alfa} (dos colas, valores críticos = {valor_critico} y {Wmax - valor_critico}). "
            f"{'Hay diferencias significativas.' if (stat <= valor_critico or stat >= (Wmax - valor_critico)) else 'No hay diferencias significativas.'}"
        )
    else:
        decision = "No implementado para n > 30"
        interpretacion = "Para n > 30, usar aproximación normal."
    return {
        "Muestra": nombre_muestra,
        "N pares válidos": n,
        "Estadístico W": stat,
        "Valor p": p_valor,
        "Valor crítico (α=0.05)": valor_critico,
        "Decisión": decision,
        "Interpretación": interpretacion
    }

def main(ruta_archivo):
    muestras = ['WILCOXON-C2-NLA', 'WILCOXON-C2-NLM', 'WILCOXON-C2-NLB']
    resultados = []
    for muestra in muestras:
        try:
            df = pd.read_excel(ruta_archivo, sheet_name=muestra)
            resultado = analizar_muestra(df, muestra)
            resultados.append(resultado)
        except Exception as e:
            resultados.append({
                "Muestra": muestra,
                "Mensaje": f"Error al procesar la muestra: {e}"
            })
    reporte = pd.DataFrame(resultados)
    print("\nREPORTE DE PRUEBA DE WILCOXON")
    print(reporte.to_string(index=False))
    reporte.to_excel("reporte_wilcoxon_mejorado.xlsx", index=False)
    print("\nReporte guardado en 'reporte_wilcoxon_mejorado.xlsx'.")
    print("SVGs de las pruebas generados en el mismo directorio.")

if __name__ == "__main__":
    main("MuestrasAplicandoFiltrado.xlsx")
