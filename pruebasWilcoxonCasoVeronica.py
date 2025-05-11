import pandas as pd
from scipy.stats import wilcoxon

# Tabla de valores críticos para Wilcoxon (α=0.05, dos colas)
valores_criticos = {
    5: 0, 6: 2, 7: 3, 8: 3, 9: 5, 10: 8,
    11: 10, 12: 13, 13: 17, 14: 21, 15: 25,
    16: 30, 17: 35, 18: 41, 19: 47, 20: 53,
    21: 60, 22: 67, 23: 75, 24: 83, 25: 91
}

archivo = 'MuestrasAplicandoFiltrado.xlsx'
hojas = {
    'WILCOXON-C2-NLA': 'Nivel Lectura Alto',
    'WILCOXON-C2-NLM': 'Nivel Lectura Medio',
    'WILCOXON-C2-NLB': 'Nivel Lectura Bajo'
}

resultados = []

for hoja, nombre in hojas.items():
    datos = pd.read_excel(archivo, sheet_name=hoja)
    datos = datos.dropna(subset=['Puntaje_Pretest', 'Puntaje_Postest'])
    pre = datos['Puntaje_Pretest']
    post = datos['Puntaje_Postest']

    # Calcular diferencias y contar pares no nulos (diferencias != 0)
    diferencias = pre - post
    n_pares = sum(diferencias != 0)

    # Si hay menos de 5 pares no nulos, no hay tabla disponible
    if n_pares < 5:
        w, p = wilcoxon(pre, post, zero_method='wilcox', correction=True)
        valor_critico = 'No disponible (pares < 5)'
        conclusion = "Muestra insuficiente para valor crítico en tablas"
    else:
        # Seleccionar solo pares no nulos para la prueba (opcional, scipy maneja bien)
        # Aquí scipy wilcoxon ignora ceros automáticamente si zero_method='wilcox'
        w, p = wilcoxon(pre, post, zero_method='wilcox')
        valor_critico = valores_criticos.get(n_pares, 'No disponible')
        if valor_critico == 'No disponible':
            conclusion = "Valor crítico no disponible para este tamaño muestral"
        else:
            if p < 0.05 and w <= valor_critico:
                conclusion = "Diferencia significativa (p < 0.05 y W ≤ valor crítico)"
            else:
                conclusion = "Sin diferencia significativa"

    resultados.append({
        'Grupo': nombre,
        'Muestra total (n)': len(datos),
        'Pares no nulos (n)': n_pares,
        'Estadístico W': w,
        'Valor crítico (α=0.05, 2 colas)': valor_critico,
        'Valor p': round(p, 5),
        'Conclusión': conclusion
    })

df_resultados = pd.DataFrame(resultados)

# Guardar reporte en Excel
df_resultados.to_excel('Reporte_Wilcoxon_Afinado.xlsx', index=False)

print("Reporte generado: Reporte_Wilcoxon_Afinado.xlsx\n")
print(df_resultados)
