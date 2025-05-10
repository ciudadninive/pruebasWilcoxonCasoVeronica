import pandas as pd
from scipy.stats import wilcoxon

# Ruta del archivo con las muestras
file_path = 'MuestrasAplicandoFiltrado.xlsx'

# Cargar las muestras de las tres hojas
df_nla = pd.read_excel(file_path, sheet_name='WILCOXON-C2-NLA')
df_nlm = pd.read_excel(file_path, sheet_name='WILCOXON-C2-NLM')
df_nlb = pd.read_excel(file_path, sheet_name='WILCOXON-C2-NLB')

def valor_critico_wilcoxon(n):
    # Tabla resumida de valores críticos para Wilcoxon bilateral, alfa = 0.05
    # Fuente: https://www.socscistatistics.com/tests/wilcoxon/default2.aspx
    tabla = {
        5: 0, 6: 2, 7: 3, 8: 4, 9: 5, 10: 8, 11: 10, 12: 13, 13: 17, 14: 21, 15: 25, 16: 30, 17: 35, 18: 41, 19: 47, 20: 53
    }
    return tabla.get(n, 'N/A')

def prueba_wilcoxon(df, nivel):
    # Elimina filas con valores faltantes
    df = df.dropna(subset=['Puntaje_Pretest', 'Puntaje_Postest'])
    pre = df['Puntaje_Pretest']
    post = df['Puntaje_Postest']
    # Elimina diferencias cero (Wilcoxon requiere esto)
    mask = (post - pre) != 0
    pre_filtrado = pre[mask]
    post_filtrado = post[mask]
    n = len(pre_filtrado)
    if n < 5:
        stat = p_value = v_critico = 'N/A'
        conclusion = 'No se puede calcular (menos de 5 pares válidos)'
    else:
        stat, p_value = wilcoxon(pre_filtrado, post_filtrado, zero_method='wilcox')
        v_critico = valor_critico_wilcoxon(n)
        # Validación de hipótesis
        alpha = 0.05
        if p_value < alpha:
            conclusion = 'Rechazamos la hipótesis nula: hay diferencia significativa pre-post.'
        else:
            conclusion = 'No se rechaza la hipótesis nula: no hay diferencia significativa pre-post.'
    # Reporte
    return {
        'Nivel de Lectura': nivel,
        'N': n,
        'Estadístico Wilcoxon (W)': stat,
        'Valor crítico (tabla)': v_critico,
        'Valor p': p_value,
        'Conclusión': conclusion
    }

# Ejecutar para cada nivel
reporte_nla = prueba_wilcoxon(df_nla, 'Alto')
reporte_nlm = prueba_wilcoxon(df_nlm, 'Medio')
reporte_nlb = prueba_wilcoxon(df_nlb, 'Bajo')

# Crear DataFrame resumen
reporte_df = pd.DataFrame([reporte_nlb, reporte_nlm, reporte_nla])

# Guardar el reporte en un archivo Excel
reporte_df.to_excel('Reporte_Wilcoxon_Resultados.xlsx', index=False)

print(reporte_df)
