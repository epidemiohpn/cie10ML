# ==================================================================
# SCRIPT PARA APLICAR MODELO CIE10 A NUEVOS DATOS SEMANALES
# ==================================================================

# --- 1. Cargar Librerías Esenciales ---
# Estas son las librerías necesarias para que el script de predicción funcione.
library(readr)
library(dplyr)
library(stringr)
library(tidytext)
library(tidyr)
library(tm)
library(e1071)


# --- 2. Cargar los Activos del Modelo Entrenado ---
# El script asume que estos archivos están en el mismo directorio de trabajo.
# Si no lo están, ajusta la ruta (ej. "C:/Modelos/modelo_svm_cie10.rds").
print("Cargando modelo y diccionario...")
modelo <- readRDS("modelo_svm_cie10.rds")
diccionario <- readRDS("diccionario_terminos_entrenamiento.rds")
print("¡Carga completa!")


# --- 3. Definir la Función Principal de Procesamiento y Predicción ---
# Esta función toma la ruta de un CSV y devuelve un dataframe con la nueva columna de predicción.
procesar_y_predecir_csv <- function(ruta_csv) {
  
  # --- Carga y Limpieza Inicial ---
  print(paste("Procesando archivo:", ruta_csv))
  df_semanal <- read_csv2(ruta_csv, 
                          locale = locale(encoding = "windows-1252"), 
                          col_types = cols(.default = "c"))
  
  # Imputar NAs en 'Diag. Definitivo' (igual que en el entrenamiento)
  if ("Diag. Presuntivo" %in% names(df_semanal)) {
    df_semanal <- df_semanal %>%
      mutate(`Diag. Definitivo` = coalesce(`Diag. Definitivo`, `Diag. Presuntivo`))
  }
  
  # Añadir un ID temporal para poder unir los resultados al final
  df_semanal <- df_semanal %>% mutate(temp_id_prediccion = row_number())
  
  # Filtrar solo las filas que tienen un diagnóstico para procesar
  df_a_predecir <- df_semanal %>%
    filter(!is.na(`Diag. Definitivo`) & `Diag. Definitivo` != "") %>%
    select(temp_id_prediccion, `Diag. Definitivo`)
  
  # Si no hay nada que predecir, devolver el dataframe original con una columna NA
  if (nrow(df_a_predecir) == 0) {
    print("No se encontraron diagnósticos para predecir en el archivo.")
    df_semanal$CIE10_ML <- NA
    df_semanal$temp_id_prediccion <- NULL
    return(df_semanal)
  }
  
  # --- Pipeline de PLN (idéntico al del entrenamiento) ---
  stopwords_es <- tibble(word = stopwords("spanish"))
  limpiar_texto <- function(texto) {
    texto <- tolower(texto)
    texto <- removePunctuation(texto)
    texto <- removeNumbers(texto)
    texto <- stripWhitespace(texto)
    return(texto)
  }
  
  df_a_predecir$Diag_Limpio <- sapply(df_a_predecir$`Diag. Definitivo`, limpiar_texto)
  
  unigrams <- df_a_predecir %>% unnest_tokens(word, Diag_Limpio) %>% filter(!word %in% stopwords_es$word)
  bigrams <- df_a_predecir %>%
    unnest_tokens(bigram, Diag_Limpio, token = "ngrams", n = 2) %>%
    filter(!is.na(bigram)) %>%
    separate(bigram, c("word1", "word2"), sep = " ") %>%
    filter(!word1 %in% stopwords_es$word, !word2 %in% stopwords_es$word) %>%
    unite(bigram, word1, word2, sep = " ")
  
  df_procesado <- bind_rows(unigrams %>% rename(token = word), bigrams %>% rename(token = bigram)) %>%
    group_by(temp_id_prediccion) %>%
    summarise(Diag_Final = paste(token, collapse = " "), .groups = "drop")
  
  # --- Creación de DTM y Predicción ---
  corpus_nuevo <- VCorpus(VectorSource(df_procesado$Diag_Final))
  dtm_nuevo <- DocumentTermMatrix(corpus_nuevo, control = list(dictionary = diccionario))
  matriz_nueva <- as.data.frame(as.matrix(dtm_nuevo))
  
  predicciones <- predict(modelo, newdata = matriz_nueva)
  
  df_resultados <- tibble(
    temp_id_prediccion = df_procesado$temp_id_prediccion,
    CIE10_ML = as.character(predicciones)
  )
  
  # --- Unir Predicciones al Dataframe Original ---
  df_final <- df_semanal %>%
    left_join(df_resultados, by = "temp_id_prediccion")
  
  # Limpiar la columna de ID temporal
  df_final$temp_id_prediccion <- NULL
  
  print(paste("¡Procesamiento completo! Se predijeron", nrow(df_resultados), "diagnósticos."))
  return(df_final)
}


# ==========================================================
# CÓMO USAR EL SCRIPT CADA SEMANA
# ==========================================================

# 1. DEFINE LA RUTA A TU NUEVO ARCHIVO CSV
#    Cambia esta ruta cada lunes por la del nuevo archivo que descargaste.
ruta_nuevo_csv <- "C:/Ruta/A/Tu/Carpeta/semana_actual.csv" # <-- ¡CAMBIA ESTO!

# 2. EJECUTA LA FUNCIÓN
#    Esta es la única línea que necesitas correr para hacer todo el trabajo.
df_semanal_con_predicciones <- procesar_y_predecir_csv(ruta_nuevo_csv)

# 3. INSPECCIONA LOS RESULTADOS (Opcional)
#    Puedes ver las primeras filas para asegurarte de que todo se vea bien.
# head(df_semanal_con_predicciones %>% select(`Diag. Definitivo`, `Código CIE10`, CIE10_ML))
# View(df_semanal_con_predicciones)

# 4. GUARDA EL NUEVO ARCHIVO CON LAS PREDICCIONES
#    Esto creará un nuevo CSV con el sufijo "_con_predicciones".
ruta_salida_csv <- str_replace(ruta_nuevo_csv, "\\.csv$", "_con_predicciones.csv")
write.csv2(df_semanal_con_predicciones, file = ruta_salida_csv, row.names = FALSE)

print(paste("Resultados guardados en:", ruta_salida_csv))