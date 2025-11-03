# Librerias
library(readr)
library(dplyr)
library(stringr)
library(tidytext)
library(tidyr)
library(tm)
library(e1071)


#Carga modelo
print("Cargando modelo y diccionario...")
modelo <- readRDS("modelo_svm_cie10.rds")
diccionario <- readRDS("diccionario_terminos_entrenamiento.rds")
print("¡Carga completa!")


#Proceso
procesar_y_predecir_csv <- function(ruta_csv) {
  
  
  print(paste("Procesando archivo:", ruta_csv))
  df_semanal <- read_csv2(ruta_csv, 
                          locale = locale(encoding = "windows-1252"), 
                          col_types = cols(.default = "c"))
  
  
  if ("Diag. Presuntivo" %in% names(df_semanal)) {
    df_semanal <- df_semanal %>%
      mutate(`Diag. Definitivo` = coalesce(`Diag. Definitivo`, `Diag. Presuntivo`))
  }
  
  
  df_semanal <- df_semanal %>% mutate(temp_id_prediccion = row_number())
  
  
  df_a_predecir <- df_semanal %>%
    filter(!is.na(`Diag. Definitivo`) & `Diag. Definitivo` != "") %>%
    select(temp_id_prediccion, `Diag. Definitivo`)
  
  
  if (nrow(df_a_predecir) == 0) {
    print("No se encontraron diagnósticos para predecir en el archivo.")
    df_semanal$CIE10_ML <- NA
    df_semanal$temp_id_prediccion <- NULL
    return(df_semanal)
  }
  
  
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
  
  
  corpus_nuevo <- VCorpus(VectorSource(df_procesado$Diag_Final))
  dtm_nuevo <- DocumentTermMatrix(corpus_nuevo, control = list(dictionary = diccionario))
  matriz_nueva <- as.data.frame(as.matrix(dtm_nuevo))
  
  predicciones <- predict(modelo, newdata = matriz_nueva)
  
  df_resultados <- tibble(
    temp_id_prediccion = df_procesado$temp_id_prediccion,
    CIE10_ML = as.character(predicciones)
  )
  
  
  df_final <- df_semanal %>%
    left_join(df_resultados, by = "temp_id_prediccion")
  
  # Limpiar la columna de ID temporal
  df_final$temp_id_prediccion <- NULL
  
  print(paste("¡Procesamiento completo! Se predijeron", nrow(df_resultados), "diagnósticos."))
  return(df_final)
}



# Usar script



ruta_nuevo_csv <- "C:/Users/usuario/Desktop/test/guardia_adultos_2025_s43.csv" # ruta del csv crudo


df_semanal_con_predicciones <- procesar_y_predecir_csv(ruta_nuevo_csv)


ruta_salida_csv <- str_replace(ruta_nuevo_csv, "\\.csv$", "_con_predicciones.csv")
write.csv2(df_semanal_con_predicciones, file = ruta_salida_csv, row.names = FALSE)

print(paste("Resultados guardados en:", ruta_salida_csv))


# debug z51



data_analisis <- data %>%
  mutate(clasificacion = substr(`Código CIE10`, 1, 3))


diagnosticos_z51 <- data_analisis %>%
  filter(clasificacion == "Z51") %>%
  select(`Diag. Definitivo`)


print("Diagnósticos más comunes para el código Z51:")
View(as.data.frame(head(sort(table(diagnosticos_z51$`Diag. Definitivo`), decreasing = TRUE), 50)))
