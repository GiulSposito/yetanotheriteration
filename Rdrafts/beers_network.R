# libs
library(tidyverse) # pipe, maps and tibble
library(lubridate) # manipulacao de datas
library(corrplot)  # correlation plot
library(ape)       # disk dendogram
library(RColorBrewer) # color palette

# bibliotecas
library(tidytext) # pacote para tratamento de textos do tidyverse 
library(ptstem)   # pacote que faz o steming de termos em português

# carrega base de avaliacoes de cerveja (gerada no post anterior)
beers <- readRDS("./content/post/data/beer_tm/beers.rds") %>%
  # adiciona um id para cerveja
  mutate( beer.id = 1:nrow(.))

# vamos selecionar os tipos de cerveja com mais de uma avaliacao
# so para diminuir o número de cervejas
selected.types <- beers %>%
  group_by(tipo) %>% 
  tally() %>% 
  filter(n>1)

print(nrow(selected.types))


# carrega stop words (do git https://gist.github.com/alopes/5358189)
pt_stopwords <- read_table("./content/post/data/beer_tm/stopwords.txt", col_names = "word")

# algumas stop words relevantes para esse problema (toque, algo, gosto, paladar)
# sao palavras que presentes nas descrições do sabor mas não agrega informação
my_stopwords <- read_table("./content/post/data/beer_tm/my_stopwords.txt", col_names = "word")

# combina as stop words
stopwords <- bind_rows(pt_stopwords, my_stopwords)

# removendo stop-words, stem e contando
beers %>%
  # filtrando por tipo
  filter(tipo %in% selected.types$tipo) %>%
  # Concatena os campos de texto Malte, Sabor e Cor
  mutate( review = paste0(malte, " ", sabor, " ", cor) ) %>%
  # Seleciona campoas de interesse para a analise
  select( beer.id, review ) %>%
  # separa as palavras do texto em vários registros
  unnest_tokens( word, review ) %>% 
  # remove as stopwords
  anti_join( stopwords ) %>% 
  # stem das palavras
  mutate( word = ptstem(word) ) %>%
  # contagem das palavars
  count( word, beer.id ) -> raw_beer_words

# removendo palavras que so aparecem
# em uma unica cerveja
selected.words <- raw_beer_words %>%
  group_by(word) %>%
  tally(wt = NULL) %>%
  filter( nn > 1) %>%
  select(word)

beer_words <- raw_beer_words %>%
  inner_join(selected.words)

# cria um dicionario
words <- beer_words %>%
  # para cada palavra
  select( word ) %>%
  distinct() %>%
  arrange( word ) %>%
  # cria um ID
  mutate( word.id = 1:nrow(.) )

# adiciona o id da palavra a relacao beer/word
beer_words <- inner_join(beer_words, words)

# constroi a lista de arestas
# para cada palavra
edge_list <- map_df(words$word.id, function(wid,bw){
  # pega as cervejas que compartilham a palavra
  nodes <- bw %>%
    filter(word.id==wid) %>%
    select( beer.id ) %>%
    distinct() %>%
    arrange( beer.id )
  
  # e cria arestas entre as cervejas
  edges <- nodes$beer.id %>%
    combn(2) %>%
    t() %>%
    as.tibble() %>%
    setNames(c("from","to")) %>%
    mutate(word.id=wid) %>%
    return()

}, bw=beers_words, .id="word.id")

str(edge_list)
# e as liga
