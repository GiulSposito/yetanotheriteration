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
beers <- readRDS("./content/post/data/beer_tm/beers.rds")

# define um "beer type"
beer.types <- beers %>%
  select( super.tipo, sub.tipo, tipo ) %>%
  distinct() %>%
  mutate(
    sub.tipo = case_when( is.na(sub.tipo) ~ super.tipo,
                          T ~sub.tipo )
  ) %>%
  arrange( super.tipo, sub.tipo ) %>%
  mutate( type.id = 1:nrow(.),
          super.tipo = as.factor(super.tipo),
          sub.tipo   = as.factor(sub.tipo) )

# atribui um "type.id" as cerverjas
beers <- beers %>%
  select(-super.tipo, -sub.tipo) %>%
  inner_join(beer.types[,], "tipo")
          
# vamos selecionar os tipos de cerveja com mais de uma avaliacao
# so para diminuir o número de cervejas
selected.types <- beers %>%
  group_by(type.id) %>% 
  tally() %>% 
  filter(n>1)

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
  filter(type.id %in% selected.types$type.id) %>%
  # Concatena os campos de texto Malte, Sabor e Cor
  mutate( review = paste0(malte, " ", sabor, " ", cor) ) %>%
  # Seleciona campoas de interesse para a analise
  select( type.id, review ) %>%
  # separa as palavras do texto em vários registros
  unnest_tokens( word, review ) %>% 
  # remove as stopwords
  anti_join( stopwords ) %>% 
  # stem das palavras
  mutate( word = ptstem(word) ) %>%
  # contagem das palavars
  count( word, type.id ) -> raw_type_words


# removendo palavras que so aparecem
# em uma unica cerveja
selected.words <- raw_type_words %>%
  group_by(word) %>%
  tally(wt = NULL) %>%
  filter( nn > 1) %>%
  select(word)

type_words <- raw_type_words %>%
  inner_join(selected.words)

# cria um dicionario
words <- type_words %>%
  # para cada palavra
  select( word ) %>%
  distinct() %>%
  arrange( word ) %>%
  # cria um ID
  mutate( word.id = 1:nrow(.) )

# adiciona o id da palavra a relacao beer/word
type_words <- inner_join(type_words, words)

# constroi a lista de arestas
# para cada palavra
raw_edge_list <- map_df(words$word.id, function(wid,bw){
  # pega as cervejas que compartilham a palavra
  nodes <- bw %>%
    filter(word.id==wid) %>%
    select( type.id ) %>%
    distinct() %>%
    arrange( type.id )
  
  # e cria arestas entre as cervejas
  # combinando 2 a dois cervejas que comparilham
  # a mesma palavra
  edges <- nodes$type.id %>%
    combn(2) %>%
    t() %>%
    as.tibble() %>%
    setNames(c("from","to")) %>%
    return()

}, bw=type_words, .id="word.id")

# overview
glimpse(raw_edge_list)

# constuindo graphs
library(igraph)    # estrutura de dados
library(tidygraph) # manipulações de redes
library(ggraph)    # visualizacoes de redes

# rede de semelhanca
edges_list <- raw_edge_list %>%
  group_by(from,to) %>%
  tally()

nids <- c(edges_list$from, edges_list$to) %>%
  unique() %>%
  sort()
 

beer.net <- tbl_graph(nodes = beer.types, edges = edges_list, directed = TRUE) %>%
  mutate( component = as.factor(group_components()))

beer.net %>%
  filter (component == names(table(component))[which.max(table(component))]) %>%
  ggraph() +
  geom_node_point(aes(color=super.tipo)) +
  geom_edge_fan(aes(alpha=n)) +
  theme_void()


# Lista de arestas, os números são identificadores dos nós
g_edgelist <- data.frame(
  from = c( 1,1,2,2,3,3,5,5),
  to   = c( 2,3,3,5,4,5,6,7)
) 

# construindo a rede a partir da lista de arestas
g <- g_edgelist %>%
  as.matrix() %>%
  graph.edgelist(directed = FALSE) %>% # usando igraph 
  as_tbl_graph() %>%
  activate("nodes") %>%
  mutate( name = LETTERS[1:7] ) # nomeando os nós

# calculando previamente as métricas
g <- g %>% 
  mutate( degree = centrality_degree(), 
          btwn   = round( centrality_betweenness(),2 ),
          clsn   = round( centrality_closeness(normalized = T),2 ),
          eign   = round( centrality_eigen(scale = F), 2 ))

# fixando o layout previamente para todos os plots terem a mesma disposicao
g_layout <- create_layout(g, layout = "kk")

# plotando o graph
ggraph(g_layout) +
  geom_edge_fan(color="black") +
  geom_node_point(color="blue",alpha=0.8, size=8) +
  geom_node_text(aes(label=name), color="white") +
  theme_void() +
  ggtitle( "Exemplo de Rede" ) + 
  theme( legend.position = "none" )
