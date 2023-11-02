open_ai_key = ""
cohere_key = ""
retriever_types = {
    #RetrievalQA': {'chain_type': 'stuff', 'search_type': 'similarity'},
    #'MapReRank': {'chain_type': 'map_rerank', 'search_type': 'similarity'},
    #'MapReduce': {'chain_type': 'map_reduce', 'search_type': 'similarity'},
    'RetrievalQA_MMR': {'chain_type': 'stuff', 'search_type': 'mmr'},
    #'MapReRank_MMR': {'chain_type': 'map_rerank', 'search_type': 'mmr'},
    'MapReduce_MMR': {'chain_type': 'map_reduce', 'search_type': 'mmr'},
}