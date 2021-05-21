
import networkx as nx
import csv
import pprint as pp

# graph creation function
def graph_create_from_tsv(tsvFilePath):
    # reading tsv file
    input_file_handler = open(tsvFilePath, 'r', encoding="utf-8")
    tsv_reader = csv.reader(input_file_handler, delimiter='\t')
    tsv_reader.__next__()
    list_adj = []
    # inserting adjacent pokemon couples in a list
    for pokemon in tsv_reader:
        pokemon1 = pokemon[0]
        pokemon2 = pokemon[1]
        list_adj.append((pokemon1, pokemon2))
    input_file_handler.close()

    #creating graph
    graph = nx.Graph()
    graph.add_edges_from(list_adj)
    return graph

# top k computation definition
def compute_top_k(map__node_id__score, remove_pkmn, k=20):
    # var remove_pkmn contains the pokemons in the topic, which we don't want in the list__node_id__score
    list__node_id__score = [(node_id, score) for node_id, score in map__node_id__score.items() if node_id not in remove_pkmn]
    list__node_id__score.sort(key=lambda x: (-x[1], x[0]))
    return list__node_id__score[:k]

# creation of the teleporting probability distribution for the selected topic
def prob_topic(graph, poke_topic):
    num_nodes_topic = len(poke_topic)
    # creating dictionary with keys the nodes of the graph, and values set to 0
    d_topic_nodes = dict.fromkeys(list(graph.nodes), 0)
    for pokemon in poke_topic:
        # setting value for the pokemon in the topic
        d_topic_nodes[pokemon] = 1/num_nodes_topic
    return d_topic_nodes


# starting code
# first given sets

graph = graph_create_from_tsv('dataset/pkmn_graph_data.tsv')

team_size = 6

# given sets
Set_A = set(["Pikachu"])
Set_B = set(["Venusaur", "Charizard", "Blastoise"])
Set_C = set(["Excadrill", "Dracovish", "Whimsicott", "Milotic"])

sets = {'Set_A': Set_A,
        'Set_B': Set_B,
        'Set_C': Set_C
}
teams = {}

# Computation of the PageRank vector for given sets
for key, val in sets.items():
    topic_pers = prob_topic(graph, val)
    map__node_id__node_pagerank_value = nx.pagerank(graph, alpha=0.33,
                                                    personalization=topic_pers)
    top_k__node_id__node_pagerank_value = compute_top_k(map__node_id__node_pagerank_value,
                                                        remove_pkmn=val,
                                                        k=team_size - len(val))
    # pp.pprint(top_k__node_id__node_pagerank_value)
    # top_k__node_id__node_pagerank_value contains the pokemon of the compute_top_k without the pokemons in the topic
    teams['team_' + key[-1]] = set([i[0] for i in top_k__node_id__node_pagerank_value] + list(val))

# Printing sets of Pokemons with respective input sets from A to C
print('Sets of Pokemons with Set_A')
print(teams['team_A'])
print('\n###########################\n')
print('Sets of Pokemons with Set_B')
print(teams['team_B'])
print('\n###########################\n')
print('Sets of Pokemons with Set_C')
print(teams['team_C'])
print('\n###########################')
print('###########################\n')

# Same code, but with different given sets than before
Set_1 = ["Charizard"]
Set_2 = ["Venusaur"]
Set_3 = ["Kingdra"]
Set_4 = set(["Charizard", "Venusaur"])
Set_5 = set(["Charizard", "Kingdra"])
Set_6 = set(["Venusaur", "Kingdra"])

sets = {'Set_1': Set_1,
        'Set_2': Set_2,
        'Set_3': Set_3,
        'Set_4': Set_4,
        'Set_5': Set_5,
        'Set_6': Set_6
}
teams = {}

for key, val in sets.items():
    topic_pers = prob_topic(graph, val)
    map__node_id__node_pagerank_value = nx.pagerank(graph, alpha=0.33,
                                                    personalization=topic_pers)
    top_k__node_id__node_pagerank_value = compute_top_k(map__node_id__node_pagerank_value,
                                                        remove_pkmn=val,
                                                        k=team_size - len(val))
    # pp.pprint(top_k__node_id__node_pagerank_value)
    # top_k__node_id__node_pagerank_value contains the pokemon of the compute_top_k without the pokemons in the topic
    teams['team_' + key[-1]] = set([i[0] for i in top_k__node_id__node_pagerank_value] + list(val))

print('Sets of Pokemons with Charizard')
print(teams['team_1'])
print('\n###########################\n')
print('Sets of Pokemons with Venusaur')
print(teams['team_2'])
print('\n###########################\n')
print('Sets of Pokemons with Kingdra')
print(teams['team_3'])
print('\n###########################\n')
print('Sets of Pokemons with Charizard and Venusaur')
print(teams['team_4'])
print('\n###########################\n')
print('Sets of Pokemons with Charizard and Kingdra')
print(teams['team_5'])
print('\n###########################\n')
print('Sets of Pokemons with Venusaur and Kingdra')
print(teams['team_6'])
print('\n###########################')
print('###########################\n')

print('Number of team members inside the Team(Charizard, Venusaur) '
      'that are neither in Team(Charizard) nor in Team(Venusaur)')
print(len(teams['team_4'].difference(teams['team_1'].union(teams['team_2']))))
print('\n###########################\n')
print('Number of team members inside the Team(Charizard, Kingdra) '
      'that are neither in Team(Charizard) nor in Team(Kingdra)')
print(len(teams['team_5'].difference(teams['team_1'].union(teams['team_3']))))
print('\n###########################\n')
print('Number of team members inside the Team(Venusaur, Kingdra) '
      'that are neither in Team(Venusaur) nor in Team(Kingdra)')
print(len(teams['team_6'].difference(teams['team_2'].union(teams['team_3']))))
print('\n###########################')
print('###########################\n')