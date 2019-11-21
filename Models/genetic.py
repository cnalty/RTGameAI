

def select_agents(agent_results, fitness_model, percent):
    num_agents = int(len(agent_results) * percent)
    best_agents = []
    for i in range(len(agent_results)):
        if len(best_agents) < num_agents:
            best_agents.append((i, fitness_model(agent_results[i])))
        else:
            list.sort(best_agents, key=lambda x: x[1])
            curr_score = fitness_model(agent_results)
            if curr_score > best_agents[0][1]:
                list.remove(0)
                list.append((i, curr_score))

    return best_agents

def crossover_agents(agent_models):
    pass

def mutate_agents(agent_models):
    pass
