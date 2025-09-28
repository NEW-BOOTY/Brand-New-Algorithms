# Copyright © 2025 Devin B. Royal. All Rights Reserved.

import networkx as nx
import logging
logging.basicConfig(level=logging.INFO)

def dependency_aware_scheduler(tasks, dependencies):
    graph = nx.DiGraph()
    for task in tasks:
        graph.add_node(task['id'])
    for task, deps in dependencies.items():
        for dep in deps:
            graph.add_edge(dep, task)
    
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Cycle detected")
    
    scheduled = []
    for task_id in nx.topological_sort(graph):
        task = next(t for t in tasks if t['id'] == task_id)
        try:
            task['run']()
            scheduled.append(task)
        except Exception as e:
            logging.error(f"Failure {task['id']}: {e}")
            # Reorder: remove failed node
            graph.remove_node(task_id)
            return dependency_aware_scheduler([t for t in tasks if t['id'] != task_id], dependencies)
    return scheduled