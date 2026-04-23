import networkx as nx
import matplotlib.pyplot as plt
import io

class AnalysisGraph:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnalysisGraph, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initializes the graph with a Start node."""
        self.G = nx.DiGraph()
        self.current_node = "Start"
        self.step_counter = 0
        self.G.add_node("Start", label="Start\n(No Data)", shape="box", color="lightgray")

    def add_step(self, action_name, description, parent_node=None):
        """
        Adds a new step to the analysis graph.
        :param parent_node: If None, continues from current_node. If set, branches off.
        """
        self.step_counter += 1
        new_node_id = f"Step_{self.step_counter}"
        
        # Determine source node (allow branching)
        source = parent_node if parent_node else self.current_node
        
        # Safety: if source doesn't exist, default back to current
        if source not in self.G.nodes:
            source = self.current_node

        # Add Node (The resulting state)
        self.G.add_node(new_node_id, label=description, color="lightblue")
        
        # Add Edge (The action taken)
        self.G.add_edge(source, new_node_id, label=action_name)
        
        # Update pointer
        self.current_node = new_node_id
        return new_node_id

    def get_lineage_text(self):
        """Returns a text summary of the path to the current node."""
        try:
            path = nx.shortest_path(self.G, source="Start", target=self.current_node)
            summary = ["\n🔹 **Current Data Lineage:**"]
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                action = self.G[u][v]['label']
                # Clean up label for text summary
                state_desc = self.G.nodes[v]['label'].replace('\n', ' ')
                summary.append(f"{i+1}. {action} -> {state_desc}")
            return "\n".join(summary)
        except:
            return "Lineage: [Complex Branching - See Graph]"

    def visualize(self, output_path="project/data/workflow_state.png"):
        """Draws the graph and saves it."""
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "workflow_graph.png")
        plt.figure(figsize=(10, 6))
        
        # Layout
        try:
            pos = nx.spring_layout(self.G, seed=42, k=1.5) # k regulates distance
        except:
            pos = nx.random_layout(self.G)

        # Draw Nodes
        colors = [self.G.nodes[n].get('color', 'lightblue') for n in self.G.nodes]
        labels = nx.get_node_attributes(self.G, 'label')
        
        nx.draw_networkx_nodes(self.G, pos, node_size=2500, node_color=colors, alpha=0.9)
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=7)
        
        # Draw Edges
        nx.draw_networkx_edges(self.G, pos, arrowstyle='->', arrowsize=20, edge_color="gray")
        edge_labels = nx.get_edge_attributes(self.G, 'label')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)
        
        plt.title(f"Analysis Workflow (Current Step: {self.current_node})")
        plt.axis('off')

        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

# Global Instance
graph_manager = AnalysisGraph()