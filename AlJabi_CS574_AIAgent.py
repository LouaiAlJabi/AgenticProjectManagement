import matplotlib.pyplot as plt
import networkx as nx
import google.generativeai as genai
import re
import tkinter as tk
from tkinter import messagebox

key = "REPLACE WITH YOUR API KEY"
genai.configure(api_key=key)

class MicrostepGenerator:

    def __init__(self, model_name='gemini-2.5-flash-lite'):
        """
        Args:
            model_name: Which Gemini model to use (I used the free one with the most daily requests)
        """
        self.model = genai.GenerativeModel(model_name)

    def createSteps(self, task):

        # Build prompt with full context
        prompt = self.build_prompt(task)

        try:
            response = self.model.generate_content(prompt)
            steps = response.text.strip()
            return steps

        except Exception as e:
            return f"gemini error: {e}"

    def build_prompt(self, task):
        prompt = f"""You're specializing in improving students' productivity.
        Analyze the task below, then break it down into microsteps following the format below. 
        Also, list which step comes after which as shown in the format below. 
        
        The task: {task} 
        
        Required format:

        - microsteps: list of (id, label, type)
        * id:    internal node id (e.g. "s1")
        * label: text shown on the node
        * type:  "start", "process", "decision", or "end"
        - transitions: list of (source_id, target_id, label)
        * label: edge label like "Yes", "No", or "" 
    
        Microsteps Example: "[("start", "Start", "start"), ("s1", "Read \nassignment", "process"), ("s2", "Break into \nmicrosteps", "process"), ("d1", "All steps \nclear?", "decision"), ("s3", "Clarify unclear \nsteps", "process"), ("end", "Done", "end")]" 
        
        Transition example: "[("start", "s1", ""), ("s1", "s2", ""), ("s2", "d1", ""), ("d1", "s3", "No"), ("d1", "end", "Yes")]" 

        The microsteps must be at most 30 characters in length. Don't include special characters anywhere. Every condition should have two options. Do not have "Understand Prompt" as your first microstep
        
        Think step by step then return the microsteps and the transition in their respective formats always as follows:
        
        Microsteps: [], Transitions: []
        
        """

        return prompt
    
class DrawFlowchart:
    """
    Class to build and draw a flowchart from microsteps + transitions.

    - microsteps: list of (id, label, type)
        * id:    internal node id (e.g. "s1")
        * label: text shown on the node
        * type:  "start", "process", "decision", or "end"
    - transitions: list of (source_id, target_id, label)
        * label: edge label like "Yes", "No", or ""
    """

    def __init__(self, microsteps, transitions, x_step=3.0, y_step=-2.0, base_size=6):
        self.microsteps = microsteps
        self.transitions = transitions
        self.x_step = x_step
        self.y_step = y_step
        self.base_size = base_size

        self.G = self.build_flowchart_graph()
        self.pos = None  # will be set by a layout method later

    # Build the graph
    def build_flowchart_graph(self):
        G = nx.DiGraph()
        for sid, label, stype in self.microsteps:
            G.add_node(sid, label=label, ntype=stype)
        for u, v, elabel in self.transitions:
            G.add_edge(u, v, label=elabel)
        return G

    # Layout using topological_generations (cycle-safe) LEFT → RIGHT with zigzag
    def flowchart_layout(self, spring_seed=42, zigzag_amplitude=3.0):
        """
        Returns a dict: {node_id: (x, y)} with a LEFT-TO-RIGHT layered layout
        that zigzags vertically to avoid a straight, cramped horizontal line.

        - Uses a copy of G (H) for layout.
        - Breaks cycles in H until it becomes a DAG.
        - Uses topological_generations(H) as columns.
        - Each column gets its own vertical baseline that alternates up/down.
        """
        G = self.G
        base_x_step = self.x_step if self.x_step is not None else 3.0
        base_y_step = abs(self.y_step) if self.y_step != 0 else 2.0

        H = G.copy()

        # Break cycles until H is a DAG
        while True:
            try:
                cycle = nx.find_cycle(H, orientation="original")
            except nx.NetworkXNoCycle:
                break
            u, v, _ = cycle[-1]
            H.remove_edge(u, v)

        # Topological generations on cycle-free copy
        try:
            generations = list(nx.topological_generations(H))
        except nx.NetworkXUnfeasible:
            self.pos = nx.spring_layout(G, seed=spring_seed)
            return self.pos

        pos = {}

        num_layers = len(generations)

        # scale x_step with number of columns so they spread more if there are many
        x_scale = 1.0 + 0.15 * max(0, num_layers - 4)
        x_scale = min(x_scale, 3.0)
        x_step_eff = base_x_step * x_scale

        for layer_index, layer in enumerate(generations):
            layer = sorted(layer)
            x = layer_index * x_step_eff

            n = len(layer)

            # --- ZIGZAG BASELINE ---
            # Things weren't spacing properly, needed to introduce some verticality
            # Alternate baseline up/down: +A, -A, +A, -A, ...
            baseline_y = zigzag_amplitude * ((-1) ** layer_index)

            if n == 1:
                # Single node in this column: just put it on the baseline
                node = layer[0]
                pos[node] = (x, baseline_y)
            else:
                # Multiple nodes in this column: stack them vertically around baseline
                spacing_y = base_y_step
                column_height = spacing_y * (n - 1)
                # center column around baseline_y
                y_start = baseline_y - column_height / 2.0

                for i, node in enumerate(layer):
                    y = y_start + i * spacing_y
                    pos[node] = (x, y)

        self.pos = pos
        return pos

    # Node-size scaling per step
    def compute_node_sizes(self):
        """
        Compute a node_size (in points^2) for each node,
        based on its label length and type.
        """
        G = self.G
        labels = nx.get_node_attributes(G, "label")
        types = nx.get_node_attributes(G, "ntype")

        node_sizes = {}

        for n in G.nodes():
            text = labels.get(n, "")
            text_len = max(len(text), 1)
            ntype = types.get(n, "process")

            # Base sizes and scaling factors per type
            if ntype == "start":
                base = 2500
                factor = 80
            elif ntype == "decision":
                base = 2000
                factor = 70
            elif ntype == "end":
                base = 2500
                factor = 80
            else:  # "process" or anything else
                base = 3000
                factor = 90

            size = base + factor * text_len
            node_sizes[n] = size

        return node_sizes

    # Square figure size based on layout complexity
    def estimate_figure_size(self, pos=None):
        """
        Estimate a square figure size dynamically based on:
        - label lengths
        - number of layers
        - widest layer
        Returns (side, side).
        """
        if pos is None:
            pos = self.pos
        if pos is None:
            raise ValueError("Position dictionary is not set. Call a layout method first.")

        G = self.G
        base_size = self.base_size

        labels = nx.get_node_attributes(G, "label")
        layers = {}

        # Average label length
        if labels:
            avg_chars = sum(len(lbl) for lbl in labels.values()) / len(labels)
        else:
            avg_chars = 10  # fallback

        # Count nodes per layer (grouped by y-coordinate)
        for node, (x, y) in pos.items():
            layers.setdefault(y, []).append(node)

        num_layers = len(layers) if layers else 1
        max_layer_size = max((len(nodes) for nodes in layers.values()), default=1)

        # "Complexity score" that controls scaling
        score = (
            base_size
            + num_layers * 2.2
            + max_layer_size * 2.5
            + avg_chars * 0.15
        )

        side = max(score, base_size)
        return (side, side)

    # Main draw function
    def draw(self, title="Flowchart", save_path="networkChart.png"):
        """
        Draw the flowchart, and saves it into the current directory
        """
        G = self.G

        if self.pos is None:
            # Use our custom LR layered layout
            self.flowchart_layout()

        pos = self.pos

        node_types = nx.get_node_attributes(G, "ntype")
        node_labels = nx.get_node_attributes(G, "label")

        start_nodes    = [n for n, t in node_types.items() if t == "start"]
        process_nodes  = [n for n, t in node_types.items() if t == "process"]
        decision_nodes = [n for n, t in node_types.items() if t == "decision"]
        end_nodes      = [n for n, t in node_types.items() if t == "end"]

        # per-node sizes
        node_sizes = self.compute_node_sizes()

        # square, auto-scaled figure size
        fig_size = self.estimate_figure_size(pos)
        plt.figure(figsize=fig_size)

        # Start
        if start_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=start_nodes,
                node_shape="o",
                node_color="#a1d99b",
                node_size=[node_sizes[n] for n in start_nodes],
                edgecolors="black"
            )

        # Process
        if process_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=process_nodes,
                node_shape="s",
                node_color="#9ecae1",
                node_size=[node_sizes[n] for n in process_nodes],
                edgecolors="black"
            )

        # Decision
        if decision_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=decision_nodes,
                node_shape="D",
                node_color="#fdd0a2",
                node_size=[node_sizes[n] for n in decision_nodes],
                edgecolors="black"
            )

        # End
        if end_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=end_nodes,
                node_shape="o",
                node_color="#fc9272",
                node_size=[node_sizes[n] for n in end_nodes],
                edgecolors="black",
                linewidths=2
            )

        # Edges + labels
        nx.draw_networkx_edges(
        G,
        pos,
        edgelist=G.edges(),
        arrows=True,               
        arrowstyle="-|>",          
        arrowsize=25,               
        width=2.5,
        connectionstyle="arc3,rad=0.2",  # curved lines or not (higher rad -> curved | 0 is straight)
        
        min_source_margin=40,       
        min_target_margin=40        
    )

        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

        edge_labels = nx.get_edge_attributes(G, "label")
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5,connectionstyle="arc3,rad=0.2",font_size=10)

        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


def add_newline(task_list):
    new_list = []
    for a, b, c in task_list:
        # replace only the first space in the 2nd element
        new_b = b.replace(" ", "\n", 1)
        new_list.append((a, new_b, c))
    return new_list


def parse_response(text: str):
    """
    Robust parser for microsteps and transitions.
    Returns two lists of tuples (id, text, type_or_label).
    """

    micro_block = re.search(r"Microsteps:\s*\[(.*?)\]", text, re.DOTALL)
    trans_block = re.search(r"Transitions:\s*\[(.*?)\]", text, re.DOTALL)

    if not micro_block or not trans_block:
        raise ValueError("Could not find microsteps or transitions section.")

    micro_raw = micro_block.group(1)
    trans_raw = trans_block.group(1)

    tuple_pattern = r"\(([^()]*?(?:\([^()]*\)[^()]*)*?)\)"
    micro_matches = re.findall(tuple_pattern, micro_raw, re.DOTALL)
    trans_matches = re.findall(tuple_pattern, trans_raw, re.DOTALL)

    def parse_tuple_string(s: str):
        """
        Safely parse a tuple string like:
        '"s7", "Draft Body (1)", "process"'
        into ('s7', 'Draft Body (1)', 'process').
        """

        # Split at commas that are NOT inside quotes
        parts = []
        current = []
        inside_quotes = False

        for char in s:
            if char == '"' and (not current or current[-1] != "\\"):
                inside_quotes = not inside_quotes
                current.append(char)
            elif char == ',' and not inside_quotes:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)

        # Add last part
        parts.append("".join(current).strip())

        # Strip surrounding quotes
        clean_parts = [p.strip().strip('"') for p in parts]
        return tuple(clean_parts)

    # Parse both lists
    microsteps = [parse_tuple_string(m) for m in micro_matches]
    microsteps = add_newline(microsteps)
    transitions = [parse_tuple_string(t) for t in trans_matches]

    return microsteps, transitions

def create_flowchart(task):
    agent = MicrostepGenerator()
    agent_data = agent.createSteps(task)
    microsteps, transitions = parse_response(agent_data)
    flow = DrawFlowchart(microsteps, transitions)
    flow.draw(title="Microsteps Flowchart", save_path="microsteps_flowchart.png")


def draw_graph(user_input):
    plt.plot([1,2,3],[1,2,3])
    plt.title(f"Graph for: {user_input}")
    plt.show()

def FlowchartCreator():
    def on_run_button():
        user_text = text_box.get("1.0", "end").strip()
        if not user_text:
            messagebox.showwarning("No input", "Please enter something.")
            return
        try:
            create_flowchart(user_text)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.minsize(400, 400)
    root.title("Flowchart Launcher")
    
    greeting = """
    Welcome! This Flowchart Generator reads the task you provide in the textbox below, then it will give you
    a flowchart showing the microsteps you could take to complete your task. This is meant to help you 
    improve your productivity and get you started on your tasks. A PNG of your flowchart will be saved to 
    your current directory (rename it before generating a new flowchart to avoid overwriting)
    """
    tk.Label(root, text=greeting).pack(pady=5)
    tk.Label(root, text="Please Enter Your Task:").pack(pady=5)
    text_box = tk.Text(root, height=10, width=40)
    text_box.pack(pady=5)

    tk.Button(root, text="Generate Flowchart", command=on_run_button).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    FlowchartCreator()