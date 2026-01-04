import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

##
# This code is a basic demo of some GNN concepts. 
## 

class Gate(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * x2

class EdgeMP(MessagePassing):
    # This class controls message passing.
    # One run of it represents one iteration of message passing
    def __init__(self, node_dim, edge_dim, hidden_dim):
        # We can define the aggregation in the inherited MessagePassing class.
        # We could write our own aggregation function, or use a premade one like sum or mean
        # I use sum here, which aggregation would you choose and why? 
        super().__init__(aggr='sum')  

        # Messages passed will go through this network
        input_edge_mlp = 2 * node_dim + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(input_edge_mlp),
            nn.Linear(input_edge_mlp,  hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Nodes will pass the aggregated messages through this network to update their value
        input_node_mlp = node_dim + hidden_dim
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(input_node_mlp),
            nn.Linear(input_node_mlp,  hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_dim]
        # edge_index: [2, E]
        # edge_attr: [E, edge_dim]

        # Propagate:
        # 1st, it will call message to compute and pass messages
        # 2nd, it will call aggregate, and aggregate the messages for each node
        # 3rd, it will run update, and update the node values
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: target node features
        # x_j: source node features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(msg_input)

    def update(self, aggr_out, x):
        # aggr_out: aggregated messages
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(update_input)



class HitGNN(nn.Module):
    def __init__(self, input_node_dim=5, edge_dim=5, hidden_dim=64, num_layers=3):
        super().__init__()

        # Initial broadcast from the initial input_node_dim to a larger hidden_dim
        # Note that hidden_dim here becomes the node_dim in EdgeMP
        self.input_mlp = nn.Sequential(
            nn.Linear(input_node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Create message passing blocks
        self.message_passing = nn.ModuleList([
            EdgeMP(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Project the output to a single logit for binary classification
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)  
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.input_mlp(x)

        # Run our message passing blocks
        # Note that we have the block learn the residual (e.g. we add in the input)
        for mp in self.message_passing:
            x = x + mp(x, edge_index, edge_attr)  
            x = F.gelu(x)

        logits = self.output_mlp(x)
        # Note, we do not pass the output through a sigmoid as our loss function (BCEWithLogitsLoss) expects logits
        return logits.squeeze(-1)

    def predict(self, data):
        # Since the forward outputs the logits, we can run this method to predict probabilities
        return nn.functional.sigmoid(self.forward(data))