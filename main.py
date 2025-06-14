import streamlit as st
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
import numpy as np
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.manifold import TSNE
import umap.umap_ as umap  
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import praw
import io

# --- Reddit Credentials ---
reddit = praw.Reddit(
    client_id="xkRF4b0_r_daXaPFpZ5z4w",
    client_secret="uCuzkijYIUlNFwir29tCDUSL8sFh4w",
    user_agent="DHANALAKSHMI 3 bot"
)

@st.cache_data(show_spinner=False)
def fetch_reddit_posts(subreddit_name="python", limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for submission in subreddit.hot(limit=limit):
        posts.append({
            "post_id": submission.id,
            "title": submission.title,
            "score": submission.score,
            "author": str(submission.author),
            "created_at": submission.created_utc,
            "url": submission.url,
            "comments": submission.num_comments
        })
    return posts

# --- Streamlit Interface ---
st.set_page_config(layout="wide")
st.sidebar.title("Graph Options")
graph_view = st.sidebar.radio("Graph View", ["Pyvis", "Matplotlib"])

subreddit_name = st.sidebar.text_input("Subreddit", value="python")
limit = st.sidebar.slider("Number of Posts", 10, 200, 100)

st.title("Reddit Anomaly Detection using GNNs")

# --- Load and Build Graph ---
posts = fetch_reddit_posts(subreddit_name=subreddit_name, limit=limit)
G = nx.Graph()
for post in posts:
    user = post.get("author")
    post_id = post.get("post_id")
    if user and post_id:
        G.add_node(user, type='user', description=f"User: {user}")
        G.add_node(post_id, type='post', description=f"Post: {post['title']}")
        G.add_edge(user, post_id)

node_mapping = {node: i for i, node in enumerate(G.nodes())}
edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()
x = torch.tensor([[G.degree(node)] for node in G.nodes()], dtype=torch.float)
pyg_data = Data(x=x, edge_index=edge_index)

# --- GCN Model ---
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(in_channels=1, hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Training ---
def train():
    model.train()
    optimizer.zero_grad()
    out = model(pyg_data)
    loss = torch.mean(torch.var(out, dim=0))
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(50):
    train()

# --- Inference & Anomaly ---
model.eval()
with torch.no_grad():
    node_embeddings = model(pyg_data)
    norms = torch.norm(node_embeddings, dim=1)
    scores = (norms - norms.mean()) / norms.std()

anomaly_df = pd.DataFrame({
    "node": list(G.nodes()),
    "score": scores.numpy(),
    "type": [G.nodes[n].get("type", "N/A") for n in G.nodes()]
})
anomaly_df = anomaly_df.sort_values(by="score", ascending=False)

# --- Embeddings ---
embeddings_np = node_embeddings.numpy()
embedding_df = pd.DataFrame(embeddings_np, index=list(G.nodes()))

projection_method = st.sidebar.radio("Embedding Projection", ["TSNE", "UMAP"])
if projection_method == "TSNE":
    reducer = TSNE(n_components=2, random_state=42)
else:
    reducer = umap.UMAP(n_components=2, random_state=42)

projection = reducer.fit_transform(embeddings_np)
proj_df = pd.DataFrame(projection, columns=["x", "y"])
proj_df["node"] = list(G.nodes())
proj_df["score"] = scores.numpy()
proj_df["type"] = anomaly_df.set_index("node").loc[proj_df["node"]]["type"].values

fig, ax = plt.subplots()
colors = proj_df["type"].map({"user": "skyblue", "post": "salmon"}).fillna("gray")
ax.scatter(proj_df["x"], proj_df["y"], c=colors, s=20, alpha=0.6)
for i, row in proj_df.iterrows():
    if abs(row["score"]) > 2:
        ax.text(row["x"], row["y"], row["node"], fontsize=8)
st.pyplot(fig)

# --- Node Search with Fuzzy Matching ---
st.sidebar.markdown("---")
st.sidebar.subheader("Fuzzy Node Search")
search_query = st.sidebar.text_input("Search node")

if search_query:
    choices = list(G.nodes())
    best_match = process.extractOne(search_query, choices)
    st.sidebar.write(f"Best match: {best_match[0]} (Score: {best_match[1]})")
    selected_node = best_match[0]
else:
    selected_node = st.selectbox("Select Node", anomaly_df["node"])

# --- Pyvis / Matplotlib Graph ---
if graph_view == "Pyvis":
    def draw_pyvis_graph(graph):
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        for node, data in graph.nodes(data=True):
            title = f"{node}<br>{data.get('description', '')}<br>Type: {data.get('type', '')}"
            color = "skyblue" if data.get('type') == 'user' else "salmon"
            net.add_node(node, label=node, title=title, color=color)
        for u, v in graph.edges():
            net.add_edge(u, v)
        net.set_options("""
            var options = {
              "nodes": {"borderWidth": 1, "size": 25, "font": {"size": 14}},
              "edges": {"color": {"inherit": true}, "smooth": false},
              "interaction": {"hover": true, "navigationButtons": true, "keyboard": true},
              "physics": {"enabled": true, "solver": "forceAtlas2Based"}
            }
        """)
        net.save_graph('graph.html')
        components.html(open('graph.html', 'r', encoding='utf-8').read(), height=620)
    draw_pyvis_graph(G)
else:
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    color_map = ["skyblue" if G.nodes[n].get("type") == "user" else "salmon" for n in G.nodes()]
    nx.draw(G, pos, node_color=color_map, with_labels=False, node_size=50)
    plt.title("Matplotlib Graph View")
    st.pyplot(plt)

# --- Node Detail Info ---
st.header("Selected Node Details")
node_info = anomaly_df[anomaly_df["node"] == selected_node].set_index("node")
st.dataframe(node_info)

if selected_node in G:
    neighbors = list(G.neighbors(selected_node))
    st.write(f"**Neighbors of {selected_node}:** {', '.join(neighbors)}")
    neighbor_info = []
    for n in neighbors:
        neighbor_info.append({
            "Node": n,
            "Type": G.nodes[n].get("type", "N/A"),
            "Description": G.nodes[n].get("description", "N/A"),
            "Anomaly Score": anomaly_df[anomaly_df["node"] == n]["score"].values[0] if n in anomaly_df["node"].values else "N/A"
        })
    st.dataframe(pd.DataFrame(neighbor_info))

st.subheader("GNN Embedding for Selected Node")
if selected_node in embedding_df.index:
    st.dataframe(embedding_df.loc[[selected_node]])
