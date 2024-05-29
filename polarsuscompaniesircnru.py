import pandas as pd
import plotly.express as px

# import altair as alt
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# from st_aggrid import AgGrid, GridUpdateMode, JsCode
# from st_aggrid.grid_options_builder import GridOptionsBuilder
import networkx as nx
import igraph as ig
from streamlit_plotly_events import plotly_events
import math
import plotly.io as pio
import altair as alt
import pickle
import pydeck as pdk
import os
import polars as pl
from langchain.chat_models import ChatOpenAI

from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from itertools import combinations
from pydeck.types import String
from typing import Dict, Any
import warnings

warnings.filterwarnings("ignore")


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["MAPBOX_TOKEN"] = st.secrets["MAPBOX_TOKEN"]
MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]


llm = ChatOpenAI(
    model_name="gpt-4o",  # 'gpt-3.5-turbo', # 'text-davinci-003' , 'gpt-3.5-turbo'
    temperature=0.3,
    max_tokens=600,
)

article_template = """
I want you to act as a scientific consultant to help intelligence 
analysts understand the if a given paper poses any kind of concern to 
United States security. 
Act like a Systems Engineering and Technical Assistance (SETA) consultant. 
The summary from you is based on article title, article abstract, the list
of authors, and the list of affiliations. 

Return a brief but detailed description of the scientific topic and applications related to
the scientific field desribed by the title, abstract, list of authors, and the list
of author affiliations. The description should be meaningful to an
new intelligence analyst. Highlight typical applications. Highlight any dual use technologies that may be of concern to the United States
Government. 

What is a good summary of the scientific paper with  title {article_title} and abstract {article_abstract}?
Take into account the list of authors {author_list} and list of affiliations {affiliation_list}. Highlight especially
any collaborations between affiliations in different countries. 
Provide the summary in about 300 words or less. 
Please end with a complete sentence.
"""

prompt_article = PromptTemplate(
    input_variables=[
        "article_title",
        "article_abstract",
        "author_list",
        "affiliation_list",
    ],
    template=article_template,
)


chain_article = LLMChain(llm=llm, prompt=prompt_article)


def get_article_llm_description(title: str, abstract: str, authors: list, affils: list):
    """
    takes in the key_phrases list, and the article title
    and returns the openai returned description.
    """
    authors = "; ".join(authors)
    affils = "; ".join(affils)
    #   return chain_article.invoke(article_title=title,article_abstract=abstract,
    #                          author_list=authors, affiliation_list=affils )

    return chain_article.run(
        article_title=title,
        article_abstract=abstract,
        author_list=authors,
        affiliation_list=affils,
    )


topic_template = """
I want you to act as a naming consultant for scientific topics based on keyphrases.
Act like a Systems Engineering and Technical Assistance (SETA) consultant. 

Return a brief but detailed description of the scientific topic and applications related to
the scientific field desribed by the list of keyphrases. The description should be meaningful to an
new intelligence analyst. Highlight typical applications. Highlight any dual use technologies that may be of concern to the United States
Government.

What is a good summary of the scientific topic related to {topic_phrases}?
Provide the summary in about 180 words. 
Please end with a complete sentence.
"""


prompt_topic = PromptTemplate(
    input_variables=["topic_phrases"],
    template=topic_template,
)

chain_topic = LLMChain(llm=llm, prompt=prompt_topic)


def get_topic_llm_description(key_phrases: list):
    """
    takes in the key_phrases list
    and returns the openai returned description.
    """
    topic_phrases = ", ".join(key_phrases)
    return chain_topic.run(topic_phrases=topic_phrases)


pio.templates.default = "plotly_dark"
st.set_page_config(layout="wide")

st.markdown(
    """
A sample of recent papers US-based companies collaborating with CN, IR, and RU affiliations.
"""
)

st.write("Topic modeling")


# kw_dict = dfinfo['keywords'].to_dict()


color_education = [228, 26, 28]
color_facility = [55, 126, 184]
color_government = [77, 175, 74]
color_other = [152, 78, 163]
color_nonprofit = [255, 127, 0]
color_company = [255, 255, 51]
color_healthcare = [166, 86, 40]
color_archive = [247, 129, 191]


fill_color_dict = {
    "education": color_education,
    "facility": color_facility,
    "government": color_government,
    "company": color_company,
    "nonprofit": color_nonprofit,
    "other": color_other,
    "healthcare": color_healthcare,
    "archive": color_archive,
}


@st.cache_data()
def load_centroids_asat():
    # dg = pd.read_csv("penguins.csv", engine="pyarrow")
    #  df = pd.read_json(df.to_json())
    # dg = pd.read_pickle('updatejammingcentroids2d.pkl.gz')
   # pl_centroids = pl.read_parquet("updatejammingcentroids2d.parquet", use_pyarrow=True)
    # return dg
    pl_centroids = pl.read_parquet("updatejammingcentroids2d.parquet", use_pyarrow=True)
    pl_centroids = pl_centroids.filter(pl.col("cluster") != -1)
    return pl_centroids


#@st.cache_data()
def load_dftriple_asat():
    pl_dftriple = pl.read_parquet("updatejammingdftriple2d.parquet", use_pyarrow=True)
   # pl_dftriple = pl.scan_parquet("updatejammingdftriple2d.parquet")
   # pl_dftriple = pl_dftriple.filter(pl.col("paper_cluster") != -1)
  #  pl_dftriple = pl_dftriple.with_columns(
  #      pl.col("type")
  #      .replace(fill_color_dict, return_dtype=pl.List(pl.Int64))
  #      .alias("fill_color")
  #  )
  #  pl_dftriple = pl_dftriple.with_columns(
  #      pl.col("fill_color").map_elements(lambda x: x[0]).alias("r"),
  #      pl.col("fill_color").map_elements(lambda x: x[1]).alias("g"),
  #      pl.col("fill_color").map_elements(lambda x: x[1]).alias("b"),
  #  )
    return pl_dftriple


@st.cache_data()
def load_dfinfo_asat():
    pl_dfinfo = pl.read_parquet("updatejammingdfinfo2d.parquet", use_pyarrow=True)
    pl_dfinfo = pl_dfinfo.filter(pl.col("cluster") != -1)
    pl_dfinfo = pl_dfinfo.with_columns(
        pl.col("cluster").cast(pl.String).alias("cluster_")
    )
    return pl_dfinfo


@st.cache_data()
def load_source_dict():
    with open("updatesource_page_dict.pkl", "rb") as f:
        source_dict = pickle.load(f)
    return source_dict


@st.cache_data()
def load_affil_geo_dict():
    with open("updateaffil_geo_dict.pkl", "rb") as f:
        affil_geo_dict = pickle.load(f)
    return affil_geo_dict


# @st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


#st.button("Rerun")  # https://discuss.streamlit.io/t/using-streamlit-cache-with-polars/38000/7

centroids = load_centroids_asat()
dftriple = load_dftriple_asat()


#dftriple = dftriple.with_columns(
#        pl.col("type")
#        .replace(fill_color_dict, return_dtype=pl.List(pl.Int64))
#        .alias("fill_color")
#    )
#dftriple = dftriple.with_columns(
#        pl.col("fill_color").map_elements(lambda x: x[0]).alias("r"),
#        pl.col("fill_color").map_elements(lambda x: x[1]).alias("g"),
#        pl.col("fill_color").map_elements(lambda x: x[1]).alias("b"),
#    )







dfinfo = load_dfinfo_asat()
source_dict = load_source_dict()
affil_geo_dict = load_affil_geo_dict()



def df_to_dict(df: pl.DataFrame,key_col: str,value_col: str) -> Dict[Any,Any]:
    """
    Get a Python dict from two columns of a DataFrame
    If the key column is not unique, the last row is used
    """
    return dict(df.select(key_col,value_col).iter_rows())



kw_dict = df_to_dict(dfinfo, "id", "keywords")

#st.write(kw_dict['https://openalex.org/W3041133507'])

#st.write(centroids.columns)


# add in the affiliations as nodes as well; that row, author, paper, affil. all three get links. ok.
def create_nx_graph(df: pl.dataframe.frame.DataFrame, cl:int) -> nx.Graph:
    """
    takes the dataframe df, and creates the undirected graph
    from the source and target columns for each row.
    """
    g = nx.Graph() # dc['paper_cluster'] == cl
    #dc = df[df['paper_cluster'] == cl]
    dc = df.filter(pl.col("paper_cluster") == cl).to_pandas()
    
    author_counts = dc['paper_author_id'].tolist()
    author_counts_dict = {c:author_counts.count(c) for c in author_counts}
    affiliation_counts = dc['id'].tolist()
    affiliation_counts_dict = {c:affiliation_counts.count(c) for c in affiliation_counts}
    source_counts = dc['source'].tolist()
    source_counts_dict = {c:source_counts.count(c) for c in source_counts}
    funder_counts = [x for row in dc['funder_list'].tolist() for x in row]
    funder_counts_dict = {c:funder_counts.count(c) for c in funder_counts}
    for index, row in dc.iterrows():
        g.add_node(row['paper_id'], group='work', title=row['paper_title'])
        g.add_node(row['paper_author_id'], title=row['paper_author_display_name'],
                   group='author',value = author_counts_dict[row['paper_author_id']])
        try:
            g.add_node(row['id'], group='affiliation',
                   title=row['display_name'] + '\n' + row['country_code'],
                  value = affiliation_counts_dict[row['id']])
        except:
            g.add_node(row['id'], group='affiliation',
                   title=row['display_name'],
                  value = affiliation_counts_dict[row['id']]) 
        if row['source']:
            g.add_node(row['source'], group=row['source_type'],
                      title=row['source'] + ' :\n ' + row['source_type'],
                      value=source_counts_dict[row['source']])
            g.add_edge(
                row['paper_id'],
                row['source'],
            )
            g.add_edge(
                row['paper_author_id'],
                row['source'],
            )
            g.add_edge(
                row['id'],
                row['source'],
            )
        if len(row['funder_list']) > 0:
            for f in row['funder_list']:
                g.add_node(f, group='funder',
                          title=str(f),
                          value=funder_counts_dict[f])
                g.add_edge(
                       row['paper_id'],
                       f,
                   )
                g.add_edge(
                       f,
                       row['paper_author_id'],
                   )
                g.add_edge(
                       f,
                       row['id'],
                   )  
                if row["source"]:
                    g.add_edge(
                        f,
                        row["source"],
                    )
        g.nodes[row['paper_id']]['title'] = (
            row['paper_title'] + ' :\n ' + str(row['paper_publication_date'] + ':\n' + 
            '\n'.join(kw_dict[row['paper_id']])) # kw_dict is the mapping in dfinfo between (id)  paper_id and keywords. ok. 
        )
        g.nodes[row['paper_author_id']]['title'] = (
            row['paper_author_display_name']
        )
        g.add_edge(
            row['paper_id'],
            row['paper_author_id'],
        )
        g.add_edge(
            row['paper_author_id'],
            row['id'],
        )
        g.add_edge(
            row['paper_id'],
            row['id'],
        )
        
    g_ig = ig.Graph.from_networkx(g) # assign 'x', and 'y' to g before ret
    layout = g_ig.layout_umap(min_dist = 2, epochs = 500)
    # https://igraph.org/python/tutorial/0.9.6/visualisation.html
    coords = layout.coords
    allnodes = list(g.nodes())
    coords_dict = {allnodes[i]:(coords[i][0], coords[i][1]) for i in range(len(allnodes))}
    for i in g.nodes():
        g.nodes[i]['x'] = 250 * coords_dict[i][0] # the scale factor needed 
        g.nodes[i]['y'] = 250 * coords_dict[i][1]
    return g






def optimized_create_nx_graph(df: pl.DataFrame, cluster_id:int) -> nx.Graph:
    """
    Creates a NetworkX graph from a filtered subset of the input DataFrame.
    
    Parameters:
    - df: A Polars DataFrame containing academic paper data.
    - cluster_id: An integer representing the cluster ID to filter the DataFrame.
    
    Returns:
    - A NetworkX graph constructed from the filtered DataFrame.
    """
    # Filter the DataFrame for the specified cluster ID
    filtered_df = df.filter(pl.col("paper_cluster") == cluster_id)
    
    # Initialize the NetworkX graph
    g = nx.Graph()
    
    # Add nodes and edges based on the filtered DataFrame
    for _, row in filtered_df.iterrows():
        # Adding nodes for papers, authors, and affiliations
        g.add_node(row['paper_id'], group='work', title=row['paper_title'])
        g.add_node(row['paper_author_id'], group='author', title=row['paper_author_display_name'])
        g.add_node(row['id'], group='affiliation', title=row['display_name'] + '\n' + row['country_code'])
        
        # Adding edges based on relationships defined in the DataFrame
        # Assuming 'source' and 'funder_list' contain identifiers that can be used as nodes
        if row['source']:
            source_node = row['source']
            g.add_edge(row['paper_id'], source_node, type='published_in')
            g.add_edge(row['paper_author_id'], source_node, type='authors_of')
            g.add_edge(row['id'], source_node, type='affiliated_with')
        
        if row['funder_list']:
            for funder in row['funder_list']:
                g.add_edge(row['paper_id'], funder, type='funded_by')
                g.add_edge(row['paper_author_id'], funder, type='authors_of_funded')
                g.add_edge(row['id'], funder, type='affiliated_with_funded')
    
    return g







def create_pyvis_html(cl: int, filename: str = "pyvis_coauthorships_graph.html"):
    """
    wrapper function that calls create_nx_graph to finally 
    produce an interactive pyvis standalone html file
    """
    g_nx = create_nx_graph(dftriple, cl) # cn cahnge back to orginal funciotn
    h = Network(height="1000px",
                width="100%",
                cdn_resources="remote", 
                bgcolor="#222222",
            neighborhood_highlight=True,
                font_color="white",
                directed=False,
                filter_menu=True,
                notebook=False,
               )
    h.from_nx(g_nx, show_edge_weights=False)
    neighbor_map = h.get_adj_list()
    h.set_options(
    """
const options = {
  "interaction": {
    "navigationButtons": false
  },
 "physics": {
     "enabled": false
 },
  "edges": {
    "color": {
        "inherit": true
    },
    "setReferenceSize": null,
    "setReference": {
        "angle": 0.7853981633974483
    },
    "smooth": {
        "forceDirection": "none"
    }
  }
  }
    """
    )
    
    try:
        path = './tmp'
        h.save_graph(f"{path}/{filename}")
        HtmlFile = open(f"{path}/{filename}","r",
                        encoding='utf-8')
    except:
        h.save_graph(f"{filename}")
        HtmlFile = open(f"{filename}", "r",
                        encoding="utf-8")
    return HtmlFile


st.dataframe(centroids.to_pandas()[['cluster','x','y','concepts','keywords']])
csv_topics = convert_df(centroids.to_pandas()[['cluster','x','y','concepts','keywords']])
st.download_button(
   "Press to Download Topics Table",
   csv_topics,
   "topics.csv",
   "text/csv",
   key='download-topics-csv'
)

def get_fig_asat():
    dfcentroids = centroids.to_pandas()
    fig_centroids = px.scatter(dfcentroids[dfcentroids.cluster != -1],
                           x='x',y='y',
                    color_discrete_sequence=['pink'],
                          hover_data=['x','y',
                                      'wrapped_keywords',
                                      'wrapped_concepts','cluster'])
    fig_centroids.update_traces(marker=dict(size=12,
                              line=dict(width=.5,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    pddfinfo = dfinfo.to_pandas()
    fig_papers = px.scatter(pddfinfo[pddfinfo.cluster != -1],
                           x='x',y='y',
                    color='cluster_',
                        hover_data = ['title','cluster',
                                      'publication_date'])
                     #     hover_data=['title','x','y',
                     #                 'z','cluster','wrapped_author_list',
                     #                 'wrapped_affil_list',
                     #                 'wrapped_keywords'])
    fig_papers.update_traces(marker=dict(size=4,
                              line=dict(width=.5,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    layout = go.Layout(
        autosize=True,
        width=1000,
        height=1000,

        #xaxis= go.layout.XAxis(linecolor = 'black',
         #                 linewidth = 1,
         #                 mirror = True),

        #yaxis= go.layout.YAxis(linecolor = 'black',
         #                 linewidth = 1,
         #                 mirror = True),

        margin=go.layout.Margin(
            l=10,
            r=10,
            b=10,
            t=10,
            pad = 4
            )
        )

    fig3 = go.Figure(data=fig_papers.data + fig_centroids.data)
    fig3.update_layout(height=700)

                   # layout=layout)  
    return fig3


bigfig = get_fig_asat()


st.subheader("Papers and Topics")
st.write("Use the navigation tools in the mode bar to pan and zoom. Papers are automatically clustered into subtopics. Topics are the bigger pink dots with representative keywords and phrases available on hover. Clicking on a topic or paper then triggers a report of the most profilic countries, affiliations, and authors associated with that topic.")
selected_point = plotly_events(bigfig, click_event=True, override_height=700)
if len(selected_point) == 0:
    st.write("Select a paper or cluster")
    st.stop()
    
#st.write(selected_point)

selected_x_value = selected_point[0]["x"]
selected_y_value = selected_point[0]["y"]


try:
   # df_selected = dfinfo[
   #     (dfinfo["x"] == selected_x_value)
   #     & (dfinfo["y"] == selected_y_value)
   # ]
    df_selected = dfinfo.filter(
        (pl.col("x") == selected_x_value) & (pl.col("y") == selected_y_value)
            ).to_pandas()
    selected_cluster = df_selected['cluster'].iloc[0]
    article_keywords = df_selected['keywords'].to_list()[0]
    article_title = df_selected['title'].iloc[0]
    article_abstract = df_selected['abstract'].iloc[0]
    article_authors = df_selected['author_list'].iloc[0]
    article_affils = df_selected['affil_list'].iloc[0]
    llm_article_description = get_article_llm_description(article_title, article_abstract,
                article_authors,  article_affils)
    st.write(f"Selected Article")
    st.data_editor(
        df_selected[['x', 'y', 'id', 'title', 'doi', 'cluster', 
       'publication_date', 'keywords', 'top_concepts', 'affil_list',
       'author_list','probability']],
        column_config={
            "doi": st.column_config.LinkColumn("doi"),
            "id": st.column_config.LinkColumn("id")
        },
        hide_index=True,
        )
    st.write(llm_article_description)
#st.write(topic_keywords)
except:
    #pass
    dfkluster = centroids.filter(
        (pl.col("x") == selected_x_value) & (pl.col("y") == selected_y_value)
            ).to_pandas()
    selected_cluster_list = dfkluster['cluster'].to_list()
    if selected_cluster_list:
        selected_cluster = selected_cluster_list[0]

df_selected_centroid = centroids.filter(
    (pl.col("cluster") == selected_cluster)
).to_pandas()


df_selected_papers = dfinfo.filter(
    (pl.col("cluster") == selected_cluster)
).to_pandas().sort_values('probability',ascending=False)
st.write(f"selected topic {selected_cluster}")
st.dataframe(df_selected_centroid[['concepts','keywords','x','y']])


csv_selected_centroid = convert_df(df_selected_centroid[['concepts','keywords',
                                                         'x','y']])
st.download_button(
   "Press to Download Selected Topic",
   csv_selected_centroid,
   "selected_topic.csv",
   "text/csv",
   key='download-selected-topic-csv'
)


topic_keywords = df_selected_centroid['keywords'].to_list()[0]
#st.write(topic_keywords)
llm_topic_description = get_topic_llm_description(topic_keywords)
st.write(llm_topic_description)


st.write(f"publications in topic {selected_cluster}")
st.data_editor(
        df_selected_papers[['x', 'y', 'id', 'title', 'doi', 'cluster', 
       'publication_date', 'keywords', 'top_concepts', 'affil_list',
       'author_list','probability']],
        column_config={
            "doi": st.column_config.LinkColumn("doi"),
            "id": st.column_config.LinkColumn("id")
        },
        hide_index=True,
        )

csv_selected_papers = convert_df(df_selected_papers[['x', 'y', 'id', 'title', 'doi', 'cluster', 
       'publication_date', 'keywords', 'top_concepts', 'affil_list',
       'author_list','probability']])

st.download_button(
   f"Press to Download Selected Papers for topic {selected_cluster}",
   csv_selected_papers,
   f"selected_papers_{selected_cluster}.csv",
   "text/csv",
   key='download-selected-topic-papers-csv'
)

def get_country_cluster_sort(dc:pl.DataFrame, cl:int):
    """
    restricts the dataframe dc to cluster value cl
    and returns the results grouped by id, ror sorted
    by the some of probablity descending
    """
    dg = dc.filter(
        (pl.col("paper_cluster") == cl)
    ).to_pandas()
    #dg = dc[dc['paper_cluster'] == cl].copy()
   # print(cl)
    dv = dg.groupby(['country_code'])['paper_cluster_score'].sum().to_frame()
    dv.sort_values('paper_cluster_score', ascending=False, inplace=True)
    return dv, df_selected_centroid['keywords'].iloc[0]



def new_get_country_cluster_sort(dc: pl.DataFrame, cl: int):
    """
    Restricts the dataframe dc to cluster value cl
    and returns the results grouped by country_code, 
    sorted by the sum of paper_cluster_score in descending order.
    Additionally, returns the first keyword from df_selected_centroid
    corresponding to the selected centroid.
    """
    # Filter the DataFrame based on the specified cluster value
    filtered_dc = dc.filter((pl.col("paper_cluster") == cl)).collect()
    
    # Group by 'country_code', sum 'paper_cluster_score', and sort the result
    grouped_and_sorted = filtered_dc.groupby('country_code').agg([
        pl.sum('paper_cluster_score').alias('total_paper_cluster_score')
    ]).sort('total_paper_cluster_score', descending=True)
    
    return grouped_and_sorted, df_selected_centroid['keywords'].iloc[0]



def get_affils_cluster_sort(dc:pl.DataFrame, cl:int):
    """
    restricts the dataframe dc to cluster value cl
    and returns the results grouped by id, ror sorted
    by the some of probablity descending
    """
    # https://learning.oreilly.com/library/view/streamlit-for-data/9781803248226/text/ch004.xhtml
    dg = dc.filter(
        (pl.col("paper_cluster") == cl) 
    ).to_pandas()
    #dg = dc[dc['paper_cluster'] == cl].copy()
    print(cl)
    dv = dg.groupby(['id','display_name','country_code',
                     'type','r','g','b'])['paper_cluster_score'].sum().to_frame()
    dv.sort_values('paper_cluster_score', ascending=False, inplace=True)
    dv.reset_index(inplace=True) # map the display_name column with the geo_dict to get lattitude, longitude
    dv['latitude'] = dv['display_name'].apply(lambda x: affil_geo_dict.get(x, (None, None))[0])
    dv['longitude'] = dv['display_name'].apply(lambda x: affil_geo_dict.get(x, (None, None))[1])
    kw = df_selected_centroid['keywords'].iloc[0]
    return dv, kw



def get_author_cluster_sort(dc:pl.DataFrame, cl:int):
    """
    restricts the dataframe dc to cluster value cl
    and returns the results grouped by id, ror sorted
    by the some of probablity descending
    """
    dg = dc.filter(
        (pl.col("paper_cluster") == cl)
    ).to_pandas()
   # dg = dc[dc['paper_cluster'] == cl].copy()
   # print(cl)
    dv = dg.groupby(['paper_author_id','paper_author_display_name',
                    'display_name',
                     'country_code'])['paper_cluster_score'].sum().to_frame()
    dv.sort_values('paper_cluster_score', ascending=False, inplace=True)
    dv.reset_index(inplace=True)
    return dv, df_selected_centroid['keywords'].iloc[0]


def get_journals_cluster_sort(dc:pl.DataFrame, cl:int):
    """
    restricts the dataframe dc to cluster value cl
    and returns the results grouped by source (where
    source_type == 'journal') sorted
    by the some of probablity descending
    """
    dg = dc.filter(
        (pl.col("paper_cluster") == cl)
    ).to_pandas()
    #dg = dc[dc['paper_cluster'] == cl].copy()
    print(cl)
    dv = dg[dg['source_type'] == 'journal'].groupby(['source'])['paper_cluster_score'].sum().to_frame()
    dv.sort_values('paper_cluster_score', ascending=False, inplace=True)
    dv['journal'] = dv.index
    dv['homepage_url'] = dv['journal'].map(source_dict)
    kw = df_selected_centroid['keywords'].iloc[0]
    return dv[['journal','homepage_url','paper_cluster_score']], kw



def get_conferences_cluster_sort(dc:pl.DataFrame, cl:int):
    """
    restricts the dataframe dc to cluster value cl
    and returns the results grouped by source (where
    source_type == 'journal') sorted
    by the some of probablity descending
    """
    dg = dc.filter(
        (pl.col("paper_cluster") == cl)
    ).to_pandas()
    #dg = dc[dc['paper_cluster'] == cl].copy()
    print(cl)
    dv = dg[dg['source_type'] == 'conference'].groupby(['source'])['paper_cluster_score'].sum().to_frame()
    dv.sort_values('paper_cluster_score', ascending=False, inplace=True)
    dv['conference'] = dv.index
   # dv['homepage_url'] = dv['conference'].map(source_dict)
    kw = df_selected_centroid['keywords'].iloc[0]
    return dv, kw



def get_country_collaborations_sort(dc:pl.DataFrame, cl:int):
    """
    resticts the dataframe dc to cluster value cl
    and returns the results of paper_id s where there is 
    more than one country_code
    """
    dg = dc.filter(
        (pl.col("paper_cluster") == cl)
    ).to_pandas()
    #dg = dc[dc['paper_cluster'] == cl].copy()
    dv = dg.groupby('paper_id')['country_code'].apply(lambda x: len(set(x.values))).to_frame()
    dc = dg.groupby('paper_id')['country_code'].apply(lambda x: list(set(x.values))).to_frame()
    dc.columns = ['collab_countries']
    dv.columns = ['country_count']
    dv['collab_countries'] = dc['collab_countries']
    dv.sort_values('country_count',ascending=False, inplace=True)
    di = dfinfo.to_pandas().set_index('id', drop=False)
    #di = dfinfo.loc[dv.index].copy()  # this is no good with polars. ok. 
    di['country_count'] = dv['country_count']
    di['collab_countries'] = dv['collab_countries']
    return di[di['country_count'] > 1]



def get_time_series(dg:pl.DataFrame, cl:int):
    """
    takes dg and the cluster number cl
    and returns a time series chart
    by month, y-axis is the article count
    """
    dftime = dg.filter(
        (pl.col("cluster") == cl)
    ).to_pandas()
    dftime = dftime[['cluster','probability','publication_date']].copy()
   # dftime = dg[dg.cluster == cl][['cluster','probability','publication_date']].copy()
    dftime['date'] = pd.to_datetime(dftime['publication_date'])
    dftime.sort_values('date', inplace=True)
    #by_month = pd.to_datetime(dftime['date']).dt.to_period('M').value_counts().sort_index()
    #by_month.index = pd.PeriodIndex(by_month.index)
    #df_month = by_month.rename_axis('month').reset_index(name='counts')
    return dftime


def generate_subsets(lst):
    return sorted(list(combinations(lst, 2)))



def get_pydeck_chart(dh:pd.DataFrame):
    """
    takes the dataframe dg (dvaffils)
    and returns a pydeck chart
    """
    dg = dh.copy()
    dg = dg.dropna(subset=["longitude","latitude"])
    dg = pd.read_json(dg.to_json())

    mean_lat = dg['latitude'].mean()
    mean_lon = dg['longitude'].mean()
    cl_initial_view = pdk.ViewState(
        latitude = dg['latitude'].iloc[0],
        longitude = dg['longitude'].iloc[0],
        zoom = 11
    )
    sp_layer = pdk.Layer(
        'ScatterplotLayer',
        data = dg,
        get_position = ['longitude','latitude'],
        get_radius = 300
    )
    return cl_initial_view, sp_layer


tab1, tab2, tab3, tab4 , tab5, tab6, tab7, tab8, tab9= st.tabs(["Countries", "Affiliations", "Authors",
                                        "Journals","Conferences",
 "Coauthorship Graph", "Country-Country Collaborations",
                    "time evolution of topic","Affiliation Map"])


dvauthor, kwwuathor = get_author_cluster_sort(dftriple, selected_cluster)
#st.dataframe(dvauthor)

dfcollab = get_country_collaborations_sort(dftriple, selected_cluster)

dvaffils, kwwaffils = get_affils_cluster_sort(dftriple, selected_cluster)
        
dc, kw = get_country_cluster_sort(dftriple, selected_cluster)


dvjournals, kwjournals = get_journals_cluster_sort(dftriple, selected_cluster)

dvconferences, kwconferences = get_conferences_cluster_sort(dftriple, selected_cluster)

htmlfile = create_pyvis_html(selected_cluster)

dftime = get_time_series(dfinfo, selected_cluster)


with tab1:
    st.dataframe(dc)

with tab2:
    st.markdown("highlight and click a value in the **id** column to be given more information")
    st.dataframe(
        dvaffils,
        column_config={
            "id": st.column_config.LinkColumn("id"),
        },
        hide_index=True,
    )
    csv_dvaffils = convert_df(dvaffils)
    st.download_button(
       f"Press to Download Affiliations for topic {selected_cluster}",
       csv_dvaffils,
       f"affils_{selected_cluster}.csv",
       "text/csv",
       key='download-affils-csv'
    )

with tab3:
    st.write("highlight and click a value in the **paper_author_id** to be given more information")
    st.dataframe(
        dvauthor,
        column_config={
            "paper_author_id": st.column_config.LinkColumn("paper_author_id")
        },
        hide_index=True,
    )
    csv_dvauthor = convert_df(dvauthor)
    st.download_button(
       f"Press to Download Authors for topic {selected_cluster}",
       csv_dvauthor,
       f"authors_{selected_cluster}.csv",
       "text/csv",
       key='download-authors-csv'
    )


with tab4:
    st.write("Journals most representative of this cluster")
   # st.dataframe(
   #     dvjournals[['journal','paper_cluster_score']],
   #     hide_index=True
   # )
    st.dataframe(
        dvjournals,
        column_config={
            "homepage_url": st.column_config.LinkColumn("homepage_url")
        },
        hide_index=True,
    )
    csv_dvjournals = convert_df(dvjournals)
    st.download_button(
       f"Press to Download Journals for topic {selected_cluster}",
       csv_dvjournals,
       f"journals_{selected_cluster}.csv",
       "text/csv",
       key='download-journals-csv'
    )

with tab5:
    st.write("Conferences most representative of this cluster")
    st.dataframe(
        dvconferences[['conference','paper_cluster_score']],
        hide_index=True
    )
    csv_dvconferences = convert_df(dvconferences)
    st.download_button(
       f"Press to Download Conferences for topic {selected_cluster}",
       csv_dvauthor,
       f"conferences_{selected_cluster}.csv",
       "text/csv",
       key='download-conferences-csv'
    )

###################################################



with tab6:
    st.write("Coauthorship Graph (Papers and Authors)")
    components.html(htmlfile.read(), height=1100)
    





#####################

with tab7:
    st.write("Country-Country Collaborations")
    st.dataframe(
        dfcollab[['x', 'y', 'id','collab_countries', 'title', 'doi', 'cluster', 'probability',
       'publication_date', 'keywords', 'top_concepts', 'affil_list',
       'author_list','funder_list']],
        column_config={
            "doi": st.column_config.LinkColumn("doi"),
        },
        hide_index=True,
    )
    csv_dvcollab = convert_df(dfcollab[['x', 'y', 'id','collab_countries', 'title', 'doi', 'cluster', 'probability',
       'publication_date', 'keywords', 'top_concepts', 'affil_list',
       'author_list','funder_list']])
    st.download_button(
       f"Press to Download Country-Country Collab for topic {selected_cluster}",
       csv_dvcollab,
       f"collab_{selected_cluster}.csv",
       "text/csv",
       key='download-collab-csv'
    )

with tab8:
    alt_chart= alt.Chart(dftime).mark_line().transform_fold(
    ['probability']
        ).encode(
        x = 'yearmonth(date):T',
        y = 'sum(value):Q',
        color='key:N'
    ).interactive()
    st.altair_chart(alt_chart, use_container_width=True)


with tab9:
    dg = dvaffils.copy()
    dg = dg.dropna(subset=["longitude","latitude"])
    dg['size'] = 100*dg['paper_cluster_score']
    dg = pd.read_json(dg.to_json())

    mean_lat = dg['latitude'].mean()
    st.write(dg.head())
    mean_lon = dg['longitude'].mean()
    cl_initial_view = pdk.ViewState(
        latitude = dg['latitude'].mean(),
        longitude = dg['longitude'].mean(),
        zoom = 3
    )
    view = pdk.data_utils.compute_view(dg[["longitude", "latitude"]])
    view.pitch = 75
    view.bearing = 60
   # da = dftriple[dftriple['paper_cluster'] == selected_cluster].copy()
    da = dftriple.filter(
        (pl.col("paper_cluster") == selected_cluster)
    ).to_pandas()
    dv = da.groupby('paper_id')['display_name'].apply(lambda x: len(set(x.values))).to_frame()
    dc = da.groupby('paper_id')['display_name'].apply(lambda x: list(set(x.values))).to_frame()
    dc.columns = ['collab_affils']
    dv.columns = ['affil_count']
    dv['collab_affils'] = dc['collab_affils']
    dv.sort_values('affil_count', ascending=False, inplace=True)
    dv = dv[dv['affil_count'] > 1].copy()
    dv['subsets'] = dv['collab_affils'].apply(generate_subsets)
    flattened_df = dv.explode('subsets').copy()
    flattened_df['source_affil'] = flattened_df['subsets'].apply(lambda x: x[0])
    flattened_df['target_affil'] = flattened_df['subsets'].apply(lambda x: x[1])
    dfarc = flattened_df['subsets'].value_counts(dropna=False).to_frame().copy()
    dfarc.rename(columns={'subsets': 'count'}, inplace=True)
    dfarc['affils'] = dfarc.index
    dfarc['source'] = dfarc['affils'].apply(lambda x: x[0])
    dfarc['target'] = dfarc['affils'].apply(lambda x: x[1])
    dfarc['source_geo'] = dfarc['source'].map(affil_geo_dict)
    dfarc['target_geo'] = dfarc['target'].map(affil_geo_dict)
    pattern = r"(-?\d+\.\d+), (-?\d+\.\d+)"
    dfarc[['source_lat', 'source_lon']] = dfarc['source_geo'].apply(str).str.extract(pattern)
    dfarc[['target_lat', 'target_lon']] = dfarc['target_geo'].apply(str).str.extract(pattern)
    dfarc['source_lon'] = dfarc['source_lon'].apply(float)
    dfarc['source_lat'] = dfarc['source_lat'].apply(float)
    dfarc['target_lon'] = dfarc['target_lon'].apply(float)
    dfarc['target_lat'] = dfarc['target_lat'].apply(float)
    GREEN_RGB = [0, 255, 0]
    RED_RGB = [240, 100, 0]

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=dfarc.dropna(),
        get_width = "count * 2",
        get_source_position = ['source_lon', 'source_lat'],
        get_target_position = ['target_lon','target_lat'],
        get_tilt=0,
        pickable=True,
        get_source_color=RED_RGB,
        get_target_color=GREEN_RGB,
        auto_highlight = True
    )
    
    sp_layer = pdk.Layer(
        'ScatterplotLayer',
        data = dg,
        get_position = ['longitude','latitude'],
        radius_scale = 75,
        radius_min_pixels=5,
        radius_max_pixels=300,
        line_width_min_pixels=1,
        get_radius = "size",
        pickable=True,
        opacity = 0.4,
        get_fill_color = [65, 182, 196]
    )
    affil_layer = pdk.Layer(
        "ColumnLayer",
        data = dg,
        get_position=["longitude","latitude"],
        get_elevation="size",
        elevation_scale = 200,
        radius = 3_000,
        line_width_min_pixels=1,
        get_radius="size",
        get_fill_color=['r','g','b'],
        auto_highlight=True,
        pickable=True,
    )
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=dg,
        opacity=0.8,
        get_position=['longitude','latitude'],
        aggregation=String('MAX'),
        get_weight='paper_cluster_score'
    )
    
    st.pydeck_chart(pdk.Deck(
        layers = [sp_layer, affil_layer, heatmap_layer, arc_layer],
        api_keys = {'mapbox': MAPBOX_TOKEN},
        map_provider='mapbox',
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=view,
        tooltip = {
            "html": "<b>{display_name}</b> <br/> <b>Strength</b>: {paper_cluster_score} <br>" + \
            "<b>source: {source} <br/> <b>target: {target} <br>" + \
            "<b>count: {count} <br/>",
            "style": {
                "backgroundColor": "white",
                "color": "black"
            }
        }
    ))

