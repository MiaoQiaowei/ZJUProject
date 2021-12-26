import dowhy
from dowhy import CausalModel
import dowhy.datasets
import pandas as pd
import cdt
from cdt import SETTINGS
import pandas as pd

# from IC.test import G
SETTINGS.verbose=True
#SETTINGS.NJOBS=16
#SETTINGS.GPU=1
import networkx as nx
import matplotlib.pyplot as plt
plt.axis('off')

# Load data
data = pd.read_csv("D:\\CODE\\TianChi\\mqw\\TianChiOCT\\IC\\TrainingAnnotation.csv")
data = data.dropna()
# CST,IRF,SRF,PED,HRF
data['VA_'] = data['VA']-data['preVA']
data['CST_'] = data['CST']-data['preCST']
data['IRF_'] = data['IRF']-data['preIRF']
data['SRF_'] = data['SRF']-data['preSRF']
data['PED_'] = data['PED']-data['prePED']
data['HRF_'] = data['HRF']-data['preHRF']


# Finding the structure of the graph
glasso = cdt.independence.graph.Glasso()
skeleton = glasso.predict(data)

# Pairwise setting
model = cdt.causality.graph.GES()
output_graph = model.predict(data, skeleton)

# Visualize causality graph
options = {
        "node_color": "#A0CBE2",
        "width": 1,
        "node_size":400,
        "edge_cmap": plt.cm.Blues,
        "with_labels": True,
    }
plt.figure()
nx.draw_networkx(output_graph,**options)
plt.tight_layout(pad=0.4, w_pad=5, h_pad=5)
plt.savefig('look.png')
nx.write_gml(output_graph, "graph.gml")

# import networkx as nx
# g = nx.read_gml("graph.gml")
# print(g.edges())
# print(len(g.edges()))
# # print(g.nodes())
# # print(g['anti-VEGF'])
# # for node in output_graph.nodes():
# #     print(node)
# #     print(output_graph[node])
# g.remove_edge('continue injection','diagnosis')
# g.remove_edge('SRF_','diagnosis')
# print(g.edges())
# print(len(g.edges()))



data = dowhy.datasets.linear_dataset(
    beta=10,
    num_common_causes=5,
    num_instruments=2,
    num_samples=10000,
    treatment_is_binary=True)
# print(data)
# print(data['common_causes_names'])

G = '''
digraph  {
    diagnosis -> d;
    anti_VEGF -> d;
    continue_injection -> d;
    age -> d;
    age -> diagnosis;
    gender -> d;
    gender -> diagnosis
}
'''
# print(G)
# print(G.replace('\n',' '))

csv = pd.read_csv('D:\\CODE\\TianChi\\mqw\\TianChiOCT\\IC\\test.csv')
csv = csv.dropna()

csv['continue_injection'] = csv['continue injection']
csv['anti_VEGF'] = csv['anti-VEGF']
csv['continue_injection'] = csv['continue_injection'].astype(bool)
csv['d'] = csv['VA']-csv['preVA']
print(csv)
model = CausalModel(
    data=csv,
    treatment=['continue_injection'],
    outcome=['anti_VEGF','diagnosis'],
    common_causes=['gender','age'],
    graph=G.replace('\n',' ')
    )

# visualize the graph
model.view_model(layout="dot")
# print(model)
# from IPython.display import Image,display
# from IPython.display im
# from IPython.display import *
# display(Image(filename="causal_model.png"))

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)
estimate = model.estimate_effect(identified_estimand, 
                                method_name="backdoor.propensity_score_matching",
                                target_units="att")
print(estimate)
# refutation = model.refute_estimate(identified_estimand, estimate,              method_name="placebo_treatment_refuter",placebo_type="permute", num_simulations=20)
# print(refutation)
refutation = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter",
                     placebo_type="permute", num_simulations=20)
print(refutation)