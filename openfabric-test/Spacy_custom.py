import spacy
from functools import lru_cache

nlp = spacy.load('en_core_web_sm')



# we train a custom entity base for scientific words


skills = [
    {'label': 'scientific', 'pattern': [{"LOWER": "science"}],'id': 'science1'},
    {'label': 'scientific', 'pattern': [{"LOWER": "biology"}], 'id': 'science2'},
    {'label': 'scientific', 'pattern': [{"LOWER": "physics"}], 'id': 'science3'},
    {'label': 'scientific', 'pattern': [{"LOWER": "robottics"}], 'id': 'science4'},
    {'label': 'scientific', 'pattern': [{"LOWER": "animal kingdom"}], 'id': 'science5'},
    {'label': 'scientific', 'pattern': [{"LOWER": "skeleton"}], 'id': 'science6'},
    {'label': 'scientific', 'pattern': [{"LOWER": "pregnancy"}, {"LOWER": "digestion"}], 'id': 'science7'},
    {'label': 'scientific', 'pattern': [{"LOWER": "expiration"}], 'id': 'science8'},
    {'label': 'scientific', 'pattern': [{"LOWER": "photosynthesis"}, {"LOWER": "flow"}], 'id': 'science9'},
    {'label': 'scientific', 'pattern': [{"LOWER": "reflection"}],'id': 'science10'},
    {'label': 'scientific', 'pattern': [{"LOWER": "atom"}],'id': 'science11'},
    {'label': 'scientific', 'pattern': [{"LOWER": "matter"}],'id': 'science12'},
    {'label': 'scientific', 'pattern': [{"LOWER": "mass"}],'id': 'science13'},
    {'label': 'scientific', 'pattern': [{"LOWER": "gravity"}],'id': 'science14'},
    {'label': 'scientific', 'pattern': [{"LOWER": "weight"}],'id': 'science15'},
    {'label': 'scientific', 'pattern': [{"LOWER": "velocity"}],'id': 'science16'},
    {'label': 'scientific', 'pattern': [{"LOWER": "speed"}],'id': 'science17'},
    {'label': 'scientific', 'pattern': [{"LOWER": "force"}],'id': 'science18'},
    {'label': 'scientific', 'pattern': [{"LOWER": "pressure"}],'id': 'science19'},
    {'label': 'scientific', 'pattern': [{"LOWER": "area"}],'id': 'science20'},
    {'label': 'scientific', 'pattern': [{"LOWER": "volume"}],'id': 'science21'},
    {'label': 'scientific', 'pattern': [{"LOWER": "wave length"}],'id': 'science22'},
    {'label': 'scientific', 'pattern': [{"LOWER": "frequency"}],'id': 'science23'},
    {'label': 'scientific', 'pattern': [{"LOWER": "momentum"}],'id': 'science24'},
    {'label': 'scientific', 'pattern': [{"LOWER": "inertia"}],'id': 'science25'},
    
    ]

ruler = nlp.add_pipe('entity_ruler', before='ner')
ruler.add_patterns(skills)

sent ='  how is the area calculated'
t = nlp(sent)
for x in t.ents:
    print(x.label, x.text, x.ent_id)

